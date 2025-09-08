#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, osr
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # 必须导入此模块才能使用3D绘图
# from matplotlib.colors import LightSource # 恢复更改，移除LightSource

# 创建输出目录
output_dir = 'processed_dem'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 打开GeoTIFF文件
dem_file = 'Data/AW3D30huangshan.tif'
ds = gdal.Open(dem_file)

if ds is None:
    print(f"Cannot open file: {dem_file}")
    exit(1)

# 获取地理变换参数
geotransform = ds.GetGeoTransform()
projection = ds.GetProjection()

# 读取DEM数据
dem_data_original = ds.ReadAsArray()

# 处理无效数据区域：海拔为0的区域视为无效
dem_data_processed = dem_data_original.copy().astype(float)
dem_data_processed[dem_data_original <= 0] = np.nan # 更健壮一些，处理<=0的值

# 获取原始坐标系统信息
origin_x = geotransform[0]
origin_y = geotransform[3]
pixel_width = geotransform[1]
pixel_height = geotransform[5]

source_srs = osr.SpatialReference()
source_srs.ImportFromWkt(ds.GetProjectionRef())

print(f"Original DEM Information:")
print(f"Size: {dem_data_original.shape}")
print(f"Top-left corner coordinates: ({origin_x}, {origin_y})")
print(f"Pixel resolution: {pixel_width} x {pixel_height}")
print(f"Spatial reference: {source_srs.ExportToProj4()}")

# 计算每个像素的经纬度坐标
rows, cols = dem_data_original.shape
lon_grid = np.zeros((rows, cols))
lat_grid = np.zeros((rows, cols))

for i in range(rows):
    for j in range(cols):
        lon_grid[i, j] = geotransform[0] + j * geotransform[1] + i * geotransform[2]
        lat_grid[i, j] = geotransform[3] + j * geotransform[4] + i * geotransform[5]

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance

ref_lat = lat_grid[0, 0]
ref_lon = lon_grid[0, 0]

x_meters = np.zeros((rows, cols))
y_meters = np.zeros((rows, cols))

for i in range(rows):
    y_meters[i, :] = haversine_distance(ref_lat, ref_lon, lat_grid[i, 0], ref_lon)
    if lat_grid[i, 0] < ref_lat:
        y_meters[i, :] *= -1

for j in range(cols):
    x_meters[:, j] = haversine_distance(ref_lat, ref_lon, ref_lat, lon_grid[0, j])
    if lon_grid[0, j] < ref_lon:
        x_meters[:, j] *= -1

x_offset = abs(min(0, np.min(x_meters)))
y_offset = abs(min(0, np.min(y_meters)))

x_meters += x_offset
y_meters += y_offset

np.save(os.path.join(output_dir, 'dem_elevation.npy'), dem_data_processed)
np.save(os.path.join(output_dir, 'dem_x_meters.npy'), x_meters)
np.save(os.path.join(output_dir, 'dem_y_meters.npy'), y_meters)

with open(os.path.join(output_dir, 'dem_info.txt'), 'w') as f:
    f.write(f"Original DEM file: {dem_file}\n")
    f.write(f"Data size: {dem_data_original.shape}\n")
    f.write(f"Original top-left coordinates: ({origin_x}, {origin_y})\n")
    f.write(f"Pixel resolution: {pixel_width} x {pixel_height}\n")
    f.write(f"X range (meters): {x_meters.min()} to {x_meters.max()}\n")
    f.write(f"Y range (meters): {y_meters.min()} to {y_meters.max()}\n")
    f.write(f"Reference point (lat, lon): ({ref_lat}, {ref_lon})\n")
    f.write(f"X offset applied: {x_offset} meters\n")
    f.write(f"Y offset applied: {y_offset} meters\n")
    f.write(f"Elevation range (valid data): {np.nanmin(dem_data_processed):.2f} to {np.nanmax(dem_data_processed):.2f} meters\n")
    f.write(f"Number of NaN (invalid) points: {np.sum(np.isnan(dem_data_processed))}\n")

# --- 2D Elevation Map Modifications ---
plt.figure(figsize=(12, 10))
terrain_cmap = cm.get_cmap('terrain').copy()
terrain_cmap.set_bad(color='lightgray', alpha=0.2) # NaN区域更透明一些

# y_meters: smaller values are at smaller row indices (map top), larger values at larger row indices (map bottom)
# imshow extent: [left, right, bottom, top]
# For origin='upper', array[0,0] is at (left, top) of extent.
# So, top extent value should be y_meters.min(), bottom extent value should be y_meters.max().
img_extent = [x_meters.min(), x_meters.max(), y_meters.max(), y_meters.min()]

im = plt.imshow(dem_data_processed, 
                cmap=terrain_cmap, 
                interpolation='nearest',
                origin='upper', # Array[0,0] is top-left
                extent=img_extent)

plt.colorbar(im, label='Elevation (m)')
plt.title('Elevation Map (Invalid areas marked in gray)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')

# 计算有效数据的边界以限制坐标轴范围
valid_rows_indices, valid_cols_indices = np.where(np.isfinite(dem_data_processed))

if len(valid_rows_indices) > 0 and len(valid_cols_indices) > 0:
    # 直接从x_meters和y_meters中获取有效点的坐标
    x_coords_of_valid_points = x_meters[valid_rows_indices, valid_cols_indices]
    y_coords_of_valid_points = y_meters[valid_rows_indices, valid_cols_indices]

    if x_coords_of_valid_points.size > 0 and y_coords_of_valid_points.size > 0:
        x_min_valid = np.min(x_coords_of_valid_points)
        x_max_valid = np.max(x_coords_of_valid_points)
        y_min_valid = np.min(y_coords_of_valid_points) # Smallest y_meter value among valid points
        y_max_valid = np.max(y_coords_of_valid_points) # Largest y_meter value among valid points

        padding_x = (x_max_valid - x_min_valid) * 0.05 
        padding_y = (y_max_valid - y_min_valid) * 0.05
        
        plt.xlim(x_min_valid - padding_x, x_max_valid + padding_x)
        # For ylim, bottom limit is larger y_meter value, top limit is smaller y_meter value
        plt.ylim(y_max_valid + padding_y, y_min_valid - padding_y)
    else: # Fallback if somehow no valid coords extracted, though previous check should cover this
        pass # Use full extent set by imshow
else:
    pass # Use full extent set by imshow if no valid data points

plt.axis('equal')
# plt.grid(False) # Explicitly ensure no grid, though not strictly needed if not called before
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'elevation_map.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'elevation_map.svg'))
plt.close()

# --- 3D Terrain Model Modifications ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

sample_rate = 5 
x_sample = x_meters[::sample_rate, ::sample_rate]
y_sample = y_meters[::sample_rate, ::sample_rate]
z_sample = dem_data_processed[::sample_rate, ::sample_rate]

# 使用原始的 cmap 和 alpha 进行渲染 (恢复)
surf = ax.plot_surface(x_sample, y_sample, z_sample, cmap=cm.terrain, 
                      linewidth=0, antialiased=True, alpha=0.8, # 稍微增加alpha使其更不透明
                      rstride=1, cstride=1) # rstride/cstride确保所有数据点被绘制

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Elevation (m)')
ax.set_title('3D Terrain Model (Invalid areas removed, Z-axis scaled)')

fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label='Elevation (m)')

# 计算有效数据的范围 (忽略NaN)
x_range_valid = np.nanmax(x_sample) - np.nanmin(x_sample)
y_range_valid = np.nanmax(y_sample) - np.nanmin(y_sample)
z_range_valid = np.nanmax(z_sample) - np.nanmin(z_sample)

# 调整Z轴比例以获得更好的视觉效果
z_scale_factor = 2.0  # 大幅增加Z轴拉伸因子
if x_range_valid > 0 and y_range_valid > 0 and z_range_valid > 0:
    max_xy_range = max(x_range_valid, y_range_valid)
    # Z轴的比例基于其自身范围相对于XY最大范围的比例，再乘以缩放因子
    # 这有助于在不同数据集上保持相对一致的视觉效果
    z_aspect = (z_range_valid / max_xy_range if max_xy_range > 0 else 0.1) * z_scale_factor
    # 限制z_aspect的最小值，避免在z_range_valid很小时过度压扁
    z_aspect = max(z_aspect, 0.3) # 保证Z轴至少有一定的可见高度
    ax.set_box_aspect([x_range_valid/max_xy_range if max_xy_range > 0 else 1, 
                       y_range_valid/max_xy_range if max_xy_range > 0 else 1, 
                       z_aspect])
else:
    ax.set_box_aspect([1,1,0.5]) # 如果没有有效数据或范围为0，则使用一个更大的默认Z轴比例

ax.view_init(elev=30, azim=-60)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3d_terrain_model.png'), dpi=300)
plt.savefig(os.path.join(output_dir, '3d_terrain_model.svg'))
plt.close()

print(f"Processing complete, results saved to directory: {output_dir}") 