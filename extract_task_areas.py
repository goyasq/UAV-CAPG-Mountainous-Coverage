#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from scipy import ndimage
import shutil
from scipy.interpolate import RectBivariateSpline
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

# Satellite image style color mapping (enhanced for better visibility)
satellite_colors = [
    (0.1, 0.3, 0.8),        # 鲜明蓝色（水体）
    (0.0, 0.5, 0.0),        # 深绿色（森林）
    (0.2, 0.7, 0.2),        # 中绿色（植被）
    (0.5, 0.8, 0.3),        # 浅绿色（草地）
    (0.8, 0.7, 0.4),        # 土黄色（裸地）
    (0.7, 0.5, 0.3),        # 棕色（岩石）
    (0.9, 0.9, 0.9)         # 白色（雪/云）
]
satellite_cmap = LinearSegmentedColormap.from_list('satellite_enhanced', satellite_colors, N=256)

# Input and base output directory
input_dir = 'processed_dem'
base_output_dir = 'task_areas_diverse'

# Task area dimensions
TASK_AREA_SIZE_METERS = 1000  # 1km x 1km
UPSCALE_FACTOR = 5
MIN_SEPARATION_PIXELS = 100 # Minimum separation between centers of selected task areas in pixel units

# Different complexity categories with their thresholds
COMPLEXITY_CATEGORIES = {
    'high': {
        'min_std': 200.0,
        'max_std': 300.0,
        'min_gradient': 0.08,
        'count': 5,
        'description': 'High Variability (Mountainous)'
    },
    'medium': {
        'min_std': 100.0,
        'max_std': 200.0,
        'min_gradient': 0.04,
        'count': 5,
        'description': 'Medium Variability (Hilly)'
    },
    'low': {
        'min_std': 30.0,
        'max_std': 100.0,
        'min_gradient': 0.02,
        'count': 5,
        'description': 'Low Variability (Rolling Hills)'
    },
    'minimal': {
        'min_std': 10.0,
        'max_std': 30.0,
        'min_gradient': 0.01,
        'count': 5,
        'description': 'Minimal Variability (Nearly Flat)'
    }
}
plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman'], 'font.size':18})

# Function to calculate pixel dimensions for a given area size
def get_pixel_dims(target_size_m, x_m, y_m):
    if x_m.shape[0] < 2 or x_m.shape[1] < 2:
        return 10, 10 # Default if DEM is too small
    pixel_height_m = np.abs(y_m[1, 0] - y_m[0, 0])
    pixel_width_m = np.abs(x_m[0, 1] - x_m[0, 0])
    
    rows_for_task = int(target_size_m / pixel_height_m) if pixel_height_m > 0 else 10
    cols_for_task = int(target_size_m / pixel_width_m) if pixel_width_m > 0 else 10
    return rows_for_task, cols_for_task

# Function to extract, process, and save a single task area
def process_and_save_task_area(dem_data_full_res, x_meters_full_res, y_meters_full_res, 
                               center_i, center_j, 
                               area_idx, complexity_value, complexity_category,
                               original_full_dem_shape):
    
    filename_prefix = f"area_{complexity_category}_{area_idx:02d}_std_{complexity_value:.1f}"
    print(f"Processing task area {filename_prefix} centered at original indices ({center_i}, {center_j})...")

    # Determine extraction window based on 1km x 1km
    task_rows, task_cols = get_pixel_dims(TASK_AREA_SIZE_METERS, x_meters_full_res, y_meters_full_res)
    
    half_rows = task_rows // 2
    half_cols = task_cols // 2

    i_min = max(0, center_i - half_rows)
    i_max = min(original_full_dem_shape[0], center_i + half_rows + (task_rows % 2))
    j_min = max(0, center_j - half_cols)
    j_max = min(original_full_dem_shape[1], center_j + half_cols + (task_cols % 2))

    current_rows = i_max - i_min
    current_cols = j_max - j_min

    if current_rows < task_rows:
        if i_min == 0: i_max = min(original_full_dem_shape[0], i_min + task_rows)
        else: i_min = max(0, i_max - task_rows)
            
    if current_cols < task_cols:
        if j_min == 0: j_max = min(original_full_dem_shape[1], j_min + task_cols)
        else: j_min = max(0, j_max - task_cols)
    
    i_min = max(0, i_min); i_max = min(original_full_dem_shape[0], i_max)
    j_min = max(0, j_min); j_max = min(original_full_dem_shape[1], j_max)

    print(f"  Extracting from DEM with original indices: rows {i_min}-{i_max}, cols {j_min}-{j_max}")

    roi_dem_original_res = dem_data_full_res[i_min:i_max, j_min:j_max].copy()
    roi_x_original_res = x_meters_full_res[i_min:i_max, j_min:j_max].copy()
    roi_y_original_res = y_meters_full_res[i_min:i_max, j_min:j_max].copy()

    if roi_dem_original_res.size == 0 or np.all(np.isnan(roi_dem_original_res)):
        print(f"  WARNING: Extracted ROI for {filename_prefix} is empty or all NaN. Skipping.")
        return None # Indicate failure

    print(f"  Extracted ROI shape (original res): {roi_dem_original_res.shape}")

    x_offset = roi_x_original_res.min()
    y_offset = roi_y_original_res.min()
    
    rect_min_x_orig_coords = x_meters_full_res[i_min, j_min]
    rect_min_y_orig_coords = y_meters_full_res[i_min, j_min] 
    
    actual_width_m = roi_x_original_res.max() - roi_x_original_res.min()
    actual_height_m = roi_y_original_res.max() - roi_y_original_res.min()
    
    print(f"  Upscaling DEM for {filename_prefix} by factor {UPSCALE_FACTOR}x...")
    orig_rows, orig_cols = roi_dem_original_res.shape
    new_rows, new_cols = orig_rows * UPSCALE_FACTOR, orig_cols * UPSCALE_FACTOR

    y_orig_grid = np.linspace(0, orig_rows - 1, orig_rows)
    x_orig_grid = np.linspace(0, orig_cols - 1, orig_cols)
    
    roi_dem_filled = roi_dem_original_res.copy()
    nan_mask = np.isnan(roi_dem_filled)
    if np.any(nan_mask):
        if np.all(nan_mask):
            print(f"  CRITICAL: ROI for {filename_prefix} is all NaN before upscaling. Skipping.")
            return None
        else:
            valid_dem_mean = np.nanmean(roi_dem_filled)
            if np.isnan(valid_dem_mean): valid_dem_mean = 0
            roi_dem_filled[nan_mask] = valid_dem_mean

    interp_func = RectBivariateSpline(y_orig_grid, x_orig_grid, roi_dem_filled, kx=1, ky=1)
    y_new_grid = np.linspace(0, orig_rows - 1, new_rows)
    x_new_grid = np.linspace(0, orig_cols - 1, new_cols)
    roi_dem_upscaled = interp_func(y_new_grid, x_new_grid)

    upscaled_x_coords_normalized = np.linspace(0, actual_width_m, new_cols)
    upscaled_y_coords_normalized = np.linspace(0, actual_height_m, new_rows)
    roi_x_upscaled_norm_mesh, roi_y_upscaled_norm_mesh = np.meshgrid(upscaled_x_coords_normalized, upscaled_y_coords_normalized)

    print(f"  Upscaled DEM shape: {roi_dem_upscaled.shape}")

    np.save(os.path.join(base_output_dir, f'{filename_prefix}_dem.npy'), roi_dem_upscaled)
    np.save(os.path.join(base_output_dir, f'{filename_prefix}_x_meters.npy'), roi_x_upscaled_norm_mesh)
    np.save(os.path.join(base_output_dir, f'{filename_prefix}_y_meters.npy'), roi_y_upscaled_norm_mesh)
    
    with open(os.path.join(base_output_dir, f'{filename_prefix}_info.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Task Area Identifier: {filename_prefix}\\n")
        f.write(f"Elevation Variability Category: {complexity_category} ({COMPLEXITY_CATEGORIES[complexity_category]['description']})\\n")
        f.write(f"Elevation Variability (Std Dev of Elevation in 1km raw ROI): {complexity_value:.2f} m\\n")
        f.write(f"Original DEM source: {input_dir}\\n")
        f.write(f"Selected Center (Original DEM Indices i,j): ({center_i}, {center_j})\\n")
        f.write(f"Selected Center (Original DEM Coordinates X,Y): ({x_meters_full_res[center_i, center_j]:.2f}, {y_meters_full_res[center_i, center_j]:.2f}) m\\n")
        f.write(f"Selected Center Elevation (Original DEM): {dem_data_full_res[center_i, center_j]:.2f} m\\n")
        f.write(f"Extracted Area (Original DEM Indices i_min,i_max, j_min,j_max): ({i_min}, {i_max}, {j_min}, {j_max})\\n")
        f.write(f"Extracted Area Size (Original Resolution): {roi_dem_original_res.shape[1]} pixels x {roi_dem_original_res.shape[0]} pixels\\n")
        f.write(f"Extracted Area Size (Meters approx): {actual_width_m:.2f}m x {actual_height_m:.2f}m\\n")
        f.write(f"X Offset for Normalization: {x_offset:.2f} m\\n")
        f.write(f"Y Offset for Normalization: {y_offset:.2f} m\\n")
        f.write(f"Upscaled DEM shape: {roi_dem_upscaled.shape}\\n")
        f.write(f"Upscaled DEM Elevation Range: {np.nanmin(roi_dem_upscaled):.2f} to {np.nanmax(roi_dem_upscaled):.2f} m\\n")

    plt.figure(figsize=(8, 7))
    extent_2d = [roi_x_upscaled_norm_mesh.min(), roi_x_upscaled_norm_mesh.max(), 
                 roi_y_upscaled_norm_mesh.max(), roi_y_upscaled_norm_mesh.min()]
    plt.imshow(roi_dem_upscaled, cmap=satellite_cmap, origin='upper', extent=extent_2d, interpolation='nearest')
    plt.colorbar(label='Elevation (m)')
    plt.title(f'Task Area: {filename_prefix}\\nUpscaled Elevation Map ({complexity_category.title()} Variability)')
    plt.xlabel('X (m, normalized)')
    plt.ylabel('Y (m, normalized)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, f'{filename_prefix}_2d.png'), dpi=150)
    plt.close()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    sample_rate_3d = max(1, new_rows // 100, new_cols // 100)
    X_3d = roi_x_upscaled_norm_mesh[::sample_rate_3d, ::sample_rate_3d]
    Y_3d = roi_y_upscaled_norm_mesh[::sample_rate_3d, ::sample_rate_3d]
    Z_3d = roi_dem_upscaled[::sample_rate_3d, ::sample_rate_3d]
    ax.plot_surface(X_3d, Y_3d, Z_3d, cmap=satellite_cmap, linewidth=0, antialiased=True, alpha=0.9)
    ax.set_title(f'Task Area: {filename_prefix}\\n3D View ({complexity_category.title()} Variability)')
    ax.set_xlabel('X (m, normalized)')
    ax.set_ylabel('Y (m, normalized)')
    ax.set_zlabel('Elevation (m)')
    ax.zaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
    ax.tick_params(axis='z', labelsize=8)
    x_data_span = np.nanmax(X_3d) - np.nanmin(X_3d)
    y_data_span = np.nanmax(Y_3d) - np.nanmin(Y_3d)
    Z_VISUAL_STRETCH_FACTOR = 0.8 
    if x_data_span > 0 and y_data_span > 0:
        max_xy_span = max(x_data_span, y_data_span)
        aspect_x = x_data_span / max_xy_span
        aspect_y = y_data_span / max_xy_span
        aspect_z = Z_VISUAL_STRETCH_FACTOR 
        ax.set_box_aspect((aspect_x, aspect_y, aspect_z))
    else:
        ax.set_box_aspect((1, 1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, f'{filename_prefix}_3d.png'), dpi=150)
    plt.close()
    
    print(f"  Successfully processed and saved data for {filename_prefix}")
    return rect_min_x_orig_coords, rect_min_y_orig_coords, actual_width_m, actual_height_m, filename_prefix, complexity_category


def main():
    if os.path.exists(base_output_dir):
        shutil.rmtree(base_output_dir)
    os.makedirs(base_output_dir)

    print("Loading original DEM data from 'processed_dem'...")
    dem_data_full = np.load(os.path.join(input_dir, 'dem_elevation.npy'))
    x_meters_full = np.load(os.path.join(input_dir, 'dem_x_meters.npy'))
    y_meters_full = np.load(os.path.join(input_dir, 'dem_y_meters.npy'))

    print(f"Loaded full DEM data with shape: {dem_data_full.shape}")

    print("Calculating gradient magnitude for the full DEM...")
    dem_for_gradient = dem_data_full.copy()
    nan_mask_full = np.isnan(dem_for_gradient)
    if np.any(nan_mask_full):
        if np.all(nan_mask_full):
            print("ERROR: Full DEM is all NaN. Cannot proceed.")
            return
        dem_mean_for_fill = np.nanmean(dem_for_gradient)
        if np.isnan(dem_mean_for_fill): dem_mean_for_fill = 0 
        dem_for_gradient[nan_mask_full] = dem_mean_for_fill
        
    gradient_y, gradient_x = np.gradient(dem_for_gradient)
    gradient_magnitude_full = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude_full[nan_mask_full] = np.nan

    task_area_rows_px, task_area_cols_px = get_pixel_dims(TASK_AREA_SIZE_METERS, x_meters_full, y_meters_full)
    print(f"Target task area size in pixels: {task_area_rows_px} rows x {task_area_cols_px} cols")
    
    stride_rows = task_area_rows_px // 4
    stride_cols = task_area_cols_px // 4

    # Collect candidates for each complexity category
    complexity_candidates = {category: [] for category in COMPLEXITY_CATEGORIES.keys()}
    
    print("Scanning DEM for candidate task areas by complexity...")
    for r_start in range(0, dem_data_full.shape[0] - task_area_rows_px + 1, stride_rows):
        for c_start in range(0, dem_data_full.shape[1] - task_area_cols_px + 1, stride_cols):
            r_end = r_start + task_area_rows_px
            c_end = c_start + task_area_cols_px

            center_i = r_start + task_area_rows_px // 2
            center_j = c_start + task_area_cols_px // 2

            if np.isnan(dem_data_full[center_i, center_j]):
                continue

            window_dem_original = dem_data_full[r_start:r_end, c_start:c_end]
            window_gradient = gradient_magnitude_full[r_start:r_end, c_start:c_end]

            num_valid_points_in_window = np.sum(~np.isnan(window_dem_original))
            if num_valid_points_in_window / window_dem_original.size < 0.75:
                continue 
            
            mean_gradient_in_window = np.nanmean(window_gradient)
            std_dev_elevation_in_window = np.nanstd(window_dem_original)

            if np.isnan(mean_gradient_in_window) or np.isnan(std_dev_elevation_in_window):
                continue

            # Categorize by complexity
            for category, criteria in COMPLEXITY_CATEGORIES.items():
                if (criteria['min_std'] <= std_dev_elevation_in_window <= criteria['max_std'] and 
                    mean_gradient_in_window >= criteria['min_gradient']):
                    complexity_candidates[category].append({
                        'std_dev': std_dev_elevation_in_window,
                        'mean_grad': mean_gradient_in_window,
                        'center_i': center_i,
                        'center_j': center_j
                    })
                    break  # Only assign to first matching category

    # Print candidate statistics
    for category, candidates in complexity_candidates.items():
        print(f"Found {len(candidates)} candidates for {category} complexity")

    selected_task_areas_info = []
    
    # Select areas from each complexity category
    for category, criteria in COMPLEXITY_CATEGORIES.items():
        candidates = complexity_candidates[category]
        if not candidates:
            print(f"Warning: No candidates found for {category} complexity")
            continue
            
        # Sort by standard deviation (higher complexity first within category)
        sorted_candidates = sorted(candidates, key=lambda x: x['std_dev'], reverse=True)
        
        selected_count = 0
        for cand in sorted_candidates:
            if selected_count >= criteria['count']:
                break

            cand_ci, cand_cj = cand['center_i'], cand['center_j']
            is_separated = True
            for existing_area_info in selected_task_areas_info:
                _, _, _, _, _, _, proc_ci, proc_cj = existing_area_info 
                dist_sq = (cand_ci - proc_ci)**2 + (cand_cj - proc_cj)**2
                if dist_sq < MIN_SEPARATION_PIXELS**2:
                    is_separated = False
                    break
            
            if is_separated:
                area_idx = selected_count + 1
                complexity_val = cand['std_dev']
                print(f"Selecting {category} complexity Area {area_idx}: center_i={cand_ci}, center_j={cand_cj}, std_dev={complexity_val:.2f}")
                
                rect_info_tuple = process_and_save_task_area(
                    dem_data_full, x_meters_full, y_meters_full, 
                    cand_ci, cand_cj, area_idx, complexity_val, category, dem_data_full.shape
                )
                if rect_info_tuple is not None:
                     rect_x, rect_y, rect_w, rect_h, f_prefix, cat = rect_info_tuple
                     selected_task_areas_info.append((rect_x, rect_y, rect_w, rect_h, f_prefix, cat, cand_ci, cand_cj))
                     selected_count += 1
                else:
                    print(f"  Processing failed for {category} candidate at ({cand_ci},{cand_cj}). Trying next.")

    if selected_task_areas_info:
        print("Creating summary plot of diverse task area locations...")
        plt.figure(figsize=(14, 10))
        img_extent_full = [x_meters_full.min(), x_meters_full.max(), 
                           y_meters_full.min(), y_meters_full.max()]
        terrain_cmap_full = satellite_cmap.copy()
        terrain_cmap_full.set_bad(color='lightgray', alpha=0.3)
        plt.imshow(dem_data_full, extent=img_extent_full, cmap=terrain_cmap_full, origin='upper', alpha=0.9)
        plt.colorbar(label='Elevation (m)')

        # Color mapping for complexity categories
        category_colors = {
            'high': '#8B0000',      # Dark red
            'medium': '#FF8C00',    # Dark orange
            'low': '#228B22',       # Forest green
            'minimal': '#000000'    # Royal blue
        }

        for idx, (rect_x, rect_y, rect_w, rect_h, filename_prefix, category, _, _) in enumerate(selected_task_areas_info):
            task_rect_patch = Rectangle((rect_x, rect_y), rect_w, rect_h,
                                     edgecolor=category_colors[category], facecolor='none', 
                                     linewidth=1.5, linestyle='-')
            plt.gca().add_patch(task_rect_patch)
            
            area_info = filename_prefix.split('_std_')
            area_id = area_info[0].replace('area_', '').replace(f'{category}_', '')
            std_val_text = area_info[1]
            
            # plt.text(rect_x + rect_w / 2, rect_y + rect_h / 2, f"{category.upper()}\\n{area_id}\\nStd:{std_val_text}",
            #          color=category_colors[category], ha='center', va='center', fontsize=9,
            #          bbox=dict(facecolor='white', alpha=0.8, pad=0.3, edgecolor='none'), fontweight='bold')

        # Create legend
        legend_elements = [Rectangle((0, 0), 1, 1, facecolor='none', edgecolor=color, linewidth=1.5, 
                                   label=f"{cat.title()} ({COMPLEXITY_CATEGORIES[cat]['description']})")
                          for cat, color in category_colors.items() if any(info[5] == cat for info in selected_task_areas_info)]
        #plt.legend(handles=legend_elements, loc='upper right', fontsize=20, frameon=False)

        plt.title(f'Diverse Task Areas by Elevation Variability ({len(selected_task_areas_info)} areas selected)', fontsize=16, fontweight='bold')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(False)
        plt.axis('equal')
        
        valid_rows_idx, valid_cols_idx = np.where(~np.isnan(dem_data_full))
        if valid_rows_idx.size > 0 and valid_cols_idx.size > 0:
            x_coords_of_valid_data = x_meters_full[valid_rows_idx, valid_cols_idx]
            y_coords_of_valid_data = y_meters_full[valid_rows_idx, valid_cols_idx]
            
            if x_coords_of_valid_data.size > 0 and y_coords_of_valid_data.size > 0:
                x_min_map, x_max_map = np.min(x_coords_of_valid_data), np.max(x_coords_of_valid_data)
                y_min_map, y_max_map = np.min(y_coords_of_valid_data), np.max(y_coords_of_valid_data)
                
                padding_x_map = (x_max_map - x_min_map) * 0.05
                padding_y_map = (y_max_map - y_min_map) * 0.05
                
                plt.xlim(x_min_map - padding_x_map, x_max_map + padding_x_map)
                plt.ylim(y_min_map - padding_y_map, y_max_map + padding_y_map)

        plt.tight_layout()
        plt.savefig(os.path.join(base_output_dir, 'summary_diverse_task_areas.png'), dpi=200)
        plt.close()
        print(f"Summary plot saved to {os.path.join(base_output_dir, 'summary_diverse_task_areas.png')}")
    else:
        print("No task areas were successfully processed, skipping summary plot.")

    print(f"Processing complete. Diverse task areas saved in: {base_output_dir}")
    
    # Print summary statistics
    print("\\nSummary of selected task areas:")
    for category, criteria in COMPLEXITY_CATEGORIES.items():
        category_areas = [info for info in selected_task_areas_info if info[5] == category]
        print(f"  {category.title()} complexity: {len(category_areas)} areas (target: {criteria['count']})")

if __name__ == '__main__':
    main() 