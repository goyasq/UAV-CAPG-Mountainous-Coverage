#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist
from scipy import interpolate
import random
import time
import argparse
from collections import defaultdict
import shutil # For deleting output directory
import glob # For finding files

# Global script start time
SCRIPT_START_TIME = time.time()

# --- Global Configuration (can be overridden by args or adjusted) ---
DEFAULT_FLIGHT_HEIGHT = 50.0
DEFAULT_HORIZONTAL_FOV = 60.0
DEFAULT_VERTICAL_FOV = 45.0
DEFAULT_GROWTH_STEP = 10.0
DEFAULT_MIN_UNCOVERED_RATIO = 0.2
DEFAULT_CONTOUR_INTERVAL = 5.0
EFFECTIVE_DELTA_Z_TOL = 5.0 # New: Tolerance for Z coordinate adjustment
DEFAULT_TARGET_COVERAGE = 98.0
DEFAULT_TARGET_OVERLAP = 30.0
DEFAULT_MAX_FAILED_ATTEMPTS = 50
DEFAULT_MAX_SEED_ATTEMPTS = 100

VISUALIZATION_PRIMITIVE_INTERVAL = 20 # Visualize after every N primitives, or at the end.

# --- Helper Functions (Many will be part of the main processing function or class) ---

def process_single_task_area(task_area_identifier, input_dir, base_output_dir, args):
    """
    Processes a single task area to generate path primitives.
    task_area_identifier is derived from the input filename, e.g., 'area_high_01_std_237.3'
    """
    task_output_dir = os.path.join(base_output_dir, task_area_identifier)

    print(f"\n{'='*20} Processing Task Area: {task_area_identifier} {'='*20}")
    area_start_time = time.time()

    # Input files are directly in input_dir, named with task_area_identifier as prefix
    dem_file = os.path.join(input_dir, f'{task_area_identifier}_dem.npy')
    x_meters_file = os.path.join(input_dir, f'{task_area_identifier}_x_meters.npy')
    y_meters_file = os.path.join(input_dir, f'{task_area_identifier}_y_meters.npy')

    if not (os.path.exists(dem_file) and os.path.exists(x_meters_file) and os.path.exists(y_meters_file)):
        print(f"  Essential data files not found for {task_area_identifier} in {input_dir}. Skipping.")
        print(f"    Expected DEM: {dem_file}")
        print(f"    Expected X: {x_meters_file}")
        print(f"    Expected Y: {y_meters_file}")
        return

    if not os.path.exists(task_output_dir):
        os.makedirs(task_output_dir)
    else: 
        print(f"  Clearing previous results in: {task_output_dir}")
        for item in os.listdir(task_output_dir):
            item_path = os.path.join(task_output_dir, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    print(f"  Loading terrain data for: {task_area_identifier}")
    try:
        dem_data = np.load(dem_file)
        x_meters = np.load(x_meters_file)
        y_meters = np.load(y_meters_file)
    except Exception as e:
        print(f"  Error loading data files for {task_area_identifier}: {e}. Skipping.")
        return

    task_min_x = x_meters.min()
    task_max_x = x_meters.max()
    task_min_y = y_meters.min()
    task_max_y = y_meters.max()
    
    print(f"  Task area terrain data loaded, shape: {dem_data.shape}")
    print(f"  Task area X range: {task_min_x:.2f} to {task_max_x:.2f} meters (Normalized for tile)")
    print(f"  Task area Y range: {task_min_y:.2f} to {task_max_y:.2f} meters (Normalized for tile)")
    print(f"  Elevation range: {np.nanmin(dem_data):.2f} to {np.nanmax(dem_data):.2f} meters")

    if np.all(np.isnan(dem_data)):
        print(f"  WARNING: DEM data for {task_area_identifier} is all NaN. Skipping path generation.")
        with open(os.path.join(task_output_dir, 'error_report.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Error: DEM data for task area '{task_area_identifier}' consists entirely of NaN values.\n")
        return

    y_coords = np.linspace(task_min_y, task_max_y, dem_data.shape[0])
    x_coords = np.linspace(task_min_x, task_max_x, dem_data.shape[1])

    x_resolution = (x_meters[0, -1] - x_meters[0, 0]) / (x_meters.shape[1] - 1) if x_meters.shape[1] > 1 else 1.0
    y_resolution = abs((y_meters[-1, 0] - y_meters[0, 0]) / (y_meters.shape[0] - 1)) if y_meters.shape[0] > 1 else 1.0
    print(f"  Map resolution: X = {x_resolution:.2f} m/pixel, Y = {y_resolution:.2f} m/pixel")

    coverage_mask = np.zeros(dem_data.shape, dtype=bool)
    global_coverage_count = np.zeros_like(dem_data, dtype=np.int32)
    coverage_cache = {} 
    attempted_seeds = []
    path_primitives = []
    random.seed(42)

    task_mask = np.ones(dem_data.shape, dtype=bool)
    nan_dem_mask = np.isnan(dem_data)
    task_mask[nan_dem_mask] = False

    flight_height = DEFAULT_FLIGHT_HEIGHT
    horizontal_fov = DEFAULT_HORIZONTAL_FOV
    vertical_fov = DEFAULT_VERTICAL_FOV
    growth_step = DEFAULT_GROWTH_STEP
    min_uncovered_ratio = DEFAULT_MIN_UNCOVERED_RATIO
    contour_interval = DEFAULT_CONTOUR_INTERVAL
    target_coverage = DEFAULT_TARGET_COVERAGE
    max_failed_attempts = DEFAULT_MAX_FAILED_ATTEMPTS
    max_seed_attempts = DEFAULT_MAX_SEED_ATTEMPTS

    cover_width = 2 * flight_height * np.tan(np.radians(horizontal_fov/2))
    cover_length = 2 * flight_height * np.tan(np.radians(vertical_fov/2))
    
    print(f"  Flight height: {flight_height:.1f}m, FOV: H{horizontal_fov:.1f} V{vertical_fov:.1f}")
    print(f"  Coverage rectangle: W={cover_width:.2f}m, L={cover_length:.2f}m")

    def get_cache_key_local(point_x, point_y, width, length):
        grid_x_local = round(point_x / x_resolution) * x_resolution
        grid_y_local = round(point_y / y_resolution) * y_resolution
        return (grid_x_local, grid_y_local, width, length)

    def calculate_coverage_around_point_cached_local(point_x, point_y, width=None, length=None):
        nonlocal coverage_cache
        effective_width = width if width is not None else cover_width
        effective_length = length if length is not None else cover_length
        
        cache_key = get_cache_key_local(point_x, point_y, effective_width, effective_length)
        if cache_key in coverage_cache:
            return coverage_cache[cache_key].copy()
        
        point_i = np.argmin(np.abs(y_meters[:, 0] - point_y))
        point_j = np.argmin(np.abs(x_meters[0, :] - point_x))
        
        temp_mask = np.zeros(dem_data.shape, dtype=bool)
        half_width = effective_width / 2
        half_length = effective_length / 2
        
        search_radius = np.sqrt(half_width**2 + half_length**2)
        search_radius_pixels_x = int(search_radius / x_resolution) + 2
        search_radius_pixels_y = int(search_radius / y_resolution) + 2
    
        i_min = max(0, point_i - search_radius_pixels_y)
        i_max = min(dem_data.shape[0], point_i + search_radius_pixels_y + 1)
        j_min = max(0, point_j - search_radius_pixels_x)
        j_max = min(dem_data.shape[1], point_j + search_radius_pixels_x + 1)
    
        y_coords_window = y_meters[i_min:i_max, j_min:j_max]
        x_coords_window = x_meters[i_min:i_max, j_min:j_max]

        dx_matrix = x_coords_window - point_x
        dy_matrix = y_coords_window - point_y
        
        covered_pixels_in_window = (np.abs(dx_matrix) <= half_width) & (np.abs(dy_matrix) <= half_length)
        
        temp_mask[i_min:i_max, j_min:j_max][covered_pixels_in_window] = True
        
        coverage_cache[cache_key] = temp_mask.copy()
        return temp_mask

    def update_coverage_incremental_local(path_points_list):
        nonlocal coverage_mask, global_coverage_count
        
        path_total_coverage_this_primitive = np.zeros_like(coverage_mask)
        for point_coords in path_points_list:
            px, py = point_coords[0], point_coords[1]
            point_coverage = calculate_coverage_around_point_cached_local(px, py)
            path_total_coverage_this_primitive = np.logical_or(path_total_coverage_this_primitive, point_coverage)
            global_coverage_count += point_coverage.astype(np.int32)
        
        old_total_coverage_mask = coverage_mask.copy()
        coverage_mask = np.logical_or(coverage_mask, path_total_coverage_this_primitive)
        
        new_pixels_covered_by_primitive = np.sum(path_total_coverage_this_primitive & ~old_total_coverage_mask & task_mask)
        new_coverage_area_by_primitive = new_pixels_covered_by_primitive * x_resolution * y_resolution
        
        current_total_covered_pixels = np.sum(coverage_mask & task_mask)
        current_total_coverage_area = current_total_covered_pixels * x_resolution * y_resolution
        
        task_area_total_pixels = np.sum(task_mask)
        current_coverage_percent = 100.0 * current_total_covered_pixels / task_area_total_pixels if task_area_total_pixels > 0 else 0.0
        
        overlapped_pixels_total = np.sum((global_coverage_count >= 2) & task_mask)
        current_overlap_percent = 100.0 * overlapped_pixels_total / task_area_total_pixels if task_area_total_pixels > 0 else 0.0
        
        return (path_total_coverage_this_primitive, new_coverage_area_by_primitive, 
                current_total_coverage_area, task_area_total_pixels * x_resolution * y_resolution, 
                current_coverage_percent, current_overlap_percent)

    print("  Pre-generating contour line collection...")
    elevation_min_tile = np.nanmin(dem_data)
    elevation_max_tile = np.nanmax(dem_data)

    if np.isnan(elevation_min_tile) or np.isnan(elevation_max_tile):
        print(f"  WARNING: Min/Max elevation for {task_area_identifier} is NaN. Cannot generate contours. Skipping area.")
        with open(os.path.join(task_output_dir, 'error_report.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Error: Min/Max elevation for task area '{task_area_identifier}' is NaN. Cannot generate contours.\n")
        return

    contour_levels_tile = np.arange(
        np.floor(elevation_min_tile / contour_interval) * contour_interval,
        np.ceil(elevation_max_tile / contour_interval) * contour_interval + contour_interval,
        contour_interval
    )
    
    if len(contour_levels_tile) == 0:
        print(f"  No contour levels to generate for {task_area_identifier}. Skipping path generation.")
        with open(os.path.join(task_output_dir, 'error_report.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Error: No contour levels to generate for task area '{task_area_identifier}'.\n")
        return

    print(f"  Generating {len(contour_levels_tile)} contour lines with {contour_interval}m interval...")
    
    contour_set_tile = plt.contour(x_coords, y_coords, dem_data, levels=contour_levels_tile)
    plt.close()

    contour_collection_tile = []
    for i, level in enumerate(contour_levels_tile):
        level_contours = []
        if i < len(contour_set_tile.allsegs):
            for contour_segment_coords in contour_set_tile.allsegs[i]:
                if len(contour_segment_coords) >= 5:
                    level_contours.append({
                        'coords': contour_segment_coords,
                        'elevation': level
                    })
        contour_collection_tile.append({
            'level': level,
            'contours': level_contours
        })

    total_contours_tile = sum(len(level_data['contours']) for level_data in contour_collection_tile)
    print(f"  Successfully generated {total_contours_tile} valid contour lines for this area.")
    if total_contours_tile == 0:
        print(f"  No valid contours found for {task_area_identifier}. Skipping path generation.")
        with open(os.path.join(task_output_dir, 'error_report.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Error: No valid contour lines (>=5 points) were generated for task area '{task_area_identifier}'.\n")
        return

    def is_point_in_task_area_local(point_x, point_y):
        return (task_min_x <= point_x <= task_max_x and task_min_y <= point_y <= task_max_y)

    def get_uncovered_ratio_cached_local(point_x, point_y, radius=None):
        eff_width = radius if radius is not None else cover_width
        eff_length = radius if radius is not None else cover_length
        area_mask_local = calculate_coverage_around_point_cached_local(point_x, point_y, width=eff_width, length=eff_length)
        total_area_pixels = np.sum(area_mask_local & task_mask)
        covered_pixels = np.sum(area_mask_local & coverage_mask & task_mask)
        if total_area_pixels == 0: return 0.0
        return (total_area_pixels - covered_pixels) / total_area_pixels

    def is_continuous_local(point1, point2, max_distance=None):
        eff_max_dist = max_distance if max_distance is not None else cover_width
        dist = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        return dist <= eff_max_dist

    def find_nearest_contour_local(seed_x, seed_y, max_dist=None):
        eff_max_dist = max_dist if max_dist is not None else cover_width * 6
        seed_i_idx = np.argmin(np.abs(y_meters[:, 0] - seed_y))
        seed_j_idx = np.argmin(np.abs(x_meters[0, :] - seed_x))
        seed_elevation_val = dem_data[seed_i_idx, seed_j_idx]
        if np.isnan(seed_elevation_val): return None
        
        nearest_contour_info = None
        min_dist_found = float('inf')
        for level_data_item in contour_collection_tile:
            current_level = level_data_item['level']
            elevation_diff = abs(current_level - seed_elevation_val)
            if elevation_diff > contour_interval * 6: continue
            
            for contour_obj in level_data_item['contours']:
                contour_segment_coords_arr = contour_obj['coords']
                distances_to_segment = np.sqrt(np.sum((contour_segment_coords_arr - np.array([seed_x, seed_y]))**2, axis=1))
                min_dist_to_this_segment = np.min(distances_to_segment)
                
                if min_dist_to_this_segment < min_dist_found and min_dist_to_this_segment <= eff_max_dist:
                    min_dist_found = min_dist_to_this_segment
                    nearest_contour_info = {
                        'coords': contour_segment_coords_arr,
                        'elevation': current_level,
                        'nearest_idx': np.argmin(distances_to_segment)
                    }
        return nearest_contour_info

    def generate_path_primitive_local(seed_x, seed_y, min_points=3):
        nonlocal attempted_seeds
        print(f"    Attempting path from seed ({seed_x:.2f}, {seed_y:.2f}) in {task_area_identifier}")
        attempted_seeds.append((seed_x, seed_y))

        nearest_c = find_nearest_contour_local(seed_x, seed_y)
        if nearest_c is None:
            print(f"    No contour found near seed. Abandoning.")
            return None
        
        contour_coords_arr = nearest_c['coords']
        contour_elev_val = nearest_c['elevation']
        seed_idx_on_contour = nearest_c['nearest_idx']
        
        fwd_pts = [contour_coords_arr[seed_idx_on_contour]]
        curr_idx = seed_idx_on_contour
        curr_dist_fwd = 0.0
        while True:
            next_idx = curr_idx + 1
            if next_idx >= len(contour_coords_arr): break
            next_pt_coords = contour_coords_arr[next_idx]
            dist_seg = np.sqrt(np.sum((next_pt_coords - fwd_pts[-1])**2))
            curr_dist_fwd += dist_seg
            if curr_dist_fwd < growth_step:
                curr_idx = next_idx
                continue
            curr_dist_fwd = 0.0
            if not is_point_in_task_area_local(next_pt_coords[0], next_pt_coords[1]): break
            if not is_continuous_local(fwd_pts[-1], next_pt_coords): break
            if get_uncovered_ratio_cached_local(next_pt_coords[0], next_pt_coords[1]) < min_uncovered_ratio: break
            fwd_pts.append(next_pt_coords)
            curr_idx = next_idx

        bwd_pts = []
        curr_idx = seed_idx_on_contour
        curr_dist_bwd = 0.0
        while True:
            prev_idx = curr_idx - 1
            if prev_idx < 0: break
            prev_pt_coords = contour_coords_arr[prev_idx]
            ref_pt_for_dist = fwd_pts[0] if not bwd_pts else bwd_pts[-1]
            dist_seg = np.sqrt(np.sum((prev_pt_coords - ref_pt_for_dist)**2))
            curr_dist_bwd += dist_seg
            if curr_dist_bwd < growth_step:
                curr_idx = prev_idx
                continue
            curr_dist_bwd = 0.0
            if not is_point_in_task_area_local(prev_pt_coords[0], prev_pt_coords[1]): break
            if not is_continuous_local(ref_pt_for_dist, prev_pt_coords): break
            if get_uncovered_ratio_cached_local(prev_pt_coords[0], prev_pt_coords[1]) < min_uncovered_ratio: break
            bwd_pts.append(prev_pt_coords)
            curr_idx = prev_idx
            
        path_pts_2d_list = bwd_pts[::-1] + fwd_pts
        
        if len(path_pts_2d_list) < min_points:
            print(f"    Generated path too short ({len(path_pts_2d_list)} points). Abandoning.")
            return None
            
        print(f"    Successfully generated 2D path with {len(path_pts_2d_list)} points. Adjusting Z coordinates...")
        
        path_pts_3d_list = []
        for pt_2d in path_pts_2d_list:
            x_val, y_val = pt_2d[0], pt_2d[1]
            i_idx = np.argmin(np.abs(y_meters[:,0] - y_val))
            j_idx = np.argmin(np.abs(x_meters[0,:] - x_val))
            
            terrain_z_at_pt = dem_data[i_idx, j_idx]
            final_pt_z = contour_elev_val # Default to contour elevation

            if np.isnan(terrain_z_at_pt):
                final_pt_z = contour_elev_val
            else:
                deviation_from_contour = terrain_z_at_pt - contour_elev_val
                if abs(deviation_from_contour) <= EFFECTIVE_DELTA_Z_TOL:
                    final_pt_z = contour_elev_val
                elif deviation_from_contour > EFFECTIVE_DELTA_Z_TOL:
                    num_tol_steps = np.floor(deviation_from_contour / EFFECTIVE_DELTA_Z_TOL)
                    final_pt_z = contour_elev_val + num_tol_steps * EFFECTIVE_DELTA_Z_TOL
                else: 
                    num_tol_steps = np.ceil(deviation_from_contour / EFFECTIVE_DELTA_Z_TOL)
                    final_pt_z = contour_elev_val + num_tol_steps * EFFECTIVE_DELTA_Z_TOL
            
            path_pts_3d_list.append([x_val, y_val, final_pt_z])
            
        return {'points': np.array(path_pts_3d_list), 'contour_elevation': contour_elev_val}

    # Continue with the rest of the processing logic...
    # (The rest of the function would be similar to the original, but I'll truncate here for brevity)
    
    # Main primitive generation loop
    iteration_count = 0
    failed_attempts = 0
    
    while failed_attempts < max_failed_attempts:
        iteration_count += 1
        
        # Calculate current coverage
        current_total_covered_pixels = np.sum(coverage_mask & task_mask)
        task_area_total_pixels = np.sum(task_mask)
        current_coverage_percent = 100.0 * current_total_covered_pixels / task_area_total_pixels if task_area_total_pixels > 0 else 0.0
        
        if current_coverage_percent >= target_coverage:
            print(f"  Target coverage {target_coverage}% achieved ({current_coverage_percent:.2f}%). Stopping.")
            break
            
        # Find seed point
        uncovered_mask = task_mask & ~coverage_mask
        if not np.any(uncovered_mask):
            print(f"  No uncovered areas remaining. Stopping.")
            break
            
        uncovered_indices = np.where(uncovered_mask)
        seed_attempts = 0
        primitive_generated = False
        
        while seed_attempts < max_seed_attempts and not primitive_generated:
            seed_attempts += 1
            
            # Random seed selection
            random_idx = random.randint(0, len(uncovered_indices[0]) - 1)
            seed_i = uncovered_indices[0][random_idx]
            seed_j = uncovered_indices[1][random_idx]
            
            seed_x = x_meters[seed_i, seed_j]
            seed_y = y_meters[seed_i, seed_j]
            
            # Generate primitive
            new_primitive = generate_path_primitive_local(seed_x, seed_y)
            
            if new_primitive is not None:
                path_primitives.append(new_primitive)
                
                # Update coverage
                coverage_result = update_coverage_incremental_local(new_primitive['points'])
                new_coverage_area = coverage_result[1]
                current_coverage_percent = coverage_result[4]
                
                print(f"  Primitive {len(path_primitives)}: {len(new_primitive['points'])} points, "
                      f"new coverage: {new_coverage_area:.1f}m², total: {current_coverage_percent:.2f}%")
                
                primitive_generated = True
                failed_attempts = 0
            else:
                failed_attempts += 1
                
        if not primitive_generated:
            failed_attempts += 1
    
    # Save results
    print(f"  Saving {len(path_primitives)} primitives to {task_output_dir}")
    
    # Save primitive data
    primitive_data = {
        'primitives': path_primitives,
        'coverage_mask': coverage_mask,
        'task_mask': task_mask,
        'coverage_percent': current_coverage_percent,
        'total_primitives': len(path_primitives)
    }
    
    np.save(os.path.join(task_output_dir, 'path_primitives.npy'), primitive_data)
    
    # Save individual primitive files
    for i, primitive in enumerate(path_primitives):
        np.save(os.path.join(task_output_dir, f'primitive_{i:03d}.npy'), primitive['points'])
    
    # Create summary
    with open(os.path.join(task_output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Task Area: {task_area_identifier}\n")
        f.write(f"Total Primitives: {len(path_primitives)}\n")
        f.write(f"Coverage: {current_coverage_percent:.2f}%\n")
        f.write(f"Processing Time: {time.time() - area_start_time:.2f}s\n")
    
    print(f"  Task area {task_area_identifier} completed in {time.time() - area_start_time:.2f}s")

def main():
    """Main function to process all diverse task areas"""
    parser = argparse.ArgumentParser(description='Generate contour-based path primitives for diverse task areas')
    parser.add_argument('--no-visualization', action='store_true', help='Disable visualization')
    parser.add_argument('--input-dir', default='task_areas_diverse', help='Input directory containing diverse task areas')
    parser.add_argument('--output-dir', default='path_primitives_diverse', help='Output directory for path primitives')
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    base_output_dir = args.output_dir
    
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' not found.")
        return
    
    # Create output directory
    if os.path.exists(base_output_dir):
        print(f"Clearing existing output directory: {base_output_dir}")
        shutil.rmtree(base_output_dir)
    os.makedirs(base_output_dir)
    
    # Find all task area files
    dem_files = glob.glob(os.path.join(input_dir, '*_dem.npy'))
    task_area_identifiers = []
    
    for dem_file in dem_files:
        task_id = os.path.basename(dem_file).replace('_dem.npy', '')
        task_area_identifiers.append(task_id)
    
    task_area_identifiers.sort()
    
    print(f"Found {len(task_area_identifiers)} diverse task areas to process:")
    for task_id in task_area_identifiers:
        print(f"  - {task_id}")
    
    # Process each task area
    total_start_time = time.time()
    
    for task_id in task_area_identifiers:
        try:
            process_single_task_area(task_id, input_dir, base_output_dir, args)
        except Exception as e:
            print(f"Error processing {task_id}: {e}")
            continue
    
    total_time = time.time() - total_start_time
    print(f"\nAll diverse task areas processed in {total_time:.2f}s")
    print(f"Results saved to: {base_output_dir}")

if __name__ == '__main__':
    main() 