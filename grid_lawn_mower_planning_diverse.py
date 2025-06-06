#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from scipy.interpolate import interp1d, RectBivariateSpline
import argparse
from mpl_toolkits.mplot3d import Axes3D
import glob
import shutil

# Parse command line arguments
parser = argparse.ArgumentParser(description='Smart Lawnmower: Intelligent Direction Selection for Coverage Path Planning - Diverse Areas')
parser.add_argument('--no-visualization', action='store_true', help='Disable visualization for faster execution')
parser.add_argument('--task_id', type=str, default=None, help='Specific task area ID to process. Process all if None.')
parser.add_argument('--task_areas_dir', type=str, default='task_areas_diverse', help='Directory containing diverse task area DEM data.')
parser.add_argument('--output_dir', type=str, default='smart_lawnmower_paths_diverse', help='Directory to save final paths and visualizations.')
args = parser.parse_args()

# Import energy model
from uav_energy_model import UAVEnergyModel

def process_single_task_area(task_area_id, task_areas_dir, base_output_dir):
    """Process a single task area for smart lawnmower path planning"""
    print(f"\n{'='*50}")
    print(f"Processing Task Area: {task_area_id}")
    print(f"Method: Smart Lawnmower - Intelligent Direction Selection (Diverse Areas)")
    print(f"{'='*50}")
    
    # Create task-specific output directory
    task_output_dir = os.path.join(base_output_dir, task_area_id)
    if not os.path.exists(task_output_dir):
        os.makedirs(task_output_dir)
    
    # Record task start time
    task_start_time = time.time()
    
    # Load task area data
    print("Loading task area terrain data...")
    dem_file = os.path.join(task_areas_dir, f'{task_area_id}_dem.npy')
    x_meters_file = os.path.join(task_areas_dir, f'{task_area_id}_x_meters.npy')
    y_meters_file = os.path.join(task_areas_dir, f'{task_area_id}_y_meters.npy')
    
    if not all(os.path.exists(f) for f in [dem_file, x_meters_file, y_meters_file]):
        print(f"ERROR: Essential data files not found for {task_area_id}. Skipping.")
        return
    
    dem_data = np.load(dem_file)
    x_meters = np.load(x_meters_file)
    y_meters = np.load(y_meters_file)
    
    # Calculate task area boundaries
    task_min_x = x_meters.min()
    task_max_x = x_meters.max()
    task_min_y = y_meters.min()
    task_max_y = y_meters.max()
    
    print(f"Task area terrain data loaded, shape: {dem_data.shape}")
    print(f"Task area X range: {task_min_x:.2f} to {task_max_x:.2f} meters")
    print(f"Task area Y range: {task_min_y:.2f} to {task_max_y:.2f} meters")
    print(f"Elevation range: {np.nanmin(dem_data):.2f} to {np.nanmax(dem_data):.2f} meters")
    
    # Create computation grid
    print("Creating computation grid...")
    y_coords = np.linspace(y_meters.min(), y_meters.max(), dem_data.shape[0])
    x_coords = np.linspace(x_meters.min(), x_meters.max(), dem_data.shape[1])
    grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing='ij')

    # Create task area mask
    task_mask = np.ones(dem_data.shape, dtype=bool)
    nan_mask = np.isnan(dem_data)
    task_mask[nan_mask] = False
    
    # Calculate x and y resolution (meters/pixel)
    x_resolution = (x_meters[0, -1] - x_meters[0, 0]) / (x_meters.shape[1] - 1) if x_meters.shape[1] > 1 else 1.0
    y_resolution = abs((y_meters[-1, 0] - y_meters[0, 0]) / (y_meters.shape[0] - 1)) if y_meters.shape[0] > 1 else 1.0
    print(f"Map resolution: X = {x_resolution:.2f} m/pixel, Y = {y_resolution:.2f} m/pixel")
    
    # Set coverage parameters
    flight_height = 50.0  # Relative flight height (meters)
    horizontal_fov = 60.0  # Horizontal field of view (degrees)
    vertical_fov = 45.0   # Vertical field of view (degrees)
    overlap_ratio = 0.3   # 30% overlap ratio
    
    # Calculate rectangular coverage width and length
    cover_width = 2 * flight_height * np.tan(np.radians(horizontal_fov/2))
    cover_length = 2 * flight_height * np.tan(np.radians(vertical_fov/2))

    print(f"Flight height: {flight_height:.1f}m")
    print(f"Field of view: H{horizontal_fov:.1f}°, V{vertical_fov:.1f}°")
    print(f"Rectangular coverage area: Width = {cover_width:.2f}m, Length = {cover_length:.2f}m")
    print(f"Overlap ratio: {overlap_ratio*100:.1f}%")
    
    # Query elevation for a point
    def query_elevation_at_point(x, y):
        """Query elevation at a specific x,y coordinate"""
        i = np.argmin(np.abs(y_meters[:, 0] - y))
        j = np.argmin(np.abs(x_meters[0, :] - x))
        terrain_z = dem_data[i, j]
        
        if np.isnan(terrain_z):
            # Use nearby valid elevation if available
            valid_elevations = dem_data[~np.isnan(dem_data)]
            if len(valid_elevations) > 0:
                terrain_z = np.mean(valid_elevations)
            else:
                terrain_z = 0  # fallback
        
        return terrain_z + flight_height
    
    # Set depot location at task area minimum coordinates for start/end point (consistent with other methods)
    depot_x = task_min_x
    depot_y = task_min_y
    depot_z = query_elevation_at_point(depot_x, depot_y)
    depot_point = np.array([depot_x, depot_y, depot_z])
    print(f"Depot location: ({depot_x:.1f}, {depot_y:.1f}, {depot_z:.1f})")
    
    # Create energy consumption model
    energy_model = UAVEnergyModel()

    # Calculate coverage around point
    def calculate_coverage_around_point(point_x, point_y, width=None, length=None):
        """Calculate rectangular coverage area mask around a point"""
        if width is None:
            width = cover_width
        if length is None:
            length = cover_length
        
        point_i = np.argmin(np.abs(y_meters[:, 0] - point_y))
        point_j = np.argmin(np.abs(x_meters[0, :] - point_x))
        
        temp_mask = np.zeros(dem_data.shape, dtype=bool)
        half_width = width / 2
        half_length = length / 2
        
        search_radius = np.sqrt(half_width**2 + half_length**2)
        search_radius_pixels_x = int(search_radius / x_resolution) + 1
        search_radius_pixels_y = int(search_radius / y_resolution) + 1
        
        i_min = max(0, point_i - search_radius_pixels_y)
        i_max = min(dem_data.shape[0], point_i + search_radius_pixels_y + 1)
        j_min = max(0, point_j - search_radius_pixels_x)
        j_max = min(dem_data.shape[1], point_j + search_radius_pixels_x + 1)
        
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                dx = x_meters[i, j] - point_x
                dy = y_meters[i, j] - point_y
                
                if abs(dx) <= half_width and abs(dy) <= half_length:
                    temp_mask[i, j] = True
        
        return temp_mask

    # Calculate total coverage
    def calculate_total_coverage(points):
        """Calculate total coverage rate and record multiple coverage areas"""
        print(f"Coverage calculation: Using {len(points)} waypoints")
        
        coverage_count = np.zeros_like(dem_data, dtype=np.int32)
        
        for point in points:
            x, y = point[0], point[1]
            point_coverage = calculate_coverage_around_point(x, y)
            coverage_count += point_coverage.astype(np.int32)
        
        coverage_mask = coverage_count > 0
        
        task_area_pixels = np.sum(task_mask)
        covered_pixels = np.sum(np.logical_and(coverage_mask, task_mask))
        
        coverage_percent = 100 * covered_pixels / task_area_pixels if task_area_pixels > 0 else 0
        
        print(f"Waypoint coverage rate: {coverage_percent:.2f}%")
        
        max_coverage = np.max(coverage_count)
        print(f"Maximum coverage count: {max_coverage}")
        
        for i in range(1, min(max_coverage + 1, 6)):
            count = np.sum((coverage_count >= i) & task_mask)
            percent = 100 * count / task_area_pixels if task_area_pixels > 0 else 0
            print(f"  Area covered >= {i} times: {percent:.2f}%")
        
        return coverage_mask, coverage_percent, coverage_count

    # Generate lawnmower path for a specific direction
    def generate_lawnmower_path_for_direction(direction):
        """Generate lawnmower pattern path for a specific direction ('x' or 'y')"""
        print(f"  Generating {direction.upper()}-direction lawnmower path...")
    
        # Calculate correct line spacing for lawnmower pattern
        if direction == 'x':
            # X-direction flight: lines are parallel to X-axis, separated in Y-direction
            # Line spacing is based on cover_width (which covers Y-direction)
            line_spacing = cover_width * (1 - overlap_ratio)
            waypoint_spacing = cover_length * (1 - overlap_ratio)  # Spacing along flight line
            
            print(f"    X-direction lawnmower: Line spacing (Y-dir) = {line_spacing:.2f}m, Waypoint spacing (X-dir) = {waypoint_spacing:.2f}m")
        else:
            # Y-direction flight: lines are parallel to Y-axis, separated in X-direction  
            # Line spacing is based on cover_length (which covers X-direction)
            line_spacing = cover_length * (1 - overlap_ratio)
            waypoint_spacing = cover_width * (1 - overlap_ratio)   # Spacing along flight line
            
            print(f"    Y-direction lawnmower: Line spacing (X-dir) = {line_spacing:.2f}m, Waypoint spacing (Y-dir) = {waypoint_spacing:.2f}m")
        
        waypoints = []
        
        if direction == 'x':
            # X-direction: Parallel lines along X-axis, separated in Y-direction
            # Calculate scan line positions to ensure complete coverage
            half_cover_width = cover_width / 2
            
            # Start from the edge that ensures coverage of task_min_y
            first_line_y = task_min_y + half_cover_width
            last_line_y = task_max_y - half_cover_width
            
            # Generate scan line Y coordinates
            scan_lines_y = []
            current_y = first_line_y
            while current_y <= last_line_y:
                scan_lines_y.append(current_y)
                current_y += line_spacing
            
            # Check if we need additional lines for complete coverage
            if len(scan_lines_y) == 0 or scan_lines_y[-1] < last_line_y - line_spacing/2:
                scan_lines_y.append(last_line_y)
            
            print(f"    Number of scan lines: {len(scan_lines_y)}")
            print(f"    Scan lines Y range: {min(scan_lines_y):.1f} to {max(scan_lines_y):.1f}")
            
            # Generate scan lines
            for line_idx, y_line in enumerate(scan_lines_y):
                # Ensure Y coordinate is within reasonable bounds
                y_line = max(task_min_y + half_cover_width, min(task_max_y - half_cover_width, y_line))
                
                # Calculate waypoint positions along this line to ensure complete coverage
                half_cover_length = cover_length / 2
                first_wp_x = task_min_x + half_cover_length
                last_wp_x = task_max_x - half_cover_length
                
                # Generate waypoint X coordinates
                waypoint_x_coords = []
                current_x = first_wp_x
                while current_x <= last_wp_x:
                    waypoint_x_coords.append(current_x)
                    current_x += waypoint_spacing
                
                # Check if we need additional waypoint for complete coverage
                if len(waypoint_x_coords) == 0 or waypoint_x_coords[-1] < last_wp_x - waypoint_spacing/2:
                    waypoint_x_coords.append(last_wp_x)
                
                # Generate waypoints along this line
                line_waypoints = []
                for x_wp in waypoint_x_coords:
                    # Ensure X coordinate is within bounds
                    x_wp = max(task_min_x + half_cover_length, min(task_max_x - half_cover_length, x_wp))
                    
                    # Check if this point covers valid task area
                    i_idx = np.argmin(np.abs(y_meters[:, 0] - y_line))
                    j_idx = np.argmin(np.abs(x_meters[0, :] - x_wp))
                    
                    if i_idx < task_mask.shape[0] and j_idx < task_mask.shape[1] and task_mask[i_idx, j_idx]:
                        z_wp = query_elevation_at_point(x_wp, y_line)
                        line_waypoints.append([x_wp, y_line, z_wp])
                
                # Add waypoints to path (alternate direction for zigzag pattern)
                if line_idx % 2 == 0:
                    # Even lines: left to right
                    waypoints.extend(line_waypoints)
                else:
                    # Odd lines: right to left  
                    waypoints.extend(reversed(line_waypoints))
                
                print(f"      Line {line_idx+1}/{len(scan_lines_y)}: Y={y_line:.1f}m, {len(line_waypoints)} waypoints")
            
        else: # direction == 'y'
            # Y-direction: Parallel lines along Y-axis, separated in X-direction
            # Calculate scan line positions to ensure complete coverage
            half_cover_length = cover_length / 2
            
            # Start from the edge that ensures coverage of task_min_x
            first_line_x = task_min_x + half_cover_length
            last_line_x = task_max_x - half_cover_length
            
            # Generate scan line X coordinates
            scan_lines_x = []
            current_x = first_line_x
            while current_x <= last_line_x:
                scan_lines_x.append(current_x)
                current_x += line_spacing
            
            # Check if we need additional lines for complete coverage
            if len(scan_lines_x) == 0 or scan_lines_x[-1] < last_line_x - line_spacing/2:
                scan_lines_x.append(last_line_x)
            
            print(f"    Number of scan lines: {len(scan_lines_x)}")
            print(f"    Scan lines X range: {min(scan_lines_x):.1f} to {max(scan_lines_x):.1f}")
            
            # Generate scan lines
            for line_idx, x_line in enumerate(scan_lines_x):
                # Ensure X coordinate is within reasonable bounds
                x_line = max(task_min_x + half_cover_length, min(task_max_x - half_cover_length, x_line))
                
                # Calculate waypoint positions along this line to ensure complete coverage
                half_cover_width = cover_width / 2
                first_wp_y = task_min_y + half_cover_width
                last_wp_y = task_max_y - half_cover_width
                
                # Generate waypoint Y coordinates
                waypoint_y_coords = []
                current_y = first_wp_y
                while current_y <= last_wp_y:
                    waypoint_y_coords.append(current_y)
                    current_y += waypoint_spacing
                
                # Check if we need additional waypoint for complete coverage
                if len(waypoint_y_coords) == 0 or waypoint_y_coords[-1] < last_wp_y - waypoint_spacing/2:
                    waypoint_y_coords.append(last_wp_y)
                
                # Generate waypoints along this line
                line_waypoints = []
                for y_wp in waypoint_y_coords:
                    # Ensure Y coordinate is within bounds
                    y_wp = max(task_min_y + half_cover_width, min(task_max_y - half_cover_width, y_wp))
                    
                    # Check if this point covers valid task area
                    i_idx = np.argmin(np.abs(y_meters[:, 0] - y_wp))
                    j_idx = np.argmin(np.abs(x_meters[0, :] - x_line))
                    
                    if i_idx < task_mask.shape[0] and j_idx < task_mask.shape[1] and task_mask[i_idx, j_idx]:
                        z_wp = query_elevation_at_point(x_line, y_wp)
                        line_waypoints.append([x_line, y_wp, z_wp])
                
                # Add waypoints to path (alternate direction for zigzag pattern)
                if line_idx % 2 == 0:
                    # Even lines: bottom to top
                    waypoints.extend(line_waypoints)
                else:
                    # Odd lines: top to bottom
                    waypoints.extend(reversed(line_waypoints))
                
                print(f"      Line {line_idx+1}/{len(scan_lines_x)}: X={x_line:.1f}m, {len(line_waypoints)} waypoints")
        
        if not waypoints:
            print(f"    ERROR: No valid waypoints generated for {direction} direction!")
            return np.array([]), np.array([])
        
        # Convert to numpy array for easier manipulation
        coverage_waypoints = np.array(waypoints)
        
        # Check coverage and add boundary waypoints if needed
        print(f"    Checking coverage completeness...")
        temp_coverage_mask, temp_coverage_percent, _ = calculate_total_coverage(coverage_waypoints)
        
        if temp_coverage_percent < 95.0:  # If coverage is less than 95%, add boundary waypoints
            print(f"    Initial coverage: {temp_coverage_percent:.1f}%, adding boundary waypoints...")
            boundary_waypoints = generate_boundary_waypoints(direction, coverage_waypoints)
            if len(boundary_waypoints) > 0:
                coverage_waypoints = np.vstack([coverage_waypoints, boundary_waypoints])
                print(f"    Added {len(boundary_waypoints)} boundary waypoints")
        else:
            print(f"    Coverage sufficient: {temp_coverage_percent:.1f}%, no boundary waypoints needed")
        
        # Add depot connections to create complete path
        complete_path = []
        
        # Start from depot
        complete_path.append(depot_point)
        
        # Add path to first waypoint
        if len(coverage_waypoints) > 0:
            first_waypoint = coverage_waypoints[0]
            # Create intermediate points if distance is large
            depot_to_first_dist = np.sqrt(np.sum((first_waypoint[:2] - depot_point[:2])**2))
            if depot_to_first_dist > waypoint_spacing:
                num_intermediate = int(depot_to_first_dist / waypoint_spacing)
                for i in range(1, num_intermediate + 1):
                    t = i / (num_intermediate + 1)
                    intermediate_x = depot_point[0] + t * (first_waypoint[0] - depot_point[0])
                    intermediate_y = depot_point[1] + t * (first_waypoint[1] - depot_point[1])
                    intermediate_z = query_elevation_at_point(intermediate_x, intermediate_y)
                    complete_path.append([intermediate_x, intermediate_y, intermediate_z])
            
            # Add all coverage waypoints
            complete_path.extend(coverage_waypoints.tolist())
            
            # Add path back to depot
            last_waypoint = coverage_waypoints[-1]
            last_to_depot_dist = np.sqrt(np.sum((depot_point[:2] - last_waypoint[:2])**2))
            if last_to_depot_dist > waypoint_spacing:
                num_intermediate = int(last_to_depot_dist / waypoint_spacing)
                for i in range(1, num_intermediate + 1):
                    t = i / (num_intermediate + 1)
                    intermediate_x = last_waypoint[0] + t * (depot_point[0] - last_waypoint[0])
                    intermediate_y = last_waypoint[1] + t * (depot_point[1] - last_waypoint[1])
                    intermediate_z = query_elevation_at_point(intermediate_x, intermediate_y)
                    complete_path.append([intermediate_x, intermediate_y, intermediate_z])
            
            # Return to depot
            complete_path.append(depot_point)
        
        complete_path = np.array(complete_path)
        
        print(f"    Generated complete path with {len(complete_path)} waypoints (including depot connections)")
        print(f"    Coverage waypoints: {len(coverage_waypoints)}")
        
        # Refine path using terrain resolution for energy calculation
        print(f"    Refining path using terrain resolution...")
        
        fine_path = []
        sample_distance = min(x_resolution, y_resolution)
        
        for i in range(len(complete_path) - 1):
            p1 = complete_path[i]
            p2 = complete_path[i + 1]
            
            dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            num_points = max(1, int(dist / sample_distance))
            
            fine_path.append(p1)
            
            if num_points > 1:
                for j in range(1, num_points):
                    t = j / num_points
                    x = p1[0] + t * (p2[0] - p1[0])
                    y = p1[1] + t * (p2[1] - p1[1])
                    
                    i_idx = np.argmin(np.abs(y_meters[:, 0] - y))
                    j_idx = np.argmin(np.abs(x_meters[0, :] - x))
                    terrain_z = dem_data[i_idx, j_idx]
                    
                    if np.isnan(terrain_z):
                        flight_z = p1[2] + t * (p2[2] - p1[2])
                    else:
                        flight_z = terrain_z + flight_height
                    
                    fine_path.append(np.array([x, y, flight_z]))
        
        fine_path.append(complete_path[-1])
        fine_path = np.array(fine_path)
        
        print(f"    Refined path has {len(fine_path)} points")
        
        return coverage_waypoints, fine_path
    
    def generate_boundary_waypoints(direction, existing_waypoints):
        """Generate boundary waypoints directly on the terrain edges for complete coverage"""
        boundary_waypoints = []
        
        # Calculate uncovered boundary areas
        existing_coverage_mask, _, _ = calculate_total_coverage(existing_waypoints)
        uncovered_mask = task_mask & ~existing_coverage_mask
        
        if not np.any(uncovered_mask):
            return np.array([])
        
        print(f"    Adding boundary waypoints to cover {np.sum(uncovered_mask)} uncovered pixels...")
        
        # Add waypoints directly on the four boundaries
        half_width = cover_width / 2
        half_length = cover_length / 2
        
        # Top boundary (Y = task_max_y)
        # For top boundary, we need waypoints that can cover the topmost area
        top_y = task_max_y - half_width  # Position waypoint so its coverage reaches the top edge
        x_positions = np.arange(task_min_x + half_length, task_max_x - half_length + 1, cover_length * 0.7)
        for x_pos in x_positions:
            if task_min_x + half_length <= x_pos <= task_max_x - half_length:
                # Check if this area needs coverage
                temp_coverage = calculate_coverage_around_point(x_pos, top_y)
                if np.any(temp_coverage & uncovered_mask):
                    wp_z = query_elevation_at_point(x_pos, top_y)
                    boundary_waypoints.append([x_pos, top_y, wp_z])
        
        # Bottom boundary (Y = task_min_y)
        bottom_y = task_min_y + half_width  # Position waypoint so its coverage reaches the bottom edge
        for x_pos in x_positions:
            if task_min_x + half_length <= x_pos <= task_max_x - half_length:
                # Check if this area needs coverage
                temp_coverage = calculate_coverage_around_point(x_pos, bottom_y)
                if np.any(temp_coverage & uncovered_mask):
                    wp_z = query_elevation_at_point(x_pos, bottom_y)
                    boundary_waypoints.append([x_pos, bottom_y, wp_z])
        
        # Left boundary (X = task_min_x)
        # For left boundary, we need waypoints that can cover the leftmost area
        left_x = task_min_x + half_length  # Position waypoint so its coverage reaches the left edge
        y_positions = np.arange(task_min_y + half_width, task_max_y - half_width + 1, cover_width * 0.7)
        for y_pos in y_positions:
            if task_min_y + half_width <= y_pos <= task_max_y - half_width:
                # Check if this area needs coverage
                temp_coverage = calculate_coverage_around_point(left_x, y_pos)
                if np.any(temp_coverage & uncovered_mask):
                    wp_z = query_elevation_at_point(left_x, y_pos)
                    boundary_waypoints.append([left_x, y_pos, wp_z])
        
        # Right boundary (X = task_max_x)
        right_x = task_max_x - half_length  # Position waypoint so its coverage reaches the right edge
        for y_pos in y_positions:
            if task_min_y + half_width <= y_pos <= task_max_y - half_width:
                # Check if this area needs coverage
                temp_coverage = calculate_coverage_around_point(right_x, y_pos)
                if np.any(temp_coverage & uncovered_mask):
                    wp_z = query_elevation_at_point(right_x, y_pos)
                    boundary_waypoints.append([right_x, y_pos, wp_z])
        
        print(f"    Generated {len(boundary_waypoints)} boundary waypoints")
        return np.array(boundary_waypoints) if boundary_waypoints else np.array([])
    
    # Test both directions and select the best one
    print("Testing both X and Y directions to find optimal path...")
    
    path_planning_start_time = time.time()
    
    # Test X direction
    print("\nTesting X-Direction (East-West sweeps):")
    coverage_waypoints_x, fine_path_x = generate_lawnmower_path_for_direction('x')
    
    energy_x = None
    if len(fine_path_x) > 0:
        energy_x = energy_model.calculate_path_energy(fine_path_x)
        total_length_x = np.sum(np.sqrt(np.diff(fine_path_x[:, 0])**2 + np.diff(fine_path_x[:, 1])**2))
        print(f"  X-direction total energy: {energy_x['E_total']:.2f} J, path length: {total_length_x:.2f} m")
    else:
        print(f"  X-direction path generation failed!")
    
    # Test Y direction
    print("\nTesting Y-Direction (North-South sweeps):")
    coverage_waypoints_y, fine_path_y = generate_lawnmower_path_for_direction('y')
    
    energy_y = None
    if len(fine_path_y) > 0:
        energy_y = energy_model.calculate_path_energy(fine_path_y)
        total_length_y = np.sum(np.sqrt(np.diff(fine_path_y[:, 0])**2 + np.diff(fine_path_y[:, 1])**2))
        print(f"  Y-direction total energy: {energy_y['E_total']:.2f} J, path length: {total_length_y:.2f} m")
    else:
        print(f"  Y-direction path generation failed!")
    
    path_planning_time = time.time() - path_planning_start_time
    
    # Select the best direction
    selected_direction = None
    if energy_x is not None and energy_y is not None:
        if energy_x['E_total'] <= energy_y['E_total']:
            selected_direction = 'x'
            selected_waypoints = coverage_waypoints_x
            selected_path = fine_path_x
            selected_energy = energy_x
            selected_length = total_length_x
            energy_saving = energy_y['E_total'] - energy_x['E_total']
            print(f"\nSelected X-direction (energy saving: {energy_saving:.2f} J, {100*energy_saving/energy_y['E_total']:.1f}%)")
        else:
            selected_direction = 'y'
            selected_waypoints = coverage_waypoints_y
            selected_path = fine_path_y
            selected_energy = energy_y
            selected_length = total_length_y
            energy_saving = energy_x['E_total'] - energy_y['E_total']
            print(f"\nSelected Y-direction (energy saving: {energy_saving:.2f} J, {100*energy_saving/energy_x['E_total']:.1f}%)")
    elif energy_x is not None:
        selected_direction = 'x'
        selected_waypoints = coverage_waypoints_x
        selected_path = fine_path_x
        selected_energy = energy_x
        selected_length = total_length_x
        print(f"\nSelected X-direction (only viable option)")
    elif energy_y is not None:
        selected_direction = 'y'
        selected_waypoints = coverage_waypoints_y
        selected_path = fine_path_y
        selected_energy = energy_y
        selected_length = total_length_y
        print(f"\nSelected Y-direction (only viable option)")
    else:
        print(f"\nERROR: Both directions failed for {task_area_id}!")
        return
    
    print(f"Smart Lawnmower path total length: {selected_length:.2f} meters")
    print(f"Smart Lawnmower path total energy consumption: {selected_energy['E_total']:.2f} joules")
    print(f"- Horizontal energy: {selected_energy['E_horizontal']:.2f} J (displacement: {selected_energy['E_d_xy']:.2f} J, acceleration: {selected_energy['E_a_xy']:.2f} J)")
    print(f"- Vertical energy: {selected_energy['E_vertical']:.2f} J (displacement: {selected_energy['E_d_z']:.2f} J, acceleration: {selected_energy['E_a_z']:.2f} J)")
    
    # Save energy data
    np.save(os.path.join(task_output_dir, 'path_energy.npy'), selected_energy)
    
    # Save path data
    print("Saving path data...")
    np.save(os.path.join(task_output_dir, 'complete_path.npy'), selected_path)
    np.save(os.path.join(task_output_dir, 'waypoints_path.npy'), selected_waypoints)
    
    # Calculate coverage using waypoints instead of refined path
    coverage_mask, coverage_percent, coverage_count = calculate_total_coverage(selected_waypoints)
    
    # Save coverage mask and coverage count
    np.save(os.path.join(task_output_dir, 'coverage_mask.npy'), coverage_mask)
    np.save(os.path.join(task_output_dir, 'coverage_count.npy'), coverage_count)
    
    # Create boundary handling visualization
    if not args.no_visualization:
        create_boundary_visualization(task_area_id, selected_waypoints, selected_path, coverage_mask, 
                                    dem_data, x_meters, y_meters, task_mask, task_output_dir)
    
    # Save runtime information
    total_runtime = time.time() - task_start_time
    runtime_info = {
        'total_runtime': total_runtime,
        'path_planning_time': path_planning_time,
        'selected_direction': selected_direction,
        'coverage_percent': coverage_percent
    }
    np.save(os.path.join(task_output_dir, 'runtime_info.npy'), runtime_info)
    
    print(f"Task area {task_area_id} completed in {total_runtime:.2f}s")
    print(f"Results saved to {task_output_dir}")

def create_boundary_visualization(task_id, waypoints, path, coverage_mask, dem_data, x_meters, y_meters, task_mask, output_dir):
    """Create visualization to check boundary handling"""
    print(f"Creating boundary handling visualization for {task_id}...")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Terrain with waypoints and task boundaries
    ax1 = axes[0, 0]
    terrain_img = ax1.imshow(dem_data, extent=[x_meters.min(), x_meters.max(), y_meters.min(), y_meters.max()], 
                            origin='lower', cmap='terrain', alpha=0.7)
    
    # Plot waypoints
    ax1.scatter(waypoints[:, 0], waypoints[:, 1], c='red', s=20, alpha=0.8, label='Waypoints')
    
    # Plot task boundaries
    task_min_x, task_max_x = x_meters.min(), x_meters.max()
    task_min_y, task_max_y = y_meters.min(), y_meters.max()
    
    # Draw boundary rectangle
    boundary_x = [task_min_x, task_max_x, task_max_x, task_min_x, task_min_x]
    boundary_y = [task_min_y, task_min_y, task_max_y, task_max_y, task_min_y]
    ax1.plot(boundary_x, boundary_y, 'k-', linewidth=3, label='Task Boundary')
    
    ax1.set_title(f'Waypoints and Task Boundaries\n{task_id}')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Flight path with boundaries
    ax2 = axes[0, 1]
    terrain_img2 = ax2.imshow(dem_data, extent=[x_meters.min(), x_meters.max(), y_meters.min(), y_meters.max()], 
                             origin='lower', cmap='terrain', alpha=0.7)
    
    # Plot flight path
    ax2.plot(path[:, 0], path[:, 1], 'blue', linewidth=2, alpha=0.8, label='Flight Path')
    ax2.scatter(path[0, 0], path[0, 1], c='green', s=100, marker='o', label='Start', zorder=10)
    ax2.scatter(path[-1, 0], path[-1, 1], c='red', s=100, marker='s', label='End', zorder=10)
    
    # Draw boundary rectangle
    ax2.plot(boundary_x, boundary_y, 'k-', linewidth=3, label='Task Boundary')
    
    ax2.set_title('Flight Path and Boundaries')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Coverage analysis
    ax3 = axes[1, 0]
    
    # Show covered and uncovered areas
    covered_display = np.zeros((dem_data.shape[0], dem_data.shape[1], 3))
    
    # Task area in light gray
    covered_display[task_mask] = [0.8, 0.8, 0.8]
    
    # Covered areas in green
    covered_display[coverage_mask & task_mask] = [0.2, 0.8, 0.2]
    
    # Uncovered areas in red
    uncovered_mask = task_mask & ~coverage_mask
    covered_display[uncovered_mask] = [0.8, 0.2, 0.2]
    
    ax3.imshow(covered_display, extent=[x_meters.min(), x_meters.max(), y_meters.min(), y_meters.max()], 
               origin='lower')
    
    # Draw boundary rectangle
    ax3.plot(boundary_x, boundary_y, 'k-', linewidth=3, label='Task Boundary')
    
    # Calculate coverage statistics
    total_task_pixels = np.sum(task_mask)
    covered_pixels = np.sum(coverage_mask & task_mask)
    uncovered_pixels = np.sum(uncovered_mask)
    coverage_percent = 100 * covered_pixels / total_task_pixels if total_task_pixels > 0 else 0
    
    ax3.set_title(f'Coverage Analysis\nCovered: {coverage_percent:.1f}% | Uncovered: {uncovered_pixels} pixels')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.legend()
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=[0.2, 0.8, 0.2], label='Covered'),
                      Patch(facecolor=[0.8, 0.2, 0.2], label='Uncovered'),
                      Patch(facecolor=[0.8, 0.8, 0.8], label='Task Area')]
    ax3.legend(handles=legend_elements, loc='upper right')
    
    # Subplot 4: Boundary detail view
    ax4 = axes[1, 1]
    
    # Find boundary regions with issues
    boundary_margin = 50  # meters
    
    # Check if there are waypoints near boundaries
    near_boundary_waypoints = []
    for wp in waypoints:
        x, y = wp[0], wp[1]
        if (x <= task_min_x + boundary_margin or x >= task_max_x - boundary_margin or
            y <= task_min_y + boundary_margin or y >= task_max_y - boundary_margin):
            near_boundary_waypoints.append(wp)
    
    near_boundary_waypoints = np.array(near_boundary_waypoints) if near_boundary_waypoints else np.array([])
    
    # Plot terrain
    terrain_img4 = ax4.imshow(dem_data, extent=[x_meters.min(), x_meters.max(), y_meters.min(), y_meters.max()], 
                             origin='lower', cmap='terrain', alpha=0.7)
    
    # Plot all waypoints in light blue
    ax4.scatter(waypoints[:, 0], waypoints[:, 1], c='lightblue', s=15, alpha=0.6, label='All Waypoints')
    
    # Highlight boundary waypoints in red
    if len(near_boundary_waypoints) > 0:
        ax4.scatter(near_boundary_waypoints[:, 0], near_boundary_waypoints[:, 1], 
                   c='red', s=30, alpha=0.9, label=f'Boundary Waypoints ({len(near_boundary_waypoints)})')
    
    # Draw boundary rectangle
    ax4.plot(boundary_x, boundary_y, 'k-', linewidth=3, label='Task Boundary')
    
    # Draw boundary margin
    margin_x = [task_min_x + boundary_margin, task_max_x - boundary_margin, 
                task_max_x - boundary_margin, task_min_x + boundary_margin, task_min_x + boundary_margin]
    margin_y = [task_min_y + boundary_margin, task_min_y + boundary_margin, 
                task_max_y - boundary_margin, task_max_y - boundary_margin, task_min_y + boundary_margin]
    ax4.plot(margin_x, margin_y, 'orange', linestyle='--', linewidth=2, label=f'Boundary Margin ({boundary_margin}m)')
    
    ax4.set_title('Boundary Waypoint Analysis')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'boundary_handling_visualization_{task_id}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Boundary visualization saved: boundary_handling_visualization_{task_id}.png")

def main():
    """Main function to process diverse task areas"""
    task_areas_dir = args.task_areas_dir
    base_output_dir = args.output_dir
    specific_task_id = args.task_id
    
    if not os.path.exists(task_areas_dir):
        print(f"Task areas directory '{task_areas_dir}' not found.")
        return
    
    # Create base output directory
    if os.path.exists(base_output_dir):
        print(f"Clearing existing output directory: {base_output_dir}")
        shutil.rmtree(base_output_dir)
    os.makedirs(base_output_dir)
    
    # Find all task area files
    if specific_task_id:
        task_area_identifiers = [specific_task_id]
    else:
        dem_files = glob.glob(os.path.join(task_areas_dir, '*_dem.npy'))
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
    successful_count = 0
    
    for task_id in task_area_identifiers:
        try:
            process_single_task_area(task_id, task_areas_dir, base_output_dir)
            successful_count += 1
        except Exception as e:
            print(f"Error processing {task_id}: {e}")
            continue
    
    total_time = time.time() - total_start_time
    print(f"\nProcessed {successful_count}/{len(task_area_identifiers)} diverse task areas in {total_time:.2f}s")
    print(f"Results saved to: {base_output_dir}")

if __name__ == '__main__':
    main() 