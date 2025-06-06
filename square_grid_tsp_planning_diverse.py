#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time
from scipy.spatial import Voronoi, Delaunay
from matplotlib.path import Path
from scipy.interpolate import RectBivariateSpline
import argparse
from mpl_toolkits.mplot3d import Axes3D
import glob
import shutil

# Parse command line arguments
parser = argparse.ArgumentParser(description='UGEOC: Uniform Grid Energy-Optimized Coverage - Diverse Areas')
parser.add_argument('--no-visualization', action='store_true', help='Disable visualization for faster execution')
parser.add_argument('--task_id', type=str, default=None, help='Specific task area ID to process. Process all if None.')
parser.add_argument('--task_areas_dir', type=str, default='task_areas_diverse', help='Directory containing diverse task area DEM data.')
parser.add_argument('--output_dir', type=str, default='ugeoc_paths_diverse', help='Directory to save final paths and visualizations.')
args = parser.parse_args()

# Import energy model
from uav_energy_model import UAVEnergyModel

def process_single_task_area(task_area_id, task_areas_dir, base_output_dir):
    """Process a single task area for square grid TSP path planning"""
    print(f"\n{'='*50}")
    print(f"Processing Task Area: {task_area_id}")
    print(f"Method: UGEOC - Uniform Grid Energy-Optimized Coverage (Diverse Areas)")
    print(f"{'='*50}")
    
    # Create task-specific output directory
    task_output_dir = os.path.join(base_output_dir, task_area_id)
    if not os.path.exists(task_output_dir):
        os.makedirs(task_output_dir)
    
    # Record task start time
    task_start_time = time.time()
    path_planning_time = 0
    
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
    
    # Calculate rectangular coverage width and length
    cover_width = 2 * flight_height * np.tan(np.radians(horizontal_fov/2))
    cover_length = 2 * flight_height * np.tan(np.radians(vertical_fov/2))

    # Use average footprint size for path planning
    average_footprint_size = (cover_width + cover_length) / 2
    overlap_ratio = 0.3  # 30% overlap ratio
    grid_spacing = average_footprint_size * (1 - overlap_ratio)
    
    print(f"Flight height: {flight_height:.1f}m")
    print(f"Field of view: H{horizontal_fov:.1f}°, V{vertical_fov:.1f}°")
    print(f"Rectangular coverage area: Width = {cover_width:.2f}m, Length = {cover_length:.2f}m")
    print(f"Average footprint size: {average_footprint_size:.2f}m, Overlap ratio: {overlap_ratio*100:.1f}%")
    print(f"Calculated waypoint spacing: {grid_spacing:.1f}m")
    print(f"Grid type: Square Grid")
    
    # Query elevation for each waypoint
    def query_elevation(points, dem, x_grid, y_grid):
        """Query elevation values for points"""
        print("Querying waypoint elevation values...")
        points_3d = []
        
        for point in points:
            x, y = point
            i = np.argmin(np.abs(y_grid[:, 0] - y))
            j = np.argmin(np.abs(x_grid[0, :] - x))
            terrain_z = dem[i, j]
            
            if np.isnan(terrain_z):
                # Use nearby valid elevation if available
                valid_elevations = dem[~np.isnan(dem)]
                if len(valid_elevations) > 0:
                    terrain_z = np.mean(valid_elevations)
                else:
                    terrain_z = 0  # fallback
            
            z = terrain_z + flight_height
            points_3d.append([x, y, z])
        
        return np.array(points_3d)

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

    # Generate square grid waypoints
    def generate_square_grid_points(x_min, x_max, y_min, y_max, spacing):
        """Generate square grid points with intelligent boundary handling"""
        print(f"Generating square grid waypoints, spacing = {spacing:.1f}m...")
        
        # Calculate coverage footprint dimensions
        half_width = cover_width / 2
        half_length = cover_length / 2
        
        # Calculate grid positions to ensure complete coverage without going outside bounds
        # Start from positions that ensure coverage of the boundaries
        first_x = x_min + half_length
        last_x = x_max - half_length
        first_y = y_min + half_width
        last_y = y_max - half_width
        
        # Generate grid coordinates
        x_coords = []
        current_x = first_x
        while current_x <= last_x:
            x_coords.append(current_x)
            current_x += spacing
        
        # Check if we need additional point for complete X coverage
        if len(x_coords) == 0 or x_coords[-1] < last_x - spacing/2:
            x_coords.append(last_x)
        
        y_coords = []
        current_y = first_y
        while current_y <= last_y:
            y_coords.append(current_y)
            current_y += spacing
        
        # Check if we need additional point for complete Y coverage
        if len(y_coords) == 0 or y_coords[-1] < last_y - spacing/2:
            y_coords.append(last_y)
        
        print(f"    Grid X range: {min(x_coords):.1f} to {max(x_coords):.1f} ({len(x_coords)} points)")
        print(f"    Grid Y range: {min(y_coords):.1f} to {max(y_coords):.1f} ({len(y_coords)} points)")
        
        # Generate all grid points
        grid_points = []
        for y in y_coords:
            for x in x_coords:
                # Ensure point is within task boundaries
                if (x_min + half_length <= x <= x_max - half_length and 
                    y_min + half_width <= y <= y_max - half_width):
                    grid_points.append([x, y])
        
        grid_points = np.array(grid_points)
        print(f"Generated {len(grid_points)} initial grid waypoints")
        
        # Check coverage and add boundary points if needed
        if len(grid_points) > 0:
            # Convert to 3D for coverage calculation
            temp_3d_points = query_elevation(grid_points, dem_data, x_meters, y_meters)
            temp_coverage_mask, temp_coverage_percent, _ = calculate_total_coverage(temp_3d_points)
            
            print(f"Initial grid coverage: {temp_coverage_percent:.1f}%")
            
            if temp_coverage_percent < 95.0:  # If coverage is less than 95%, add boundary points
                print(f"Adding boundary points to improve coverage...")
                boundary_points = generate_boundary_grid_points(x_min, x_max, y_min, y_max, grid_points)
                if len(boundary_points) > 0:
                    grid_points = np.vstack([grid_points, boundary_points])
                    print(f"Added {len(boundary_points)} boundary points")
            else:
                print(f"Coverage sufficient, no boundary points needed")
        
        print(f"Final grid has {len(grid_points)} waypoints")
        return grid_points
    
    def generate_boundary_grid_points(x_min, x_max, y_min, y_max, existing_points):
        """Generate boundary grid points directly on the terrain edges for complete coverage"""
        boundary_points = []
        
        # Calculate current coverage
        temp_3d_points = query_elevation(existing_points, dem_data, x_meters, y_meters)
        existing_coverage_mask, _, _ = calculate_total_coverage(temp_3d_points)
        uncovered_mask = task_mask & ~existing_coverage_mask
        
        if not np.any(uncovered_mask):
            return np.array([])
        
        print(f"    Adding boundary grid points to cover {np.sum(uncovered_mask)} uncovered pixels...")
        
        # Add grid points directly on the four boundaries
        half_width = cover_width / 2
        half_length = cover_length / 2
        
        # Top boundary (Y = y_max)
        # For top boundary, we need grid points that can cover the topmost area
        top_y = y_max - half_width  # Position grid point so its coverage reaches the top edge
        x_positions = np.arange(x_min + half_length, x_max - half_length + 1, cover_length * 0.7)
        for x_pos in x_positions:
            if x_min + half_length <= x_pos <= x_max - half_length:
                # Check if this area needs coverage
                temp_coverage = calculate_coverage_around_point(x_pos, top_y)
                if np.any(temp_coverage & uncovered_mask):
                    # Check if this point is not too close to existing ones
                    min_dist = float('inf')
                    for existing_pt in existing_points:
                        dist = np.sqrt((x_pos - existing_pt[0])**2 + (top_y - existing_pt[1])**2)
                        min_dist = min(min_dist, dist)
                    
                    if min_dist > grid_spacing * 0.3:  # Minimum separation
                        boundary_points.append([x_pos, top_y])
        
        # Bottom boundary (Y = y_min)
        bottom_y = y_min + half_width  # Position grid point so its coverage reaches the bottom edge
        for x_pos in x_positions:
            if x_min + half_length <= x_pos <= x_max - half_length:
                # Check if this area needs coverage
                temp_coverage = calculate_coverage_around_point(x_pos, bottom_y)
                if np.any(temp_coverage & uncovered_mask):
                    # Check if this point is not too close to existing ones
                    min_dist = float('inf')
                    for existing_pt in existing_points:
                        dist = np.sqrt((x_pos - existing_pt[0])**2 + (bottom_y - existing_pt[1])**2)
                        min_dist = min(min_dist, dist)
                    
                    if min_dist > grid_spacing * 0.3:  # Minimum separation
                        boundary_points.append([x_pos, bottom_y])
        
        # Left boundary (X = x_min)
        # For left boundary, we need grid points that can cover the leftmost area
        left_x = x_min + half_length  # Position grid point so its coverage reaches the left edge
        y_positions = np.arange(y_min + half_width, y_max - half_width + 1, cover_width * 0.7)
        for y_pos in y_positions:
            if y_min + half_width <= y_pos <= y_max - half_width:
                # Check if this area needs coverage
                temp_coverage = calculate_coverage_around_point(left_x, y_pos)
                if np.any(temp_coverage & uncovered_mask):
                    # Check if this point is not too close to existing ones
                    min_dist = float('inf')
                    for existing_pt in existing_points:
                        dist = np.sqrt((left_x - existing_pt[0])**2 + (y_pos - existing_pt[1])**2)
                        min_dist = min(min_dist, dist)
                    
                    if min_dist > grid_spacing * 0.3:  # Minimum separation
                        boundary_points.append([left_x, y_pos])
        
        # Right boundary (X = x_max)
        right_x = x_max - half_length  # Position grid point so its coverage reaches the right edge
        for y_pos in y_positions:
            if y_min + half_width <= y_pos <= y_max - half_width:
                # Check if this area needs coverage
                temp_coverage = calculate_coverage_around_point(right_x, y_pos)
                if np.any(temp_coverage & uncovered_mask):
                    # Check if this point is not too close to existing ones
                    min_dist = float('inf')
                    for existing_pt in existing_points:
                        dist = np.sqrt((right_x - existing_pt[0])**2 + (y_pos - existing_pt[1])**2)
                        min_dist = min(min_dist, dist)
                    
                    if min_dist > grid_spacing * 0.3:  # Minimum separation
                        boundary_points.append([right_x, y_pos])
        
        # Remove duplicates
        if boundary_points:
            boundary_points = np.array(boundary_points)
            # Simple duplicate removal based on distance
            unique_points = []
            for point in boundary_points:
                is_duplicate = False
                for unique_point in unique_points:
                    if np.sqrt(np.sum((point - unique_point)**2)) < grid_spacing * 0.3:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_points.append(point)
            boundary_points = np.array(unique_points) if unique_points else np.array([])
        
        print(f"    Generated {len(boundary_points)} boundary grid points")
        return boundary_points if len(boundary_points) > 0 else np.array([])

    # Set depot location at task area minimum coordinates for start/end point (consistent with other methods)
    depot_x = task_min_x
    depot_y = task_min_y
    depot_terrain_z = dem_data[np.argmin(np.abs(y_meters[:, 0] - depot_y)), np.argmin(np.abs(x_meters[0, :] - depot_x))]
    if np.isnan(depot_terrain_z):
        depot_terrain_z = np.nanmean(dem_data)
    depot_z = depot_terrain_z + flight_height
    depot_point = np.array([depot_x, depot_y, depot_z])
    print(f"Depot location: ({depot_x:.1f}, {depot_y:.1f}, {depot_z:.1f})")
    
    # Generate grid waypoints
    grid_points = generate_square_grid_points(task_min_x, task_max_x, task_min_y, task_max_y, grid_spacing)
    
    # Add elevation information
    grid_points_3d = query_elevation(grid_points, dem_data, x_meters, y_meters)
    
    # Insert depot as the first point to ensure it's used as start/end
    grid_points_3d = np.vstack([depot_point, grid_points_3d])
    print(f"Total waypoints including depot: {len(grid_points_3d)}")
    
    # Calculate coverage area
    coverage_mask, coverage_percent, coverage_count = calculate_total_coverage(grid_points_3d)
    
    # Create energy consumption model
    energy_model = UAVEnergyModel()
    
    # TSP path planning using OR-Tools
    def solve_tsp_with_ortools(points):
        """Solve TSP using OR-Tools"""
        print("Solving TSP using OR-Tools...")
        
        path_gen_start_time = time.time()
        
        if len(points) < 2:
            print("Not enough points for TSP")
            return points, 0.0
        
        # Create distance matrix using energy consumption
        n = len(points)
        distance_matrix = np.zeros((n, n), dtype=np.int64)
        
        print("Computing energy cost matrix...")
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Calculate energy consumption between points
                    energy = energy_model.calculate_energy_between_points(points[i], points[j])
                    # Convert to integer (energy in millijoules)
                    distance_matrix[i][j] = int(energy * 1000)
                else:
                    distance_matrix[i][j] = 0
        
        # Create routing index manager
        manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # Start and end at depot (point 0)
        
        # Create routing model
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = 60  # Time limit for optimization

        print("Solving TSP...")
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            print("TSP solution found!")
            
            # Get the solution path
            tsp_path = []
            index = routing.Start(0)
            route_distance = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                tsp_path.append(points[node_index])
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
            
            # Add return to start
            tsp_path.append(points[manager.IndexToNode(routing.Start(0))])
            
            tsp_path = np.array(tsp_path)
            total_energy = route_distance / 1000.0  # Convert back to joules
            
            path_planning_time = time.time() - path_gen_start_time
            print(f"TSP solved in {path_planning_time:.2f} seconds")
            print(f"Total path points: {len(tsp_path)}")
            print(f"Total energy cost: {total_energy:.2f} J")
            
            return tsp_path, path_planning_time
        else:
            print("No TSP solution found!")
            # Return points in original order as fallback
            path_planning_time = time.time() - path_gen_start_time
            return points, path_planning_time
    
    # Solve TSP
    tsp_path, path_planning_time = solve_tsp_with_ortools(grid_points_3d)
    
    # Refine path using terrain resolution for energy calculation
    print("Refining path using terrain resolution...")
    
    fine_path = []
    sample_distance = min(x_resolution, y_resolution)
    
    for i in range(len(tsp_path) - 1):
        p1 = tsp_path[i]
        p2 = tsp_path[i + 1]
        
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
    
    fine_path.append(tsp_path[-1])
    fine_path = np.array(fine_path)
    print(f"Refined path has {len(fine_path)} points")
    
    # Calculate path length and energy consumption
    total_length = np.sum(np.sqrt(np.diff(fine_path[:, 0])**2 + np.diff(fine_path[:, 1])**2))
    energy = energy_model.calculate_path_energy(fine_path)
    print(f"UGEOC path total length: {total_length:.2f} meters")
    print(f"UGEOC path total energy consumption: {energy['E_total']:.2f} joules")
    print(f"- Horizontal energy: {energy['E_horizontal']:.2f} J (displacement: {energy['E_d_xy']:.2f} J, acceleration: {energy['E_a_xy']:.2f} J)")
    print(f"- Vertical energy: {energy['E_vertical']:.2f} J (displacement: {energy['E_d_z']:.2f} J, acceleration: {energy['E_a_z']:.2f} J)")
    
    # Save energy data
    np.save(os.path.join(task_output_dir, 'path_energy.npy'), energy)
    
    # Save path data
    print("Saving path data...")
    np.save(os.path.join(task_output_dir, 'complete_path.npy'), fine_path)
    np.save(os.path.join(task_output_dir, 'waypoints_path.npy'), tsp_path)
    
    # Calculate coverage using waypoints
    coverage_mask, coverage_percent, coverage_count = calculate_total_coverage(grid_points_3d)
    
    # Save coverage mask and coverage count
    np.save(os.path.join(task_output_dir, 'coverage_mask.npy'), coverage_mask)
    np.save(os.path.join(task_output_dir, 'coverage_count.npy'), coverage_count)
    
    # Create boundary handling visualization
    if not args.no_visualization:
        create_boundary_visualization(task_area_id, grid_points_3d, tsp_path, fine_path, coverage_mask, 
                                    dem_data, x_meters, y_meters, task_mask, task_output_dir)
    
    # Save runtime information
    total_runtime = time.time() - task_start_time
    runtime_info = {
        'total_runtime': total_runtime,
        'path_planning_time': path_planning_time,
        'coverage_percent': coverage_percent
    }
    np.save(os.path.join(task_output_dir, 'runtime_info.npy'), runtime_info)
    
    print(f"Task area {task_area_id} completed in {total_runtime:.2f}s")
    print(f"Results saved to {task_output_dir}")

def create_boundary_visualization(task_id, grid_points, tsp_path, fine_path, coverage_mask, dem_data, x_meters, y_meters, task_mask, output_dir):
    """Create visualization to check boundary handling for grid-based method"""
    print(f"Creating boundary handling visualization for {task_id}...")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Terrain with grid points and task boundaries
    ax1 = axes[0, 0]
    terrain_img = ax1.imshow(dem_data, extent=[x_meters.min(), x_meters.max(), y_meters.min(), y_meters.max()], 
                            origin='lower', cmap='terrain', alpha=0.7)
    
    # Plot grid points
    ax1.scatter(grid_points[:, 0], grid_points[:, 1], c='red', s=25, alpha=0.8, label='Grid Points')
    
    # Plot task boundaries
    task_min_x, task_max_x = x_meters.min(), x_meters.max()
    task_min_y, task_max_y = y_meters.min(), y_meters.max()
    
    # Draw boundary rectangle
    boundary_x = [task_min_x, task_max_x, task_max_x, task_min_x, task_min_x]
    boundary_y = [task_min_y, task_min_y, task_max_y, task_max_y, task_min_y]
    ax1.plot(boundary_x, boundary_y, 'k-', linewidth=3, label='Task Boundary')
    
    ax1.set_title(f'Grid Points and Task Boundaries\n{task_id}')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: TSP path with boundaries
    ax2 = axes[0, 1]
    terrain_img2 = ax2.imshow(dem_data, extent=[x_meters.min(), x_meters.max(), y_meters.min(), y_meters.max()], 
                             origin='lower', cmap='terrain', alpha=0.7)
    
    # Plot TSP path
    ax2.plot(tsp_path[:, 0], tsp_path[:, 1], 'blue', linewidth=2, alpha=0.8, label='TSP Path')
    ax2.scatter(tsp_path[0, 0], tsp_path[0, 1], c='green', s=100, marker='o', label='Start', zorder=10)
    ax2.scatter(tsp_path[-1, 0], tsp_path[-1, 1], c='red', s=100, marker='s', label='End', zorder=10)
    
    # Plot grid points
    ax2.scatter(grid_points[:, 0], grid_points[:, 1], c='orange', s=15, alpha=0.6, label='Grid Points')
    
    # Draw boundary rectangle
    ax2.plot(boundary_x, boundary_y, 'k-', linewidth=3, label='Task Boundary')
    
    ax2.set_title('TSP Path and Boundaries')
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
    
    # Subplot 4: Grid distribution and boundary analysis
    ax4 = axes[1, 1]
    
    # Find boundary regions with issues
    boundary_margin = 50  # meters
    
    # Check if there are grid points near boundaries
    near_boundary_points = []
    for gp in grid_points:
        x, y = gp[0], gp[1]
        if (x <= task_min_x + boundary_margin or x >= task_max_x - boundary_margin or
            y <= task_min_y + boundary_margin or y >= task_max_y - boundary_margin):
            near_boundary_points.append(gp)
    
    near_boundary_points = np.array(near_boundary_points) if near_boundary_points else np.array([])
    
    # Plot terrain
    terrain_img4 = ax4.imshow(dem_data, extent=[x_meters.min(), x_meters.max(), y_meters.min(), y_meters.max()], 
                             origin='lower', cmap='terrain', alpha=0.7)
    
    # Plot all grid points in light blue
    ax4.scatter(grid_points[:, 0], grid_points[:, 1], c='lightblue', s=20, alpha=0.6, label='All Grid Points')
    
    # Highlight boundary grid points in red
    if len(near_boundary_points) > 0:
        ax4.scatter(near_boundary_points[:, 0], near_boundary_points[:, 1], 
                   c='red', s=40, alpha=0.9, label=f'Boundary Points ({len(near_boundary_points)})')
    
    # Draw boundary rectangle
    ax4.plot(boundary_x, boundary_y, 'k-', linewidth=3, label='Task Boundary')
    
    # Draw boundary margin
    margin_x = [task_min_x + boundary_margin, task_max_x - boundary_margin, 
                task_max_x - boundary_margin, task_min_x + boundary_margin, task_min_x + boundary_margin]
    margin_y = [task_min_y + boundary_margin, task_min_y + boundary_margin, 
                task_max_y - boundary_margin, task_max_y - boundary_margin, task_min_y + boundary_margin]
    ax4.plot(margin_x, margin_y, 'orange', linestyle='--', linewidth=2, label=f'Boundary Margin ({boundary_margin}m)')
    
    ax4.set_title('Grid Distribution and Boundary Analysis')
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