#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import glob
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time
import argparse
import shutil

from uav_energy_model import UAVEnergyModel

# --- Script Constants ---
EFFECTIVE_DELTA_Z_TOL = 5.0
INTER_PRIMITIVE_SAMPLE_STEP = 10.0
RELATIVE_FLIGHT_HEIGHT_FINAL_ADJUST = 50.0
LARGE_PENALTY_FOR_SKIPPING_PRIMITIVE = np.iinfo(np.int64).max // 1000 # Significantly larger penalty

# Parse command line arguments
parser = argparse.ArgumentParser(description="CAPG-TCO: Contour-aligned Primitive Generation with Terminal Connectivity Optimization for Diverse Task Areas")
parser.add_argument('--task_areas_dir', type=str, default='task_areas_diverse', help='Directory containing diverse task area DEM data.')
parser.add_argument('--primitives_dir', type=str, default='path_primitives_diverse', help='Base directory containing path primitives for each diverse task area.')
parser.add_argument('--output_dir', type=str, default='capg_tco_paths_diverse', help='Directory to save final paths and visualizations.')
parser.add_argument('--task_id', type=str, default=None, help='Specific task area ID to process. Processes all if None.')

# --- Helper Functions ---

def load_task_area_data(task_area_id, base_task_areas_dir='task_areas_diverse'):
    print(f"  Loading terrain data for task area: {task_area_id}...")
    dem_file = os.path.join(base_task_areas_dir, f'{task_area_id}_dem.npy')
    x_meters_file = os.path.join(base_task_areas_dir, f'{task_area_id}_x_meters.npy')
    y_meters_file = os.path.join(base_task_areas_dir, f'{task_area_id}_y_meters.npy')

    if not all(os.path.exists(f) for f in [dem_file, x_meters_file, y_meters_file]):
        print(f"  ERROR: Essential terrain data not found for {task_area_id}. Skipping.")
        return None, None, None, None

    dem_data = np.load(dem_file)
    x_meters = np.load(x_meters_file)
    y_meters = np.load(y_meters_file)
    
    min_x_coord = np.min(x_meters)
    min_y_coord = np.min(y_meters)
    
    depot_idx_r = np.argmin(np.abs(y_meters[:, 0] - min_y_coord))
    depot_idx_c = np.argmin(np.abs(x_meters[0, :] - min_x_coord))
    dem_depot_z = dem_data[depot_idx_r, depot_idx_c]
    
    if np.isnan(dem_depot_z):
        print("WARNING: DEM is NaN at min_x, min_y for depot. Using mean elevation of area.")
        valid_dem_mean = np.nanmean(dem_data)
        dem_depot_z = valid_dem_mean if not np.isnan(valid_dem_mean) else 0
        
    depot_base_z = np.round(dem_depot_z / EFFECTIVE_DELTA_Z_TOL) * EFFECTIVE_DELTA_Z_TOL
    depot_3d_location = np.array([min_x_coord, min_y_coord, depot_base_z])
    
    print(f"  Terrain data loaded: DEM shape {dem_data.shape}. Depot at {depot_3d_location}")
    return dem_data, x_meters, y_meters, depot_3d_location

def load_path_primitives(task_area_id, base_primitives_dir='path_primitives_diverse'):
    print(f"  Loading path primitives for task area: {task_area_id}...")
    primitives_dir = os.path.join(base_primitives_dir, task_area_id)
    
    # Try to load from the new format first
    primitive_data_file = os.path.join(primitives_dir, 'path_primitives.npy')
    if os.path.exists(primitive_data_file):
        try:
            data = np.load(primitive_data_file, allow_pickle=True).item()
            primitives = data['primitives']
            path_primitives_data = []
            for idx, prim in enumerate(primitives):
                path_primitives_data.append({
                    'id': idx,
                    'points': prim['points'],
                    'base_contour_elevation': prim['contour_elevation']
                })
            print(f"  Loaded {len(path_primitives_data)} primitives from new format.")
            return path_primitives_data
        except Exception as e:
            print(f"  Error loading from new format: {e}")
    
    # Fallback to individual primitive files
    primitive_files = sorted(glob.glob(os.path.join(primitives_dir, 'primitive_*.npy')))
    
    if not primitive_files:
        print(f"  WARNING: No path primitive files found in {primitives_dir} for {task_area_id}.")
        return []

    path_primitives_data = []
    for idx, f_path in enumerate(primitive_files):
        try:
            points = np.load(f_path)
            if len(points) >= 2:  # Only include primitives with at least 2 points
                path_primitives_data.append({
                    'id': idx, 
                    'points': points, 
                    'base_contour_elevation': points[0, 2]  # Use first point's Z as contour elevation
                })
        except Exception as e:
            print(f"  Error loading primitive file {f_path}: {e}")
    
    print(f"  Loaded {len(path_primitives_data)} physical primitives.")
    return path_primitives_data

def generate_tsp_nodes_from_primitives(physical_primitives, energy_model, depot_3d_location):
    print("  Generating TSP nodes from physical primitives...")
    tsp_nodes = []
    
    # Add depot node (index 0)
    tsp_nodes.append({
        'id': 0,
        'type': 'depot',
        'physical_primitive_id': -1, 
        'direction': None,
        'entry_point_3d': depot_3d_location,
        'exit_point_3d': depot_3d_location,
        'internal_points_3d_ordered': np.array([depot_3d_location]),
        'traversal_energy': 0.0
    })
    print(f"    TSP Node 0 (Depot): phys_id=-1, entry={depot_3d_location}, exit={depot_3d_location}")

    tsp_node_idx_counter = 1
    for p_prim in physical_primitives:
        if len(p_prim['points']) < 2:
            print(f"    Skipping physical primitive {p_prim['id']} as it has < 2 points.")
            continue

        # Forward direction node
        points_fwd = p_prim['points']
        traversal_energy_fwd = energy_model.calculate_path_energy(points_fwd)['E_total'] if len(points_fwd) > 1 else 0
        tsp_nodes.append({
            'id': tsp_node_idx_counter,
            'type': 'primitive',
            'physical_primitive_id': p_prim['id'],
            'direction': 'forward',
            'entry_point_3d': points_fwd[0],
            'exit_point_3d': points_fwd[-1],
            'internal_points_3d_ordered': points_fwd,
            'traversal_energy': traversal_energy_fwd
        })
        print(f"    TSP Node {tsp_node_idx_counter}: type=primitive, phys_id={p_prim['id']}, dir=forward, entry={points_fwd[0]}, exit={points_fwd[-1]}, trav_E={traversal_energy_fwd:.2f}")
        tsp_node_idx_counter += 1

        # Backward direction node
        points_bwd = p_prim['points'][::-1]
        traversal_energy_bwd = energy_model.calculate_path_energy(points_bwd)['E_total'] if len(points_bwd) > 1 else 0
        tsp_nodes.append({
            'id': tsp_node_idx_counter,
            'type': 'primitive',
            'physical_primitive_id': p_prim['id'], 
            'direction': 'backward',
            'entry_point_3d': points_bwd[0],
            'exit_point_3d': points_bwd[-1],
            'internal_points_3d_ordered': points_bwd,
            'traversal_energy': traversal_energy_bwd
        })
        print(f"    TSP Node {tsp_node_idx_counter}: type=primitive, phys_id={p_prim['id']}, dir=backward, entry={points_bwd[0]}, exit={points_bwd[-1]}, trav_E={traversal_energy_bwd:.2f}")
        tsp_node_idx_counter += 1
        
    print(f"  Generated {len(tsp_nodes)} TSP nodes (1 depot + {len(tsp_nodes)-1} directed primitive nodes).")
    return tsp_nodes

def calculate_inter_primitive_connection_path(ep1_3d, ep2_3d, dem_data, x_m, y_m, sample_step, tol):
    p1_2d = ep1_3d[:2]
    p2_2d = ep2_3d[:2]
    dist_2d = np.linalg.norm(p2_2d - p1_2d)

    if dist_2d < 1e-3:
        return np.array([ep1_3d])

    num_samples = max(2, int(np.ceil(dist_2d / sample_step)) + 1)
    path_3d_points = []
    for i in range(num_samples):
        ratio = i / (num_samples - 1)
        curr_x = p1_2d[0] + ratio * (p2_2d[0] - p1_2d[0])
        curr_y = p1_2d[1] + ratio * (p2_2d[1] - p1_2d[1])

        idx_r = np.argmin(np.abs(y_m[:, 0] - curr_y))
        idx_c = np.argmin(np.abs(x_m[0, :] - curr_x))
        terrain_z = dem_data[idx_r, idx_c]
        
        final_z = 0
        if np.isnan(terrain_z):
            # If terrain data is invalid, interpolate between endpoints
            final_z = ep1_3d[2] + ratio * (ep2_3d[2] - ep1_3d[2])
        else:
            # Adjust terrain height to nearest tol increment for contour-like behavior
            final_z = np.round(terrain_z / tol) * tol
        path_3d_points.append([curr_x, curr_y, final_z])
    return np.array(path_3d_points)

def solve_tsp_with_disjunctions(tsp_nodes, energy_model, dem_data, x_m, y_m):
    num_total_tsp_nodes = len(tsp_nodes)
    depot_or_tools_idx = 0

    energy_matrix = np.full((num_total_tsp_nodes, num_total_tsp_nodes), np.inf)

    print("  Building energy cost matrix for TSP with disjunctions...")
    max_raw_energy_cost = 0.0
    for i in range(num_total_tsp_nodes):
        tsp_node_i = tsp_nodes[i]
        exit_of_i = tsp_node_i['exit_point_3d']
        
        for j in range(num_total_tsp_nodes):
            if i == j:
                energy_matrix[i, j] = 0 if tsp_node_i['type'] == 'depot' else np.inf
                continue

            tsp_node_j = tsp_nodes[j]
            entry_of_j = tsp_node_j['entry_point_3d']
            traversal_energy_of_j = tsp_node_j['traversal_energy']

            connection_path_3d = calculate_inter_primitive_connection_path(
                exit_of_i, entry_of_j, dem_data, x_m, y_m, 
                INTER_PRIMITIVE_SAMPLE_STEP, EFFECTIVE_DELTA_Z_TOL
            )
            
            connection_energy = 0
            if len(connection_path_3d) > 1:
                energy_dict = energy_model.calculate_path_energy(connection_path_3d)
                connection_energy = energy_dict['E_total']
            elif len(connection_path_3d) == 1 and not np.allclose(exit_of_i, entry_of_j): 
                connection_energy = 0.001 
            
            total_cost_arc_ij = connection_energy + traversal_energy_of_j
            energy_matrix[i, j] = total_cost_arc_ij if total_cost_arc_ij > 0 else 0.001
            if not np.isinf(energy_matrix[i,j]):
                 max_raw_energy_cost = max(max_raw_energy_cost, energy_matrix[i,j])

    print(f"    Max raw energy cost observed in matrix (before scaling): {max_raw_energy_cost:.2f}")
    cost_scale_factor = 100.0 
    energy_matrix_int = (energy_matrix * cost_scale_factor).astype(np.int64)
    large_int_for_inf = np.iinfo(np.int64).max // (num_total_tsp_nodes + 10)
    energy_matrix_int[np.isinf(energy_matrix)] = large_int_for_inf
    
    manager = pywrapcp.RoutingIndexManager(num_total_tsp_nodes, 1, depot_or_tools_idx)
    routing = pywrapcp.RoutingModel(manager)

    def energy_callback_internal(from_index, to_index):
        from_node_our_idx = manager.IndexToNode(from_index)
        to_node_our_idx = manager.IndexToNode(to_index)
        return energy_matrix_int[from_node_our_idx, to_node_our_idx]

    transit_callback_idx = routing.RegisterTransitCallback(energy_callback_internal)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_idx)

    print("  Applying Disjunctions...")
    physical_prim_to_tsp_nodes = {}
    for node_data in tsp_nodes:
        if node_data['type'] == 'primitive':
            phys_id = node_data['physical_primitive_id']
            if phys_id not in physical_prim_to_tsp_nodes:
                physical_prim_to_tsp_nodes[phys_id] = []
            physical_prim_to_tsp_nodes[phys_id].append(node_data['id'])

    num_disjunctions_applied = 0
    for phys_id, associated_tsp_node_indices in physical_prim_to_tsp_nodes.items():
        if len(associated_tsp_node_indices) == 2: 
            or_tools_indices_for_disjunction = [manager.NodeToIndex(our_idx) for our_idx in associated_tsp_node_indices]
            routing.AddDisjunction(or_tools_indices_for_disjunction, LARGE_PENALTY_FOR_SKIPPING_PRIMITIVE, 1) 
            print(f"    Applied Disjunction for physical_primitive_id {phys_id} between TSP nodes {associated_tsp_node_indices}")
            num_disjunctions_applied += 1
        else:
            print(f"    WARNING: Physical primitive {phys_id} does not have exactly 2 TSP nodes ({associated_tsp_node_indices}). Skipping disjunction application for it.")
    print(f"  Total disjunctions applied: {num_disjunctions_applied}")

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 60

    print("  Solving TSP with Disjunctions...")
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        print(f"  TSP solution found. Objective value: {solution.ObjectiveValue() / cost_scale_factor:.2f} (raw energy)")
        if solution.ObjectiveValue() >= LARGE_PENALTY_FOR_SKIPPING_PRIMITIVE:
            num_penalties_paid = solution.ObjectiveValue() // LARGE_PENALTY_FOR_SKIPPING_PRIMITIVE
            print(f"    WARNING: TSP solution objective value ({solution.ObjectiveValue()}) is very high, suggesting {num_penalties_paid} disjunction penalties might have been paid.")
        
        ordered_our_indices = []
        or_tools_current_idx = routing.Start(0)
        route_output = "Route: "
        while not routing.IsEnd(or_tools_current_idx):
            node_our_idx = manager.IndexToNode(or_tools_current_idx)
            ordered_our_indices.append(node_our_idx)
            route_output += f" {node_our_idx} ->"
            or_tools_current_idx = solution.Value(routing.NextVar(or_tools_current_idx))
        
        node_our_idx = manager.IndexToNode(or_tools_current_idx)
        ordered_our_indices.append(node_our_idx)
        route_output += f" {node_our_idx}"
        print(f"    {route_output}")
        return ordered_our_indices
    else:
        print("  ERROR: No TSP solution found with disjunctions.")
        return None

def reconstruct_full_path_from_tsp_solution(ordered_tsp_node_indices, tsp_nodes_list, 
                                            dem_data, x_m, y_m, energy_model):
    print(f"  Reconstructing full path from {len(ordered_tsp_node_indices)} TSP nodes in sequence...")
    full_path_segments = []
    
    if not ordered_tsp_node_indices or len(ordered_tsp_node_indices) < 2:
        print("  ERROR: TSP node sequence is too short for reconstruction.")
        return np.array([])

    for k in range(len(ordered_tsp_node_indices) - 1):
        tsp_node_idx_i = ordered_tsp_node_indices[k]
        tsp_node_idx_j = ordered_tsp_node_indices[k+1]

        current_tsp_node_obj = tsp_nodes_list[tsp_node_idx_i]
        next_tsp_node_obj = tsp_nodes_list[tsp_node_idx_j]
        
        # Add internal path of current TSP node (if it's a primitive)
        if current_tsp_node_obj['type'] == 'depot' and k == 0:
            full_path_segments.append(np.array([current_tsp_node_obj['exit_point_3d']]))
        elif current_tsp_node_obj['type'] == 'primitive':
            current_prim_points = current_tsp_node_obj['internal_points_3d_ordered']
            if full_path_segments and np.allclose(full_path_segments[-1][-1], current_prim_points[0]):
                 if len(current_prim_points) > 1: full_path_segments.append(current_prim_points[1:])
            else:
                 full_path_segments.append(current_prim_points)

        # Add connection path from exit of current to entry of next
        exit_i = current_tsp_node_obj['exit_point_3d']
        entry_j = next_tsp_node_obj['entry_point_3d']
        
        connection_segment_3d = calculate_inter_primitive_connection_path(
            exit_i, entry_j, dem_data, x_m, y_m, 
            INTER_PRIMITIVE_SAMPLE_STEP, EFFECTIVE_DELTA_Z_TOL
        )
        
        if len(connection_segment_3d) > 0:
            if full_path_segments and np.allclose(full_path_segments[-1][-1], connection_segment_3d[0]):
                if len(connection_segment_3d) > 1:
                    full_path_segments.append(connection_segment_3d[1:])
            else:
                 full_path_segments.append(connection_segment_3d)
        
    if not full_path_segments:
        print("  Warning: No path segments generated during reconstruction. Path may be empty.")
        return np.array([])
        
    final_path = np.concatenate(full_path_segments, axis=0)
    
    if len(final_path) > 1:
        unique_rows_idx = [0] + [i for i in range(1, len(final_path)) if not np.allclose(final_path[i], final_path[i-1], atol=1e-2)]
        final_path = final_path[unique_rows_idx]
        
    return final_path

def calculate_coverage_mask(final_flight_path_3d, dem_data, x_m, y_m):
    """Calculate coverage mask for the flight path"""
    # Coverage parameters
    flight_height = RELATIVE_FLIGHT_HEIGHT_FINAL_ADJUST
    horizontal_fov = 60.0  # degrees
    vertical_fov = 45.0    # degrees
    cover_width = 2 * flight_height * np.tan(np.radians(horizontal_fov/2))
    cover_length = 2 * flight_height * np.tan(np.radians(vertical_fov/2))
    
    # Create task mask
    task_mask = np.ones(dem_data.shape, dtype=bool)
    nan_mask = np.isnan(dem_data)
    task_mask[nan_mask] = False
    
    coverage_mask = np.zeros(dem_data.shape, dtype=bool)
    
    def calculate_coverage_around_point(point_x, point_y):
        """Calculate rectangular coverage area mask around a point"""
        point_i = np.argmin(np.abs(y_m[:, 0] - point_y))
        point_j = np.argmin(np.abs(x_m[0, :] - point_x))
        
        temp_mask = np.zeros(dem_data.shape, dtype=bool)
        half_width = cover_width / 2
        half_length = cover_length / 2
        
        # Calculate coverage area
        y_coords_window = y_m
        x_coords_window = x_m
        
        dx_matrix = x_coords_window - point_x
        dy_matrix = y_coords_window - point_y
        
        covered_pixels = (np.abs(dx_matrix) <= half_width) & (np.abs(dy_matrix) <= half_length)
        temp_mask[covered_pixels] = True
        
        return temp_mask
    
    # Calculate coverage for each point in the flight path
    for point in final_flight_path_3d:
        point_coverage = calculate_coverage_around_point(point[0], point[1])
        coverage_mask = np.logical_or(coverage_mask, point_coverage)
    
    # Calculate coverage percentage
    total_covered_pixels = np.sum(coverage_mask & task_mask)
    total_task_pixels = np.sum(task_mask)
    coverage_percent = 100.0 * total_covered_pixels / total_task_pixels if total_task_pixels > 0 else 0.0
    
    return coverage_mask, coverage_percent

def process_task_area(task_area_id, base_task_areas_dir, base_primitives_dir, base_output_dir, energy_model_instance):
    print(f"\nProcessing Task Area: {task_area_id}")
    print(f"Mode: CAPG-TCO (Terminal Connectivity Optimization) - Diverse Areas")
    
    # Record overall start time
    overall_start_time = time.time()
    
    dem_data, x_m, y_m, depot_3d_loc = load_task_area_data(task_area_id, base_task_areas_dir)
    if dem_data is None: return

    # Record primitive loading time
    primitive_loading_start = time.time()
    physical_primitives = load_path_primitives(task_area_id, base_primitives_dir)
    primitive_loading_time = time.time() - primitive_loading_start
    
    if not physical_primitives:
        print(f"  No primitives to process for {task_area_id}. Skipping.")
        return

    # Record TSP solving time
    tsp_start_time = time.time()
    
    # Use terminal connectivity mode (CAPG-TCO)
    tsp_nodes = generate_tsp_nodes_from_primitives(physical_primitives, energy_model_instance, depot_3d_loc)
    if len(tsp_nodes) <= 1: 
        print(f"  Too few TSP nodes ({len(tsp_nodes)}) generated for {task_area_id}. Skipping TSP.")
        return
        
    tsp_ordered_indices = solve_tsp_with_disjunctions(tsp_nodes, energy_model_instance, dem_data, x_m, y_m)

    if tsp_ordered_indices:
        print(f"  TSP node sequence (our indices): {tsp_ordered_indices}")
        
        # Verify which physical primitives were included
        visited_physical_primitive_ids = set()
        for tsp_node_idx in tsp_ordered_indices:
            node_obj = tsp_nodes[tsp_node_idx]
            if node_obj['type'] == 'primitive':
                visited_physical_primitive_ids.add(node_obj['physical_primitive_id'])
        
        print(f"  Physical primitives included in TSP solution: {len(visited_physical_primitive_ids)} out of {len(physical_primitives)}")
        if len(visited_physical_primitive_ids) < len(physical_primitives):
            all_physical_ids = {p['id'] for p in physical_primitives}
            missed_ids = all_physical_ids - visited_physical_primitive_ids
            print(f"    WARNING: Missed physical primitive IDs: {sorted(list(missed_ids))}")

        final_base_path_3d = reconstruct_full_path_from_tsp_solution(
            tsp_ordered_indices, tsp_nodes, dem_data, x_m, y_m, energy_model_instance
        )
    else:
        print(f"  Could not generate TSP path for {task_area_id}.")
        return
    
    tsp_solving_time = time.time() - tsp_start_time
    
    if final_base_path_3d.size == 0:
        print(f"  ERROR: Path reconstruction failed for {task_area_id}.")
        return

    final_flight_path_3d = final_base_path_3d.copy()
    final_flight_path_3d[:, 2] += RELATIVE_FLIGHT_HEIGHT_FINAL_ADJUST
    print(f"  Final path generated with {len(final_flight_path_3d)} points (after Z adjust).")

    final_path_energy_dict = energy_model_instance.calculate_path_energy(final_flight_path_3d)
    print(f"  Energy of final assembled path: {final_path_energy_dict['E_total']:.2f} J")

    # Calculate coverage
    coverage_mask, coverage_percent = calculate_coverage_mask(final_flight_path_3d, dem_data, x_m, y_m)
    print(f"  Coverage: {coverage_percent:.2f}%")

    task_specific_output_dir = os.path.join(base_output_dir, task_area_id)
    if not os.path.exists(task_specific_output_dir): 
        os.makedirs(task_specific_output_dir)
    
    # Save in standard format for comprehensive analysis
    np.save(os.path.join(task_specific_output_dir, 'complete_path.npy'), final_flight_path_3d)
    np.save(os.path.join(task_specific_output_dir, 'path_energy.npy'), final_path_energy_dict)
    np.save(os.path.join(task_specific_output_dir, 'coverage_mask.npy'), coverage_mask)
    
    # Save runtime information
    total_runtime = time.time() - overall_start_time
    runtime_info = {
        'total_runtime': total_runtime,
        'primitive_loading_time': primitive_loading_time,
        'tsp_solving_time': tsp_solving_time,
        'coverage_percent': coverage_percent
    }
    np.save(os.path.join(task_specific_output_dir, 'runtime_info.npy'), runtime_info)
    
    print(f"  Task area {task_area_id} completed in {total_runtime:.2f}s")
    print(f"  Results saved to {task_specific_output_dir}")

def main():
    """Main function to process diverse task areas"""
    args = parser.parse_args()
    
    base_task_areas_dir = args.task_areas_dir
    base_primitives_dir = args.primitives_dir
    base_output_dir = args.output_dir
    specific_task_id = args.task_id
    
    if not os.path.exists(base_task_areas_dir):
        print(f"Task areas directory '{base_task_areas_dir}' not found.")
        return
    
    if not os.path.exists(base_primitives_dir):
        print(f"Primitives directory '{base_primitives_dir}' not found.")
        return
    
    # Create output directory
    if os.path.exists(base_output_dir):
        print(f"Clearing existing output directory: {base_output_dir}")
        shutil.rmtree(base_output_dir)
    os.makedirs(base_output_dir)
    
    # Initialize energy model
    energy_model = UAVEnergyModel()
    
    # Find all task area files
    if specific_task_id:
        task_area_identifiers = [specific_task_id]
    else:
        dem_files = glob.glob(os.path.join(base_task_areas_dir, '*_dem.npy'))
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
            process_task_area(task_id, base_task_areas_dir, base_primitives_dir, base_output_dir, energy_model)
            successful_count += 1
        except Exception as e:
            print(f"Error processing {task_id}: {e}")
            continue
    
    total_time = time.time() - total_start_time
    print(f"\nProcessed {successful_count}/{len(task_area_identifiers)} diverse task areas in {total_time:.2f}s")
    print(f"Results saved to: {base_output_dir}")

if __name__ == '__main__':
    main() 