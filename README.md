# Energy-Aware UAV Coverage Path Planning in Mountainous Terrain

This repository contains the implementation of **Contour-Aligned Path Generation with Terminal Connectivity Optimization (CAPG-TCO)**, a novel framework for energy-efficient UAV coverage path planning in mountainous terrain.

## Overview

Coverage path planning in mountainous terrain is highly energy-intensive for multirotor UAVs due to frequent altitude changes. This work proposes a CAPG framework that leverages terrain structure to minimize energy use while ensuring full coverage. CAPG reduces the 3D problem to efficient 2D planning by extracting elevation contour primitives from digital elevation models and sequencing them via an asymmetric Traveling Salesperson Problem (ATSP) that accounts for the higher cost of climbs versus descents.

## Key Features

- **Contour-Aligned Path Generation**: Extracts sparse elevation contour primitives from DEM data
- **Energy-Aware ATSP Optimization**: Models asymmetric climb/descent energy costs
- **Altitude Quantization**: Constrains UAVs to discrete altitude bands for stability
- **Comprehensive Validation**: Tested on 20 diverse mountainous terrains

## Performance Results

- **35.4% energy reduction** compared to grid-based methods
- **98.3% coverage** maintained across all test areas
- **±1.1m AGL deviation** for stable flight
- **48-52 Wh/km** energy efficiency vs 64-94 Wh/km for baselines

## Installation

### Prerequisites

- Python 3.7+
- Required packages (see requirements.txt)

### Setup

```bash
# Clone the repository
git clone https://github.com/goyasq/UAV-CAPG-Mountainous-Coverage.git
cd UAV-CAPG-Mountainous-Coverage

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

**Step 1: Generate Contour Primitives**
```bash
python contour_path_generator.py
```

**Step 2: Run CAPG-TCO Path Planning**
```bash
python primitive_tsp_planner.py
```

**Step 3: Run Baseline Methods (for comparison)**
```bash
# Square Grid TSP
python square_grid_tsp_planning.py

# Adaptive Boustrophedon  
python grid_lawn_mower_planning.py
```

**Step 4: Collect Performance Data**
```bash
python collect_runtime_data.py
```

**Step 5: Generate Analysis and Visualizations**
```bash
python comprehensive_analysis.py
```

### Advanced Usage

**Ablation Studies** (Optional):
```bash
# Generate ablation study scripts
python ablation_study_design.py

# Run ablation experiments
python ablation_quantization_study.py
python ablation_connection_study.py
```

## File Structure

```
├── contour_path_generator.py            # Contour primitive generation
├── primitive_tsp_planner.py             # CAPG-TCO main algorithm
├── square_grid_tsp_planning.py          # Grid-based baseline
├── grid_lawn_mower_planning.py          # Boustrophedon baseline
├── uav_energy_model.py                  # Energy consumption model
├── collect_runtime_data.py              # Performance data collection
├── comprehensive_analysis.py            # Results analysis and visualization
├── ablation_study_design.py             # Ablation study framework
├── task_areas_diverse/                  # Test terrain data
├── path_primitives_diverse/             # Generated contour primitives
└── requirements.txt                     # Python dependencies
```

## Data Requirements

The framework requires Digital Elevation Model (DEM) data in the following format:
- `{area_id}_dem.npy`: Elevation data (2D numpy array)
- `{area_id}_x_meters.npy`: X coordinates (2D numpy array)
- `{area_id}_y_meters.npy`: Y coordinates (2D numpy array)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{shao2024energy,
  title={Energy-Aware UAV Coverage Path Planning in Mountainous Terrain via Contour-Aligned Path Generation},
  author={Shao, Qi and Mao, Xuefei and Xu, Wenbin},
  journal={IEEE Transactions on Robotics},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Qi Shao: 3220231166@bit.edu.cn
- Xuefei Mao: maoxuefei21@163.com (Corresponding Author)
- Wenbin Xu: xuwenbinking13@126.com

## Acknowledgments

This work was supported by the School of Automation, Beijing Institute of Technology, Beijing, China.
