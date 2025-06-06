# UAV Coverage Path Planning in Mountainous Terrain (CAPG)

[![arXiv](https://img.shields.io/badge/arXiv-Submitted-b31b1b.svg)](#)
[![IEEE RAL](https://img.shields.io/badge/IEEE%20RAL-Under%20Review-yellow.svg)](#)

This repository contains the implementation and experimental data for the paper **"Contour-Aligned Path Generation for Energy-Aware UAV Coverage in Mountainous Terrain"** submitted to IEEE Robotics and Automation Letters (RAL).

## Overview

CAPG (Contour-Aligned Path Generation) is a novel framework for energy-efficient UAV coverage path planning in mountainous terrain. Our method achieves **46% energy reduction** compared to traditional grid-based approaches while maintaining **98.2% coverage completeness**.

### Key Features

- **Terrain-following primitives**: Generates contour-aligned path segments from Digital Elevation Models
- **Energy-aware optimization**: ATSP formulation captures asymmetric UAV energy consumption  
- **Quantized altitude adherence**: Maintains discrete elevation levels (±1.1m AGL variation)
- **Theoretical guarantees**: Proven coverage completeness and polynomial computational complexity

## Main Results

| Terrain Type | CAPG Energy (kJ) | UG-EAC Energy (kJ) | Energy Reduction |
|--------------|------------------|-------------------|------------------|
| Minimal      | 2387±35         | 3124±80          | 23.6%           |
| Low          | 2327±38         | 3647±107         | 36.2%           |
| Medium       | 2171±35         | 3984±121         | 45.5%           |
| High         | 2448±47         | 4382±142         | 44.1%           |

## Repository Structure

```
├── manuscript.tex                          # Main paper LaTeX source
├── references.bib                          # Bibliography
├── comprehensive_analysis_diverse.py        # Main experimental analysis
├── contour_path_generator_diverse.py      # Core CAPG algorithm
├── primitive_tsp_planner_diverse.py       # ATSP optimization
├── uav_energy_model.py                    # DJI Matrice 300 RTK energy model
├── Data/                                  # Huangshan Mountain DEM data
├── comprehensive_analysis_diverse_results/ # Experimental results
└── task_areas_diverse/                    # Test terrain areas
```

## Quick Start

1. **Install Dependencies**
```bash
pip install numpy pandas matplotlib scipy scikit-image networkx ortools
```

2. **Run Experiments**
```bash
python comprehensive_analysis_diverse.py
```

3. **Generate Plots**
```bash
python framework_demo.py
```

## Citation

If you use this work in your research, please cite:

```bibtex
@article{shao2024contour,
  title={Contour-Aligned Path Generation for Energy-Aware UAV Coverage in Mountainous Terrain},
  author={Shao, Qi and Mao, Xuefei and Xu, Wenbin},
  journal={IEEE Robotics and Automation Letters},
  note={Submitted},
  year={2024}
}
```

## Authors

- **Qi Shao** - Beijing Institute of Technology
- **Xuefei Mao** - Beijing Institute of Technology (Corresponding author)
- **Wenbin Xu** - Beijing Institute of Technology

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by the School of Automation, Beijing Institute of Technology. Experimental validation used ALOS World 3D-30m DEM data and DJI Matrice 300 RTK specifications. 