# Phase Diagram Workflows

🔬 **Advanced workflows for phase diagram calculations using Calphy and LAMMPS**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue)](LICENSE)
[![Calphy](https://img.shields.io/badge/built_with-calphy-orange)](https://github.com/ICAMS/calphy)

[![Tests](https://github.com/prabhath-c/phase_diagram_workflows/actions/workflows/tests.yml/badge.svg)](https://github.com/prabhath-c/phase_diagram_workflows/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/prabhath-c/phase_diagram_workflows/graph/badge.svg?token=W47QH377QH)](https://codecov.io/gh/prabhath-c/phase_diagram_workflows)

## 📋 **Overview**

**Phase Diagram Workflows** is a powerful Python package for computational materials science that provides robust workflows for calculating free energies and phase diagrams using [Calphy](https://github.com/ICAMS/calphy) and LAMMPS.

The package offers both traditional serial execution and advanced parallel execution via executorlib, making it suitable for both small-scale testing and large-scale HPC calculations.

## 🚀 **Features**

### **Core Functionality**
- ✅ **Free Energy Calculations**: Thermodynamic-integration (TI) free-energy calculations via Calphy, with automatic temperature-scaling bracket convergence (`free_energies`)
- 🧬 **Point Defects**: Vacancy/substitution/interstitial structure creation, interstitial site discovery, and formation-energy bookkeeping (`structures.point_defects`)
- 📈 **Phase Diagrams**: Composition-energy convex hull construction, with DFT overlay support (`phase_diagram`)
- 🌐 **Materials Project Integration**: Structure queries against the Materials Project database (`structures.materials_project`)
- 🔗 **Workflows**: Multi-step pipelines chaining the building blocks above into end-to-end calculations, runnable step-by-step or all at once (`workflows`)
- 🚀 **Executor Support**: Parallel execution via executorlib for HPC environments

## 📦 **Installation**

### **From Source**
```bash
git clone https://github.com/prabhath-c/phase_diagram_workflows.git
cd phase_diagram_workflows
pip install -e .
```

## 🛠️ **Dependencies**

### **Core Dependencies**
- `ase` - Atomic Simulation Environment
- `calphy` - Calphy free energy calculation library
- `pandas` - Data analysis and manipulation
- `lammpsparser` - LAMMPS structure and output parsing utilities
- `ruamel.yaml` - YAML configuration support
- `pydantic` - Data validation and settings management

### **Optional Dependencies**
- `executorlib`, `pylammpsmpi` - Parallel execution framework (`executor` extra)
- `atomistics[lammps]` - LAMMPS-based calculators (`atomistics` extra)
- `mp-api`, `pymatgen` - Materials Project queries (`materials_project` extra)
- `plotly`, `seaborn`, `matplotlib` - Plotting (`plotting` extra)
- `spglib` - Symmetry-unique sublattice discovery for point defects (`point_defects` extra)


## 📂 **Project Structure**

The package separates the actual work -- single-purpose building-block
functions -- from workflows -- multi-step pipelines that chain those
functions together:

```
phase_diagram_workflows/
├── src/
│   └── phase_diagram_workflows/
│       ├── free_energies/        # TI free-energy calculations (Calphy)
│       │   ├── ti_calculator.py      # Run a TI calculation, gather results
│       │   ├── ti_helpers.py         # Structure/composition validation, LAMMPS I/O
│       │   └── ts_convergence/       # Temperature-scaling bracket refinement
│       ├── phase_diagram/        # Phase diagram construction
│       │   ├── convex_hull.py        # Composition-energy convex hull, DFT overlay
│       │   └── defect_diagram.py
│       ├── structures/           # Structure generation and management
│       │   ├── materials_project.py  # Materials Project structure queries
│       │   └── point_defects/        # Vacancy/substitution/interstitial creation,
│       │                             # interstitial site discovery, formation-energy
│       │                             # bookkeeping
│       ├── workflows/            # Multi-step pipelines built on the modules above
│       │                         # (e.g. TI convergence -> formation-energy calculation)
│       └── utils/                # Shared helpers (nested batch execution, ...)
├── notebooks/                    # Example notebooks
├── tests/                        # Unit + integration tests (mirrors src/ layout)
├── pyproject.toml                # Project configuration
└── README.md                     # This file
```

## 🔬 **Examples**

Check out the example notebooks in the `notebooks/` directory:

- **[Al_free_energy_executor_demo.ipynb](notebooks/Al_free_energy_executor_demo.ipynb)** - Demonstrates executor integration
- **[Al_point_defects_demo.ipynb](notebooks/Al_point_defects_demo.ipynb)** - Point-defect structure creation and formation energies
- **[ts_convergence_single_demo.ipynb](notebooks/ts_convergence_single_demo.ipynb)** - Temperature-scaling bracket convergence
- **[ConvexHull_MaterialsProject.ipynb](notebooks/ConvexHull_MaterialsProject.ipynb)** - Convex hull construction with Materials Project data

## 🤝 **Contributing**

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'feat: your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a pull request

## 📜 **License**

This project is licensed under the **BSD 3-Clause License** - see the [LICENSE](LICENSE) file for details.

## 📞 **Support**

For issues, questions, or feature requests:
- Open an issue on GitHub
- Contact: p.chilakalapudi@mpi-susmat.de

---

**© 2026 Phase Diagram Workflows | MPI for Sustainable Materials**
