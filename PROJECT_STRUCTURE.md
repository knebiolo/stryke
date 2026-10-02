# STRYKE Project Structure

```
stryke/
├── README.md                    # Main project documentation
├── LICENSE.txt                  # Project license
├── setup.py                     # Package installation script
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment specification
│
├── Stryke/                      # Core simulation package
│   ├── __init__.py
│   ├── stryke.py                # Main simulation engine
│   └── ...
│
├── Scripts/                     # Analysis and utility scripts
│   ├── barotrauma.py
│   ├── entrainment_density_comparisons.py
│   └── ...
│
├── tests/                       # Unit tests
│   └── test_*.py
│
├── tools/                       # Development and debugging tools
│   ├── report_probe.py
│   ├── inspect_survival.py
│   └── diagnose_mapping.py
│
├── Data/                        # Reference data and lookup tables
│   ├── Barotrauma_Values_Pflugrath2021.xlsx
│   ├── fish_class.csv
│   └── ...
│
├── docs/                        # User documentation
│   ├── ROUTE_AGG_README.md
│   └── html/                    # Built documentation
│
├── dev-notes/                   # Development logs and fix notes
│   ├── README.md
│   └── *.md                     # Progress logs, fix documentation
│
├── examples/                    # Example notebooks and projects
│   ├── README.md
│   └── stryke_project_notebook.ipynb
│
├── temp/                        # Temporary outputs (not in git)
│   ├── README.md
│   ├── *.h5                     # Simulation output files
│   └── simulation_report_*/     # Generated reports
│
├── pics/                        # Screenshots and diagrams
│   └── *.jpg, *.pdf
│
├── simulation_project/          # Active simulation projects (runtime, not in git)
│
└── source/                      # Sphinx documentation source
    └── *.rst
```

## Key Directories

### Core Application
- **Stryke/** - The main simulation engine and business logic

### Development
- **tests/** - Automated tests (run with pytest)
- **tools/** - Debug scripts and development utilities
- **dev-notes/** - Historical fix logs and implementation notes

### Data & Configuration
- **Data/** - Reference data for species, facilities, etc.
- **Scripts/** - Standalone analysis scripts

### Documentation
- **docs/** - User-facing documentation
- **examples/** - Example notebooks and tutorials
- **pics/** - Visual assets and diagrams

### Runtime (Not in Git)
- **temp/** - Temporary simulation outputs
- **simulation_project/** - Active simulation data
