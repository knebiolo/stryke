# STRYKE Project Structure

```
stryke/
├── README.md                    # Main project documentation
├── LICENSE.txt                  # Project license
├── setup.py                     # Package installation script
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment specification
├── Dockerfile                   # Docker container definition
├── railway.toml                 # Railway deployment config
├── Procfile.txt                 # Process definitions for deployment
│
├── Stryke/                      # Core simulation package
│   ├── __init__.py
│   ├── stryke.py                # Main simulation engine
│   └── ...
│
├── webapp/                      # Flask web application
│   ├── app.py                   # Main Flask app
│   ├── templates/               # HTML templates
│   └── static/                  # CSS, JS, images
│       ├── autosave.js
│       ├── validation.js
│       └── enable-on-change.js
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
├── instance/                    # Flask instance folder (runtime)
│   └── sessions/
│
├── uploads/                     # User file uploads (runtime, not in git)
│
├── simulation_project/          # Active simulation projects (runtime, not in git)
│
└── source/                      # Sphinx documentation source
    └── *.rst
```

## Key Directories

### Core Application
- **Stryke/** - The main simulation engine and business logic
- **webapp/** - Web interface built with Flask

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
- **uploads/** - User-uploaded files
- **simulation_project/** - Active simulation data
- **instance/** - Flask session data
