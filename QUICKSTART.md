# STRYKE Quick Start Guide

## Project Overview

STRYKE is a fish passage and survival simulation tool for hydroelectric facilities. The codebase is organized as follows:

```
stryke/
├── Stryke/          # Core simulation engine
├── webapp/          # Web interface
├── Data/            # Reference data
├── Scripts/         # Analysis tools
├── tests/           # Unit tests
├── tools/           # Debug utilities
├── examples/        # Example notebooks
├── dev-notes/       # Development logs
└── temp/            # Temporary outputs
```

See [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) for detailed directory descriptions.

## Getting Started

### 1. Environment Setup

```powershell
# Create conda environment
conda env create -f environment.yml
conda activate stryke

# Or use pip
pip install -r requirements.txt
```

### 2. Run the Web Application

```powershell
# From project root
cd webapp
python app.py
```

Then open http://localhost:5000 in your browser.

### 3. Run Example Notebook

```powershell
jupyter lab examples/stryke_project_notebook.ipynb
```

## Common Tasks

### Running Tests

```powershell
pytest tests/
```

### Debugging Simulations

```powershell
# Analyze latest simulation output
python tools/report_probe.py

# Inspect HDF5 structure
python tools/inspect_survival.py
```

### Cleaning Temporary Files

```powershell
Remove-Item temp\* -Recurse -Force -Exclude README.md
```

## Development Workflow

1. **Make changes** to code in `Stryke/` or `webapp/`
2. **Test locally** with `pytest` and manual testing
3. **Check outputs** in `temp/` directory
4. **Review logs** in `dev-notes/` for previous fixes
5. **Update docs** if adding new features

## Key Files

- **README.md** - Main project documentation
- **PROJECT_STRUCTURE.md** - Directory layout guide
- **setup.py** - Package installation
- **requirements.txt** - Python dependencies
- **environment.yml** - Conda environment

## Troubleshooting

### Simulation not running?
Check `temp/simulation_debug.log` for errors.

### Web interface issues?
Enable Flask debug mode and check browser console.

### Data loading errors?
Verify file paths in `Data/` directory.

## Resources

- **Development Notes:** See `dev-notes/` for fix logs and implementation notes
- **API Documentation:** Run `python setup.py build_sphinx` to build docs
- **Example Usage:** See `examples/stryke_project_notebook.ipynb`

## Support

For bugs and feature requests, use the GitHub issue tracker or contact the development team.
