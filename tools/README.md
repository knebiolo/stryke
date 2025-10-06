# Development Tools

Debugging and analysis utilities for the STRYKE simulation engine.

## HDF5 Analysis Tools

### `report_probe.py`
Main diagnostic tool for inspecting simulation outputs.

**Usage:**
```powershell
python tools\report_probe.py
```

**What it does:**
- Reads HDF5 simulation files
- Computes empirical survival statistics
- Validates survival-by-route aggregation
- Prints bootstrap confidence intervals

### `inspect_survival.py`
Inspects state and survival columns in simulation tables.

**Usage:**
```powershell
python tools\inspect_survival.py
```

### `diagnose_mapping.py`
Diagnoses survival column mapping for fish passage events.

**Usage:**
```powershell
python tools\diagnose_mapping.py
```

### `pandas_analyze_h5.py`
Analyzes HDF5 files using pandas.

### `inspect_h5_h5py.py`
Low-level HDF5 inspection using h5py.

### `analyze_sim_table_h5py.py`
Analyzes simulation tables with h5py.

### `analyze_latest_h5.py`
Quick analysis of the most recent simulation output.

## Template Testing

### `render_template_test.py`
Tests Jinja template rendering without running Flask.

**Usage:**
```powershell
python tools\render_template_test.py
```

## Statistical Tools

### `compute_one_in_n.py`
Computes "1-in-N day" entrainment statistics.

### `inspect_numeric_blocks.py`
Inspects numeric data blocks in outputs.

## Usage Tips

Most tools are designed to run against `temp/same_old.h5` or the latest simulation output. Update the file path in the script if needed.

To run any tool:
```powershell
cd "c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\stryke"
python tools\<tool_name>.py
```
