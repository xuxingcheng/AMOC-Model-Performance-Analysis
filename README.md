# AMOC Model Performance Analysis

Tools and workflows for computing and comparing **AMOC-relevant surface-forcing diagnostics** across ocean/climate models.

This repository focuses on deriving water-mass transformation style metrics (notably `Fgen`) from model output, with a primary emphasis on:

- Sea-surface density and thermodynamic coefficients (`rho`, `alpha`, `beta`)
- Buoyancy-relevant surface forcing (`fsurf`) from heat + freshwater fluxes
- Density-class integration north of the subpolar North Atlantic (typically `lat > 45°N`)
- Cross-model comparison of resulting forcing profiles

---

## Repository layout

```text
scripts/
  FgenCalculation/
    Fgenrun2.py                  # Main multi-model Fgen workflow
    Fgen_calculation_new.ipynb   # Notebook experimentation
    old_methods/                 # Earlier versions and exploratory notebooks
  FgenEvaluation/                # Evaluation notebooks for model-by-model and all-model comparisons
  CESM1_Fgen.py                  # CESM1-specific single-model workflow
  CESM1_FWF_Calc.py              # CESM1 freshwater flux assembly helper
  sourceID/                      # Variable metadata/source notes
output/
  AMOC_Strength_all_models.png
  AMOC_Strength_normalized_all_models.png
  available_models.txt
```

---

## Core workflow

The primary end-to-end script is:

- `scripts/FgenCalculation/Fgenrun2.py`

### 1) Load and concatenate model data
The script groups netCDF files by model and variable, filters unreadable/empty files, and opens valid files with xarray/dask.

Expected variables are pulled from scenario folders like:

- `sea_surface_temperature` (`tos`)
- `sea_surface_salinity` (`sos`)
- `heatflux` (`hfds`)
- `waterflux` (`wfo`)

### 2) Align time windows
For each model, all required fields are aligned to a shared trailing window (commonly last 20 years).

### 3) Compute surface density properties
Using the Gibbs SeaWater package (`gsw`), the script computes:

- `rho` (surface density)
- `alpha` (thermal expansion coefficient)
- `beta` (saline contraction coefficient)

### 4) Compute buoyancy-relevant forcing (`fsurf`)
`fsurf` is computed as:

\[
fsurf = \frac{\alpha}{c_p} \cdot HF + \frac{\rho_0}{\rho_{fw}} \cdot \beta \cdot S_0 \cdot WF
\]

with component terms retained separately as `heat_comp` and `fw_comp`.

### 5) Integrate by density class
For each model, the script:

- Applies a latitude mask (`> 45°N`)
- Uses `areacello` to area-weight forcing
- Bins by density classes (`rho_classes`)
- Integrates and averages over time
- Stores per-density profiles of:
  - `Fgen`
  - `HeatFlux`
  - `FreshwaterFlux`
  - `AreaSum`

### 6) Save output
Final outputs are serialized to pickle (`Fgen_Allmodels.pkl`) for later plotting/evaluation.

---

## CESM1-specific scripts

- `scripts/CESM1_FWF_Calc.py` builds total freshwater flux (`FWF`) from CESM1 components:
  - `ROFF_F + IOFF_F + MELT_F + PREC_F + EVAP_F`
- `scripts/CESM1_Fgen.py` performs a focused CESM1 version of `fsurf` and `Fgen` calculation and saves a model-specific pickle.

These are useful references for debugging and method validation against the general multi-model pipeline.

---

## Requirements

Typical Python dependencies used in scripts/notebooks:

- `numpy`
- `xarray`
- `dask`
- `pandas`
- `matplotlib`
- `gsw`
- `netCDF4`
- `cartopy`
- `pyproj`

Install with your preferred environment manager (conda recommended for geoscience stacks).

---

## Data assumptions

Current scripts are configured for local HPC paths (e.g., `/glade/work/...`) and CMIP-style netCDF naming conventions. You will likely need to update path constants before running in a different environment.

In particular, check and edit:

- Base directories for input variables
- Scenario folder names
- `areacello` location
- Output save paths

---

## Running

A typical run is currently script/notebook-driven:

```bash
python scripts/FgenCalculation/Fgenrun2.py
```

If running on a workstation, consider adapting dask chunking and file loading strategy to available memory.

---

## Outputs and evaluation

- Intermediate and final diagnostics are stored as pickles and visualized in notebooks under:
  - `scripts/FgenEvaluation/`
- Pre-generated figures and model lists are in:
  - `output/`

---

## Notes

- This is an active research workspace and includes exploratory notebooks.
- Some scripts share repeated logic for different grid conventions (`i/j`, `x/y`, `lat/lon`) and may be consolidated in future refactors.
- Paths and model exclusions are currently hard-coded in places for reproducibility of in-progress experiments.
