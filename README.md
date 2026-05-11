# AMOC Model Performance Analysis

This repository is an active NCAR/HPC research workspace for comparing
AMOC-relevant surface-forcing diagnostics across CMIP6 and CESM ocean output.

The current workflow does four main things:

- downloads and stages CMIP6 piControl ocean variables,
- computes sea-surface density and surface forcing diagnostics,
- integrates forcing by density class for each model,
- compares those diagnostics with AMOC strength from `msftmz`.

## Current Workflow

The preferred multi-model calculation is:

```bash
python scripts/FgenCalculation/Fgenrun2_streaming.py
```

The preferred corrected AreaSum calculation is:

```bash
python scripts/FgenCalculation/AreaSumrun_streaming.py
```

Both scripts are sequential streaming drivers intended to avoid keeping every
model in memory at once. They save after each model by default and can resume
from existing pickle output.

PBS wrappers are available for Derecho/Casper-style runs:

```bash
qsub scripts/FgenCalculation/run
qsub scripts/FgenCalculation/run2
```

`run` launches the CMIP6 downloader. `run2` launches
`Fgenrun2_streaming.py`.

## Data Layout

Most scripts assume data under:

```text
/glade/work/stevenxu/AMOC_models
```

The downloader first writes to:

```text
/glade/work/stevenxu/AMOC_models/downloads
```

After verification/staging, the calculation scripts expect this layout:

```text
/glade/work/stevenxu/AMOC_models/
  sea_surface_temperature/scenarios/PIControl/   # tos
  sea_surface_salinity/scenarios/PIControl/      # sos
  heatflux/scenarios/PIControl/                  # hfds
  waterflux/scenarios/PIControl/                 # wfo
  areacello/                                     # ocean cell area
  downloads/                                     # raw ESGF download tree
```

`msftmz` remains under the download tree for the AMOC evaluation notebooks.

## Download And Stage Data

Download monthly CMIP6 piControl files for the strict `r1i1p1f1` member:

```bash
python scripts/FgenCalculation/dataDownload.py
```

By default this searches ESGF for models with:

- `tos`
- `sos`
- `hfds`
- `wfo`
- `msftmz`

and downloads the last 30 years at file granularity.

Useful downloader checks:

```bash
python scripts/FgenCalculation/dataDownload.py --dry-run
python scripts/FgenCalculation/dataDownload.py --discover-models-only
python scripts/FgenCalculation/dataDownload.py --models MIROC6 CanESM5
```

Download matching `areacello` files for models already present locally:

```bash
python scripts/FgenCalculation/areacelloDownload.py
```

Verify downloaded files and stage passing models into the calculation layout:

```bash
python scripts/FgenCalculation/verifyDownloads.py
```

Useful verification modes:

```bash
python scripts/FgenCalculation/verifyDownloads.py --verify-only
python scripts/FgenCalculation/verifyDownloads.py --deep-check --verify-only
python scripts/FgenCalculation/verifyDownloads.py --dry-run
```

## Fgen Calculation

`scripts/FgenCalculation/Fgenrun2_streaming.py` is the active Fgen driver.

For each model, it:

1. opens `tos`, `sos`, `hfds`, `wfo`, and `areacello`;
2. aligns all variables to a shared trailing time window;
3. computes surface `rho`, `alpha`, and `beta` with `gsw`;
4. computes

```text
fsurf = (alpha / cp) * hfds + (rho0 / rho_fw) * beta * S0 * wfo
```

5. applies the North Atlantic geographic mask:

```text
latitude > 45
normalized longitude between -90 and 60
```

6. bins points by density and integrates area-weighted forcing.

Default settings:

```text
scenario:       PIControl
last years:     20
last months:    240
density range:  1015.0 to 1030.0 kg/m^3
density step:   0.05 kg/m^3
output:         /glade/work/stevenxu/AMOC_models/Fgen_Allmodels_streaming.pkl
excluded model: CESM2
```

Example focused runs:

```bash
python scripts/FgenCalculation/Fgenrun2_streaming.py --models MIROC6
python scripts/FgenCalculation/Fgenrun2_streaming.py --max-models 2
python scripts/FgenCalculation/Fgenrun2_streaming.py --resume
python scripts/FgenCalculation/Fgenrun2_streaming.py --time-chunk 240
```

The output pickle is a dictionary:

```text
model name -> pandas.DataFrame
```

Each Fgen DataFrame is averaged over time by density class and contains:

```text
rho, Fgen, HeatFlux, FreshwaterFlux, AreaSum
```

Important: the `AreaSum` column in this Fgen output is the mean area in each
density bin. It is not the corrected whole-model AreaSum metric described
below.

## Corrected AreaSum Calculation

`scripts/FgenCalculation/AreaSumrun_streaming.py` computes the corrected
AreaSum as a separate output.

For each model and each time step, it sums `areacello` over grid cells where:

```text
fsurf is finite
rho is finite
area is finite
fsurf < 0
rho > rho_min
the same North Atlantic geographic mask is true
```

The default threshold is:

```text
rho_min = 1025.0 kg/m^3
```

Default output:

```text
/glade/work/stevenxu/AMOC_models/AreaSum_Allmodels_streaming.pkl
```

Example runs:

```bash
python scripts/FgenCalculation/AreaSumrun_streaming.py --models MIROC6
python scripts/FgenCalculation/AreaSumrun_streaming.py --rho-min 1025.0
python scripts/FgenCalculation/AreaSumrun_streaming.py --resume
python scripts/FgenCalculation/AreaSumrun_streaming.py --max-models 2
```

The output pickle is a dictionary:

```text
model name -> pandas.DataFrame
```

Each AreaSum DataFrame contains one row per time step:

```text
time_index, time, AreaSum
```

The "whole model" AreaSum used in evaluation is the time mean of that monthly
`AreaSum` series.

## Evaluation Notebooks

Active evaluation notebooks are in `scripts/FgenEvaluation/`.

- `AMOC_evaluation.ipynb` loads `Fgen_Allmodels_streaming.pkl`, computes AMOC
  strength from `msftmz`, and makes AMOC-vs-Fgen, heat-flux, freshwater-flux,
  and negative-area-integral plots.
- `AreaSum_evaluation.ipynb` loads `AreaSum_Allmodels_streaming.pkl`, inspects
  one model by default (`MIROC6`), plots its corrected AreaSum time series, and
  makes the AMOC strength vs mean AreaSum plot across models.
- `FgenCloud_evaluation.ipynb` and `Fgen_evaluation_allModels.ipynb` are older
  plotting notebooks for previous pickle formats.
- `oldEvaluation/` contains earlier model-specific evaluation notebooks.

AMOC strength is evaluated from `msftmz`, using the `atlantic_arctic_ocean`
basin/sector when that coordinate exists. The Fgen and AreaSum calculations do
not use a basin mask; they use the geographic latitude/longitude mask above.

Generated plot directories include:

```text
scripts/FgenEvaluation/AMOCvsFgen_plots/
scripts/FgenEvaluation/AreaSum_plots/
output/
```

## Repository Layout

```text
AGENTS.md
README.md
scripts/
  FgenCalculation/
    Fgenrun2_streaming.py        # active streaming Fgen driver
    AreaSumrun_streaming.py      # active corrected AreaSum driver
    dataDownload.py              # ESGF CMIP6 piControl downloader
    areacelloDownload.py         # matching areacello downloader
    verifyDownloads.py           # download validation and staging
    run                          # PBS wrapper for downloader
    run2                         # PBS wrapper for streaming Fgen
    Fgenrun2.py                  # older all-model Fgen workflow
    cloud_methods/               # experimental Pangeo/cloud workflows
    old_methods/                 # historical notebook/script attempts
  FgenEvaluation/
    AMOC_evaluation.ipynb
    AreaSum_evaluation.ipynb
    FgenCloud_evaluation.ipynb
    Fgen_evaluation_allModels.ipynb
    oldEvaluation/
  CESM1_Fgen.py                  # CESM1-specific Fgen reference workflow
  CESM1_FWF_Calc.py              # CESM1 freshwater-flux assembly helper
  example_watermass_calc.py      # reference water-mass transformation code
  example_northpolemap.py        # plotting/interpolation example
  sourceID/                      # model availability notes by CMIP variable
output/
  available_models.txt
  AMOC_Strength_all_models.png
  AMOC_Strength_normalized_all_models.png
```

## Environment

On NCAR systems, the existing wrappers use:

```bash
module load conda/latest
conda activate amoc-env
```

Common Python dependencies used across the active scripts and notebooks:

- `numpy`
- `pandas`
- `xarray`
- `dask`
- `gsw`
- `scipy`
- `h5netcdf`
- `netCDF4`
- `matplotlib`
- `requests`
- `cartopy`
- `pyproj`

Cloud/experimental notebooks may also need `intake`, `gcsfs`, and zarr-related
packages.

## Validation

Basic syntax checks:

```bash
python -m py_compile scripts/FgenCalculation/Fgenrun2_streaming.py
python -m py_compile scripts/FgenCalculation/AreaSumrun_streaming.py
python -m py_compile scripts/FgenCalculation/dataDownload.py
python -m py_compile scripts/FgenCalculation/areacelloDownload.py
python -m py_compile scripts/FgenCalculation/verifyDownloads.py
```

Data checks:

```bash
python scripts/FgenCalculation/verifyDownloads.py --verify-only
python scripts/FgenCalculation/verifyDownloads.py --deep-check --verify-only
```

Small calculation smoke tests:

```bash
python scripts/FgenCalculation/Fgenrun2_streaming.py --max-models 1
python scripts/FgenCalculation/AreaSumrun_streaming.py --max-models 1
```

Full execution depends on the `/glade/work/stevenxu/AMOC_models` data tree.

## Legacy And Experimental Code

`Fgenrun2.py`, `old_methods/`, the cloud notebooks/scripts, and the CESM1
scripts are useful references but are not the preferred multi-model production
path. Keep them when comparing methods or recovering older experiments, but use
the streaming drivers for new CMIP6 piControl Fgen and corrected AreaSum runs.

Large NetCDF files, generated pickles, dask scratch files, and `__pycache__`
artifacts should stay out of git.
