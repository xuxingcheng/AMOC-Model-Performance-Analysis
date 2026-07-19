# Codex Guide for AMOCproject

## Project Purpose

This repository is an active NCAR/HPC research workspace for AMOC-relevant
surface-forcing diagnostics. The main workflows download and stage CMIP6
piControl ocean data, compute `Fgen` and related regional metrics on each
model's native grid, evaluate those metrics against AMOC strength, and compare
native-grid results with shared latitude/longitude grid experiments.

The code is script/notebook driven, not packaged as an installable Python
module. There is no checked-in `pyproject.toml`, requirements file, conda
environment file, or automated test configuration.

## Setup

Run commands from the repository root unless a script says otherwise:

```bash
cd /glade/u/home/stevenxu/AMOCproject
module load conda/latest
conda activate amoc-env
```

Current scripts assume NCAR paths such as
`/glade/work/stevenxu/AMOC_models` and
`/glade/derecho/scratch/stevenxu/tmp`. Prefer CLI flags or top-level constants
for new paths instead of scattering more hard-coded paths. CESM1 is not part of
the active workflow; do not add it to new model loops unless the user asks.

## Current Numerical Defaults

`scripts/FgenCalculation/Fgenrun2_streaming.py` is the source of truth for
shared calculation defaults:

- scenario: `PIControl`;
- trailing window: 20 years / 240 months;
- density range: 1015 to 1030 kg m^-3;
- density step: 0.01 kg m^-3;
- excluded model: `CESM2`;
- North Atlantic mask: latitude greater than 45 degrees and normalized
  longitude from -90 to 60 degrees.

The latest all-model grid-alignment result in `girdtest_Fgen.ipynb` used a
1-degree target (`lat: 180`, `lon: 360`), 240 months, a 0.01 density step, and
the `original`, `nearest`, and `binned` methods. This is an experiment setting,
not a universal CLI default: `standard_grid()`, the standalone regridding
scripts, `test_small_batch.py`, and `comparason_test.py` still default to a
2-degree target unless `--resolution 1` is passed.

### Raw-First Multi-Model-Mean Experiment

`scripts/FgenCalculation/Fgen_mean_calculation.py` is a separate, intentional
raw-first experiment. It does not replace the native-grid workflow above. Its
current production configuration is:

- 16 equal-weight models with matching native `areacello`; exclude `CESM2`,
  `FGOALS-f3-L`, `FGOALS-g3`, and `SAM0-UNICON`;
- the final 240 consecutive year-months shared by `tos`, `sos`, `hfds`, and
  `wfo` for each model, giving 20 samples per calendar month;
- 12-month source chunks regridded with
  `grids.regrid_area_weighted_bins()` to the 1-degree `180 x 360` grid before
  forming per-model January-December climatologies;
- an equal-model mean only where all four variables are finite for a given
  model, cell, and month, with the shared contributor count saved as
  `model_count`;
- effective `areacello` formed by averaging each model's binned native ocean
  area over the complete 16-model cohort, treating unassigned cells as zero;
- Fgen calculated with the helpers in `Fgenrun2_streaming.py` at 0.05 kg m^-3
  density spacing, producing 302 bin centers that exactly match the legacy
  streaming profiles.

The default output and checkpoint directory is
`/glade/work/stevenxu/AMOC_models/MMM_binned_1deg_no_SAM0`. Its four field
products are `MMM_{tos,sos,hfds,wfo}_binned_1deg.nc`; the diagnostic is
`MMM_Fgen_binned_1deg.pkl`. The checkpoint fingerprints the scientific
configuration and every input path, size, and modification time. Do not reuse
this directory after changing the cohort or inputs; choose a new output
directory instead. `--resume` reuses validated per-model caches but rebuilds
the final MMM fields and Fgen result.

The latest validated production run is PBS job `6739144` (exit status 0,
16/16 models, no checkpoint errors). Preserve its report at
`/glade/u/home/stevenxu/Fgen_MMM.o6739144`.

## Important Folders and Files

- `README.md`: high-level project description, data layout, and method summary.
- `scripts/FgenCalculation/Fgenrun2_streaming.py`: preferred sequential
  multi-model Fgen driver. It processes one model at a time, saves atomically,
  supports resume/model filtering, and supplies helpers reused by the other
  streaming calculations.
- `scripts/FgenCalculation/Fgen_mean_calculation.py`: resumable raw-first
  binned MMM driver described above. It keeps 12-month climatology caches,
  requires the complete configured cohort before aggregation, writes four MMM
  NetCDFs, reopens them, and calls the existing streaming Fgen helpers.
- `scripts/FgenCalculation/run_Fgen_mean`: Derecho PBS wrapper for the current
  16-model no-SAM0 experiment. The model list and output directory are pinned
  explicitly so they cannot silently inherit a different checkpoint cohort.
- `scripts/FgenCalculation/AreaSumrun_streaming.py`: area where `fsurf < 0`,
  `rho > rho_min`, and the North Atlantic mask is true. The default threshold
  is 1025 kg m^-3.
- `scripts/FgenCalculation/AreaSumrun_streaming_no_fsurf_filter.py`: alternate
  AreaSum calculation with the density/spatial conditions but no `fsurf` sign
  filter. Keep its output distinct from the corrected filtered AreaSum.
- `scripts/FgenCalculation/AverageHeatFluxrun_streaming.py`: area-weighted raw
  `hfds` mean over a configurable region, defaulting to 50-70N and normalized
  longitude from -90 to 60 degrees.
- `scripts/FgenCalculation/Fgenrun2.py`: older all-model local workflow. Keep
  behavior-compatible unless the user explicitly asks to modernize it.
- `scripts/FgenCalculation/dataDownload.py`: ESGF CMIP6 piControl downloader
  for `tos`, `sos`, `hfds`, `wfo`, and `msftmz`.
- `scripts/FgenCalculation/areacelloDownload.py`: downloads matching
  `areacello` files for models already present under the downloads tree.
- `scripts/FgenCalculation/verifyDownloads.py`: verifies downloads and can
  stage passing files into the `/glade/work/stevenxu/AMOC_models` layout. Use
  `--verify-only` or `--dry-run` before moving data.
- `scripts/FgenCalculation/run` and `run2`: PBS wrappers for the downloader and
  streaming Fgen driver. `run2` uses `$PBS_O_WORKDIR`, so submit it from the
  intended working directory.
- `scripts/GridAlignment/README.md`: detailed method contract, limitations,
  tested grid families, and example commands. Read it before changing grid
  behavior.
- `scripts/GridAlignment/grids.py`: shared coordinate detection and regridding
  implementation for rectilinear, curvilinear, and unstructured grids.
- `scripts/GridAlignment/regrid_nearest.py`: spherical SciPy KD-tree nearest
  neighbor with a maximum-distance cutoff and optional ocean-area mask.
- `scripts/GridAlignment/regrid_binned.py`: source-area-weighted center-bin
  aggregation; also returns `source_area_sum` for integral checks.
- `scripts/GridAlignment/regrid_xesmf.py`: general xESMF experiment driver.
- `scripts/GridAlignment/regrid_regridder.py`: notebook-style periodic
  `xe.Regridder`, defaulting to `nearest_s2d`.
- `scripts/GridAlignment/test_small_batch.py`: one-variable, small-model method
  comparison with optional JSON reporting.
- `scripts/GridAlignment/comparason_test.py`: compares native Fgen integration
  with aligned methods after computing nonlinear diagnostics on the native
  grid. The misspelling is the checked-in filename; do not silently rename it.
- `scripts/GridAlignment/check_grid_sizes.py`: actually regrids one `tos`
  timestep and prints the returned shape. It defaults to binned 1-degree output.
- `scripts/GridAlignment/girdtest_Fgen.ipynb`: active grid-method evaluation
  notebook. Its misspelled filename is also intentional repository history.
- `scripts/GridAlignment/MMM_Comparason.ipynb`: executed comparison of the
  raw-first MMM Fgen with the equal-model mean of the same 16 legacy streaming
  profiles. Its misspelled filename is intentional; do not rename it.
- `scripts/FgenEvaluation/`: active Fgen, filtered/unfiltered AreaSum, average
  heat-flux, and AMOC comparison notebooks plus checked-in plots.
- `scripts/FgenCalculation/cloud_methods/`: Pangeo/intake alternatives. Treat
  these as experimental paths.
- `scripts/FgenCalculation/old_methods/` and
  `scripts/FgenEvaluation/oldEvaluation/`: historical work. Avoid editing
  unless the task specifically targets it.
- `scripts/sourceID/`: CMIP variable/source metadata notes.
- `output/`: small curated figures/model lists. Do not place large generated
  data here.
- `dask-scratch-space/`, `wget/`, and `__pycache__/`: generated or temporary
  workspace content; do not treat them as source.

## Run Commands

Use scratch output for focused Fgen checks:

```bash
python scripts/FgenCalculation/Fgenrun2_streaming.py \
  --models MIROC6 \
  --output /glade/derecho/scratch/stevenxu/tmp/MIROC6_Fgen.pkl
```

Resume a run or submit the existing PBS job:

```bash
python scripts/FgenCalculation/Fgenrun2_streaming.py --resume --save-every 1
qsub scripts/FgenCalculation/run2
```

Run or resume the current 16-model raw-first MMM experiment:

```bash
python scripts/FgenCalculation/Fgen_mean_calculation.py \
  --resume \
  --time-chunk 12
qsub /glade/u/home/stevenxu/AMOCproject/scripts/FgenCalculation/run_Fgen_mean
```

`run_Fgen_mean` uses `$PBS_O_WORKDIR` and joins stderr into stdout. Submit it
from the directory where the persistent `Fgen_MMM.o<job-id>` report should be
written. A configuration-fingerprint error means the current cohort or input
manifest differs from the checkpoint; do not bypass it by editing the stored
fingerprint or deleting validated production data.

Run the filtered and unfiltered AreaSum variants with different outputs:

```bash
python scripts/FgenCalculation/AreaSumrun_streaming.py --models MIROC6
python scripts/FgenCalculation/AreaSumrun_streaming_no_fsurf_filter.py --models MIROC6
```

Run the regional raw heat-flux calculation:

```bash
python scripts/FgenCalculation/AverageHeatFluxrun_streaming.py --models MIROC6
```

Download and staging checks:

```bash
python scripts/FgenCalculation/dataDownload.py --dry-run --models ACCESS-CM2 --max-files-per-variable 1
python scripts/FgenCalculation/areacelloDownload.py --list-targets
python scripts/FgenCalculation/verifyDownloads.py --verify-only --models ACCESS-CM2
```

Representative grid-alignment smoke batch:

```bash
python scripts/GridAlignment/test_small_batch.py \
  --models E3SM-1-0 MIROC6 ICON-ESM-LR \
  --methods nearest binned \
  --resolution 2 \
  --time-steps 1 \
  --report /glade/derecho/scratch/stevenxu/tmp/grid_alignment_report.json
```

Focused Fgen alignment comparison at the current experiment resolution:

```bash
python scripts/GridAlignment/comparason_test.py \
  --models MIROC6 \
  --methods original nearest binned \
  --resolution 1 \
  --last-n-months 12 \
  --output /glade/derecho/scratch/stevenxu/tmp/gridtest_Fgen_smoke.pkl
```

Verify actual returned grid sizes instead of inferring them from configuration:

```bash
python scripts/GridAlignment/check_grid_sizes.py \
  --models E3SM-1-0 MIROC6 ICON-ESM-LR \
  --method binned \
  --resolution 1
```

Only run non-dry-run downloads, staging, full multi-model calculations, or
multi-century regridding when the user has confirmed the data path and expected
runtime.

## Test and Validation Commands

There is no formal test suite. Start with syntax validation for Python edits:

```bash
find scripts -name '*.py' -exec python -m py_compile {} +
```

Then use the narrowest data-backed check that covers the change:

- downloader/staging edits: a single-model `--dry-run`, `--list-targets`, or
  `--verify-only` command;
- Fgen helper edits: one model with explicit scratch output;
- grid utility edits: `test_small_batch.py` with one timestep and models
  representing rectilinear (`E3SM-1-0`), curvilinear (`MIROC6`), and
  unstructured (`ICON-ESM-LR`) grids;
- target-grid edits: `check_grid_sizes.py` and verify exact `lat`/`lon` sizes;
- raw-first MMM edits: use a new scratch output directory, test checkpoint
  resume, and cover rectilinear (`E3SM-1-0`), curvilinear (`MIROC6`), and
  unstructured (`ICON-ESM-LR`) inputs before a full run. For production,
  require four identical `time=12, lat=180, lon=360` grids, January-December
  order, identical area/count fields, maximum assigned-area relative error
  below `1e-6`, 302 density bins, and exact density-coordinate equality with
  the selected legacy profiles;
- xESMF edits: run xESMF separately because ESMPy initialization can depend on
  the HPC environment. Test structured/curvilinear and unstructured inputs
  separately.

Use `--deep-check` with `verifyDownloads.py` only when NetCDF open/variable
validation is necessary; it is slower. A successful one-timestep regrid does
not establish conservation, so also check assigned/source area totals when the
change affects binned or conservative methods.

## Grid-Alignment Rules

- The shared target contract uses dimensions exactly `lat` and `lon`,
  longitude in `[-180, 180)`, cell-center coordinates, and `lat_b`/`lon_b`
  edges. A global 2-degree grid is `90 x 180`; a 1-degree grid is `180 x 360`.
- For `comparason_test.py` and the native-first grid-method comparison, compute
  nonlinear `rho`, `fsurf`, `heat_comp`, and `fw_comp` on the native model grid
  before alignment. The raw-first MMM experiment is the documented exception:
  it intentionally regrids `tos`, `sos`, `hfds`, and `wfo` first to measure a
  different operation order. Do not mix these two experiment definitions.
- Use native `areacello` for reference integrations. Nearest and bilinear
  remapping are not area-conserving.
- `regrid_area_weighted_bins()` is center-bin aggregation, not exact polygon
  overlap. Use `source_area_sum` for aligned integrations and validate the
  assigned-area error.
- `regrid_nearest()` supports all local grid families and applies a default
  maximum distance of three target cells. Pass an area dataset for ocean fields
  so finite land placeholders are excluded.
- The notebook-style `regridder` method maps `areacello` with xESMF
  `nearest_s2d`; its area result is not conservative. Keep it as a comparison
  method, not the integral-sensitive reference.
- xESMF bilinear/conservative methods require structured or curvilinear input
  and usable bounds. The installed xESMF does not support bilinear or
  conservative LocStream regridding for ICON's unstructured grid.
- Reuse xESMF weights for chunked production work. `grids.py` contains a scoped
  ESMPy compatibility shim for the installed environment; do not replace it
  with a global package or environment mutation.
- FGOALS area handling uses an `ACCESS-CM2` fallback alias in active helpers,
  but the grids do not truly match every alignment path. Treat FGOALS aligned
  output as method-specific and validate it rather than assuming conservation.

## Code Style

- Keep Python compatible with the `amoc-env` Python 3.11 stack.
- Prefer `argparse`; put runnable logic behind `main()` and
  `if __name__ == "__main__":`.
- For new or heavily edited scripts, order imports, constants, dataclasses,
  helpers, then `main()`.
- Prefer `pathlib.Path` in newer downloader/grid utilities. Follow the existing
  `os.path` style for small edits to older calculation scripts.
- Preserve CMIP names and staged-directory conventions: `tos`, `sos`, `hfds`,
  `wfo`, `msftmz`, and `areacello`; staged directories include
  `sea_surface_temperature`, `sea_surface_salinity`, `heatflux`, `waterflux`,
  and `areacello`.
- Keep xarray/dask work lazy where practical. Process one model at a time,
  choose bounded time chunks, and close datasets in `finally` blocks.
- Preserve support for `i/j`, `x/y`, `lat/lon`, and unstructured spatial
  conventions unless validation covers every affected grid family.
- Do not add broad warning suppression. Keep unavoidable suppression scoped and
  documented.
- Avoid notebook reformatting or clearing/re-executing outputs unless the task
  specifically requires notebook results.

## Generated Output Rules

- Do not commit large climate data or derived binary output: `*.nc`, `*.nc4`,
  `*.zarr/`, `*.pkl`, `*.part`, Dask scratch data, xESMF weight files, or ESGF
  downloads.
- Put smoke outputs and JSON reports under
  `/glade/derecho/scratch/stevenxu/tmp`; established project data belongs under
  `/glade/work/stevenxu/AMOC_models`.
- Streaming calculation and comparison scripts write through a temporary file
  and replace the destination atomically. Preserve this behavior.
- Checked-in figures and model lists under `output/` and
  `scripts/FgenEvaluation/*_plots/` are curated results. Regenerate them only
  when the user asks.
- Notebook outputs can be very large. Do not add execution noise or embedded
  images unless the task is specifically to regenerate notebook analysis.
- Preserve PBS stdout/stderr reports such as `Fgen_MMM.o<job-id>` unless the
  user explicitly asks to remove them. They are generated artifacts, but they
  are also the human-readable production record and should not be included in
  routine cleanup.
- Before editing, run `git status --short` and preserve unrelated user changes,
  including tracked notebook outputs and any currently tracked cache files.

## Secret and API-Key Safety

- Current ESGF and Pangeo catalog workflows use public HTTP/HTTPS endpoints and
  should not require API keys.
- Never commit credentials, tokens, cookies, private keys, `.netrc`, or
  environment dumps.
- If a future source needs credentials, read them from environment variables or
  user-managed files outside the repository. Document variable names, never
  their values.
- Treat PBS accounts, email addresses, and user-specific `/glade/...` paths as
  environment-specific configuration. Do not add or change personal values
  unless the user explicitly requests it.
