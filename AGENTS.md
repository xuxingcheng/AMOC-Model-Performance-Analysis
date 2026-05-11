# Codex Guide for AMOCproject

## Project Purpose

This repository is an active research workspace for AMOC-relevant surface-forcing diagnostics. Most workflows compute or evaluate `Fgen` from CMIP/CESM ocean output using `xarray`, `dask`, `gsw`, and related geoscience packages.

The code is script/notebook driven, not packaged as an installable Python module. There is no checked-in `pyproject.toml`, `requirements.txt`, conda environment file, or automated test config.

## Setup

Run commands from the repository root unless a script says otherwise:

```bash
cd /glade/u/home/stevenxu/AMOCproject
module load conda/latest
conda activate amoc-env
```


Current scripts assume NCAR/HPC paths such as `/glade/work/stevenxu/AMOC_models` and `/glade/work/stevenxu/CESM1`. Prefer adding CLI flags or top-level constants for new paths instead of scattering new hard-coded paths. The CESM1 model is unused for now, ignore it when coding to avoid redundency.

## Important Folders and Files

- `README.md`: high-level project description and method summary.
- `scripts/FgenCalculation/Fgenrun2_streaming.py`: preferred sequential multi-model Fgen driver. It processes one model at a time, checkpoints with pickle, supports `--resume`, `--models`, `--exclude-models`, `--max-models`, and custom `--output`.
- `scripts/FgenCalculation/Fgenrun2.py`: older all-model local workflow. Keep behavior-compatible unless the user explicitly asks to modernize it.
- `scripts/FgenCalculation/dataDownload.py`: ESGF CMIP6 piControl downloader for `tos`, `sos`, `hfds`, `wfo`, and `msftmz`.
- `scripts/FgenCalculation/areacelloDownload.py`: downloads matching `areacello` files for models already present under the downloads tree.
- `scripts/FgenCalculation/verifyDownloads.py`: verifies downloaded files and can stage passing files into the target `/glade/work/stevenxu/AMOC_models` layout. Use `--verify-only` or `--dry-run` before moving data.
- `scripts/FgenCalculation/run`: PBS wrapper for `dataDownload.py`.
- `scripts/FgenCalculation/run2`: PBS wrapper for `Fgenrun2_streaming.py`; submit from the desired working directory because it uses `$PBS_O_WORKDIR`.
- `scripts/FgenCalculation/cloud_methods/`: Pangeo/intake cloud workflows. Treat these as alternate or experimental paths.
- `scripts/FgenCalculation/old_methods/`: historical scripts and notebooks. Avoid editing unless the task specifically targets them.
- `scripts/FgenEvaluation/`: evaluation notebooks and plots.
- `scripts/sourceID/`: CMIP variable/source metadata notes.
- `output/`: small checked-in figures and model lists. Do not place large generated data here.
- `dask-scratch-space/`, `wget/`, `__pycache__/`: generated or temporary workspace content.

## Run Commands

Local smoke run for the streaming Fgen workflow:

```bash
python scripts/FgenCalculation/Fgenrun2_streaming.py --max-models 1 --output /glade/derecho/scratch/stevenxu/tmp/Fgen_smoke.pkl --save-every 1
```

Resume or restrict a production-style run:

```bash
python scripts/FgenCalculation/Fgenrun2_streaming.py --resume --save-every 1
python scripts/FgenCalculation/Fgenrun2_streaming.py --models ACCESS-CM2 --output /glade/derecho/scratch/stevenxu/tmp/ACCESS-CM2_Fgen.pkl
```

Submit the existing PBS streaming job:

```bash
qsub scripts/FgenCalculation/run2
```

Download and staging workflow:

```bash
python scripts/FgenCalculation/dataDownload.py --dry-run --models ACCESS-CM2 --max-files-per-variable 1
python scripts/FgenCalculation/areacelloDownload.py --list-targets
python scripts/FgenCalculation/verifyDownloads.py --verify-only --models ACCESS-CM2
```

Only run non-dry-run download, staging, or full production commands when the user has confirmed the target data path and expected runtime.

## Test and Validation Commands

There is no formal test suite. Use the narrowest validation that matches the change:

```bash
find scripts -name '*.py' -exec python -m py_compile {} +
```

For changes to download or staging logic:

```bash
python scripts/FgenCalculation/dataDownload.py --dry-run --models ACCESS-CM2 --max-files-per-variable 1
python scripts/FgenCalculation/areacelloDownload.py --list-targets --models ACCESS-CM2
python scripts/FgenCalculation/verifyDownloads.py --verify-only --models ACCESS-CM2
```

For changes to Fgen computation:

```bash
python scripts/FgenCalculation/Fgenrun2_streaming.py --max-models 1 --output /glade/derecho/scratch/stevenxu/tmp/Fgen_smoke.pkl --save-every 1
```

Use `--deep-check` with `verifyDownloads.py` only when NetCDF open/variable validation is necessary; it is slower.

## Code Style

- Keep Python code compatible with the `amoc-env` Python 3.11 stack.
- Prefer `argparse` for script options and put runnable logic behind `main()` plus `if __name__ == "__main__":`.
- Keep imports, constants, dataclasses, helpers, then `main()` in that order for new or heavily edited scripts.
- Prefer `pathlib.Path` in downloader/staging utilities; follow existing `os.path` style when making small edits to older calculation scripts.
- Preserve existing CMIP variable names and directory conventions: `tos`, `sos`, `hfds`, `wfo`, `msftmz`, and `areacello`; staged directories include `sea_surface_temperature`, `sea_surface_salinity`, `heatflux`, `waterflux`, and `areacello`.
- Keep xarray/dask work lazy where possible. Avoid loading all models at once; the streaming driver intentionally processes one model at a time and closes datasets in `finally` blocks.
- Be careful with grid handling. Existing code supports multiple spatial conventions such as `i/j`, `x/y`, and `lat/lon`; do not simplify this unless the user asks and validation covers affected models.
- Do not add broad warning suppression unless it is scoped and explained. Existing scripts suppress known serialization/ID warnings.
- Keep comments useful and short. Avoid reformatting notebooks or legacy scripts just for style.

## Generated Output Rules

- Do not commit large climate data or derived binary outputs: `*.nc`, `*.nc4`, `*.zarr/`, `*.pkl`, `*.part`, dask scratch data, or downloaded ESGF files.
- Prefer scratch or work locations for generated outputs, especially `/glade/derecho/scratch/stevenxu/tmp` for smoke runs and `/glade/work/stevenxu/AMOC_models` for established project data.
- `Fgenrun2_streaming.py` writes atomically through `<output>.tmp`; leave this behavior intact.
- The small files currently in `output/` are curated figures/model lists. Update them only when the user asks for regenerated evaluation outputs.
- Notebook outputs can be very large. When editing notebooks, avoid adding execution noise or embedded images unless the task is specifically to regenerate notebook results.
- Before editing, check `git status --short` and preserve unrelated user changes.

## Secret and API-Key Safety

- Current ESGF and Pangeo catalog workflows use public HTTP/HTTPS endpoints and should not require API keys.
- Never commit credentials, tokens, cookies, private keys, `.netrc`, or environment dumps.
- If a future data source needs credentials, read them from environment variables or user-managed files outside the repository, and document the variable names without recording values.
- Treat PBS account IDs, email addresses, and user-specific `/glade/...` paths as environment-specific configuration. Do not change or add personal values unless the user explicitly requests it.
