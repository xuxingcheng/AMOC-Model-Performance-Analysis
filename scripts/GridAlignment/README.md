# Standard latitude/longitude grid experiments

These scripts convert CMIP ocean fields from rectilinear, curvilinear, or
unstructured native grids to a shared regular latitude/longitude grid.

## Standard grid contract

The recommended and default target is a global 2-degree cell-center grid:

- dimensions are exactly `lat` and `lon`;
- shape is `lat: 90, lon: 180`;
- `lat` runs from `-89` to `89`;
- `lon` runs from `-179` to `179`;
- longitude uses the `[-180, 180)` convention;
- `lat_b` and `lon_b` contain target cell edges.

Two degrees is no finer than the coarsest tested native grids and reduces empty
cells in center-bin aggregation. Set `--resolution 1` when a 1-degree map is
specifically required.

The standalone scripts are eager experimental tools and default to one
timestep. Pass `--time-steps 0` only when intentionally processing the full
input file. A production multi-century workflow should call these methods in
time chunks and reuse xESMF weights.

## Methods

`regrid_nearest.py`

- Supports every local grid type, including ICON's unstructured grid.
- Uses a spherical SciPy KD-tree and a maximum source-to-target distance.
- Accepts `--area-input`; use it for ocean fields so finite land placeholders
  are excluded using `areacello > 0`.
- Good robust baseline and useful for masks or categorical fields.
- Does not conserve area integrals and should not be used for `areacello`.

`regrid_binned.py`

- Supports every local grid type.
- Assigns each source cell center to a target cell and computes a
  source-area-weighted mean.
- Writes `source_area_sum`, the conservative sum of source area assigned to
  each target cell.
- Accepts source/area coordinate differences below `0.001` degrees to account
  for float32/float64 serialization differences.
- Conservatively sums integer-factor finer area grids to the source field grid;
  this handles the staged GISS `180 x 288` area with `90 x 144` fields.
- Recommended local method for integral-sensitive experiments when exact
  polygon-overlap conservative weights are unavailable.
- It is center-bin aggregation, not exact polygon-overlap remapping. Use a
  target resolution no finer than the source grid to reduce empty cells.

`regrid_xesmf.py`

- Supports bilinear, nearest, and conservative xESMF methods for structured
  and curvilinear grids.
- The installed xESMF 0.8.7 only supports nearest methods for unstructured
  LocStream input.
- Conservative methods require usable cell bounds and can fail on malformed
  native-grid polygons.
- Accepts `--area-input` to provide an ocean mask for structured
  non-conservative methods. Installed xESMF LocStream and conservative paths
  do not reliably support this mask.
- xESMF nearest has no maximum-distance cutoff and can fill distant target
  land cells from the nearest ocean point. Prefer `regrid_nearest.py` when
  retaining an approximate coastline matters.
- The local ESMPy 8.4.2 package omits optional metadata fields that its import
  code requires. `grids.py` applies a scoped import-time compatibility shim;
  it does not modify the conda environment.

`regrid_regridder.py`

- Implements the notebook-style pattern from `DemonstrateRegridding.ipynb`:
  construct `xe.Regridder(source, target, method, periodic=True)` and apply it
  to the field.
- Defaults to `nearest_s2d`, matching the model-loop example in the notebook.
- Uses the shared standard grid coordinates and local coordinate detection, so
  it can be run through the same command-line interface as the other methods.
- In `comparason_test.py`, the `regridder` method applies the same
  `nearest_s2d` regridder to `rho`, `fsurf`, `heat_comp`, `fw_comp`, and
  `areacello`.
- It is useful for comparing against the notebook workflow. Regridded
  `areacello` is not area-conserving, so use `binned` or exact conservative
  weights for integral-sensitive production calculations.

## Small-batch comparison

The default batch represents all local grid families:

```bash
python scripts/GridAlignment/test_small_batch.py \
  --models E3SM-1-0 MIROC6 ICON-ESM-LR \
  --methods nearest binned \
  --resolution 2 \
  --time-steps 1 \
  --report /glade/derecho/scratch/stevenxu/tmp/grid_alignment_report.json
```

Run xESMF separately because ESMPy initialization can depend on the HPC
execution environment:

```bash
python scripts/GridAlignment/test_small_batch.py \
  --models E3SM-1-0 MIROC6 ICON-ESM-LR \
  --methods regridder xesmf-bilinear xesmf-nearest xesmf-conservative \
  --resolution 2 \
  --time-steps 1
```

## Recommendation for Fgen calculations

Do not automatically regrid `tos`, `sos`, `hfds`, and `wfo` before computing
nonlinear diagnostics. Regridding first can change density and surface forcing,
and bilinear or nearest-neighbor remapping does not conserve area integrals.

The safer workflow is:

1. align each model's variables on that model's native ocean grid;
2. compute `rho`, `fsurf`, and other nonlinear diagnostics on the native grid;
3. use native `areacello` for native-grid integrations;
4. regrid the resulting diagnostic only when a shared map or cell-by-cell
   cross-model comparison is required;
5. use `source_area_sum` or exact conservative xESMF weights for integrals.

## Test results

Small batches used one timestep per model on the default 2-degree target.

- `tos`: nearest produced the standard grid for all 20 staged models. Binned
  passed for all 18 models with local `areacello`; only the two FGOALS models
  lack area files.
- `sos` and `hfds`: nearest and binned passed for representative rectilinear,
  curvilinear, and unstructured models.
- `wfo`: nearest and binned passed for E3SM-1-0, MIROC6, ICON-ESM-LR,
  ACCESS-ESM1-5, and CanESM5.
- Binned valid-area relative error was at most about `1.3e-7` in these runs.
- Binned aggregation requires a matching local `areacello`; the current
  FGOALS-f3-L and FGOALS-g3 staging does not provide one.
- xESMF bilinear and nearest passed for structured/curvilinear models; xESMF
  nearest passed for ICON. xESMF conservative passed for E3SM-1-0 and MIROC6.
- Installed xESMF cannot apply bilinear or conservative methods to ICON's
  unstructured LocStream grid.

JSON reports from the development runs are under:

```text
/glade/derecho/scratch/stevenxu/tmp/grid_alignment*_report.json
```

## Fgen method comparison

`comparason_test.py` compares the production-style native-grid Fgen result with
the configured alignment methods:

- `original`: integrate native-grid `rho` and `fsurf` with native `areacello`;
- `nearest`: regrid native-grid diagnostics by spherical nearest neighbor, then
  integrate with native area assigned to the standard grid;
- `binned`: source-area-weight native-grid diagnostics into standard-grid cells,
  then integrate with the same assigned standard-grid area.
- `regridder`: use the `DemonstrateRegridding.ipynb` xESMF `nearest_s2d`
  periodic regridder pattern for diagnostics and `areacello`.

All methods compute nonlinear `rho` and `fsurf` diagnostics on the native model
grid first. This isolates the effect of grid alignment from nonlinear diagnostic
calculation.

Run the tested default small batch:

```bash
python scripts/GridAlignment/comparason_test.py \
  --models E3SM-1-0 MIROC6 ICON-ESM-LR \
  --methods original nearest binned regridder \
  --last-n-months 12 \
  --output /glade/derecho/scratch/stevenxu/tmp/gridtest_Fgen.pkl
```

The output is a pickle payload containing:

```text
metadata
results -> method -> model -> Fgen DataFrame
timings
errors
```

Open `girdtest_Fgen.ipynb` to compare density profiles, heat/freshwater
components, minimum Fgen, negative-density integrals, area totals, differences
from the original method, and runtime.
