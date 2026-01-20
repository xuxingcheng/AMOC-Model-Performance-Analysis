# AMOC FAIR Emission Modeling

Personal workbench for simplifying FAIR model inputs by ranking/ filtering emission species and feeding FAIR with forecasted emissions (ARIMA and Chronos time-series models). Not intended as a polished release.

## What this repo is for
- Rank species importance and trim FAIR inputs to the essentials.
- Generate emission forecasts with ARIMA and Chronos ML models.
- Run FAIR with these pared-down inputs and compare against the original model.
- Inspect radiative forcing/temperature impacts and keep plots for quick review.



## Key notebooks
- `Original Model.ipynb` — Baseline FAIR runs.
- `Simplified_Model_Extension.ipynb` / `Simplified_Model_RCIMP.ipynb` — FAIR with species filtering and scenario variants.
- `Max Forcing Analysis.ipynb` — Ranks species (max forcing, coeff. of variation) to drop low-impact species.
- `ModelEvaluation_Extension.ipynb` / `ModelEvaluation_RCIMP.ipynb` — Compare simplified/extended models to the baseline.
- `Fair Tests.ipynb` — Scratchpad experiments.
- `examples/` — FAIR reference runs (CO₂ effects, CMIP6 SSPs, data import, equilibrium checks).
- `graph_outputs/` — Saved plots for before/after comparisons.
- `TOP_species_extension.txt`, `TOP_species_rcimp_modified.txt` — Current ranked/selected species lists.

## Typical workflow
1) Rank species in `Max Forcing Analysis.ipynb`; update the TOP_species files if the cutoff changes.
2) Fit ARIMA and Chronos forecasts for emissions; export forecasted series.
3) Feed forecasted/filtered emissions into the simplified FAIR notebooks and run.
4) Review outputs in `graph_outputs/` to see performance vs the original model.

## Notes
- This is for my own analyses; paths and assumptions are hard-coded in places.
- Ensure `graph_outputs/` exists before running notebooks that save figures.
- `.idea/` and other IDE artifacts are safe to ignore.
