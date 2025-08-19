# AMOC-Model-Performance-Analysis

This project analyzes the Atlantic Meridional Overturning Circulation (AMOC) using climate model outputs. It provides scripts and notebooks for data import, processing, visualization, and evaluation of AMOC-related variables under different scenarios.

## Directory Structure

- `output/`  
	Contains model availability information and summary plots.
	- `available_models.txt`: List of available models.
	- `msftmz_all_models.png`: Visualization of AMOC streamfunction across models.

- `scripts/`  
	Python and Jupyter scripts for data analysis and visualization.
	- `findModels.ipynb`: Identifies available models and their data.
	- `model_test.ipynb`: Tests and explores model data.
	- `ssp370_model_eva.ipynb`: Evaluates AMOC under SSP370 scenario.
	- `sourceID/`: Contains raw data files for various ocean variables (e.g., temperature, salinity, streamfunction).

## Main Features

- **Data Import & Concatenation**: Loads and concatenates NetCDF files for multiple models and scenarios using xarray.
- **Unit Conversion**: Converts physical units (e.g., kg/s to Sverdrup) for AMOC streamfunction.
- **Visualization**: Plots AMOC strength and trends for individual models and across all models, including rolling mean smoothing.
- **Model Evaluation**: Compares AMOC performance and variability between models under different climate scenarios.

## Requirements

- Python 3.x
- xarray
- numpy
- matplotlib
- glob
- cartopy (for map visualizations)
- gsw (for seawater calculations)

Install dependencies with:
```bash
pip install xarray numpy matplotlib cartopy gsw
```

## Usage

1. Place NetCDF files for each model and scenario in the appropriate directory (e.g., `/glade/work/<username>/AMOC_models/scenarios/ssp370`).
2. Open and run the Jupyter notebooks in `scripts/` to process and visualize the data.
3. Review output plots and summary files in the `output/` directory.

## Example Workflow

- Run `ssp370_model_eva.ipynb` to evaluate AMOC under the SSP370 scenario.
- Use `findModels.ipynb` to list available models and their data coverage.
- Visualize AMOC streamfunction and trends using the provided plotting scripts.

## License

This project is for research and educational purposes. Please cite appropriately if used in publications.
# AMOC-Model-Performance-Analysis