"""Shared utilities for converting CMIP ocean grids to regular latitude/longitude.

The standard grid uses cell-center coordinates and always returns dimensions named
``lat`` and ``lon``.  The helpers here support rectilinear, curvilinear, and
one-dimensional unstructured source grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr


LATITUDE_NAMES = ("lat", "latitude", "nav_lat", "TLAT")
LONGITUDE_NAMES = ("lon", "longitude", "nav_lon", "TLONG")
COORDINATE_MATCH_TOLERANCE_DEGREES = 1.0e-3


@dataclass(frozen=True)
class SourceGeometry:
    latitude: xr.DataArray
    longitude: xr.DataArray
    spatial_dims: tuple[str, ...]
    leading_dims: tuple[str, ...]
    grid_type: str


def normalize_longitude(longitude):
    """Normalize longitude values to the half-open interval [-180, 180)."""
    return ((longitude + 180.0) % 360.0) - 180.0


def standard_grid(
    resolution: float = 2.0,
    lat_min: float = -90.0,
    lat_max: float = 90.0,
    lon_min: float = -180.0,
    lon_max: float = 180.0,
) -> xr.Dataset:
    """Create a regular global grid with center coordinates and cell bounds."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    lat_edges = np.arange(lat_min, lat_max + resolution * 0.5, resolution)
    lon_edges = np.arange(lon_min, lon_max + resolution * 0.5, resolution)
    if not np.isclose(lat_edges[-1], lat_max) or not np.isclose(lon_edges[-1], lon_max):
        raise ValueError("resolution must divide both latitude and longitude ranges")

    lat = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    lon = (lon_edges[:-1] + lon_edges[1:]) / 2.0
    return xr.Dataset(
        coords={
            "lat": (
                "lat",
                lat,
                {"standard_name": "latitude", "units": "degrees_north", "bounds": "lat_b"},
            ),
            "lon": (
                "lon",
                lon,
                {"standard_name": "longitude", "units": "degrees_east", "bounds": "lon_b"},
            ),
            "lat_b": ("lat_b", lat_edges),
            "lon_b": ("lon_b", lon_edges),
        },
        attrs={
            "grid_type": "regular_lat_lon",
            "resolution_degrees": resolution,
            "longitude_convention": "[-180, 180)",
        },
    )


def _find_coordinate_name(
    dataset: xr.Dataset,
    candidates: Iterable[str],
    standard_name: str,
) -> str:
    for name in candidates:
        if name in dataset:
            return name

    for name, variable in dataset.variables.items():
        if variable.attrs.get("standard_name") == standard_name:
            return name

    raise KeyError(f"Could not find a {standard_name} coordinate")


def source_geometry(dataset: xr.Dataset, variable: str | xr.DataArray) -> SourceGeometry:
    """Find and broadcast source latitude/longitude to a variable's spatial grid."""
    data_array = dataset[variable] if isinstance(variable, str) else variable
    lat_name = _find_coordinate_name(dataset, LATITUDE_NAMES, "latitude")
    lon_name = _find_coordinate_name(dataset, LONGITUDE_NAMES, "longitude")
    latitude = dataset[lat_name]
    longitude = dataset[lon_name]
    if latitude.ndim == longitude.ndim == 1:
        grid_type = "unstructured" if latitude.dims == longitude.dims else "rectilinear"
    else:
        grid_type = "curvilinear"

    coordinate_dims = set(latitude.dims) | set(longitude.dims)
    spatial_dims = tuple(dim for dim in data_array.dims if dim in coordinate_dims)
    leading_dims = tuple(dim for dim in data_array.dims if dim not in spatial_dims)
    if not spatial_dims:
        raise ValueError(
            f"{data_array.name or 'variable'} has no dimensions shared with "
            f"{lat_name}/{lon_name}"
        )
    if any(dim not in data_array.dims for dim in coordinate_dims):
        raise ValueError(
            f"Coordinate dimensions {sorted(coordinate_dims)} are not all present in "
            f"{data_array.name or 'variable'} dimensions {data_array.dims}"
        )

    latitude, longitude = xr.broadcast(latitude, longitude)
    latitude = latitude.transpose(*spatial_dims)
    longitude = longitude.transpose(*spatial_dims)
    return SourceGeometry(latitude, longitude, spatial_dims, leading_dims, grid_type)


def describe_grid(dataset: xr.Dataset, variable: str | xr.DataArray) -> dict[str, object]:
    geometry = source_geometry(dataset, variable)
    return {
        "grid_type": geometry.grid_type,
        "spatial_dims": geometry.spatial_dims,
        "shape": tuple(geometry.latitude.sizes[dim] for dim in geometry.spatial_dims),
        "latitude_name": geometry.latitude.name,
        "longitude_name": geometry.longitude.name,
    }


def _leading_coords(data_array: xr.DataArray, leading_dims: tuple[str, ...]):
    return {
        dim: data_array.coords[dim]
        for dim in leading_dims
        if dim in data_array.coords and data_array.coords[dim].dims == (dim,)
    }


def _target_mesh(target: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    longitude, latitude = np.meshgrid(target["lon"].values, target["lat"].values)
    return latitude, longitude


def _lat_lon_to_xyz(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat_radians = np.deg2rad(latitude)
    lon_radians = np.deg2rad(longitude)
    cos_lat = np.cos(lat_radians)
    return np.column_stack(
        (
            cos_lat * np.cos(lon_radians),
            cos_lat * np.sin(lon_radians),
            np.sin(lat_radians),
        )
    )


def regrid_nearest(
    dataset: xr.Dataset,
    variable: str,
    target: xr.Dataset,
    max_distance_degrees: float | None = None,
    area_dataset: xr.Dataset | None = None,
    area_variable: str = "areacello",
) -> xr.DataArray:
    """Regrid any supported source grid using spherical nearest neighbors."""
    from scipy.spatial import cKDTree

    data_array = dataset[variable]
    geometry = source_geometry(dataset, data_array)
    transposed = data_array.transpose(*geometry.leading_dims, *geometry.spatial_dims)
    values = np.asarray(transposed.values)
    leading_shape = values.shape[: len(geometry.leading_dims)]
    source_shape = values.shape[len(geometry.leading_dims) :]
    flat_values = values.reshape((-1, int(np.prod(source_shape))))

    source_lat = np.asarray(geometry.latitude.values).ravel()
    source_lon = normalize_longitude(np.asarray(geometry.longitude.values).ravel())
    target_lat, target_lon = _target_mesh(target)
    target_xyz = _lat_lon_to_xyz(target_lat.ravel(), target_lon.ravel())
    valid_geometry = np.isfinite(source_lat) & np.isfinite(source_lon)
    if area_dataset is not None:
        source_area = _matching_source_weights(dataset, variable, area_dataset, area_variable)
        valid_geometry &= np.isfinite(source_area) & (source_area > 0)

    if max_distance_degrees is None:
        resolution = float(target.attrs.get("resolution_degrees", 1.0))
        max_distance_degrees = 3.0 * resolution
    max_chord_distance = 2.0 * np.sin(np.deg2rad(max_distance_degrees) / 2.0)

    output = np.full((flat_values.shape[0], target_xyz.shape[0]), np.nan, dtype=np.float64)
    for index, source_slice in enumerate(flat_values):
        valid = valid_geometry & np.isfinite(source_slice)
        if not np.any(valid):
            continue
        tree = cKDTree(_lat_lon_to_xyz(source_lat[valid], source_lon[valid]))
        distances, source_indices = tree.query(
            target_xyz,
            k=1,
            distance_upper_bound=max_chord_distance,
            workers=1,
        )
        mapped = np.isfinite(distances)
        valid_values = source_slice[valid]
        output[index, mapped] = valid_values[source_indices[mapped]]

    output_shape = leading_shape + (target.sizes["lat"], target.sizes["lon"])
    coords = _leading_coords(transposed, geometry.leading_dims)
    coords.update({"lat": target["lat"], "lon": target["lon"]})
    result = xr.DataArray(
        output.reshape(output_shape),
        dims=geometry.leading_dims + ("lat", "lon"),
        coords=coords,
        name=data_array.name,
        attrs=data_array.attrs.copy(),
    )
    result.attrs.update(
        regrid_method="scipy_spherical_nearest",
        max_distance_degrees=max_distance_degrees,
        source_ocean_mask=int(area_dataset is not None),
    )
    return result


def _matching_source_weights(
    dataset: xr.Dataset,
    variable: str,
    area_dataset: xr.Dataset,
    area_variable: str,
) -> np.ndarray:
    source = source_geometry(dataset, variable)
    area = source_geometry(area_dataset, area_variable)
    if source.spatial_dims != area.spatial_dims:
        raise ValueError(
            f"Area spatial dimensions {area.spatial_dims} do not match "
            f"source dimensions {source.spatial_dims}"
        )

    source_lat = np.asarray(source.latitude.values)
    source_lon = normalize_longitude(np.asarray(source.longitude.values))
    area_lat = np.asarray(area.latitude.values)
    area_lon = normalize_longitude(np.asarray(area.longitude.values))
    weights = area_dataset[area_variable].transpose(*area.spatial_dims).squeeze(drop=True)

    if source_lat.shape != area_lat.shape:
        if source.spatial_dims != area.spatial_dims:
            raise ValueError("Area latitude/longitude do not match the source grid")
        factors = {}
        for dim in source.spatial_dims:
            source_size = source.latitude.sizes[dim]
            area_size = area.latitude.sizes[dim]
            if area_size % source_size != 0:
                raise ValueError("Area latitude/longitude do not match the source grid")
            factors[dim] = area_size // source_size
        if any(factor < 1 for factor in factors.values()):
            raise ValueError("Area latitude/longitude do not match the source grid")

        weights = weights.coarsen(factors, boundary="exact").sum()
        area_latitude = area.latitude.coarsen(factors, boundary="exact").mean()
        longitude_radians = np.deg2rad(area.longitude)
        area_longitude = np.rad2deg(
            np.arctan2(
                np.sin(longitude_radians).coarsen(factors, boundary="exact").mean(),
                np.cos(longitude_radians).coarsen(factors, boundary="exact").mean(),
            )
        )
        area_lat = np.asarray(area_latitude.values)
        area_lon = normalize_longitude(np.asarray(area_longitude.values))

    longitude_difference = normalize_longitude(source_lon - area_lon)
    if not np.allclose(
        source_lat,
        area_lat,
        rtol=0.0,
        atol=COORDINATE_MATCH_TOLERANCE_DEGREES,
        equal_nan=True,
    ) or not np.allclose(
        longitude_difference,
        0.0,
        rtol=0.0,
        atol=COORDINATE_MATCH_TOLERANCE_DEGREES,
        equal_nan=True,
    ):
        raise ValueError("Area latitude/longitude do not match the source grid")

    return np.asarray(weights.values).ravel()


def regrid_area_weighted_bins(
    dataset: xr.Dataset,
    variable: str,
    target: xr.Dataset,
    area_dataset: xr.Dataset,
    area_variable: str = "areacello",
) -> xr.Dataset:
    """Aggregate source cell centers into target bins using source-cell area."""
    data_array = dataset[variable]
    geometry = source_geometry(dataset, data_array)
    transposed = data_array.transpose(*geometry.leading_dims, *geometry.spatial_dims)
    values = np.asarray(transposed.values)
    leading_shape = values.shape[: len(geometry.leading_dims)]
    source_shape = values.shape[len(geometry.leading_dims) :]
    flat_values = values.reshape((-1, int(np.prod(source_shape))))
    source_area = _matching_source_weights(dataset, variable, area_dataset, area_variable)

    source_lat = np.asarray(geometry.latitude.values).ravel()
    source_lon = normalize_longitude(np.asarray(geometry.longitude.values).ravel())
    lat_edges = np.asarray(target["lat_b"].values)
    lon_edges = np.asarray(target["lon_b"].values)
    lat_index = np.searchsorted(lat_edges, source_lat, side="right") - 1
    lon_index = np.searchsorted(lon_edges, source_lon, side="right") - 1
    n_lat = target.sizes["lat"]
    n_lon = target.sizes["lon"]
    n_target = n_lat * n_lon
    valid_geometry = (
        np.isfinite(source_lat)
        & np.isfinite(source_lon)
        & np.isfinite(source_area)
        & (source_area > 0)
        & (lat_index >= 0)
        & (lat_index < n_lat)
        & (lon_index >= 0)
        & (lon_index < n_lon)
    )
    flat_target_index = lat_index * n_lon + lon_index

    output = np.full((flat_values.shape[0], n_target), np.nan, dtype=np.float64)
    area_sum = np.zeros((flat_values.shape[0], n_target), dtype=np.float64)
    for index, source_slice in enumerate(flat_values):
        valid = valid_geometry & np.isfinite(source_slice)
        if not np.any(valid):
            continue
        bins = flat_target_index[valid]
        weights = source_area[valid]
        weighted_sum = np.bincount(
            bins,
            weights=source_slice[valid] * weights,
            minlength=n_target,
        )
        area_sum[index] = np.bincount(bins, weights=weights, minlength=n_target)
        covered = area_sum[index] > 0
        output[index, covered] = weighted_sum[covered] / area_sum[index, covered]

    output_shape = leading_shape + (n_lat, n_lon)
    dims = geometry.leading_dims + ("lat", "lon")
    coords = _leading_coords(transposed, geometry.leading_dims)
    coords.update({"lat": target["lat"], "lon": target["lon"]})
    result = xr.Dataset(
        {
            variable: xr.DataArray(
                output.reshape(output_shape),
                dims=dims,
                coords=coords,
                attrs=data_array.attrs.copy(),
            ),
            "source_area_sum": xr.DataArray(
                area_sum.reshape(output_shape),
                dims=dims,
                coords=coords,
                attrs={
                    "long_name": "sum of valid source-cell area assigned to target cell",
                    "units": area_dataset[area_variable].attrs.get("units", "m2"),
                },
            ),
        },
        attrs={
            "regrid_method": "source_area_weighted_center_bin_average",
            "note": (
                "Source cells are assigned by their centers. source_area_sum is "
                "conservative, but this is not polygon-overlap conservative remapping."
            ),
        },
    )
    return result


def source_valid_area_totals(
    dataset: xr.Dataset,
    variable: str,
    area_dataset: xr.Dataset,
    area_variable: str = "areacello",
) -> np.ndarray:
    """Return source-cell area totals where a field and its geometry are valid."""
    data_array = dataset[variable]
    geometry = source_geometry(dataset, data_array)
    transposed = data_array.transpose(*geometry.leading_dims, *geometry.spatial_dims)
    values = np.asarray(transposed.values)
    source_shape = values.shape[len(geometry.leading_dims) :]
    flat_values = values.reshape((-1, int(np.prod(source_shape))))
    source_area = _matching_source_weights(dataset, variable, area_dataset, area_variable)
    source_lat = np.asarray(geometry.latitude.values).ravel()
    source_lon = normalize_longitude(np.asarray(geometry.longitude.values).ravel())
    valid_geometry = (
        np.isfinite(source_lat)
        & np.isfinite(source_lon)
        & np.isfinite(source_area)
        & (source_area > 0)
        & (source_lat >= -90.0)
        & (source_lat < 90.0)
        & (source_lon >= -180.0)
        & (source_lon < 180.0)
    )
    return np.asarray(
        [
            source_area[valid_geometry & np.isfinite(source_slice)].sum()
            for source_slice in flat_values
        ]
    )


def regrid_with_xesmf(
    dataset: xr.Dataset,
    variable: str,
    target: xr.Dataset,
    method: str = "bilinear",
    weights_file: str | Path | None = None,
    area_dataset: xr.Dataset | None = None,
    area_variable: str = "areacello",
) -> xr.DataArray:
    """Regrid with xESMF; unstructured inputs use LocStream nearest-neighbor."""
    xe = _import_xesmf()

    # xESMF/cf-xarray may update coordinate metadata while constructing weights.
    dataset = dataset.copy(deep=False)
    data_array = dataset[variable]
    geometry = source_geometry(dataset, data_array)
    locstream_in = geometry.grid_type == "unstructured"
    if locstream_in and method not in {"nearest_s2d", "nearest_d2s"}:
        raise ValueError(
            "Installed xESMF LocStream support only permits nearest_s2d or nearest_d2s "
            "for unstructured source grids"
        )

    source_mask = None
    can_apply_source_mask = (
        area_dataset is not None
        and not locstream_in
        and method not in {"conservative", "conservative_normed"}
    )
    if can_apply_source_mask:
        source_area = _matching_source_weights(dataset, variable, area_dataset, area_variable)
        source_mask = np.isfinite(source_area) & (source_area > 0)

    if method in {"conservative", "conservative_normed"}:
        if locstream_in:
            raise ValueError("Conservative xESMF regridding is unavailable for LocStream input")
        source_grid = dataset
    else:
        source_grid = xr.Dataset(
            coords={
                "lat": geometry.latitude,
                "lon": geometry.longitude,
            }
        )

    if source_mask is not None and not locstream_in:
        mask = xr.DataArray(
            source_mask.reshape(tuple(data_array.sizes[dim] for dim in geometry.spatial_dims)),
            dims=geometry.spatial_dims,
        )
        source_grid = source_grid.assign(mask=mask.astype(np.int32))

    kwargs = {
        "locstream_in": locstream_in,
        "periodic": False,
        "unmapped_to_nan": True,
        "ignore_degenerate": True,
    }
    if weights_file is not None:
        weights_path = Path(weights_file)
        kwargs["filename"] = str(weights_path)
        kwargs["reuse_weights"] = weights_path.exists()

    regridder = xe.Regridder(source_grid, target, method, **kwargs)
    result = regridder(data_array)
    result = result.assign_coords(lat=target["lat"], lon=target["lon"])
    result.attrs = data_array.attrs.copy()
    result.attrs["regrid_method"] = f"xesmf_{method}"
    result.attrs["source_ocean_mask"] = int(source_mask is not None)
    return result


def _import_xesmf():
    """Import xESMF despite missing optional fields in ESMPy 8.4.2 metadata."""
    import importlib.metadata as importlib_metadata

    original_metadata = importlib_metadata.metadata

    def metadata_with_esmpy_defaults(distribution_name):
        metadata = original_metadata(distribution_name)
        if distribution_name == "esmpy":
            defaults = {
                "Author": "ESMF Core Team",
                "Home-page": "https://earthsystemmodeling.org/",
                "obsoletes": "",
            }
            for key, value in defaults.items():
                if metadata.get(key) is None:
                    metadata[key] = value
        return metadata

    try:
        importlib_metadata.metadata = metadata_with_esmpy_defaults
        import xesmf
    finally:
        importlib_metadata.metadata = original_metadata
    return xesmf


def open_dataset(path: str | Path) -> xr.Dataset:
    """Open a NetCDF source without loading its full data payload."""
    errors = {}
    for engine in ("h5netcdf", "scipy", "netcdf4"):
        try:
            return xr.open_dataset(path, engine=engine, decode_times=False)
        except Exception as exc:
            errors[engine] = repr(exc)
    raise RuntimeError(f"Unable to open {path} with available backends: {errors}")


def select_first_steps(dataset: xr.Dataset, variable: str, time_steps: int | None) -> xr.Dataset:
    if time_steps is None or time_steps <= 0 or "time" not in dataset[variable].dims:
        return dataset
    return dataset.isel(time=slice(0, time_steps))


def write_dataset(dataset: xr.Dataset | xr.DataArray, output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path, engine="h5netcdf")


def as_standard_dataset(
    data: xr.Dataset | xr.DataArray,
    target: xr.Dataset,
    variable_name: str | None = None,
) -> xr.Dataset:
    """Attach the standard grid bounds and metadata to an output dataset."""
    if isinstance(data, xr.DataArray):
        name = variable_name or data.name
        if name is None:
            raise ValueError("variable_name is required for an unnamed DataArray")
        dataset = data.to_dataset(name=name)
    else:
        dataset = data

    dataset = dataset.assign_coords(lat_b=target["lat_b"], lon_b=target["lon_b"])
    attrs = target.attrs.copy()
    attrs.update(dataset.attrs)
    dataset.attrs = attrs
    return dataset


def summarize(data_array: xr.DataArray) -> dict[str, object]:
    values = np.asarray(data_array.values)
    finite = np.isfinite(values)
    return {
        "dims": data_array.dims,
        "shape": data_array.shape,
        "finite_fraction": float(finite.mean()),
        "minimum": float(np.nanmin(values)) if np.any(finite) else None,
        "maximum": float(np.nanmax(values)) if np.any(finite) else None,
        "mean": float(np.nanmean(values)) if np.any(finite) else None,
    }
