#!/usr/bin/env python3
"""Verify local CMIP6 piControl downloads and matching areacello files."""

from __future__ import annotations

import re
import argparse
import importlib
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


DOWNLOADS_ROOT = Path("/glade/work/stevenxu/AMOC_models/downloads")
AREACELLO_DIR = DOWNLOADS_ROOT / "areacello"
TARGET_ROOT = Path("/glade/work/stevenxu/AMOC_models")
REQUIRED_VARIABLES = ["tos", "sos", "hfds", "wfo", "msftmz"]
SURFACE_VARIABLES = ["tos", "sos", "hfds", "wfo"]
AREACELLO_VARIABLE = "areacello"
LAST_N_YEARS = 30
EXCLUDED_DIRS = {AREACELLO_VARIABLE}
TARGET_SUBDIR_BY_VARIABLE = {
    "tos": Path("sea_surface_temperature/scenarios/PIControl"),
    "sos": Path("sea_surface_salinity/scenarios/PIControl"),
    "hfds": Path("heatflux/scenarios/PIControl"),
    "wfo": Path("waterflux/scenarios/PIControl"),
}
AREACELLO_TARGET_SUBDIR = Path("areacello")

TIME_FILE_PATTERN = re.compile(
    r"^(?P<variable>[^_]+)_(?P<table>[^_]+)_(?P<model>.+?)_"
    r"(?P<experiment>piControl)_(?P<variant>r\d+i\d+p\d+f\d+)_"
    r"(?P<grid>[^_]+)_(?P<start>\d{6})[-_](?P<end>\d{6})\.(?:nc|nc4)$"
)
STATIC_FILE_PATTERN = re.compile(
    r"^(?P<variable>[^_]+)_(?P<table>[^_]+)_(?P<model>.+?)_"
    r"(?P<experiment>piControl)_(?P<variant>r\d+i\d+p\d+f\d+)_"
    r"(?P<grid>[^_]+)\.(?:nc|nc4)$"
)


@dataclass(frozen=True)
class TimeFileRecord:
    path: Path
    model: str
    variable: str
    grid_label: str
    start_ym: int
    end_ym: int


@dataclass(frozen=True)
class StaticFileRecord:
    path: Path
    model: str
    variable: str
    grid_label: str


@dataclass
class GridValidation:
    grid_label: str
    passed: bool
    max_end_year: int | None = None
    threshold_year: int | None = None
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ModelResult:
    model: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    valid_variable_grids: dict[str, set[str]] = field(default_factory=dict)
    valid_variable_records: dict[str, dict[str, list[TimeFileRecord]]] = field(default_factory=dict)
    valid_areacello_grids: set[str] = field(default_factory=set)
    valid_areacello_records: dict[str, list[StaticFileRecord]] = field(default_factory=dict)
    common_surface_grids: set[str] = field(default_factory=set)
    transfer_surface_grids: set[str] = field(default_factory=set)

    @property
    def passed(self) -> bool:
        return not self.errors


@dataclass
class StageSummary:
    eligible_models: int = 0
    planned_files: int = 0
    moved_files: int = 0
    skipped_model_targets: int = 0
    skipped_existing_files: int = 0
    failures: int = 0


def log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"

    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that locally downloaded CMIP6 piControl models have the five "
            "required variables for the last 30 years and matching areacello files."
        )
    )
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--downloads-root", default=str(DOWNLOADS_ROOT))
    parser.add_argument("--areacello-dir", default=str(AREACELLO_DIR))
    parser.add_argument("--target-root", default=str(TARGET_ROOT))
    parser.add_argument(
        "--deep-check",
        action="store_true",
        help="Attempt to open NetCDF files and confirm the expected variable exists.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only report verification results; do not move files into target folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview verification-driven moves without modifying files.",
    )
    return parser.parse_args()


def ym_to_month_index(year_month: int) -> int:
    year = year_month // 100
    month = year_month % 100
    return year * 12 + month - 1


def month_index_to_ym(month_index: int) -> int:
    year, month_offset = divmod(month_index, 12)
    return year * 100 + month_offset + 1


def format_ym(year_month: int) -> str:
    year = year_month // 100
    month = year_month % 100
    return f"{year:04d}-{month:02d}"


def iter_netcdf_files(root: Path) -> list[Path]:
    files = list(root.rglob("*.nc"))
    files.extend(root.rglob("*.nc4"))
    return sorted(files)


def parse_time_file(path: Path) -> TimeFileRecord | None:
    match = TIME_FILE_PATTERN.match(path.name)
    if match is None:
        return None

    return TimeFileRecord(
        path=path,
        model=match.group("model"),
        variable=match.group("variable"),
        grid_label=match.group("grid"),
        start_ym=int(match.group("start")),
        end_ym=int(match.group("end")),
    )


def parse_static_file(path: Path) -> StaticFileRecord | None:
    match = STATIC_FILE_PATTERN.match(path.name)
    if match is None:
        return None

    return StaticFileRecord(
        path=path,
        model=match.group("model"),
        variable=match.group("variable"),
        grid_label=match.group("grid"),
    )


def discover_local_models(downloads_root: Path, requested_models: list[str] | None) -> list[str]:
    requested = set(requested_models or [])
    discovered: list[str] = []

    if not downloads_root.exists():
        raise FileNotFoundError(f"Downloads root does not exist: {downloads_root}")

    for entry in sorted(downloads_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in EXCLUDED_DIRS:
            continue
        if requested and entry.name not in requested:
            continue
        discovered.append(entry.name)

    if requested:
        missing = sorted(requested - set(discovered))
        if missing:
            log(
                "WARN",
                f"Requested model(s) were not found locally and will be skipped: {', '.join(missing)}",
            )

    return discovered


def load_dataset_class():
    netcdf4_module = importlib.import_module("netCDF4")
    return netcdf4_module.Dataset


def check_netcdf_variable(path: Path, variable_name: str, require_time: bool) -> str | None:
    try:
        dataset_class = load_dataset_class()
    except Exception as exc:  # pragma: no cover - environment-dependent
        return f"NetCDF deep-check is unavailable: {exc!r}"

    try:
        with dataset_class(path, mode="r") as dataset:
            if variable_name not in dataset.variables:
                return f"{path.name}: variable {variable_name!r} is missing"
            if require_time and "time" not in dataset.variables and "time" not in dataset.dimensions:
                return f"{path.name}: time coordinate/dimension is missing"
    except Exception as exc:  # pragma: no cover - depends on file content
        return f"{path.name}: failed to open as NetCDF: {exc!r}"

    return None


def find_coverage_gap(records: list[TimeFileRecord], threshold_ym: int, max_end_ym: int) -> str | None:
    threshold_idx = ym_to_month_index(threshold_ym)
    max_end_idx = ym_to_month_index(max_end_ym)
    intervals = sorted(
        (
            max(ym_to_month_index(record.start_ym), threshold_idx),
            ym_to_month_index(record.end_ym),
        )
        for record in records
        if ym_to_month_index(record.end_ym) >= threshold_idx
    )

    if not intervals:
        return (
            f"no file covers the required last-30-year window "
            f"{format_ym(threshold_ym)} to {format_ym(max_end_ym)}"
        )

    cursor = threshold_idx
    for start_idx, end_idx in intervals:
        if end_idx < cursor:
            continue
        if start_idx > cursor:
            return (
                f"coverage gap from {format_ym(month_index_to_ym(cursor))} to "
                f"{format_ym(month_index_to_ym(start_idx - 1))}"
            )
        cursor = max(cursor, end_idx + 1)
        if cursor > max_end_idx:
            return None

    if cursor <= max_end_idx:
        return (
            f"coverage ends early at {format_ym(month_index_to_ym(cursor - 1))}, "
            f"expected through {format_ym(max_end_ym)}"
        )

    return None


def has_overlap(records: list[TimeFileRecord], threshold_ym: int) -> bool:
    threshold_idx = ym_to_month_index(threshold_ym)
    intervals = sorted(
        (
            max(ym_to_month_index(record.start_ym), threshold_idx),
            ym_to_month_index(record.end_ym),
        )
        for record in records
        if ym_to_month_index(record.end_ym) >= threshold_idx
    )
    previous_end: int | None = None
    for start_idx, end_idx in intervals:
        if previous_end is not None and start_idx <= previous_end:
            return True
        previous_end = max(previous_end or end_idx, end_idx)
    return False


def validate_time_grid(
    variable: str,
    grid_label: str,
    records: list[TimeFileRecord],
    deep_check: bool,
) -> GridValidation:
    validation = GridValidation(grid_label=grid_label, passed=False)

    if deep_check:
        for record in records:
            error = check_netcdf_variable(
                path=record.path,
                variable_name=variable,
                require_time=True,
            )
            if error:
                validation.messages.append(error)

    max_end_ym = max(record.end_ym for record in records)
    max_end_year = max_end_ym // 100
    threshold_year = max_end_year - LAST_N_YEARS + 1
    threshold_ym = threshold_year * 100 + 1

    validation.max_end_year = max_end_year
    validation.threshold_year = threshold_year

    gap_message = find_coverage_gap(records, threshold_ym=threshold_ym, max_end_ym=max_end_ym)
    if gap_message:
        validation.messages.append(gap_message)

    if has_overlap(records, threshold_ym=threshold_ym):
        validation.warnings.append(
            (
                f"last-30-year coverage for grid {grid_label} contains overlapping files "
                f"within {threshold_year}-{max_end_year}"
            )
        )

    validation.passed = not validation.messages
    return validation


def validate_static_grid(
    grid_label: str,
    records: list[StaticFileRecord],
    deep_check: bool,
) -> GridValidation:
    validation = GridValidation(grid_label=grid_label, passed=False)

    if len(records) > 1:
        validation.warnings.append(
            f"multiple areacello files were found for grid {grid_label}; using them as duplicates"
        )

    if deep_check:
        for record in records:
            error = check_netcdf_variable(
                path=record.path,
                variable_name=AREACELLO_VARIABLE,
                require_time=False,
            )
            if error:
                validation.messages.append(error)

    validation.passed = not validation.messages
    return validation


def collect_variable_files(model_dir: Path, variable: str) -> tuple[dict[str, list[TimeFileRecord]], list[str]]:
    variable_dir = model_dir / variable
    groups: dict[str, list[TimeFileRecord]] = defaultdict(list)
    findings: list[str] = []

    if not variable_dir.exists():
        findings.append(f"missing directory {variable_dir}")
        return groups, findings

    files = iter_netcdf_files(variable_dir)
    if not files:
        findings.append(f"no NetCDF files found in {variable_dir}")
        return groups, findings

    for file_path in files:
        record = parse_time_file(file_path)
        if record is None:
            findings.append(f"{file_path.name}: unable to parse CMIP6 time range")
            continue
        if record.variable != variable:
            findings.append(
                f"{file_path.name}: expected variable {variable!r}, found {record.variable!r}"
            )
            continue
        groups[record.grid_label].append(record)

    return groups, findings


def collect_areacello_files(
    areacello_dir: Path, model: str
) -> tuple[dict[str, list[StaticFileRecord]], list[str]]:
    groups: dict[str, list[StaticFileRecord]] = defaultdict(list)
    findings: list[str] = []

    if not areacello_dir.exists():
        findings.append(f"missing areacello directory {areacello_dir}")
        return groups, findings

    for file_path in sorted(areacello_dir.glob(f"*_{model}_*.nc")) + sorted(
        areacello_dir.glob(f"*_{model}_*.nc4")
    ):
        record = parse_static_file(file_path)
        if record is None:
            findings.append(f"{file_path.name}: unable to parse areacello filename")
            continue
        if record.model != model or record.variable != AREACELLO_VARIABLE:
            continue
        groups[record.grid_label].append(record)

    return groups, findings


def validate_model(
    model: str,
    downloads_root: Path,
    areacello_dir: Path,
    deep_check: bool,
) -> ModelResult:
    result = ModelResult(model=model)
    model_dir = downloads_root / model

    if not model_dir.exists():
        result.errors.append(f"model directory is missing: {model_dir}")
        return result

    for variable in REQUIRED_VARIABLES:
        grouped_records, findings = collect_variable_files(model_dir=model_dir, variable=variable)
        valid_grids: set[str] = set()
        invalid_grid_messages: list[str] = []

        for grid_label, records in sorted(grouped_records.items()):
            sorted_records = sorted(
                records,
                key=lambda item: (item.start_ym, item.end_ym, item.path.name),
            )
            validation = validate_time_grid(
                variable=variable,
                grid_label=grid_label,
                records=sorted_records,
                deep_check=deep_check,
            )

            if validation.passed:
                valid_grids.add(grid_label)
                result.valid_variable_records.setdefault(variable, {})[grid_label] = sorted_records
                result.warnings.extend(
                    f"{variable} grid {grid_label}: {warning}"
                    for warning in validation.warnings
                )
            else:
                invalid_grid_messages.extend(
                    f"{variable} grid {grid_label}: {message}"
                    for message in validation.messages
                )

        result.valid_variable_grids[variable] = valid_grids

        if findings:
            prefixed_findings = [f"{variable}: {message}" for message in findings]
            if valid_grids:
                result.warnings.extend(prefixed_findings)
            else:
                result.errors.extend(prefixed_findings)

        if invalid_grid_messages:
            if valid_grids:
                result.warnings.extend(invalid_grid_messages)
            else:
                result.errors.extend(invalid_grid_messages)

        if not valid_grids:
            result.errors.append(
                f"{variable}: no grid has complete local coverage for the last {LAST_N_YEARS} years"
            )

    common_surface_grids: set[str] | None = None
    for variable in SURFACE_VARIABLES:
        grids = result.valid_variable_grids.get(variable, set())
        common_surface_grids = set(grids) if common_surface_grids is None else common_surface_grids & grids

    result.common_surface_grids = common_surface_grids or set()
    if not result.common_surface_grids:
        result.errors.append(
            "no common valid surface grid exists across tos, sos, hfds, and wfo"
        )

    grouped_areacello, areacello_findings = collect_areacello_files(
        areacello_dir=areacello_dir,
        model=model,
    )

    valid_areacello_grids: set[str] = set()
    invalid_areacello_messages: list[str] = []
    for grid_label, records in sorted(grouped_areacello.items()):
        validation = validate_static_grid(
            grid_label=grid_label,
            records=records,
            deep_check=deep_check,
        )
        if validation.passed:
            valid_areacello_grids.add(grid_label)
            result.valid_areacello_records[grid_label] = sorted(
                records,
                key=lambda item: item.path.name,
            )
            result.warnings.extend(
                f"areacello grid {grid_label}: {warning}"
                for warning in validation.warnings
            )
        else:
            invalid_areacello_messages.extend(
                f"areacello grid {grid_label}: {message}"
                for message in validation.messages
            )

    result.valid_areacello_grids = valid_areacello_grids

    if areacello_findings:
        if valid_areacello_grids:
            result.warnings.extend(areacello_findings)
        else:
            result.errors.extend(areacello_findings)

    if invalid_areacello_messages:
        if valid_areacello_grids:
            result.warnings.extend(invalid_areacello_messages)
        else:
            result.errors.extend(invalid_areacello_messages)

    if result.common_surface_grids:
        matched_areacello_grids = result.common_surface_grids & result.valid_areacello_grids
        result.transfer_surface_grids = matched_areacello_grids
        if not matched_areacello_grids:
            result.errors.append(
                (
                    "no valid areacello file matches the common surface grid(s): "
                    f"{', '.join(sorted(result.common_surface_grids))}"
                )
            )
        missing_surface_areacello = result.common_surface_grids - result.valid_areacello_grids
        if missing_surface_areacello:
            result.warnings.append(
                (
                    "missing areacello for some common surface grid(s): "
                    f"{', '.join(sorted(missing_surface_areacello))}"
                )
            )

    return result


def print_model_result(result: ModelResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    log(
        "INFO",
        (
            f"{status} {result.model}: "
            f"surface_grids={sorted(result.common_surface_grids)} "
            f"areacello_grids={sorted(result.valid_areacello_grids)} "
            f"transfer_grids={sorted(result.transfer_surface_grids)}"
        ),
    )

    for variable in REQUIRED_VARIABLES:
        grids = sorted(result.valid_variable_grids.get(variable, set()))
        log("INFO", f"  {result.model} {variable}: valid grids={grids}")

    for error in result.errors:
        log("ERROR", f"  {result.model}: {error}")
    for warning in result.warnings:
        log("WARN", f"  {result.model}: {warning}")


def variable_target_dir(variable: str, target_root: Path) -> Path:
    return target_root / TARGET_SUBDIR_BY_VARIABLE[variable]


def areacello_target_dir(target_root: Path) -> Path:
    return target_root / AREACELLO_TARGET_SUBDIR


def target_has_model_files(target_dir: Path, variable: str, model: str) -> bool:
    return any(target_dir.glob(f"{variable}_*_{model}_piControl_*.nc")) or any(
        target_dir.glob(f"{variable}_*_{model}_piControl_*.nc4")
    )


def move_paths(
    paths: list[Path],
    target_dir: Path,
    dry_run: bool,
    summary: StageSummary,
) -> None:
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    for source_path in paths:
        destination_path = target_dir / source_path.name
        summary.planned_files += 1

        if destination_path.exists():
            summary.skipped_existing_files += 1
            log(
                "WARN",
                f"Target file already exists and will be left in place: {destination_path}",
            )
            continue

        if dry_run:
            log("INFO", f"DRY RUN would move {source_path} -> {destination_path}")
            continue

        try:
            shutil.move(str(source_path), str(destination_path))
        except Exception as exc:
            summary.failures += 1
            log("ERROR", f"Failed to move {source_path} -> {destination_path}: {exc!r}")
            continue

        summary.moved_files += 1
        log("INFO", f"Moved {source_path} -> {destination_path}")


def stage_variable_for_model(
    model: str,
    variable: str,
    paths: list[Path],
    target_root: Path,
    dry_run: bool,
    summary: StageSummary,
) -> None:
    if not paths:
        return

    target_dir = variable_target_dir(variable=variable, target_root=target_root)
    if target_has_model_files(target_dir=target_dir, variable=variable, model=model):
        summary.skipped_model_targets += 1
        log(
            "INFO",
            (
                f"Skipping move for {model} {variable}: target folder already contains "
                f"files for this model at {target_dir}"
            ),
        )
        return

    move_paths(paths=paths, target_dir=target_dir, dry_run=dry_run, summary=summary)


def stage_areacello_for_model(
    model: str,
    paths: list[Path],
    target_root: Path,
    dry_run: bool,
    summary: StageSummary,
) -> None:
    if not paths:
        return

    target_dir = areacello_target_dir(target_root=target_root)
    if target_has_model_files(target_dir=target_dir, variable=AREACELLO_VARIABLE, model=model):
        summary.skipped_model_targets += 1
        log(
            "INFO",
            (
                f"Skipping move for {model} areacello: target folder already contains "
                f"files for this model at {target_dir}"
            ),
        )
        return

    move_paths(paths=paths, target_dir=target_dir, dry_run=dry_run, summary=summary)


def stage_passed_models(
    results: list[ModelResult],
    target_root: Path,
    dry_run: bool,
) -> StageSummary:
    summary = StageSummary()

    for result in results:
        if not result.passed:
            continue

        summary.eligible_models += 1
        surface_grids = sorted(result.transfer_surface_grids)

        for variable in SURFACE_VARIABLES:
            variable_paths: list[Path] = []
            for grid_label in surface_grids:
                variable_paths.extend(
                    record.path
                    for record in result.valid_variable_records.get(variable, {}).get(grid_label, [])
                )
            variable_paths = sorted(variable_paths)
            stage_variable_for_model(
                model=result.model,
                variable=variable,
                paths=variable_paths,
                target_root=target_root,
                dry_run=dry_run,
                summary=summary,
            )

        areacello_paths = sorted(
            record.path
            for records in result.valid_areacello_records.values()
            for record in records
        )
        stage_areacello_for_model(
            model=result.model,
            paths=areacello_paths,
            target_root=target_root,
            dry_run=dry_run,
            summary=summary,
        )

    return summary


def print_stage_summary(summary: StageSummary, dry_run: bool) -> None:
    label = "Staging preview" if dry_run else "Staging summary"
    log("INFO", f"{label}:")
    log("INFO", f"  eligible passed models: {summary.eligible_models}")
    log("INFO", f"  files considered for move: {summary.planned_files}")
    log("INFO", f"  files moved: {summary.moved_files}")
    log("INFO", f"  skipped existing model targets: {summary.skipped_model_targets}")
    log("INFO", f"  skipped existing files: {summary.skipped_existing_files}")
    log("INFO", f"  move failures: {summary.failures}")


def main() -> int:
    run_start = time.perf_counter()
    args = parse_args()

    downloads_root = Path(args.downloads_root)
    areacello_dir = Path(args.areacello_dir)
    target_root = Path(args.target_root)

    if args.deep_check:
        log(
            "WARN",
            "Deep NetCDF checks are enabled. This may fail in MPI-linked environments.",
        )
    else:
        log("INFO", "Using filename-only validation. Pass --deep-check to open NetCDF files.")
    if args.verify_only:
        log("INFO", "Verification-only mode is enabled. No files will be moved.")
    elif args.dry_run:
        log("INFO", "Dry-run mode is enabled. Move actions will be logged only.")

    models = discover_local_models(
        downloads_root=downloads_root,
        requested_models=list(args.models) if args.models is not None else None,
    )
    if not models:
        log("ERROR", "No local models were found to verify.")
        return 1

    log(
        "INFO",
        (
            f"Verifying {len(models)} model(s) under {downloads_root} with "
            f"areacello directory {areacello_dir}"
        ),
    )

    results: list[ModelResult] = []
    for model in models:
        results.append(
            validate_model(
                model=model,
                downloads_root=downloads_root,
                areacello_dir=areacello_dir,
                deep_check=args.deep_check,
            )
        )

    passed = sum(1 for result in results if result.passed)
    failed = len(results) - passed
    warning_count = sum(len(result.warnings) for result in results)

    for result in results:
        print_model_result(result)

    stage_summary = StageSummary()
    if not args.verify_only:
        stage_summary = stage_passed_models(
            results=results,
            target_root=target_root,
            dry_run=args.dry_run,
        )
        print_stage_summary(summary=stage_summary, dry_run=args.dry_run)

    elapsed = time.perf_counter() - run_start
    log("INFO", "Verification summary:")
    log("INFO", f"  models checked: {len(results)}")
    log("INFO", f"  passed: {passed}")
    log("INFO", f"  failed: {failed}")
    log("INFO", f"  warnings: {warning_count}")
    log("INFO", f"  wall time: {format_duration(elapsed)}")
    return 1 if failed or stage_summary.failures else 0


if __name__ == "__main__":
    sys.exit(main())
