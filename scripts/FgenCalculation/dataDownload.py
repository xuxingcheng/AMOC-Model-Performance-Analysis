#!/usr/bin/env python3
"""Download CMIP6 piControl files from ESGF into the NCAR workspace.

The script is intentionally model-agnostic. By default it discovers every
CMIP6 model that has all requested variables for the strict r1i1p1f1 member,
then downloads the selected files for those models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry


PROJECT = "CMIP6"
ACTIVITY_ID = "CMIP"
EXPERIMENT = "piControl"
FREQUENCY = "mon"
PREFERRED_VARIANT = "r1i1p1f1"
LAST_N_YEARS = 30

OUTPUT_DIR = Path("/glade/work/stevenxu/AMOC_models/downloads")
# Leave empty to auto-discover every qualifying model across CMIP6.
TARGET_MODELS: list[str] = []
TARGET_VARIABLES = ["tos", "sos", "hfds", "wfo", "msftmz"]
TABLE_ID_BY_VARIABLE = {
    "tos": "Omon",
    "sos": "Omon",
    "hfds": "Omon",
    "wfo": "Omon",
    "msftmz": "Omon",
}

SEARCH_URLS = [
    "https://esgf.ceda.ac.uk/esg-search/search",
    "https://esgf-data.dkrz.de/esg-search/search",
    "https://esgf-node.ipsl.upmc.fr/esg-search/search",
    "https://esgf.nci.org.au/esg-search/search",
    "https://esgf-node.llnl.gov/esg-search/search",
]
SEARCH_PAGE_SIZE = 200
SEARCH_CONNECT_TIMEOUT_SECONDS = 10
SEARCH_READ_TIMEOUT_SECONDS = 30
CONNECT_TIMEOUT_SECONDS = 30
READ_TIMEOUT_SECONDS = 300
RETRY_TOTAL = 5
RETRY_BACKOFF_SECONDS = 1.0
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
USER_AGENT = "AMOCproject-CMIP6-downloader/1.0"

TIME_RANGE_PATTERN = re.compile(
    r"_(?P<start>\d{6})-(?P<end>\d{6})\.(?:nc|nc4)$"
)
VERSION_PATTERN = re.compile(r"\.v(?P<version>\d+)(?:[.|]|$)")


@dataclass(frozen=True)
class FileRecord:
    model: str
    variable: str
    filename: str
    download_url: str
    size_bytes: int | None
    checksum: str | None
    checksum_type: str | None
    start_year: int
    end_year: int
    version: int | None


@dataclass
class Summary:
    discovered_models: int = 0
    model_variables_seen: int = 0
    files_selected: int = 0
    downloaded: int = 0
    skipped_existing: int = 0
    missing_dataset: int = 0
    parse_failed: int = 0
    failed: int = 0
    discovery_seconds: float = 0.0
    download_seconds: float = 0.0
    run_seconds: float = 0.0


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


def node_label(search_url: str) -> str:
    parsed = urlparse(search_url)
    return parsed.netloc or search_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search ESGF for CMIP6 piControl NetCDF files and download the last "
            "30 years of files at file granularity."
        )
    )
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--variables", nargs="*", default=None)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--discover-models-only", action="store_true")
    parser.add_argument("--max-files-per-variable", type=int, default=None)
    return parser.parse_args()


def build_retry() -> Retry:
    return Retry(
        total=RETRY_TOTAL,
        connect=RETRY_TOTAL,
        read=RETRY_TOTAL,
        status=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF_SECONDS,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    adapter = HTTPAdapter(max_retries=build_retry())
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def first_value(value: object) -> str | None:
    if isinstance(value, list):
        if not value:
            return None
        first = value[0]
        return str(first) if first is not None else None
    if value is None:
        return None
    return str(value)


def coerce_int(value: object) -> int | None:
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_filename_years(filename: str) -> tuple[int, int] | None:
    match = TIME_RANGE_PATTERN.search(filename)
    if match is None:
        return None
    start = match.group("start")
    end = match.group("end")
    return int(start[:4]), int(end[:4])


def pick_http_download_url(url_entries: object) -> str | None:
    if not isinstance(url_entries, list):
        return None

    for entry in url_entries:
        if not isinstance(entry, str):
            continue
        parts = entry.split("|")
        if len(parts) >= 3 and parts[2] == "HTTPServer":
            return parts[0]

    for entry in url_entries:
        if isinstance(entry, str) and entry.startswith("http"):
            return entry.split("|", 1)[0]

    return None


def extract_filename(doc: dict[str, object]) -> str | None:
    title = first_value(doc.get("title"))
    if title:
        return os.path.basename(title)

    http_url = pick_http_download_url(doc.get("url"))
    if http_url:
        return os.path.basename(http_url)

    return None


def build_search_params(model: str, variable: str, offset: int) -> dict[str, object]:
    table_id = TABLE_ID_BY_VARIABLE[variable]
    return {
        "project": PROJECT,
        "activity_id": ACTIVITY_ID,
        "experiment_id": EXPERIMENT,
        "source_id": model,
        "variable_id": variable,
        "table_id": table_id,
        "member_id": PREFERRED_VARIANT,
        "type": "File",
        "distrib": "true",
        "format": "application/solr+json",
        "limit": SEARCH_PAGE_SIZE,
        "offset": offset,
    }


def build_model_discovery_params(variable: str) -> dict[str, object]:
    return {
        "project": PROJECT,
        "activity_id": ACTIVITY_ID,
        "experiment_id": EXPERIMENT,
        "variable_id": variable,
        "table_id": TABLE_ID_BY_VARIABLE[variable],
        "member_id": PREFERRED_VARIANT,
        "type": "File",
        "distrib": "true",
        "format": "application/solr+json",
        "limit": 0,
        "facets": "source_id",
    }


def fetch_search_payload(
    session: requests.Session, search_url: str, params: dict[str, object]
) -> dict[str, object]:
    response = session.get(
        search_url,
        params=params,
        timeout=(SEARCH_CONNECT_TIMEOUT_SECONDS, SEARCH_READ_TIMEOUT_SECONDS),
    )
    response.raise_for_status()

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        snippet = response.text[:200].replace("\n", " ")
        raise RuntimeError(
            f"Search endpoint returned non-JSON content: {snippet!r}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Search response did not decode into a JSON object.")

    return payload


def fetch_search_page(
    session: requests.Session,
    search_url: str,
    params: dict[str, object],
) -> tuple[int, list[dict[str, object]]]:
    payload = fetch_search_payload(
        session=session,
        search_url=search_url,
        params=params,
    )

    search_response = payload.get("response")
    if not isinstance(search_response, dict):
        raise RuntimeError("Search response is missing the expected 'response' block.")

    num_found = search_response.get("numFound")
    docs = search_response.get("docs")
    if not isinstance(num_found, int) or not isinstance(docs, list):
        raise RuntimeError("Search response is missing 'numFound' or 'docs'.")

    return num_found, docs


def parse_source_id_models(values: object) -> set[str]:
    if not isinstance(values, list):
        raise RuntimeError("Facet response is missing the source_id list.")

    models: set[str] = set()
    for index in range(0, len(values), 2):
        model_name = values[index]
        count = values[index + 1] if index + 1 < len(values) else None

        if not isinstance(model_name, str):
            continue
        if not isinstance(count, int) or count <= 0:
            continue
        models.add(model_name)

    return models


def discover_models_for_variable_from_node(
    session: requests.Session,
    search_url: str,
    variable: str,
) -> set[str]:
    payload = fetch_search_payload(
        session=session,
        search_url=search_url,
        params=build_model_discovery_params(variable),
    )

    facet_counts = payload.get("facet_counts")
    if not isinstance(facet_counts, dict):
        raise RuntimeError("Facet discovery response is missing 'facet_counts'.")

    facet_fields = facet_counts.get("facet_fields")
    if not isinstance(facet_fields, dict):
        raise RuntimeError("Facet discovery response is missing 'facet_fields'.")

    return parse_source_id_models(facet_fields.get("source_id"))


def discover_models_for_variable(
    session: requests.Session,
    variable: str,
) -> set[str]:
    union_models: set[str] = set()
    successes = 0
    last_error: Exception | None = None

    for search_url in SEARCH_URLS:
        try:
            models = discover_models_for_variable_from_node(
                session=session,
                search_url=search_url,
                variable=variable,
            )
        except Exception as exc:
            last_error = exc
            log(
                "WARN",
                f"Model discovery for {variable} failed on {node_label(search_url)}: {exc!r}",
            )
            continue

        successes += 1
        union_models |= models
        log(
            "INFO",
            (
                f"Model discovery for {variable} on {node_label(search_url)}: "
                f"{len(models)} candidate model(s)"
            ),
        )

    if successes == 0:
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"No search nodes succeeded for variable {variable}.")

    return union_models


def discover_available_models(
    session: requests.Session,
    variables: list[str],
    summary: Summary,
) -> list[str]:
    discovery_start = time.perf_counter()
    common_models: set[str] | None = None

    for variable in variables:
        variable_start = time.perf_counter()
        models = discover_models_for_variable(session=session, variable=variable)
        elapsed = time.perf_counter() - variable_start

        log(
            "INFO",
            (
                f"Model discovery for {variable}: found {len(models)} candidate "
                f"model(s) in {format_duration(elapsed)}"
            ),
        )

        common_models = models if common_models is None else common_models & models

    summary.discovery_seconds = time.perf_counter() - discovery_start
    available_models = sorted(common_models or set())
    summary.discovered_models = len(available_models)

    if available_models:
        log(
            "INFO",
            (
                f"Discovered {len(available_models)} model(s) with all requested "
                f"variables in {format_duration(summary.discovery_seconds)}: "
                f"{', '.join(available_models)}"
            ),
        )
    else:
        log(
            "WARN",
            (
                "No models were found with all requested variables after "
                f"{format_duration(summary.discovery_seconds)}."
            ),
        )

    return available_models


def fetch_all_file_docs_from_node(
    session: requests.Session,
    search_url: str,
    model: str,
    variable: str,
) -> list[dict[str, object]]:
    docs: list[dict[str, object]] = []
    offset = 0
    total_hits: int | None = None

    while True:
        params = build_search_params(model=model, variable=variable, offset=offset)
        num_found, page_docs = fetch_search_page(
            session=session,
            search_url=search_url,
            params=params,
        )

        if total_hits is None:
            total_hits = num_found
            log(
                "INFO",
                (
                    f"{model} {variable}: {node_label(search_url)} returned "
                    f"{total_hits} file record(s) for "
                    f"{TABLE_ID_BY_VARIABLE[variable]} ({FREQUENCY}) / "
                    f"{PREFERRED_VARIANT}"
                ),
            )

        if not page_docs:
            break

        docs.extend(page_docs)
        offset += len(page_docs)

        if offset >= num_found:
            break

    return docs


def fetch_all_file_docs(
    session: requests.Session, model: str, variable: str
) -> list[dict[str, object]]:
    combined_docs: list[dict[str, object]] = []
    successes = 0
    last_error: Exception | None = None

    for search_url in SEARCH_URLS:
        try:
            node_docs = fetch_all_file_docs_from_node(
                session=session,
                search_url=search_url,
                model=model,
                variable=variable,
            )
        except Exception as exc:
            last_error = exc
            log(
                "WARN",
                f"{model} {variable}: search failed on {node_label(search_url)}: {exc!r}",
            )
            continue

        combined_docs.extend(node_docs)
        successes += 1

    if successes == 0:
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"No search nodes succeeded for {model} {variable}.")

    log(
        "INFO",
        (
            f"{model} {variable}: collected {len(combined_docs)} raw file record(s) "
            f"across {successes}/{len(SEARCH_URLS)} search node(s)"
        ),
    )
    return combined_docs


def extract_version(doc: dict[str, object]) -> int | None:
    for key in ("dataset_id", "instance_id", "id"):
        raw_value = first_value(doc.get(key))
        if not raw_value:
            continue
        match = VERSION_PATTERN.search(raw_value)
        if match:
            return int(match.group("version"))
    return None


def record_from_doc(
    doc: dict[str, object], model: str, variable: str
) -> tuple[FileRecord | None, str | None]:
    filename = extract_filename(doc)
    if not filename:
        return None, "missing filename/title"

    years = parse_filename_years(filename)
    if years is None:
        return None, f"unable to parse time range from {filename}"

    download_url = pick_http_download_url(doc.get("url"))
    if not download_url:
        return None, f"missing HTTPServer URL for {filename}"

    return (
        FileRecord(
            model=model,
            variable=variable,
            filename=filename,
            download_url=download_url,
            size_bytes=coerce_int(doc.get("size")),
            checksum=first_value(doc.get("checksum")),
            checksum_type=first_value(doc.get("checksum_type")),
            start_year=years[0],
            end_year=years[1],
            version=extract_version(doc),
        ),
        None,
    )


def deduplicate_records(records: Iterable[FileRecord]) -> list[FileRecord]:
    by_filename: dict[str, FileRecord] = {}
    warned_duplicates: set[str] = set()
    for record in records:
        existing = by_filename.get(record.filename)
        if existing is None:
            by_filename[record.filename] = record
            continue

        existing_version = existing.version if existing.version is not None else -1
        record_version = record.version if record.version is not None else -1
        if record_version > existing_version:
            by_filename[record.filename] = record
            continue

        if (
            existing.download_url != record.download_url
            and record_version == existing_version
            and record.filename not in warned_duplicates
        ):
            log(
                "WARN",
                (
                    f"{record.model} {record.variable}: duplicate filename "
                    f"{record.filename} has multiple URLs at version "
                    f"{existing.version}; keeping the first one."
                ),
            )
            warned_duplicates.add(record.filename)
    return sorted(
        by_filename.values(),
        key=lambda item: (item.start_year, item.end_year, item.filename),
    )


def select_last_n_year_files(
    records: list[FileRecord], last_n_years: int = LAST_N_YEARS
) -> tuple[list[FileRecord], int, int]:
    max_end_year = max(record.end_year for record in records)
    threshold_year = max_end_year - last_n_years + 1
    selected = [
        record
        for record in records
        if record.end_year >= threshold_year
    ]
    selected.sort(key=lambda item: (item.start_year, item.end_year, item.filename))
    return selected, max_end_year, threshold_year


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown size"

    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def should_skip_existing(path: Path, expected_size: int | None) -> bool:
    if not path.exists():
        return False
    if expected_size is None:
        return True
    return path.stat().st_size == expected_size


def remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def verify_checksum(path: Path, checksum: str, checksum_type: str) -> None:
    try:
        hasher = hashlib.new(checksum_type.lower())
    except ValueError as exc:
        raise RuntimeError(f"Unsupported checksum type {checksum_type!r}") from exc

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            hasher.update(chunk)

    digest = hasher.hexdigest()
    if digest.lower() != checksum.lower():
        raise RuntimeError(
            f"Checksum mismatch for {path.name}: expected {checksum}, got {digest}"
        )


def download_record(
    session: requests.Session,
    record: FileRecord,
    output_root: Path,
    dry_run: bool,
) -> tuple[str, float]:
    destination_dir = output_root / record.model / record.variable
    destination_path = destination_dir / record.filename
    partial_path = destination_dir / f"{record.filename}.part"

    if should_skip_existing(destination_path, record.size_bytes):
        detail = format_bytes(record.size_bytes)
        log(
            "INFO",
            f"Skipping existing file {destination_path} ({detail})",
        )
        return "skipped", 0.0

    if destination_path.exists():
        actual_size = destination_path.stat().st_size
        expected = format_bytes(record.size_bytes)
        log(
            "WARN",
            (
                f"Existing file has size mismatch and will be replaced: "
                f"{destination_path} ({format_bytes(actual_size)} on disk vs {expected})"
            ),
        )

    if dry_run:
        log(
            "INFO",
            (
                f"DRY RUN would download {record.filename} "
                f"({record.start_year}-{record.end_year}, {format_bytes(record.size_bytes)})"
            ),
        )
        return "planned", 0.0

    destination_dir.mkdir(parents=True, exist_ok=True)
    remove_if_exists(partial_path)
    remove_if_exists(destination_path)

    log(
        "INFO",
        (
            f"Downloading {record.filename} "
            f"({record.start_year}-{record.end_year}, {format_bytes(record.size_bytes)})"
        ),
    )

    bytes_written = 0
    next_percent_report = 10
    download_start = time.perf_counter()

    try:
        with session.get(
            record.download_url,
            stream=True,
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
        ) as response:
            response.raise_for_status()

            with partial_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    bytes_written += len(chunk)

                    if record.size_bytes and record.size_bytes > 0:
                        percent = int((bytes_written / record.size_bytes) * 100)
                        if percent >= next_percent_report:
                            log(
                                "INFO",
                                (
                                    f"  {record.filename}: {percent}% "
                                    f"({format_bytes(bytes_written)} / "
                                    f"{format_bytes(record.size_bytes)})"
                                ),
                            )
                            next_percent_report += 10

        if record.size_bytes is not None and bytes_written != record.size_bytes:
            raise RuntimeError(
                f"Expected {record.size_bytes} bytes but wrote {bytes_written} bytes"
            )

        if record.checksum and record.checksum_type:
            verify_checksum(
                path=partial_path,
                checksum=record.checksum,
                checksum_type=record.checksum_type,
            )

        os.replace(partial_path, destination_path)
        elapsed = time.perf_counter() - download_start
        log("INFO", f"Finished {destination_path} in {format_duration(elapsed)}")
        return "downloaded", elapsed
    except Exception:
        remove_if_exists(partial_path)
        raise


def resolve_records(
    session: requests.Session,
    model: str,
    variable: str,
    summary: Summary,
) -> list[FileRecord]:
    try:
        docs = fetch_all_file_docs(session=session, model=model, variable=variable)
    except RequestException as exc:
        summary.failed += 1
        log("ERROR", f"{model} {variable}: ESGF search failed: {exc!r}")
        return []
    except Exception as exc:
        summary.failed += 1
        log("ERROR", f"{model} {variable}: unexpected search error: {exc!r}")
        return []

    if not docs:
        summary.missing_dataset += 1
        log(
            "WARN",
            (
                f"{model} {variable}: no ESGF file records matched "
                f"{PREFERRED_VARIANT} and {TABLE_ID_BY_VARIABLE[variable]}"
            ),
        )
        return []

    records: list[FileRecord] = []
    for doc in docs:
        record, error_message = record_from_doc(doc=doc, model=model, variable=variable)
        if record is None:
            summary.parse_failed += 1
            log("WARN", f"{model} {variable}: skipping ESGF record: {error_message}")
            continue
        records.append(record)

    if not records:
        log(
            "WARN",
            f"{model} {variable}: no usable file records remained after parsing.",
        )
        return []

    return deduplicate_records(records)


def process_model_variable(
    session: requests.Session,
    model: str,
    variable: str,
    output_root: Path,
    dry_run: bool,
    max_files_per_variable: int | None,
    summary: Summary,
) -> None:
    variable_start = time.perf_counter()
    summary.model_variables_seen += 1
    log("INFO", f"Processing model={model} variable={variable}")
    try:
        records = resolve_records(
            session=session,
            model=model,
            variable=variable,
            summary=summary,
        )
        if not records:
            return

        selected_records, max_end_year, threshold_year = select_last_n_year_files(records)
        if max_files_per_variable is not None:
            selected_records = selected_records[:max_files_per_variable]
            log(
                "INFO",
                (
                    f"{model} {variable}: limiting to first {len(selected_records)} selected "
                    f"file(s) because --max-files-per-variable was set."
                ),
            )

        summary.files_selected += len(selected_records)
        log(
            "INFO",
            (
                f"{model} {variable}: max end year={max_end_year}, "
                f"threshold year={threshold_year}, selected {len(selected_records)} "
                f"of {len(records)} file(s)"
            ),
        )

        for record in selected_records:
            try:
                status, elapsed = download_record(
                    session=session,
                    record=record,
                    output_root=output_root,
                    dry_run=dry_run,
                )
            except RequestException as exc:
                summary.failed += 1
                log("ERROR", f"{record.filename}: request failed: {exc!r}")
                continue
            except Exception as exc:
                summary.failed += 1
                log("ERROR", f"{record.filename}: download failed: {exc!r}")
                continue

            summary.download_seconds += elapsed
            if status == "downloaded":
                summary.downloaded += 1
            elif status == "skipped":
                summary.skipped_existing += 1
    finally:
        elapsed = time.perf_counter() - variable_start
        log(
            "INFO",
            f"Completed model={model} variable={variable} in {format_duration(elapsed)}",
        )


def validate_variables(variables: Iterable[str]) -> list[str]:
    validated: list[str] = []
    for variable in variables:
        if variable not in TABLE_ID_BY_VARIABLE:
            raise ValueError(
                f"Unsupported variable {variable!r}. "
                f"Choose from {sorted(TABLE_ID_BY_VARIABLE)}."
            )
        validated.append(variable)
    return validated


def resolve_models_to_process(
    available_models: list[str],
    requested_models: list[str],
) -> list[str]:
    if not requested_models:
        log(
            "INFO",
            f"Using all {len(available_models)} discovered model(s) for download.",
        )
        return available_models

    requested_set = set(requested_models)
    resolved = [model for model in available_models if model in requested_set]
    missing = [model for model in requested_models if model not in set(resolved)]

    if missing:
        log(
            "WARN",
            (
                "Requested model(s) do not have all requested variables and will be "
                f"skipped: {', '.join(missing)}"
            ),
        )

    log("INFO", f"Using requested available model(s): {resolved}")
    return resolved


def print_summary(summary: Summary) -> None:
    log("INFO", "Run summary:")
    log("INFO", f"  discovered models: {summary.discovered_models}")
    log("INFO", f"  model-variable pairs processed: {summary.model_variables_seen}")
    log("INFO", f"  files selected: {summary.files_selected}")
    log("INFO", f"  downloaded: {summary.downloaded}")
    log("INFO", f"  skipped existing: {summary.skipped_existing}")
    log("INFO", f"  missing dataset matches: {summary.missing_dataset}")
    log("INFO", f"  parse failures: {summary.parse_failed}")
    log("INFO", f"  hard failures: {summary.failed}")
    log("INFO", f"  discovery time: {format_duration(summary.discovery_seconds)}")
    log("INFO", f"  download time: {format_duration(summary.download_seconds)}")
    log("INFO", f"  total wall time: {format_duration(summary.run_seconds)}")


def main() -> int:
    run_start = time.perf_counter()
    args = parse_args()
    requested_models = list(args.models) if args.models is not None else list(TARGET_MODELS)

    try:
        variables = validate_variables(args.variables if args.variables else TARGET_VARIABLES)
    except ValueError as exc:
        log("ERROR", str(exc))
        return 2

    output_root = Path(args.output_dir)
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    summary = Summary()
    session = build_session()
    try:
        log(
            "INFO",
            (
                f"Starting CMIP6 search for variables={variables}, project={PROJECT}, "
                f"experiment={EXPERIMENT}, member={PREFERRED_VARIANT}, "
                f"output_dir={output_root}"
            ),
        )

        try:
            available_models = discover_available_models(
                session=session,
                variables=variables,
                summary=summary,
            )
        except RequestException as exc:
            summary.failed += 1
            log("ERROR", f"Model discovery failed: {exc!r}")
            summary.run_seconds = time.perf_counter() - run_start
            print_summary(summary)
            return 1
        except Exception as exc:
            summary.failed += 1
            log("ERROR", f"Unexpected model discovery error: {exc!r}")
            summary.run_seconds = time.perf_counter() - run_start
            print_summary(summary)
            return 1

        models = resolve_models_to_process(
            available_models=available_models,
            requested_models=requested_models,
        )

        if args.discover_models_only:
            summary.run_seconds = time.perf_counter() - run_start
            print_summary(summary)
            return 0

        if not models:
            log("ERROR", "No models remain to process after discovery/filtering.")
            summary.run_seconds = time.perf_counter() - run_start
            print_summary(summary)
            return 1

        log("INFO", f"Beginning download phase for model(s): {models}")
        for model in models:
            for variable in variables:
                process_model_variable(
                    session=session,
                    model=model,
                    variable=variable,
                    output_root=output_root,
                    dry_run=args.dry_run,
                    max_files_per_variable=args.max_files_per_variable,
                    summary=summary,
                )
    finally:
        session.close()

    summary.run_seconds = time.perf_counter() - run_start
    print_summary(summary)
    return 1 if summary.failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
