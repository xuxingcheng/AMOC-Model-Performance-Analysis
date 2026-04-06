#!/usr/bin/env python3
"""Download CMIP6 piControl areacello files for models already on disk.

The script scans the existing download tree to find models that already have
CMIP6 ocean data downloaded, infers the horizontal grid labels in use for each
model, and downloads matching areacello files into a shared output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
VARIABLE_ID = "areacello"
TABLE_ID = "Ofx"
PREFERRED_VARIANT = "r1i1p1f1"

DOWNLOADS_ROOT = Path("/glade/work/stevenxu/AMOC_models/downloads")
OUTPUT_DIR = DOWNLOADS_ROOT / VARIABLE_ID
SURFACE_VARIABLES = {"tos", "sos", "hfds", "wfo"}
EXCLUDED_MODEL_DIRS = {VARIABLE_ID}

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
USER_AGENT = "AMOCproject-areacello-downloader/1.0"


@dataclass(frozen=True)
class ModelTarget:
    model: str
    grid_labels: tuple[str, ...]


@dataclass(frozen=True)
class FileRecord:
    model: str
    grid_label: str
    filename: str
    download_urls: tuple[str, ...]
    size_bytes: int | None
    checksum: str | None
    checksum_type: str | None
    version: int | None


@dataclass
class Summary:
    local_models_found: int = 0
    model_grid_pairs: int = 0
    files_selected: int = 0
    downloaded: int = 0
    skipped_existing: int = 0
    missing_grid_matches: int = 0
    parse_failed: int = 0
    failed: int = 0
    target_scan_seconds: float = 0.0
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


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown size"

    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def node_label(search_url: str) -> str:
    parsed = urlparse(search_url)
    return parsed.netloc or search_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download CMIP6 areacello files for model/grid combinations already "
            "present under the NCAR downloads tree."
        )
    )
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--downloads-root", default=str(DOWNLOADS_ROOT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-targets", action="store_true")
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


def extract_version(doc: dict[str, object]) -> int | None:
    for key in ("dataset_id", "instance_id", "id"):
        raw_value = first_value(doc.get(key))
        if not raw_value:
            continue
        marker = ".v"
        if marker not in raw_value:
            continue
        version_text = raw_value.split(marker, 1)[1].split("|", 1)[0].split(".", 1)[0]
        if version_text.isdigit():
            return int(version_text)
    return None


def iter_netcdf_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.nc")
    yield from root.rglob("*.nc4")


def parse_grid_label_from_filename(filename: str) -> str | None:
    stem = filename
    if stem.endswith(".nc4"):
        stem = stem[:-4]
    elif stem.endswith(".nc"):
        stem = stem[:-3]

    parts = stem.split("_")
    for index, part in enumerate(parts):
        if part == PREFERRED_VARIANT and index + 1 < len(parts):
            return parts[index + 1]
    return None


def collect_grid_labels_from_dir(model_dir: Path) -> set[str]:
    grid_labels: set[str] = set()
    for file_path in iter_netcdf_files(model_dir):
        grid_label = parse_grid_label_from_filename(file_path.name)
        if grid_label:
            grid_labels.add(grid_label)
    return grid_labels


def infer_target_grid_labels(model_dir: Path) -> tuple[str, ...]:
    preferred_grid_labels: set[str] = set()
    for variable_dir in sorted(model_dir.iterdir()):
        if not variable_dir.is_dir() or variable_dir.name not in SURFACE_VARIABLES:
            continue
        preferred_grid_labels |= collect_grid_labels_from_dir(variable_dir)

    if preferred_grid_labels:
        return tuple(sorted(preferred_grid_labels))

    fallback_grid_labels = collect_grid_labels_from_dir(model_dir)
    return tuple(sorted(fallback_grid_labels))


def discover_local_targets(
    downloads_root: Path,
    requested_models: list[str] | None,
    summary: Summary,
) -> list[ModelTarget]:
    scan_start = time.perf_counter()
    targets: list[ModelTarget] = []
    requested_set = set(requested_models or [])
    found_requested: set[str] = set()

    if not downloads_root.exists():
        raise FileNotFoundError(f"Downloads root does not exist: {downloads_root}")

    for model_dir in sorted(downloads_root.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name in EXCLUDED_MODEL_DIRS:
            continue
        if requested_set and model_dir.name not in requested_set:
            continue

        has_data = any(iter_netcdf_files(model_dir))
        if not has_data:
            continue

        grid_labels = infer_target_grid_labels(model_dir)
        if not grid_labels:
            log(
                "WARN",
                f"{model_dir.name}: no CMIP6 grid labels could be inferred from local files.",
            )
            continue

        targets.append(ModelTarget(model=model_dir.name, grid_labels=grid_labels))
        found_requested.add(model_dir.name)

    summary.target_scan_seconds = time.perf_counter() - scan_start
    summary.local_models_found = len(targets)
    summary.model_grid_pairs = sum(len(target.grid_labels) for target in targets)

    if requested_set:
        missing = sorted(requested_set - found_requested)
        if missing:
            log(
                "WARN",
                (
                    "Requested model(s) were not found under the local downloads tree "
                    f"and will be skipped: {', '.join(missing)}"
                ),
            )

    return targets


def build_search_params(model: str, offset: int) -> dict[str, object]:
    return {
        "project": PROJECT,
        "activity_id": ACTIVITY_ID,
        "experiment_id": EXPERIMENT,
        "source_id": model,
        "variable_id": VARIABLE_ID,
        "table_id": TABLE_ID,
        "member_id": PREFERRED_VARIANT,
        "type": "File",
        "distrib": "true",
        "format": "application/solr+json",
        "limit": SEARCH_PAGE_SIZE,
        "offset": offset,
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
    payload = fetch_search_payload(session=session, search_url=search_url, params=params)

    search_response = payload.get("response")
    if not isinstance(search_response, dict):
        raise RuntimeError("Search response is missing the expected 'response' block.")

    num_found = search_response.get("numFound")
    docs = search_response.get("docs")
    if not isinstance(num_found, int) or not isinstance(docs, list):
        raise RuntimeError("Search response is missing 'numFound' or 'docs'.")

    return num_found, docs


def fetch_all_file_docs_from_node(
    session: requests.Session,
    search_url: str,
    model: str,
) -> list[dict[str, object]]:
    docs: list[dict[str, object]] = []
    offset = 0
    total_hits: int | None = None

    while True:
        params = build_search_params(model=model, offset=offset)
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
                    f"{model} {VARIABLE_ID}: {node_label(search_url)} returned "
                    f"{total_hits} file record(s) for {TABLE_ID} / {PREFERRED_VARIANT}"
                ),
            )

        if not page_docs:
            break

        docs.extend(page_docs)
        offset += len(page_docs)

        if offset >= num_found:
            break

    return docs


def fetch_all_file_docs(session: requests.Session, model: str) -> list[dict[str, object]]:
    combined_docs: list[dict[str, object]] = []
    successes = 0
    last_error: Exception | None = None

    for search_url in SEARCH_URLS:
        try:
            node_docs = fetch_all_file_docs_from_node(
                session=session,
                search_url=search_url,
                model=model,
            )
        except Exception as exc:
            last_error = exc
            log(
                "WARN",
                f"{model} {VARIABLE_ID}: search failed on {node_label(search_url)}: {exc!r}",
            )
            continue

        combined_docs.extend(node_docs)
        successes += 1

    if successes == 0:
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"No search nodes succeeded for {model} {VARIABLE_ID}.")

    log(
        "INFO",
        (
            f"{model} {VARIABLE_ID}: collected {len(combined_docs)} raw file record(s) "
            f"across {successes}/{len(SEARCH_URLS)} search node(s)"
        ),
    )
    return combined_docs


def extract_filename(doc: dict[str, object]) -> str | None:
    title = first_value(doc.get("title"))
    if title:
        return os.path.basename(title)

    for url in extract_http_download_urls(doc.get("url")):
        return os.path.basename(url)
    return None


def extract_http_download_urls(url_entries: object) -> tuple[str, ...]:
    if not isinstance(url_entries, list):
        return ()

    urls: list[str] = []
    seen: set[str] = set()
    for entry in url_entries:
        if not isinstance(entry, str):
            continue
        parts = entry.split("|")
        if len(parts) >= 3 and parts[2] != "HTTPServer":
            continue
        url = parts[0]
        if not url.startswith("http"):
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)

    return tuple(urls)


def record_from_doc(doc: dict[str, object], model: str) -> tuple[FileRecord | None, str | None]:
    filename = extract_filename(doc)
    if not filename:
        return None, "missing filename/title"

    grid_label = first_value(doc.get("grid_label")) or parse_grid_label_from_filename(filename)
    if not grid_label:
        return None, f"unable to infer grid label from {filename}"

    download_urls = extract_http_download_urls(doc.get("url"))
    if not download_urls:
        return None, f"missing HTTPServer URL for {filename}"

    return (
        FileRecord(
            model=model,
            grid_label=grid_label,
            filename=filename,
            download_urls=download_urls,
            size_bytes=coerce_int(doc.get("size")),
            checksum=first_value(doc.get("checksum")),
            checksum_type=first_value(doc.get("checksum_type")),
            version=extract_version(doc),
        ),
        None,
    )


def merge_urls(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for url in left + right:
        if url in seen:
            continue
        seen.add(url)
        merged.append(url)
    return tuple(merged)


def merge_records(existing: FileRecord, incoming: FileRecord) -> FileRecord:
    return FileRecord(
        model=existing.model,
        grid_label=existing.grid_label,
        filename=existing.filename,
        download_urls=merge_urls(existing.download_urls, incoming.download_urls),
        size_bytes=existing.size_bytes if existing.size_bytes is not None else incoming.size_bytes,
        checksum=existing.checksum or incoming.checksum,
        checksum_type=existing.checksum_type or incoming.checksum_type,
        version=existing.version,
    )


def deduplicate_records(records: Iterable[FileRecord]) -> list[FileRecord]:
    by_filename: dict[str, FileRecord] = {}
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
        if record_version < existing_version:
            continue

        by_filename[record.filename] = merge_records(existing, record)

    return sorted(
        by_filename.values(),
        key=lambda item: (item.grid_label, item.filename),
    )


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


def attempt_download(
    session: requests.Session,
    record: FileRecord,
    destination_path: Path,
    url: str,
) -> float:
    partial_path = destination_path.with_suffix(destination_path.suffix + ".part")
    remove_if_exists(partial_path)

    bytes_written = 0
    download_start = time.perf_counter()
    with session.get(
        url,
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

    if record.size_bytes is not None and bytes_written != record.size_bytes:
        remove_if_exists(partial_path)
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
    return time.perf_counter() - download_start


def download_record(
    session: requests.Session,
    record: FileRecord,
    output_root: Path,
    dry_run: bool,
) -> tuple[str, float]:
    destination_path = output_root / record.filename

    if should_skip_existing(destination_path, record.size_bytes):
        log(
            "INFO",
            f"Skipping existing file {destination_path} ({format_bytes(record.size_bytes)})",
        )
        return "skipped", 0.0

    if destination_path.exists():
        actual_size = destination_path.stat().st_size
        log(
            "WARN",
            (
                f"Existing file has size mismatch and will be replaced: {destination_path} "
                f"({format_bytes(actual_size)} on disk vs {format_bytes(record.size_bytes)})"
            ),
        )

    if dry_run:
        log(
            "INFO",
            (
                f"DRY RUN would download {record.filename} "
                f"(grid={record.grid_label}, {format_bytes(record.size_bytes)})"
            ),
        )
        return "planned", 0.0

    output_root.mkdir(parents=True, exist_ok=True)
    remove_if_exists(destination_path)

    errors: list[str] = []
    for attempt_number, download_url in enumerate(record.download_urls, start=1):
        host = urlparse(download_url).netloc or download_url
        log(
            "INFO",
            (
                f"Downloading {record.filename} from {host} "
                f"(attempt {attempt_number}/{len(record.download_urls)}, "
                f"grid={record.grid_label}, {format_bytes(record.size_bytes)})"
            ),
        )

        try:
            elapsed = attempt_download(
                session=session,
                record=record,
                destination_path=destination_path,
                url=download_url,
            )
        except Exception as exc:
            errors.append(f"{host}: {exc!r}")
            log("WARN", f"{record.filename}: download attempt from {host} failed: {exc!r}")
            continue

        log("INFO", f"Finished {destination_path} in {format_duration(elapsed)}")
        return "downloaded", elapsed

    raise RuntimeError(
        f"All download URLs failed for {record.filename}: {'; '.join(errors)}"
    )


def resolve_records(
    session: requests.Session,
    model: str,
    summary: Summary,
) -> list[FileRecord]:
    try:
        docs = fetch_all_file_docs(session=session, model=model)
    except RequestException as exc:
        summary.failed += 1
        log("ERROR", f"{model} {VARIABLE_ID}: ESGF search failed: {exc!r}")
        return []
    except Exception as exc:
        summary.failed += 1
        log("ERROR", f"{model} {VARIABLE_ID}: unexpected search error: {exc!r}")
        return []

    if not docs:
        log(
            "WARN",
            f"{model} {VARIABLE_ID}: no ESGF file records matched {TABLE_ID} and {PREFERRED_VARIANT}.",
        )
        return []

    records: list[FileRecord] = []
    for doc in docs:
        record, error_message = record_from_doc(doc=doc, model=model)
        if record is None:
            summary.parse_failed += 1
            log("WARN", f"{model} {VARIABLE_ID}: skipping ESGF record: {error_message}")
            continue
        records.append(record)

    if not records:
        log(
            "WARN",
            f"{model} {VARIABLE_ID}: no usable file records remained after parsing.",
        )
        return []

    return deduplicate_records(records)


def select_records_for_target(
    records: list[FileRecord], target: ModelTarget
) -> tuple[list[FileRecord], list[str]]:
    wanted_grids = set(target.grid_labels)
    selected = [record for record in records if record.grid_label in wanted_grids]
    selected.sort(key=lambda item: (item.grid_label, item.filename))

    found_grids = {record.grid_label for record in selected}
    missing_grids = [grid for grid in target.grid_labels if grid not in found_grids]
    return selected, missing_grids


def process_target(
    session: requests.Session,
    target: ModelTarget,
    output_root: Path,
    dry_run: bool,
    summary: Summary,
) -> None:
    target_start = time.perf_counter()
    log(
        "INFO",
        f"Processing model={target.model} grid_labels={list(target.grid_labels)}",
    )

    try:
        records = resolve_records(session=session, model=target.model, summary=summary)
        if not records:
            summary.missing_grid_matches += len(target.grid_labels)
            return

        selected_records, missing_grids = select_records_for_target(records=records, target=target)
        summary.files_selected += len(selected_records)

        if missing_grids:
            summary.missing_grid_matches += len(missing_grids)
            log(
                "WARN",
                (
                    f"{target.model} {VARIABLE_ID}: no matching areacello file was found for "
                    f"grid(s): {', '.join(missing_grids)}"
                ),
            )

        if selected_records:
            log(
                "INFO",
                (
                    f"{target.model} {VARIABLE_ID}: selected {len(selected_records)} file(s) "
                    f"for grid(s): {', '.join(target.grid_labels)}"
                ),
            )
        else:
            log(
                "WARN",
                f"{target.model} {VARIABLE_ID}: no files matched the local target grids.",
            )
            return

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
        elapsed = time.perf_counter() - target_start
        log(
            "INFO",
            f"Completed model={target.model} in {format_duration(elapsed)}",
        )


def print_targets(targets: list[ModelTarget]) -> None:
    if not targets:
        log("WARN", "No local model targets were found.")
        return

    log("INFO", f"Local model targets ({len(targets)}):")
    for target in targets:
        log("INFO", f"  {target.model}: {', '.join(target.grid_labels)}")


def print_summary(summary: Summary) -> None:
    log("INFO", "Run summary:")
    log("INFO", f"  local models found: {summary.local_models_found}")
    log("INFO", f"  model-grid pairs processed: {summary.model_grid_pairs}")
    log("INFO", f"  files selected: {summary.files_selected}")
    log("INFO", f"  downloaded: {summary.downloaded}")
    log("INFO", f"  skipped existing: {summary.skipped_existing}")
    log("INFO", f"  missing grid matches: {summary.missing_grid_matches}")
    log("INFO", f"  parse failures: {summary.parse_failed}")
    log("INFO", f"  hard failures: {summary.failed}")
    log("INFO", f"  target scan time: {format_duration(summary.target_scan_seconds)}")
    log("INFO", f"  download time: {format_duration(summary.download_seconds)}")
    log("INFO", f"  total wall time: {format_duration(summary.run_seconds)}")


def main() -> int:
    run_start = time.perf_counter()
    args = parse_args()

    downloads_root = Path(args.downloads_root)
    output_root = Path(args.output_dir)
    requested_models = list(args.models) if args.models is not None else None

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    summary = Summary()

    try:
        targets = discover_local_targets(
            downloads_root=downloads_root,
            requested_models=requested_models,
            summary=summary,
        )
    except Exception as exc:
        summary.failed += 1
        log("ERROR", f"Failed to discover local model targets: {exc!r}")
        summary.run_seconds = time.perf_counter() - run_start
        print_summary(summary)
        return 1

    log(
        "INFO",
        (
            f"Starting areacello download for {len(targets)} local model target(s), "
            f"output_dir={output_root}"
        ),
    )
    print_targets(targets)

    if args.list_targets:
        summary.run_seconds = time.perf_counter() - run_start
        print_summary(summary)
        return 0

    if not targets:
        log("ERROR", "No local models remain to process.")
        summary.run_seconds = time.perf_counter() - run_start
        print_summary(summary)
        return 1

    session = build_session()
    try:
        for target in targets:
            process_target(
                session=session,
                target=target,
                output_root=output_root,
                dry_run=args.dry_run,
                summary=summary,
            )
    finally:
        session.close()

    summary.run_seconds = time.perf_counter() - run_start
    print_summary(summary)
    return 1 if summary.failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
