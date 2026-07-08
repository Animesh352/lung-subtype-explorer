"""
Download TCGA lung cohort datasets from UCSC Xena into data/raw/.

Expression matrices -- tcga.xenahubs.net (cohort-specific HiSeqV2, gzip):
  TCGA.LUAD.sampleMap/HiSeqV2.gz  -- LUAD HiSeqV2 expression
  TCGA.LUSC.sampleMap/HiSeqV2.gz  -- LUSC HiSeqV2 expression
  Falls back to toil.xenahubs.net on 403 (pan-TCGA RSEM log2 counts, gzip).

Clinical matrices -- tcga.xenahubs.net (plain TSV, no authentication required):
  TCGA.LUAD.sampleMap/LUAD_clinicalMatrix  -- LUAD demographics and staging
  TCGA.LUSC.sampleMap/LUSC_clinicalMatrix  -- LUSC demographics and staging
  Both return HTTP 200 without credentials. The .gz variants return 403.

URLs confirmed 2026-07-01.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

_TCGA = "https://tcga.xenahubs.net/download"
_TOIL = "https://toil.xenahubs.net/download"


@dataclass
class Dataset:
    name: str
    url: str
    filename: str
    gzipped: bool = True
    fallback_url: str | None = None
    fallback_filename: str | None = None


DATASETS: list[Dataset] = [
    Dataset(
        name="luad_expression",
        url=f"{_TCGA}/TCGA.LUAD.sampleMap/HiSeqV2.gz",
        filename="luad_expression.tsv.gz",
        fallback_url=f"{_TOIL}/tcga_RSEM_Hugo_norm_count.gz",
        fallback_filename="tcga_expression_rsem.tsv.gz",
    ),
    Dataset(
        name="lusc_expression",
        url=f"{_TCGA}/TCGA.LUSC.sampleMap/HiSeqV2.gz",
        filename="lusc_expression.tsv.gz",
        fallback_url=f"{_TOIL}/tcga_RSEM_Hugo_norm_count.gz",
        fallback_filename="tcga_expression_rsem.tsv.gz",
    ),
    Dataset(
        name="luad_clinical",
        url=f"{_TCGA}/TCGA.LUAD.sampleMap/LUAD_clinicalMatrix",
        filename="luad_clinical.tsv",
        gzipped=False,
    ),
    Dataset(
        name="lusc_clinical",
        url=f"{_TCGA}/TCGA.LUSC.sampleMap/LUSC_clinicalMatrix",
        filename="lusc_clinical.tsv",
        gzipped=False,
    ),
]

# ---------------------------------------------------------------------------
# Download primitives
# ---------------------------------------------------------------------------

_CHUNK = 8 * 1024 * 1024  # 8 MB streaming chunks
_MAX_RETRIES = 4
_TIMEOUT = httpx.Timeout(connect=15.0, read=300.0, write=None, pool=15.0)

_TRANSIENT_ERRORS = (
    httpx.NetworkError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
)


def _stream_to_file(
    client: httpx.Client,
    url: str,
    dest: Path,
) -> tuple[int, str]:
    """Stream url to dest; return (bytes_written, sha256hex).

    Raises PermissionError on HTTP 401/403 so callers can trigger a fallback
    without burning through retry budget.
    """
    sha = hashlib.sha256()
    written = 0
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with client.stream("GET", url) as resp:
            if resp.status_code in (401, 403):
                raise PermissionError(f"HTTP {resp.status_code} from {url}")
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=_CHUNK):
                    fh.write(chunk)
                    sha.update(chunk)
                    written += len(chunk)
                    log.debug("  streamed %.1f MB", written / 1e6)
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return written, sha.hexdigest()


def _download_with_retry(
    client: httpx.Client,
    url: str,
    dest: Path,
) -> tuple[int, str]:
    """_stream_to_file with exponential backoff on transient network errors."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return _stream_to_file(client, url, dest)
        except PermissionError:
            raise
        except _TRANSIENT_ERRORS as exc:
            if attempt == _MAX_RETRIES:
                raise
            delay = 2.0**attempt
            log.warning(
                "attempt %d/%d failed (%s) -- retrying in %.0fs",
                attempt,
                _MAX_RETRIES,
                exc,
                delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def _count_tsv_dims(path: Path, gzipped: bool = True) -> tuple[int, int]:
    """Return (row_count, col_count) for a TSV (gzipped or plain) including the header row."""
    rows = 0
    cols = 0
    opener = gzip.open(path, "rt", errors="replace") if gzipped else path.open(errors="replace")
    with opener as fh:
        for i, line in enumerate(fh):
            if i == 0:
                cols = len(line.rstrip("\n").split("\t"))
            rows = i + 1
    return rows, cols


def _already_complete(manifest: dict, name: str, dest: Path) -> bool:
    entry = manifest.get(name)
    if not entry or not dest.exists():
        return False
    return entry.get("size_bytes") == dest.stat().st_size


# ---------------------------------------------------------------------------
# Per-dataset download logic
# ---------------------------------------------------------------------------


def download_dataset(
    ds: Dataset,
    manifest: dict,
    raw_dir: Path,
    client: httpx.Client,
) -> None:
    dest = raw_dir / ds.filename

    if _already_complete(manifest, ds.name, dest):
        log.info("[%s] already complete, skipping", ds.name)
        return

    source = "tcga_hub"
    url = ds.url
    filename = ds.filename
    gzipped = ds.gzipped

    log.info("[%s] fetching %s", ds.name, url)
    try:
        size, sha256 = _download_with_retry(client, url, dest)
    except PermissionError as exc:
        if not ds.fallback_url:
            raise
        log.warning(
            "[%s] %s -- switching to public fallback.",
            ds.name,
            exc,
        )
        source = "public_fallback"
        url = ds.fallback_url
        filename = ds.fallback_filename or ds.filename
        gzipped = True  # all fallback expression files are gzipped
        dest = raw_dir / filename

        # A prior dataset in this run may have already downloaded the same fallback file.
        if dest.exists() and any(
            e.get("url") == url and e.get("size_bytes") == dest.stat().st_size
            for e in manifest.values()
        ):
            log.info("[%s] fallback file already present from prior dataset, reusing", ds.name)
            prior = next(e for e in manifest.values() if e.get("url") == url)
            size = prior["size_bytes"]
            sha256 = prior["sha256"]
        else:
            log.info("[%s] fetching fallback %s", ds.name, url)
            size, sha256 = _download_with_retry(client, url, dest)

    log.info("[%s] counting dimensions...", ds.name)
    rows, cols = _count_tsv_dims(dest, gzipped=gzipped)

    manifest[ds.name] = {
        "filename": filename,
        "url": url,
        "sha256": sha256,
        "size_bytes": size,
        "rows": rows,
        "columns": cols,
        "source": source,
        "downloaded_at": datetime.now(UTC).isoformat(),
    }
    log.info(
        "[%s] done: %d rows x %d cols  %.1f MB  source=%s",
        ds.name,
        rows,
        cols,
        size / 1e6,
        source,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_all(
    datasets: list[Dataset],
    raw_dir: Path,
    client: httpx.Client,
    manifest_path: Path,
) -> dict:
    """Download all datasets; persist manifest after each file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for ds in datasets:
        download_dataset(ds, manifest, raw_dir, client)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    with httpx.Client(follow_redirects=True, timeout=_TIMEOUT) as client:
        manifest = download_all(DATASETS, RAW_DIR, client, MANIFEST_PATH)

    log.info("Manifest: %s", MANIFEST_PATH)
    for name, entry in manifest.items():
        log.info(
            "  %-20s  %8.1f MB  %d rows x %d cols  [%s]",
            name,
            entry["size_bytes"] / 1e6,
            entry["rows"],
            entry["columns"],
            entry["source"],
        )


if __name__ == "__main__":
    main()
