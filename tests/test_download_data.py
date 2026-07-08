import gzip
import json
from pathlib import Path

import pytest

from pipeline.download_data import (
    DATASETS,
    _count_tsv_dims,
    download_all,
    download_dataset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gz(rows: int = 5, cols: int = 4) -> bytes:
    header = "\t".join(["gene"] + [f"s{i}" for i in range(cols - 1)])
    lines = [header] + [
        "\t".join([f"GENE{r}"] + [f"{r * 0.1:.4f}" for _ in range(cols - 1)])
        for r in range(rows - 1)
    ]
    return gzip.compress("\n".join(lines).encode())


def _make_tsv(rows: int = 5, cols: int = 4) -> bytes:
    header = "\t".join(["sampleID"] + [f"col{i}" for i in range(cols - 1)])
    lines = [header] + [
        "\t".join([f"TCGA-XX-000{r}-01"] + [f"val{r}" for _ in range(cols - 1)])
        for r in range(rows - 1)
    ]
    return "\n".join(lines).encode()


class _FakeStreamResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.status_code = status_code
        self._content = content

    def raise_for_status(self) -> None:
        pass

    def iter_bytes(self, chunk_size: int | None = None):
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _FakeClient:
    """Returns gz content for .gz URLs; plain TSV for all others."""

    def __init__(self, gz_content: bytes, plain_content: bytes | None = None):
        self._gz = gz_content
        self._plain = plain_content or _make_tsv()

    def stream(self, method: str, url: str, **kwargs):
        content = self._gz if url.endswith(".gz") else self._plain
        return _FakeStreamResponse(content, 200)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _Auth403Client:
    """Returns 403 for tcga.xenahubs.net .gz URLs; 200 for everything else."""

    def __init__(self, public_gz: bytes, plain_content: bytes | None = None):
        self._gz = public_gz
        self._plain = plain_content or _make_tsv()

    def stream(self, method: str, url: str, **kwargs):
        if "tcga.xenahubs.net" in url and url.endswith(".gz"):
            return _FakeStreamResponse(b"<Error>AccessDenied</Error>", status_code=403)
        if url.endswith(".gz"):
            return _FakeStreamResponse(self._gz, status_code=200)
        return _FakeStreamResponse(self._plain, status_code=200)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "raw"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def fake_gz() -> bytes:
    return _make_gz(rows=5, cols=4)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_files_land_in_raw_dir_with_expected_names(raw_dir: Path, fake_gz: bytes) -> None:
    """All datasets are downloaded to their expected filenames."""
    client = _FakeClient(fake_gz)
    manifest_path = raw_dir / "manifest.json"

    download_all(DATASETS, raw_dir, client, manifest_path)

    found = {p.name for p in raw_dir.iterdir() if p.name != "manifest.json"}
    for ds in DATASETS:
        assert ds.filename in found, f"Expected {ds.filename!r} in {found}"


def test_manifest_written_with_all_datasets(raw_dir: Path, fake_gz: bytes) -> None:
    """manifest.json contains one entry per dataset with required fields."""
    client = _FakeClient(fake_gz)
    manifest_path = raw_dir / "manifest.json"

    download_all(DATASETS, raw_dir, client, manifest_path)

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest.keys()) == {ds.name for ds in DATASETS}

    required = {
        "filename",
        "url",
        "sha256",
        "size_bytes",
        "rows",
        "columns",
        "downloaded_at",
        "source",
    }
    for name, entry in manifest.items():
        missing = required - entry.keys()
        assert not missing, f"{name}: missing keys {missing}"
        assert entry["size_bytes"] > 0
        assert entry["rows"] == 5
        assert entry["columns"] == 4


def test_fallback_activated_on_403(raw_dir: Path, fake_gz: bytes) -> None:
    """When TCGA hub returns 403, the public fallback URL is used."""
    client = _Auth403Client(fake_gz)
    ds = DATASETS[0]  # luad_expression -> fallback is tcga_expression_rsem.tsv.gz
    manifest: dict = {}

    download_dataset(ds, manifest, raw_dir, client)

    assert ds.fallback_filename is not None
    fallback_file = raw_dir / ds.fallback_filename
    assert fallback_file.exists(), f"Fallback file {ds.fallback_filename!r} not found in {raw_dir}"
    assert manifest[ds.name]["source"] == "public_fallback"
    assert manifest[ds.name]["filename"] == ds.fallback_filename


def test_expression_fallback_file_reused_for_same_url(raw_dir: Path, fake_gz: bytes) -> None:
    """LUAD and LUSC expression datasets sharing the same Toil fallback download it only once."""
    download_count = {"n": 0}

    class _CountingAuth403Client:
        def stream(self, method, url, **kwargs):
            if "tcga.xenahubs.net" in url and url.endswith(".gz"):
                return _FakeStreamResponse(b"err", status_code=403)
            download_count["n"] += 1
            return _FakeStreamResponse(fake_gz, status_code=200)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    client = _CountingAuth403Client()
    manifest_path = raw_dir / "manifest.json"

    expression = [ds for ds in DATASETS if "expression" in ds.name]
    download_all(expression, raw_dir, client, manifest_path)

    assert download_count["n"] == 1, (
        f"Expected 1 HTTP download for shared fallback, got {download_count['n']}"
    )


def test_skip_already_complete(raw_dir: Path, fake_gz: bytes) -> None:
    """A file that is already present and size-matches the manifest is not re-downloaded."""
    manifest_path = raw_dir / "manifest.json"
    client = _FakeClient(fake_gz)

    m1 = download_all(DATASETS, raw_dir, client, manifest_path)
    sha_before = m1[DATASETS[0].name]["sha256"]

    # Second run would produce different content if re-downloaded
    different_content = _make_gz(rows=99, cols=50)
    client2 = _FakeClient(different_content)
    m2 = download_all(DATASETS, raw_dir, client2, manifest_path)

    assert m2[DATASETS[0].name]["sha256"] == sha_before


def test_count_tsv_dims_gzipped(tmp_path: Path) -> None:
    gz_file = tmp_path / "test.tsv.gz"
    gz_file.write_bytes(_make_gz(rows=10, cols=6))
    rows, cols = _count_tsv_dims(gz_file, gzipped=True)
    assert rows == 10
    assert cols == 6


def test_count_tsv_dims_plain(tmp_path: Path) -> None:
    tsv_file = tmp_path / "test.tsv"
    tsv_file.write_bytes(_make_tsv(rows=8, cols=3))
    rows, cols = _count_tsv_dims(tsv_file, gzipped=False)
    assert rows == 8
    assert cols == 3
