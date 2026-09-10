"""Shared compressed-SWC provenance contract.

The `# DROCAT simpl:` / `# DROCAT source:` header pair is written and parsed
through one module (`skeleton_provenance`) so the NeuPrint, BANC, and FAFB
skeleton backends agree. These tests pin the parser/writer semantics and the
provenance aliases the backends still expose under their private names.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import skeleton_provenance as sp  # noqa: E402


def test_header_constants_values():
    assert sp.SIMPLIFICATION_HEADER == "DROCAT simpl:"
    assert sp.SOURCE_HEADER == "DROCAT source:"


def test_read_stored_simplification_variants():
    read = sp.read_stored_simplification
    assert read("# DROCAT simpl: 50\n1 1 0 0 0 1 -1") == 50
    assert read(b"# DROCAT simpl: 90\n") == 90
    assert read("# DROCAT simpl: bogus\n") == 0        # malformed -> raw
    assert read("1 1 0 0 0 1 -1\n") == 0               # headerless -> raw
    assert read("") == 0


def test_read_stored_source_variants():
    read = sp.read_stored_source
    assert read("# DROCAT source: banc_gcs_full\n") == "banc_gcs_full"
    assert read(b"# DROCAT source: cave_mesh_wavefront\nx") == \
        "cave_mesh_wavefront"
    assert read("# SWC body\n1 1 0 0 0 1 -1\n") == ""
    # only the leading header block is scanned
    body = "1 1 0 0 0 1 -1\n" * 20 + "# DROCAT source: late\n"
    assert read(body) == ""


def test_parse_provenance_both_values():
    prov = sp.parse_provenance(
        "# DROCAT simpl: 0\n# DROCAT source: local_extrusion_fix\nx")
    assert prov.simplification == 0
    assert prov.source == "local_extrusion_fix"


def test_resolution_from_source():
    rf = sp.resolution_from_source
    assert rf("banc_gcs_full") == "full"
    assert rf("banc_gcs_l2") == "l2"
    assert rf("banc_gcs_pcg_um_x1000") == "l2"
    assert rf("") == "l2"
    assert rf("cave_mesh_wavefront") == "l2"
    assert rf(None) == "l2"


def test_is_repaired_source():
    assert sp.is_repaired_source("cave_mesh_wavefront")
    assert sp.is_repaired_source("local_extrusion_fix")
    assert not sp.is_repaired_source("banc_gcs_full")
    assert not sp.is_repaired_source("")


def test_source_of_reads_attribute():
    class _N:
        _drocat_source = "banc_gcs_full"

    assert sp.source_of(_N()) == "banc_gcs_full"
    assert sp.source_of(object()) == ""


def test_write_roundtrip_and_source_line(tmp_path):
    body = b"1 1 0 0 0 1 -1\n2 1 1 0 0 1 1\n"
    path = tmp_path / "x.swc.zst"
    sp.write_compressed_swc_zst(path, sp.make_source_line("banc_gcs_full")
                                + body, simplification=7)

    import zstandard as zstd
    with open(path, "rb") as handle:
        with zstd.ZstdDecompressor().stream_reader(handle) as reader:
            content = reader.read().decode("utf-8", "replace")
    lines = content.splitlines()
    assert lines[0] == "# DROCAT simpl: 7"
    assert lines[1] == "# DROCAT source: banc_gcs_full"
    assert sp.read_stored_simplification(content) == 7
    assert sp.read_stored_source(content) == "banc_gcs_full"


def test_backend_aliases_point_at_shared_module():
    """The backends keep their private names but share one implementation."""
    import cave_data_fetcher as cdf
    import morphology as morph
    import banc_public_data as bpd

    assert cdf._SIMPLIFICATION_HEADER == sp.SIMPLIFICATION_HEADER
    assert cdf._SOURCE_HEADER == sp.SOURCE_HEADER
    assert cdf._read_stored_source is not None
    assert morph.SIMPLIFICATION_HEADER == sp.SIMPLIFICATION_HEADER
    assert morph.SOURCE_HEADER == sp.SOURCE_HEADER
    # BANC's resolution helper delegates to the shared classifier
    assert bpd._resolution_from_source_header(
        "# DROCAT source: banc_gcs_full\n") == "full"
    assert bpd._resolution_from_source_header(
        "# DROCAT source: banc_gcs_l2\n") == "l2"
    assert bpd._SOURCE_HEADER == sp.SOURCE_HEADER


def test_backend_parsers_agree_with_shared_module():
    import morphology as morph
    import cave_data_fetcher as cdf

    sample = "# DROCAT simpl: 42\n# DROCAT source: cave_mesh_wavefront\nx"
    for mod in (morph, cdf):
        assert mod._read_stored_simplification(sample) == 42
        assert mod._read_stored_source(sample) == "cave_mesh_wavefront"
