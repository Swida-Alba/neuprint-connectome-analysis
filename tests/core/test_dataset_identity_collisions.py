"""Regression tests for release-aware cross-dataset identities."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison import ComparisonParameters  # noqa: E402
from comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper  # noqa: E402


def test_comparison_labels_and_output_codes_disambiguate_releases():
    params = ComparisonParameters(
        datasets=["male-cns:v1.0", "male-cns:v0.9"],
        source_neurons=["aMe12"],
        target_neurons=["PPL101"],
        auto_type_mapping=False,
        verbose=False,
    )

    assert params.get_dataset_nicknames() == ["MCNS_v1_0", "MCNS_v0_9"]
    assert params.get_nickname_map() == {
        "male-cns:v1.0": "MCNS_v1_0",
        "male-cns:v0.9": "MCNS_v0_9",
    }
    assert params._get_dataset_short_codes() == "M_v1_0M_v0_9"


def test_cross_dataset_mapper_preserves_explicit_release_tokens(tmp_path):
    neuron_df = tmp_path / "neurons.csv"
    neuron_df.write_text(
        "bodyId,type,flywireType,hemibrainType,mancType\n"
        "1,MeVPLo2,MTe07,MeVPLo2,\n",
        encoding="utf-8",
    )
    mapper = CrossDatasetTypeMapper(
        neuron_df_path=str(neuron_df),
        verbose=False,
    )

    assert mapper._normalize_dataset_name("male-cns:v0.9") == "male-cns:v0.9"
    assert mapper._normalize_dataset_name("banc_v888") == "banc_v888"
    # §version control: releases are per-release mapping namespaces —
    # male-cns:v0.9 keeps its own namespace instead of silently borrowing
    # the v1.0 crosswalk, and banc_v888 resolves against its own neuron
    # tables, never through banc_v626.
    assert mapper._get_type_mapping_key("male-cns:v0.9") == "male-cns:v0.9"
    assert mapper._get_type_mapping_key("banc_v888") == "banc_v888"

    # v1.0 <-> banc_v888 pairs still resolve hermetically from the temp
    # table's type <-> flywireType columns.
    assert mapper.get_mapped_type(
        "MeVPLo2", "male-cns:v1.0", "banc_v888"
    ) == "MTe07"
    assert mapper.get_mapped_type(
        "MTe07", "banc_v888", "male-cns:v1.0"
    ) == "MeVPLo2"
    assert mapper.get_canonical_type("MTe07", "banc_v888") == "MeVPLo2"

    # v0.9 borrows nothing: no resolution through the v1.0 namespace, and
    # the unsupported-release warning fires exactly for it.
    assert mapper.get_mapped_type(
        "MeVPLo2", "male-cns:v0.9", "flywire_FAFB_v783"
    ) is None
    assert mapper._unsupported_dataset_warnings == {"male-cns:v0.9"}

    resolved = mapper.resolve_type_across_datasets(
        "MeVPLo2",
        ["male-cns:v1.0", "banc_v888"],
        source_dataset="male-cns:v1.0",
    )
    assert resolved == {
        "male-cns:v1.0": "MeVPLo2",
        "banc_v888": "MTe07",
    }
    assert mapper.resolve_type_across_datasets(
        "MeVPLo2",
        ["male-cns:v0.9"],
        source_dataset="male-cns:v1.0",
    ) == {"male-cns:v0.9": None}


def test_mapper_legends_keep_both_colliding_releases():
    mapper = CrossDatasetTypeMapper(verbose=False)

    assert mapper.get_all_dataset_short_codes(
        ["male-cns:v1.0", "male-cns:v0.9"]
    ) == {
        "M_v1_0": "male-cns v1.0",
        "M_v0_9": "male-cns v0.9",
    }
    assert mapper.get_all_dataset_short_codes(
        ["banc_v626", "banc_v888"]
    ) == {
        "B_v626": "BANC v626",
        "B_v888": "BANC v888",
    }
