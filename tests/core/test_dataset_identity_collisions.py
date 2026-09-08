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
    neuron_df = tmp_path / "neurons_v10.csv"
    neuron_df.write_text(
        "bodyId,type,flywireType,hemibrainType,mancType\n"
        "1,MeVPLo2,MTe07,MeVPLo2,\n",
        encoding="utf-8",
    )
    neuron_df_v09 = tmp_path / "neurons_v09.csv"
    neuron_df_v09.write_text(
        "bodyId,type,flywireType,hemibrainType,mancType\n"
        "901,MeVPLo2,MTe07,MeVPLo2,\n"
        "902,LegacyOnly,LegacyFW,,\n",
        encoding="utf-8",
    )
    banc_v888 = tmp_path / "banc_v888.csv"
    banc_v888.write_text(
        "bodyId,type,Alternative Cell Type(s),malecns_cell_type,"
        "fafb_cell_type\n"
        "8881,BancMe,,MeVPLo2,MTe07\n",
        encoding="utf-8",
    )
    fafb = tmp_path / "fafb.csv"
    fafb.write_text(
        "bodyId,type,additional_type(s)\n"
        "f1,MTe07,\n"
        "f2,LegacyFW,\n",
        encoding="utf-8",
    )
    mapper = CrossDatasetTypeMapper(
        neuron_df_path=str(neuron_df),
        mcns_v09_neuron_df_path=str(neuron_df_v09),
        flywire_neuron_df_paths={
            "banc_v888": str(banc_v888),
            "flywire_FAFB_v783": str(fafb),
        },
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

    # v1.0 <-> BANC v888 is grounded by the BANC malecns_cell_type label.
    assert mapper.get_mapped_type(
        "MeVPLo2", "male-cns:v1.0", "banc_v888"
    ) == "BancMe"
    assert mapper.get_mapped_type(
        "BancMe", "banc_v888", "male-cns:v1.0"
    ) == "MeVPLo2"
    assert mapper.get_canonical_type("BancMe", "banc_v888") == "MeVPLo2"

    # A shared v0.9 name exposes an explicit same-name release alias before
    # delegating to the v1.0 crosswalk.
    assert mapper.get_mapped_type(
        "MeVPLo2", "male-cns:v0.9", "flywire_FAFB_v783"
    ) == "MTe07"
    assert mapper.get_mapped_type(
        "MeVPLo2", "male-cns:v0.9", "banc_v888"
    ) == "BancMe"
    assert mapper.get_type_bridges(
        "MeVPLo2", "male-cns:v0.9", "flywire_FAFB_v783"
    )[0][1]["column"] == "release_alias"
    assert mapper._unsupported_dataset_warnings == set()

    # A v0.9-only name never invents a v1.0 name; it uses the native v0.9
    # flywireType fallback when that target is grounded.
    assert mapper.get_mapped_type(
        "LegacyOnly", "male-cns:v0.9", "flywire_FAFB_v783"
    ) == "LegacyFW"
    assert mapper.get_mapped_type(
        "LegacyOnly", "male-cns:v0.9", "male-cns:v1.0"
    ) is None

    resolved = mapper.resolve_type_across_datasets(
        "MeVPLo2",
        ["male-cns:v1.0", "banc_v888"],
        source_dataset="male-cns:v1.0",
    )
    assert resolved == {
        "male-cns:v1.0": "MeVPLo2",
        "banc_v888": "BancMe",
    }
    assert mapper.resolve_type_across_datasets(
        "MeVPLo2",
        ["male-cns:v0.9"],
        source_dataset="male-cns:v1.0",
    ) == {"male-cns:v0.9": "MeVPLo2"}

    # banc_v888 sits in DATASET_PRIORITY (after banc_v626), so a
    # v888-only type auto-detects its own namespace in the priority walk
    # instead of falling through to "unknown".
    assert mapper._detect_type_source("BancMe") == "banc_v888"
    assert mapper._detect_type_source("MeVPLo2") == "male-cns:v1.0"


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
