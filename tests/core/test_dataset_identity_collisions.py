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
    assert mapper._normalize_dataset_name("flywire_BANC_v888") == "flywire_BANC_v888"
    # Releases remain distinct dataset identities, but Male-CNS releases
    # share the Male-CNS type namespace, FAFB releases share the FAFB
    # namespace, and BANC releases share the BANC namespace.
    assert mapper._get_type_mapping_key("male-cns:v0.9") == "male-cns:v1.0"
    assert mapper._get_type_mapping_key("flywire_BANC_v888") == "flywire_BANC_v626"
    assert mapper.get_mapped_type(
        "MeVPLo2", "male-cns:v0.9", "flywire_FAFB_v783"
    ) == "MTe07"
    assert mapper.get_mapped_type(
        "MeVPLo2", "male-cns:v0.9", "flywire_BANC_v888"
    ) == "MTe07"
    assert mapper.get_mapped_type(
        "MTe07", "flywire_BANC_v888", "male-cns:v0.9"
    ) == "MeVPLo2"
    assert mapper.get_canonical_type("MTe07", "flywire_BANC_v888") == "MeVPLo2"
    assert mapper._unsupported_dataset_warnings == set()

    resolved = mapper.resolve_type_across_datasets(
        "MeVPLo2",
        ["male-cns:v0.9", "flywire_BANC_v888"],
        source_dataset="male-cns:v0.9",
    )
    assert resolved == {
        "male-cns:v0.9": "MeVPLo2",
        "flywire_BANC_v888": "MTe07",
    }


def test_mapper_legends_keep_both_colliding_releases():
    mapper = CrossDatasetTypeMapper(verbose=False)

    assert mapper.get_all_dataset_short_codes(
        ["male-cns:v1.0", "male-cns:v0.9"]
    ) == {
        "M_v1_0": "male-cns v1.0",
        "M_v0_9": "male-cns v0.9",
    }
    assert mapper.get_all_dataset_short_codes(
        ["flywire_BANC_v626", "flywire_BANC_v888"]
    ) == {
        "B_v626": "FlyWire BANC v626",
        "B_v888": "FlyWire BANC v888",
    }
