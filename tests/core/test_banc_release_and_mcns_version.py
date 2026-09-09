"""Hermetic tests for the BANC label/release overlays and MCNS aliases."""

from pathlib import Path

import pandas as pd

from comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper


MCNS = "male-cns:v1.0"
MCNS_09 = "male-cns:v0.9"
FAFB = "flywire_FAFB_v783"
BANC626 = "banc_v626"
BANC888 = "banc_v888"


def _build_mapper(
    tmp_path: Path, *, include_relation: bool = True
) -> CrossDatasetTypeMapper:
    v10 = tmp_path / "mcns_v10.csv"
    v10.write_text(
        "bodyId,type,flywireType,hemibrainType,mancType\n"
        "1,Shared,FwShared,HbShared,MnShared\n"
        "2,Other,FwOther,HbOther,MnOther\n",
        encoding="utf-8",
    )
    v09 = tmp_path / "mcns_v09.csv"
    v09.write_text(
        "bodyId,type,flywireType,hemibrainType,mancType\n"
        "101,Shared,FwShared,HbShared,MnShared\n"
        "2,Shared,FwShared,HbShared,MnShared\n"
        "102,LegacyOnly,LegacyFW,,\n",
        encoding="utf-8",
    )
    fafb = tmp_path / "fafb.csv"
    fafb.write_text(
        "bodyId,type,additional_type(s)\n"
        "f1,FType,\n"
        "f2,FwShared,\n"
        "f3,LegacyFW,\n",
        encoding="utf-8",
    )

    banc626 = tmp_path / "banc626.csv"
    banc626.write_text(
        "bodyId,type,Alternative Cell Type(s),malecns_cell_type,"
        "malecns_match,fafb_cell_type,fafb_match\n"
        "6261,T626,,Shared,1,FType,f1\n"
        "6262,T626,,Shared,1,FType,f1\n"
        "6263,T626,,Other,2,FType,f1\n"
        "6264,TBad,,Shared,2,,\n"
        "6265,AutoType,,auto:machine,1,,\n"
        "6266,SameName,,,,,\n"
        "6267,AutoShared,,auto:Shared,1,,\n",
        encoding="utf-8",
    )
    banc888 = tmp_path / "banc888.csv"
    banc888.write_text(
        "bodyId,type,Alternative Cell Type(s),malecns_cell_type,"
        "malecns_match,fafb_cell_type,fafb_match\n"
        "8881,T888,,Shared,1,FType,f1\n"
        "8882,T888,,Shared,1,FType,f1\n"
        "8883,T888b,,Shared,1,FType,f1\n"
        "8884,SameName,,,,,\n",
        encoding="utf-8",
    )

    if include_relation:
        relation_path = (
            tmp_path / "compiled_data" / "banc_888" / "banc_888_meta.feather"
        )
        relation_path.parent.mkdir(parents=True)
        # The repeated 6261→8881 row is intentional: the release overlay must
        # retain repeated root_626 rows rather than collapsing the relation.
        pd.DataFrame(
            {
                "root_626": [6261, 6261, 6261, 6262, 6263],
                "root_888": [8881, 8881, 8883, 8881, 8882],
            }
        ).to_feather(relation_path)

    mapper = CrossDatasetTypeMapper(
        workspace_path=str(tmp_path),
        neuron_df_path=str(v10),
        mcns_v09_neuron_df_path=str(v09),
        flywire_neuron_df_paths={
            FAFB: str(fafb),
            BANC626: str(banc626),
            BANC888: str(banc888),
        },
        verbose=False,
    )
    assert mapper.load() is True
    return mapper


def test_banc_label_overlay_uses_curated_columns_and_verification(tmp_path):
    mapper = _build_mapper(tmp_path)

    # MCNS↔BANC is the curated MCNS-side label, not the MCNS flywireType
    # cell.  In this fixture the evidence fans out to three BANC primaries,
    # so it remains visible as a split but no single target is accepted.
    assert mapper.get_mapped_type("Shared", MCNS, BANC626) is None
    decision = mapper.get_mapping_decision("Shared", MCNS, BANC626)
    assert decision["status"] == "valid_split_evidence"
    assert set(decision["target_types"]) == {"T626", "TBad", "AutoShared"}
    assert decision["relationship"] == "1-to-N"
    assert mapper.get_mapping_conflicts(MCNS, BANC626, "Shared")
    assert mapper.get_mapped_type("T626", BANC626, MCNS) == "Shared"
    provenance = mapper._bridge_provenance[(BANC626, "T626", MCNS)]
    assert provenance["kind"] == "cross-dataset cell type"
    assert provenance["column"] == "malecns_cell_type"
    assert provenance["votes"] == {"Shared": 2, "Other": 1}
    assert provenance["verified_votes"] == {"Shared": 2, "Other": 1}

    # The match bodyId disagrees with row 6264's Shared label.  The match is
    # retained as a diagnostic conflict, but it does not gate the type-level
    # label bridge.
    bad_provenance = mapper._bridge_provenance[(BANC626, "TBad", MCNS)]
    assert bad_provenance["verification_conflicts"] == {"Other": 1}
    assert mapper.get_mapped_type("TBad", BANC626, MCNS) == "Shared"
    # Machine-only labels are not automatic evidence.
    assert mapper.get_mapped_type("AutoType", BANC626, MCNS) is None

    # ``auto:`` is a provenance tier, not a separate type namespace.  A
    # stripped machine label that names a current MCNS type participates in
    # the vote and records that provenance on both bridge directions.
    assert mapper.get_mapped_type("AutoShared", BANC626, MCNS) == "Shared"
    auto_provenance = mapper._bridge_provenance[(
        BANC626, "AutoShared", MCNS)]
    assert auto_provenance["winner_derived_from_auto"] is True
    assert auto_provenance["auto_stripped_votes"] == {"Shared": 1}
    assert mapper._banc_label_votes[(
        BANC626, "malecns_cell_type", "AutoShared"
    )]["auto_stripped_votes"] == {"Shared": 1}

    # The same policy applies to the optional HEMI/MANC label columns: a
    # normalized auto label must resolve to a known target type, while a
    # curated non-auto label may remain usable without a local target table.
    assert mapper._banc_label_candidate_details(
        "auto:HbShared", "hemibrain_cell_type") == [("HbShared", True)]
    assert mapper._banc_label_candidate_details(
        "auto:MnShared", "manc_cell_type") == [("MnShared", True)]
    assert mapper._banc_label_candidate_details(
        "auto:not-a-known-type", "hemibrain_cell_type") == []
    assert mapper._banc_label_candidate_details(
        "auto:not-a-known-type", "manc_cell_type") == []

    # Exercise the normal Polars aggregation path as well as the compatibility
    # candidate helper: raw auto provenance must survive normalization before
    # unknown HEMI/MANC tokens are filtered.
    import polars as pl

    labels = pl.DataFrame({
        "type": ["AutoHemiUnknown", "AutoHemiKnown",
                 "AutoMancUnknown", "AutoMancKnown"],
        "hemibrain_cell_type": ["auto:not-known", "auto:HbShared", "", ""],
        "manc_cell_type": ["", "", "auto:not-known", "auto:MnShared"],
    })
    hemi_batches = mapper._banc_label_vote_batches_polars(
        BANC626, labels, "hemibrain_cell_type", ["hemibrain:v1.2.1"])
    manc_batches = mapper._banc_label_vote_batches_polars(
        BANC626, labels, "manc_cell_type", ["manc:v1.0", "manc:v1.2.1"])
    assert [batch[0] for batch in hemi_batches] == ["AutoHemiKnown"]
    assert [batch[0] for batch in manc_batches] == ["AutoMancKnown"]

    # A direct FAFB label is separately exposed as FAFB↔BANC evidence.
    assert mapper.get_mapped_type("FType", FAFB, BANC626) == "T626"
    chains = mapper.get_type_bridges("FType", FAFB, BANC626)
    assert any(
        any(hop["column"] == "fafb_cell_type" for hop in chain[1:])
        for chain in chains
    )


def test_banc_release_overlay_uses_full_root_relation(tmp_path):
    mapper = _build_mapper(tmp_path)

    assert mapper.get_mapped_type("T626", BANC626, BANC888) == "T888"
    assert mapper.get_mapped_type("T888", BANC888, BANC626) == "T626"
    votes = mapper._banc_release_votes[(BANC626, "T626", BANC888)]
    assert votes == {"T888": 4, "T888b": 1}

    chains = mapper.get_type_bridges("T626", BANC626, BANC888)
    assert chains
    assert any(
        [hop["column"] for hop in chain[1:]] == ["banc_release_crosswalk"]
        for chain in chains
    )
    assert all(
        all(hop["column"] not in {
            "flywireType",
            "Alternative Cell Type(s)",
            "malecns_cell_type",
        } for hop in chain[1:])
        for chain in chains
    )
    provenance = mapper._bridge_provenance[(BANC626, "T626", BANC888)]
    assert provenance["source_id_column"] == "root_626"
    assert provenance["target_id_column"] == "root_888"


def test_banc_same_name_fallback_does_not_require_release_relation(tmp_path):
    mapper = _build_mapper(tmp_path, include_relation=False)

    assert mapper.get_mapped_type("SameName", BANC626, BANC888) == "SameName"
    chains = mapper.get_type_bridges("SameName", BANC626, BANC888)
    assert any(
        chain[1]["column"] == "type"
        and chain[1]["value"] == "SameName"
        for chain in chains
        if len(chain) > 1
    )


def test_mcns_v09_shared_and_native_only_resolution(tmp_path):
    mapper = _build_mapper(tmp_path)

    assert mapper.get_mapped_type("Shared", MCNS_09, MCNS) == "Shared"
    assert mapper.get_mapped_type("Shared", MCNS_09, BANC888) is None
    decision = mapper.get_mapping_decision("Shared", MCNS_09, BANC888)
    assert decision["status"] == "valid_split_evidence"
    assert set(decision["target_types"]) == {"T888", "T888b"}
    alias_chain = mapper.get_type_bridges("Shared", MCNS_09, FAFB)[0]
    assert alias_chain[0]["dataset"] == MCNS_09
    assert alias_chain[1]["column"] == "release_alias"

    # The old-only name does not invent a v1.0 identity, but its native
    # flywireType column remains usable as a direct, lower-tier fallback.
    assert mapper.get_mapped_type("LegacyOnly", MCNS_09, MCNS) is None
    assert mapper.get_mapped_type("LegacyOnly", MCNS_09, FAFB) == "LegacyFW"
    assert mapper._release_alias_diagnostics["shared_type_names"] == 1
    assert mapper._release_alias_diagnostics[
        "release_alias_disagreement"]["body_id_count"] == 1
    assert mapper._release_alias_diagnostics[
        "release_alias_disagreement"]["shared_names"] == ["Shared"]
