"""BANC canonical naming: legacy ``flywire_BANC_*`` identifiers stay working.

DROCAT analyzes BANC from its own public release data, so the dataset
identifiers dropped the ``flywire_`` prefix.  Every dataset-name entry point
must keep accepting the legacy spellings (see
``utils.naming_utils.canonical_dataset_name``).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.naming_utils import canonical_dataset_name


class TestCanonicalDatasetName:
    def test_legacy_banc_names_map_to_canonical(self):
        assert canonical_dataset_name("flywire_BANC_v626") == "banc_v626"
        assert canonical_dataset_name("flywire_BANC_v888") == "banc_v888"
        assert canonical_dataset_name("flywire_banc_v626") == "banc_v626"

    def test_bare_legacy_name_pins_default_release(self):
        # The unversioned alias must not straddle the two BANC releases
        # (v626/v888 have distinct id spaces): it pins to the historical
        # default release, matching the type mapper's own fold.
        assert canonical_dataset_name("flywire_BANC") == "banc_v626"
        assert canonical_dataset_name("banc") == "banc_v626"

    def test_non_banc_names_pass_through(self):
        assert canonical_dataset_name("flywire_FAFB_v783") == "flywire_FAFB_v783"
        assert canonical_dataset_name("male-cns:v1.0") == "male-cns:v1.0"
        assert canonical_dataset_name("banc_v888") == "banc_v888"
        assert canonical_dataset_name("manc:v1.2.3") == "manc:v1.2.3"
        assert canonical_dataset_name("") == ""
        assert canonical_dataset_name(None) == ""

    def test_hidden_neuprint_colon_form_is_not_touched(self):
        assert canonical_dataset_name("banc:v888") == "banc:v888"


class TestDatasetFolderAlias:
    def test_legacy_name_resolves_to_canonical_folder(self):
        from flywire_ids import dataset_folder

        assert dataset_folder("banc_v626") == "banc_v626"
        assert dataset_folder("flywire_BANC_v626") == "banc_v626"
        assert dataset_folder("flywire_FAFB_v783") == "flywire_FAFB_v783"
        assert dataset_folder("male-cns:v1.0") == "male-cns_v1_0"

    def test_resolve_dataset_dir_accepts_legacy_name(self, tmp_path):
        from flywire_ids import resolve_flywire_dataset_dir

        datasets_root = tmp_path / "datasets"
        banc_dir = datasets_root / "banc_v626"
        banc_dir.mkdir(parents=True)
        (banc_dir / "banc_v626_allneurons_neuron_df.parquet").write_text("x")
        # Legacy spelling resolves the renamed directory.
        assert resolve_flywire_dataset_dir(tmp_path, "flywire_BANC_v626") == banc_dir
        assert resolve_flywire_dataset_dir(tmp_path, "banc_v626") == banc_dir


class TestCacheNamespaces:
    def test_cave_fetcher_cache_namespace_canonical(self, tmp_path):
        from cave_data_fetcher import CAVEDataFetcher

        fetcher = CAVEDataFetcher(dataset="flywire_BANC_v626",
                                  project_root=str(tmp_path), verbose=False)
        assert fetcher._cache_dataset_name() == "banc_v626"
        fetcher = CAVEDataFetcher(dataset="banc_v888",
                                  project_root=str(tmp_path), verbose=False)
        assert fetcher._cache_dataset_name() == "banc_v888"
        # FAFB keeps its historical namespace.
        fetcher = CAVEDataFetcher(dataset="flywire_FAFB_v783",
                                  project_root=str(tmp_path), verbose=False)
        assert fetcher._cache_dataset_name() == "flywire_FAFB_v783"

    def test_mesh_cache_dataset_folder_canonical(self):
        from flywire_mesh_cache import _dataset_folder

        assert _dataset_folder("flywire_BANC_v888") == "banc_v888"


class TestBancPublicDataAliases:
    def test_public_release_version_uses_canonical_alias(self):
        import banc_public_data as bpd

        assert bpd._connection_version("banc") == "v626"
        assert bpd._connection_version("flywire_BANC") == "v626"
        assert bpd._connection_version("flywire_BANC_v626") == "v626"
        assert bpd._connection_version("flywire_BANC_v888") == "v888"

    def test_v626_crosswalk_applies_to_bare_legacy_alias(self, tmp_path,
                                                         monkeypatch):
        import banc_public_data as bpd

        monkeypatch.setattr(
            bpd, "get_id_crosswalk",
            lambda dataset, project_root=None: {"626-id": "888-id"},
        )
        assert bpd.resolve_banc_stem(
            "flywire_BANC", "626-id", project_root=tmp_path
        ) == "888-id"


class TestMapperNamespace:
    def _mapper(self):
        from comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper

        mapper = CrossDatasetTypeMapper.__new__(CrossDatasetTypeMapper)
        mapper._unsupported_dataset_warnings = set()
        return mapper

    def test_legacy_and_canonical_share_namespace(self):
        mapper = self._mapper()
        assert mapper._normalize_dataset_name("flywire_BANC_v888") == "banc_v888"
        assert mapper._normalize_dataset_name("banc") == "banc_v626"
        assert mapper._get_type_mapping_key("flywire_BANC_v626") == "banc_v626"
        # §version control: per-release namespaces — banc_v888 resolves
        # against ITS OWN tables, never through the banc_v626 namespace.
        assert mapper._get_type_mapping_key("banc_v888") == "banc_v888"

    def test_fafb_namespace_untouched(self):
        mapper = self._mapper()
        assert mapper._get_type_mapping_key("flywire_FAFB_v783") == "flywire_FAFB_v783"
        # §version control: male-cns v0.9 keeps its own namespace (the v1.0
        # crosswalk cannot verify v0.9 names).
        assert mapper._get_type_mapping_key("male-cns:v0.9") == "male-cns:v0.9"


class TestReadinessPredicates:
    def test_predicates_on_canonical_names(self):
        from utils.flywire_readiness import is_banc_dataset, is_fafb_dataset

        assert is_banc_dataset("banc_v626")
        assert is_banc_dataset("flywire_BANC_v626")
        assert not is_fafb_dataset("banc_v888")
        assert is_fafb_dataset("flywire_FAFB_v783")


class TestUiDatasetNames:
    def test_folder_round_trip(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        import ui.dataset_service as ds_mod

        assert ds_mod.folder_to_dataset("banc_v626") == "banc_v626"
        assert ds_mod.dataset_to_folder("banc_v626") == "banc_v626"
        # The generic NeuPrint rule must not invent a colon form.
        assert ds_mod.folder_to_dataset("banc_v626") != "banc:v626"

    def test_ui_predicates_keep_banc_out_of_flywire(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        import ui.dataset_service as ds_mod

        assert ds_mod.is_flywire_dataset("banc_v888") is False
        assert ds_mod.is_flywire_dataset("flywire_BANC_v888") is False
        assert ds_mod.is_banc_dataset("banc_v888") is True
        assert ds_mod.is_banc_dataset("flywire_BANC_v888") is True
        # The hidden, non-queryable NeuPrint entry stays excluded.
        assert ds_mod.is_flywire_dataset("banc:v888") is False
        assert ds_mod.is_banc_dataset("banc:v888") is True
        assert ds_mod.is_flywire_dataset("male-cns:v1.0") is False

    def test_static_catalog_uses_canonical_names(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from ui.config import BANC_DATASETS, DATASETS, FLYWIRE_DATASETS

        assert "banc_v626" in DATASETS and "banc_v888" in DATASETS
        assert "banc_v626" in BANC_DATASETS
        assert "banc_v626" not in FLYWIRE_DATASETS
        assert not any(name.startswith("flywire_BANC") for name in DATASETS)

    def test_ui_label_uses_independent_banc_tag(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from ui.components.common import _dataset_label_parts

        class _Svc:
            _cache = {}

            def _check_local_prepared(self, ds):
                return True

            def _check_local_cache(self, ds):
                return False

        svc = _Svc()
        assert _dataset_label_parts("banc_v888", svc)[1] == "[BANC]"
        assert _dataset_label_parts("flywire_BANC_v888", svc)[1] == "[BANC]"
        assert _dataset_label_parts("flywire_FAFB_v783", svc)[1] == "[FAFB]"
        assert _dataset_label_parts("male-cns:v1.0", svc)[1] == "[NP]" 


class TestBancSynapseDefaults:
    """BANC defaults to skipped synapses; opting in carries an explicit
    warning about the one-time table download and pre-site-only markers."""

    def test_default_view(self):
        from ui.tabs.visualization import banc_synapse_view_default

        assert banc_synapse_view_default("banc_v888") == "skip"
        assert banc_synapse_view_default("banc_v626") == "skip"
        assert banc_synapse_view_default("flywire_FAFB_v783") == "synapse"

    def test_warning_only_when_enabled(self):
        from ui.tabs.visualization import banc_synapse_warning

        assert banc_synapse_warning("banc_v888", "skip") is None
        warning = banc_synapse_warning("banc_v888", "synapse")
        assert warning is not None
        assert "3.9 GB" in warning
        assert "PRE-synaptic site coordinates" in warning
        # non-BANC datasets are never flagged
        assert banc_synapse_warning("flywire_FAFB_v783", "synapse") is None
