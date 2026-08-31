"""FindSimilar Tab - Morphological and connection-profile similarity search."""

import re

from nicegui import ui

from ..config import (
    CANDIDATE_SOURCE_OPTIONS,
    DEFAULTS,
    MORPH_LEVEL_OPTIONS,
    MORPH_METHOD_OPTIONS,
    PROJECT_ROOT,
    SIMILARITY_METRICS,
    SRC_DIR,
    get_user_default,
)
from ..components.common import (
    dataset_selector, neuron_list_input, number_input, select_input,
    checkbox_input, dir_input, section_header, param_grid, tool_page,
    apply_filter_mode,
)
from ..components.output_panel import OutputPanel
from ..components.skeleton_visualization_settings import skeleton_visualization_settings
from ..runner import ScriptRunner
from ..type_suggestions import dataset_suggestions
from ..dataset_service import is_banc_dataset

# Option lists live centrally in ui/config; labels for the method select
# stay local because the backend only knows the raw keys.
_MORPH_METHOD_LABELS = {
    "vector_v2": "Vector (spatial)",
    "nblast": "NBLAST",
}
MORPH_METHODS = {
    method: _MORPH_METHOD_LABELS.get(method, method)
    for method in MORPH_METHOD_OPTIONS
}


def create_find_similar_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("Similarity Output")
    dataset = None
    source_dataset = None

    def _morph_suggest(text):
        dataset_name = dataset.value if dataset is not None else ""
        return dataset_suggestions(text, dataset_name, limit=None)

    def _profile_suggest(text):
        dataset_name = source_dataset.value if source_dataset is not None else ""
        return dataset_suggestions(text, dataset_name, limit=None)

    form_col, results_col = tool_page(
        "Similar Neurons",
        "Find morphologically or connectivity-profile similar neurons.",
        icon="science",
        doc="find_similar.md",
    )

    with form_col:
        mode_value = {"value": "Morphological similarity"}
        with ui.row().classes(
            "w-full items-center justify-between gap-8 px-2"
        ):
            morph_mode_button = ui.button(
                "Morphological similarity",
            ).props("outline no-caps").classes("w-5/12")
            connectivity_mode_button = ui.button(
                "Connectivity similarity",
            ).props("outline no-caps").classes("w-5/12")
            for button in (morph_mode_button, connectivity_mode_button):
                button.style(
                    "min-height: 3.5rem; font-size: 1.1rem; "
                    "font-weight: 700;"
                )

        # ================= Morphological similarity panel =================
        with ui.column().classes("w-full gap-1") as morph_panel:
            with ui.card().classes("w-full drocat-card").props('id="card-findsimilar-morphology-dataset"'):
                section_header("Dataset", "storage")
                dataset = dataset_selector(
                    disable_banc=True,
                    hint="Dataset to search for similar neurons in.",
                )
                morph_output_dir = dir_input(scope="find_similar_morphology")
                morph_dataset_warning = ui.label(
                    "⚠️ BANC morphological similarity is unavailable because "
                    "FlyWire does not provide BANC skeletons. Select a non-BANC dataset."
                ).classes("text-caption text-amber-8").set_visibility(False)
                morph_best_dataset_warning = ui.label(
                    "⚠️ Morphological similarity works best with the "
                    "male-cns:v1.0 dataset; results on other datasets may be "
                    "less reliable."
                ).classes("text-caption text-amber-8").set_visibility(False)

            with ui.card().classes("w-full drocat-card").props('id="card-findsimilar-morphology-neurons"'):
                section_header("Query", "search")
                query_input = neuron_list_input(
                    label="Query Neuron(s)",
                    placeholder="Type or upload CSV/TSV/Excel (e.g., aMe12, 1005174948)",
                    hint="Neuron types, bodyIds, or patterns. Multiple queries "
                         "are searched independently and saved as separate runs.",
                    suggestions=_morph_suggest,
                    available_neurons=lambda: dataset.value
                    if dataset is not None else "",
                ).classes("drocat-fixed-neuron-input")

            with ui.card().classes("w-full drocat-card"):
                section_header("Similarity Parameters", "tune")
                with param_grid(3):
                    level = select_input(
                        "Level", MORPH_LEVEL_OPTIONS, get_user_default("morph_level"),
                        hint="'auto' (recommended): a type query returns "
                             "type-to-type results, a bodyId query returns "
                             "bodyId-to-bodyId results. 'bodyid': rank "
                             "individual neurons. 'type': aggregate candidates "
                             "by neuron type.",
                    )
                    # Legacy saved default ("vector") maps to the current method.
                    saved_method = get_user_default("morph_method")
                    if saved_method not in MORPH_METHODS:
                        saved_method = "vector_v2"
                    method = select_input(
                        "Method", MORPH_METHODS, saved_method,
                        hint="'Vector (spatial)' (default): shape + brain-position/"
                             "expansion blocks with ZCA whitening — finer "
                             "discrimination. 'NBLAST': canonical NBLAST "
                             "(slower; runs on vector-prefiltered candidates).",
                    )
                    # The legacy cosine/Pearson metric selector was removed:
                    # vector_v2 scoring is per-block whitened cosine, and
                    # NBLAST carries its own normalized score.
                with param_grid(2):
                    candidate_source = select_input(
                        "Candidate Source", CANDIDATE_SOURCE_OPTIONS,
                        get_user_default("candidate_source"),
                        hint="'auto' (recommended): NeuPrint datasets screen "
                             "candidates by primary-ROI distribution "
                             "similarity (every neuron reachable; a one-time "
                             "matrix is cached), FlyWire searches the vector "
                             "cache directly. 'roi': ROI-distribution screen "
                             "only. 'combined': union of the ROI screen and "
                             "the connectivity (shared-partner) screen — "
                             "widest pool, most skeleton fetches. 'profile': "
                             "connectivity screen only (misses neurons "
                             "without shared partners). 'cache': full-"
                             "morphology over the local skeleton population "
                             "(download skeletons first).",
                    )
                    candidate_cap = number_input(
                        "Candidate Cap", get_user_default("candidate_cap"), 10, 5000,
                        hint="Maximum number of candidates entering the "
                             "morphological comparison: the sorted candidate "
                             "list is truncated to this many neurons (all "
                             "source modes; also the NBLAST prefilter in "
                             "cache mode). ALL compared candidates are "
                             "returned and written; the visualize Top N "
                             "controls rendering only.",
                    )
                roi_filter = select_input(
                    "ROI Filter", ["All ROIs"], "All ROIs",
                    hint="Restrict candidate discovery to synapse rows in "
                         "the selected ROIs (only shown when the dataset's "
                         "connection cache carries ROI data). 'All ROIs' = "
                         "no restriction.",
                )
                with ui.row().classes("w-full items-center gap-4"):
                    visualize = checkbox_input(
                        "Visualize Top Results",
                        DEFAULTS["morph_visualize_top_n"] > 0,
                        hint="Render optional 3D skeletons for the highest-ranked results.",
                    )
                    visualization_settings = skeleton_visualization_settings(
                        default_top_n=DEFAULTS["morph_visualize_top_n"],
                        top_n_label="Visualize Top N Types / Neurons",
                        top_n_hint=(
                            "Number of top results to render. The grouping choice "
                            "controls whether types or individual bodyIds are shown."
                        ),
                        default_visualize_by=DEFAULTS["morph_visualize_by"],
                        show_high_quality_warning=True,
                        dataset_provider=lambda: dataset.value,
                        dataset_watchers=[dataset],
                    )

            with ui.card().classes("w-full drocat-card"):
                section_header("Skeleton Vector Cache", "memory")
                coverage_label = ui.label("Coverage: checking...").classes(
                    "text-caption drocat-muted"
                )
                with ui.row().classes("items-center gap-2"):
                    build_button = ui.button(
                        "Build Vector Cache", icon="auto_awesome"
                    ).props("color=secondary outline").on_click(
                        lambda: build_cache()
                    )
                    ui.label(
                        "One-time build from cached skeletons (auto-triggered on "
                        "first query); incremental afterwards."
                    ).classes("text-caption drocat-muted")

                ui.label(
                    "Raw skeletons fetched by Find Similar, visualization, and "
                    "dataset pulls are always stored as reusable .swc.zst files "
                    "under cache/<dataset>/skeletons/raw_skeletons/ (legacy "
                    ".swc.gz remains readable). Use "
                    "Settings → Dataset Cache → Download All Skeletons "
                    "to prefetch "
                    "the shared population."
                ).classes("text-caption drocat-muted")

            def refresh_roi_options():
                # ROI data availability differs per dataset (male-cns has 114
                # ROIs; hemibrain's connection cache has none).
                try:
                    from pathlib import Path
                    import polars as pl
                    conn_path = (Path(PROJECT_ROOT) / "cache"
                                 / dataset.value.replace(":", "_").replace(".", "_")
                                 / "connections.parquet")
                    rois = ["All ROIs"]
                    if conn_path.exists():
                        conn = pl.read_parquet(conn_path)
                        if "roi" in conn.columns:
                            vals = (conn["roi"].drop_nulls()
                                    .filter(pl.col("roi") != "")
                                    .unique().sort().to_list())
                            if vals:
                                rois = ["All ROIs"] + [str(v) for v in vals]
                    roi_filter.options = rois
                    if roi_filter.value not in rois:
                        roi_filter.value = "All ROIs"
                    roi_filter.set_visibility(len(rois) > 1)
                except Exception:
                    roi_filter.set_visibility(False)

            def refresh_coverage():
                # Lightweight: no navis/statvis import at page build (they are
                # heavy and would slow the first page response). The vector
                # cache is a plain parquet file, countable with polars.
                try:
                    from pathlib import Path
                    dataset_folder = (Path(PROJECT_ROOT) / "cache"
                                      / dataset.value.replace(":", "_").replace(".", "_"))
                    vector_folder = dataset_folder / "find_similar"
                    # recursive: raw-cache downloads may be grouped in nested
                    # folders by a dataset-specific source.
                    raw_dir = dataset_folder / "skeletons" / "raw_skeletons"
                    raw_files = (list(raw_dir.rglob("*.pkl"))
                                 + list(raw_dir.rglob("*.swc.gz"))
                                 + list(raw_dir.rglob("*.swc.zst")))
                    legacy_raw_dir = vector_folder / "raw_skeletons"
                    raw_files += (list(legacy_raw_dir.rglob("*.pkl"))
                                  + list(legacy_raw_dir.rglob("*.swc.gz"))
                                  + list(legacy_raw_dir.rglob("*.swc.zst")))
                    n_skel = len({
                        p.name.removesuffix(".swc.zst").removesuffix(".swc.gz")
                        .removesuffix(".pkl")
                        for p in raw_files
                    })
                    # FAFB v783: the healed bundle is the real skeleton
                    # source (.zst first; ZIP fallback with lazy conversion;
                    # the pickle cache holds meshes).
                    dataset_folder_name = dataset.value.replace(":", "_").replace(".", "_")
                    dataset_dir = Path(PROJECT_ROOT) / "datasets" / dataset_folder_name
                    bundle_path = dataset_dir / "sk_lod1_783_healed.zst"
                    zip_path = dataset_dir / "sk_lod1_783_healed.zip"
                    if bundle_path.exists() or zip_path.exists():
                        try:
                            import sys as _sys
                            if str(SRC_DIR) not in _sys.path:
                                _sys.path.insert(0, str(SRC_DIR))
                            from fafb_bundle import FAFBSkeletonBundle
                            reader = FAFBSkeletonBundle(
                                bundle_path, zip_path=zip_path if zip_path.exists() else None,
                                lazy_convert=False)
                            try:
                                n_skel = reader.count()
                            finally:
                                reader.close()
                        except Exception:
                            pass
                    n_vec = 0
                    vec_file = (vector_folder / "morphology"
                                / "skeleton__vectors_v2.parquet")
                    if vec_file.exists():
                        import polars as pl
                        n_vec = pl.read_parquet(vec_file).height
                    coverage_label.text = (
                        f"Dataset skeletons: {n_skel}  ·  vectorized: {n_vec}"
                    )
                except Exception:
                    coverage_label.text = "Coverage unavailable."

            async def build_cache():
                if is_banc_dataset(dataset.value):
                    morph_dataset_warning.set_visibility(True)
                    ui.notify(
                        "BANC morphological similarity is unavailable; select a non-BANC dataset.",
                        type="warning",
                    )
                    return
                build_button.disable()
                ui.notify(
                    "Building skeleton vector cache (this can take a few minutes)...",
                    type="info",
                )
                try:
                    import asyncio
                    import sys
                    sys.path.insert(0, str(SRC_DIR))
                    from morphology import find_similar_dataset_cache_v2

                    def _run():
                        cache = find_similar_dataset_cache_v2(
                            dataset.value, n_workers=8, verbose=False
                        )
                        return cache.build(fetch_missing=0)

                    stats = await asyncio.to_thread(_run)
                    ui.notify(
                        f"Vector cache ready: {stats['rows']} rows "
                        f"({stats['new']} new)"
                    )
                    refresh_coverage()
                except Exception as ex:
                    ui.notify(f"Cache build failed: {ex}", type="negative")
                finally:
                    build_button.enable()

        # ============ Connectivity similarity panel ============
        with ui.column().classes("w-full gap-1") as profile_panel:
            with ui.card().classes("w-full drocat-card").props('id="card-findsimilar-profile-dataset"'):
                section_header("Dataset", "storage")
                source_dataset = dataset_selector(
                    label="Dataset",
                    hint="Dataset used for both the query and candidate search. "
                         "Connectivity similarity is intra-dataset only.",
                )
                profile_output_dir = dir_input(scope="find_similar_profiling")

            with ui.card().classes("w-full drocat-card").props('id="card-findsimilar-profile-neurons"'):
                section_header("Query", "search")
                source_input = neuron_list_input(
                    label="Query Neuron (type or bodyId)",
                    show_filter=False,
                    show_upload=True,
                    hint="Enter one or more neuron types or bodyIds. Each input is searched independently.",
                    suggestions=_profile_suggest,
                    available_neurons=lambda: source_dataset.value
                    if source_dataset is not None else "",
                ).classes("drocat-fixed-neuron-input")

            with ui.card().classes("w-full drocat-card"):
                section_header("Search Parameters", "tune")
                with param_grid(3):
                    profile_top_n = number_input(
                        "Top N Candidates", get_user_default("top_n"), 5, 100,
                        hint="Number of top candidates to return.",
                    )
                    min_shared_partners = number_input(
                        "Min Shared Partners", 2, 1, 10,
                        hint="Minimum shared partners for a candidate (adjacency "
                             "expansion). Lower = looser discovery (1 = any shared "
                             "partner makes a candidate).",
                    )
                    candidate_prune = number_input(
                        "Candidate Prune %", 5, 5, 100,
                        hint="Keep the top N% of cosine-positive candidates after "
                             "vector pre-filtering. 100 = keep all (loosest search).",
                    )
                with param_grid(2):
                    similarity_metric = select_input(
                        "Similarity Metric", SIMILARITY_METRICS,
                        get_user_default("similarity_metric"),
                        hint="Metric for comparing connectivity profiles.",
                    )
                    top_k = number_input(
                        "Top K Partners", get_user_default("top_k"), 5, 50,
                        hint="Top K partners per direction for profile construction.",
                    )
                with ui.row().classes("gap-4"):
                    use_fast = checkbox_input(
                        "Fast Search", get_user_default("fast_search"),
                        hint="Adjacency-expansion discovery (recommended).",
                    )
                    vector_prefilter = checkbox_input(
                        "Vector Pre-filtering", get_user_default("vector_prefilter"),
                        hint="Cosine pre-filter of candidates for speed.",
                    )
                    expand_2hop = checkbox_input(
                        "2-Hop Expansion", get_user_default("expand_2hop"),
                        hint="Include untyped partners via 2-hop typed partners.",
                    )
                    use_cache = checkbox_input(
                        "Use Cache", get_user_default("use_cache"),
                        hint="Cache profiles and connections locally.",
                    )
                saveas = ui.input(
                    label="Save Folder Name (optional)",
                    placeholder="e.g., aMe12_similar",
                ).classes("w-full drocat-input").tooltip(
                    "Custom output folder name. Leave empty for the auto name."
                )
                full_cache = checkbox_input(
                    "Pre-build Full Dataset Cache", False,
                    hint="Fetch connections for EVERY uncached neuron before "
                         "searching. Very slow on first use (can take hours).",
                )
                with ui.row().classes("w-full items-center gap-4"):
                    profile_visualize = checkbox_input(
                        "Visualize Top Candidates",
                        True,
                        hint="Generate a separate 3D visualization list for the "
                             "highest-ranked connectivity-similar candidates.",
                    )
                    profile_visualization_settings = skeleton_visualization_settings(
                        default_top_n=5,
                        top_n_label="Visualize Top N Candidates",
                        top_n_hint=(
                            "Number of top connectivity-similar candidates to "
                            "render. This list is independent of morphological "
                            "similarity results."
                        ),
                        default_visualize_by="type",
                        show_high_quality_warning=True,
                        dataset_provider=lambda: [source_dataset.value],
                        dataset_watchers=[source_dataset],
                    )

        def sync_mode():
            is_morph = mode_value["value"] == "Morphological similarity"
            morph_panel.set_visibility(is_morph)
            profile_panel.set_visibility(not is_morph)
            morph_mode_button.props(
                "color=primary" if is_morph else "color=grey-7"
            )
            connectivity_mode_button.props(
                "color=grey-7" if is_morph else "color=primary"
            )

        def set_mode(value: str):
            mode_value["value"] = value
            sync_mode()

        def on_dataset_change(_e=None):
            morph_dataset_warning.set_visibility(is_banc_dataset(dataset.value))
            morph_best_dataset_warning.set_visibility(
                str(dataset.value or "").strip().lower() != "male-cns:v1.0"
            )
            refresh_coverage()
            refresh_roi_options()

        morph_mode_button.on_click(
            lambda _event: set_mode("Morphological similarity")
        )
        connectivity_mode_button.on_click(
            lambda _event: set_mode("Connectivity similarity")
        )
        dataset.on_value_change(on_dataset_change)
        sync_mode()
        on_dataset_change()

    with results_col:
        output_panel.create(run_label="Run Similarity Search", run_icon="play_arrow")

    def _unique_queries(values):
        """Return the entered queries in order, without duplicate chips."""
        queries = []
        seen = set()
        for value in values or []:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            queries.append(value)
        return queries

    def _saveas_for_query(base_value, index, query, total):
        """Keep custom-named multi-query runs separate on disk."""
        base = str(base_value or "").strip()
        if not base or total == 1:
            return base
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(query)).strip("_")
        safe = safe[:60] or f"query_{index + 1}"
        return f"{base}_{index + 1}_{safe}"

    def _collect_files(results):
        """Merge per-query output files while preserving their paths."""
        files = {}
        for result in results:
            for file_info in result.get("files", []):
                path = file_info.get("path")
                if path:
                    files[path] = file_info
        return list(files.values())

    async def run_morphological():
        if is_banc_dataset(dataset.value):
            morph_dataset_warning.set_visibility(True)
            ui.notify(
                "BANC morphological similarity is unavailable; select a non-BANC dataset.",
                type="warning",
            )
            return
        mode, neurons = query_input.get_value()
        raw_queries = _unique_queries(neurons)
        queries = _unique_queries(apply_filter_mode(neurons, mode))
        if not queries:
            ui.notify("Please enter at least one query neuron", type="warning")
            return
        if len(queries) > 50:
            ui.notify("Please limit the query to 50 neurons", type="warning")
            return

        output_panel.clear()
        output_panel.set_running(True)

        visualization_values = visualization_settings.values()
        if visualize.value:
            visualization_settings.warn_empty_custom_palettes()
        base_params = {
            "dataset": dataset.value,
            "level": level.value,
            "method": method.value,
            "candidate_cap": int(candidate_cap.value),
            "candidate_source": candidate_source.value,
            "roi_filter": None if roi_filter.value == "All ROIs" else [roi_filter.value],
            "visualize_top_n": (
                visualization_values["visualize_top_n"] if visualize.value else 0
            ),
            "visualize_by": visualization_values["visualize_by"],
            "visualization_settings": visualization_values,
            "output_dir": morph_output_dir.value,
            "saveas": "",
            "verbose": True,
            "n_workers": 8,
            "use_cache": get_user_default("use_cache"),
            # Raw skeleton persistence is now unconditional and shared with
            # visualization and Settings cache pulls.
            "cache_fetched_skeletons": True,
        }
        results = []
        last_output_folder = None
        try:
            for index, query in enumerate(queries):
                if len(queries) > 1:
                    output_panel.log(
                        f"--- Morphology query {index + 1}/{len(queries)}: {query} ---",
                        "system",
                    )
                constructor_params = dict(base_params)
                constructor_params["query"] = query
                result = await output_panel.run(
                    runner, "find_similar_morphology", constructor_params,
                    "find_similar", output_dir=morph_output_dir.value,
                )
                results.append(result)
                # A completed per-query run means the query resolved in the
                # dataset; keep the raw chip (pre-pattern) in the history.
                if result.get("returncode") == 0:
                    from ..history_store import record as _record_history
                    raw = raw_queries[index] if index < len(raw_queries) else query
                    _record_history(
                        [str(raw)],
                        datasets=[dataset.value] if dataset.value else [],
                    )
                last_output_folder = result.get("output_folder") or last_output_folder
                if result.get("cancelled"):
                    break

            cancelled = any(result.get("cancelled") for result in results)
            succeeded = bool(results) and all(
                result.get("returncode") == 0 for result in results
            )
            if cancelled:
                output_panel.set_status("Cancelled", "red")
            else:
                output_panel.set_status(
                    "Completed" if succeeded else "Failed",
                    "green" if succeeded else "red",
                )
            files = _collect_files(results)
            if files:
                output_panel.show_files(
                    files,
                    morph_output_dir.value if len(queries) > 1
                    else last_output_folder or morph_output_dir.value,
                )
        finally:
            output_panel.set_running(False)

    async def run_profile():
        source_vals = source_input.get_value()[1]
        sources = _unique_queries(source_vals)
        if not sources:
            ui.notify("Please enter at least one query neuron", type="warning")
            return

        output_panel.clear()
        output_panel.set_running(True)

        visualization_values = profile_visualization_settings.values()
        if profile_visualize.value:
            profile_visualization_settings.warn_empty_custom_palettes()
        base_params = {
            "source_dataset": source_dataset.value,
            "target_dataset": source_dataset.value,
            "output_dir": profile_output_dir.value,
            "top_n": int(profile_top_n.value),
            "top_k": int(top_k.value),
            "min_shared_partners": int(min_shared_partners.value),
            "vector_prune_fraction": float(candidate_prune.value) / 100.0,
            "similarity_metric": similarity_metric.value,
            "vector_prefiltering": vector_prefilter.value,
            "include_untyped_partners": expand_2hop.value,
            "use_cache": use_cache.value,
            "saveas": saveas.value.strip() or "",
            "min_synapse_threshold": get_user_default("min_synapse_num"),
            "ensure_cache_complete": full_cache.value,
            "morphological_enrichment": True,
            "output_folder_prefix": "similar-connectivity",
            "visualize_skeleton": profile_visualize.value,
            "visualize_top_n": (
                visualization_values["visualize_top_n"]
                if profile_visualize.value else 0
            ),
            "visualization_settings": visualization_values,
            "verbose": True,
        }
        method_name = "find_homologs_fast" if use_fast.value else "find_novel_homologs"
        results = []
        last_output_folder = None
        try:
            for index, source in enumerate(sources):
                if len(sources) > 1:
                    output_panel.log(
                        f"--- Connectivity query {index + 1}/{len(sources)}: {source} ---",
                        "system",
                    )
                constructor_params = dict(base_params)
                constructor_params.update({
                    "source": source,
                    "saveas": _saveas_for_query(saveas.value, index, source, len(sources)),
                })
                result = await output_panel.run(
                    runner, "find_similar_profile", constructor_params,
                    method_name, output_dir=profile_output_dir.value,
                )
                results.append(result)
                # A completed per-query run means the source resolved in the
                # dataset; keep the raw chip in the query history.
                if result.get("returncode") == 0:
                    from ..history_store import record as _record_history
                    _record_history(
                        [str(source)],
                        datasets=[source_dataset.value]
                        if source_dataset.value else [],
                    )
                last_output_folder = result.get("output_folder") or last_output_folder
                if result.get("cancelled"):
                    break

            cancelled = any(result.get("cancelled") for result in results)
            succeeded = bool(results) and all(
                result.get("returncode") == 0 for result in results
            )
            if cancelled:
                output_panel.set_status("Cancelled", "red")
            else:
                output_panel.set_status(
                    "Completed" if succeeded else "Failed",
                    "green" if succeeded else "red",
                )
            files = _collect_files(results)
            if files:
                output_panel.show_files(
                    files,
                    profile_output_dir.value if len(sources) > 1
                    else last_output_folder or profile_output_dir.value,
                )
        finally:
            output_panel.set_running(False)

    async def run_similar():
        if mode_value["value"] == "Morphological similarity":
            await run_morphological()
        else:
            await run_profile()

    output_panel.run_button.on_click(run_similar)
    output_panel.cancel_button.on_click(runner.cancel)
