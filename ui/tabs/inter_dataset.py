"""Cross-Dataset Comparison Tab - runs ComparisonAnalyzer over N datasets."""

from nicegui import ui
from ..config import COMPARISON_MODES, PATH_MODES, PATHFINDING_ALGORITHMS, SEARCH_COLUMNS, get_user_default
from ..components.common import (
    dataset_multi_selector, neuron_list_input, number_input, select_input,
    checkbox_input, dir_input, section_header, param_grid, tool_page,
    apply_filter_mode,
)
from ..components.mapping_editor import custom_grouping_block
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner
from ..type_suggestions import dataset_aware_suggestions


def create_inter_dataset_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("Comparison Output")
    datasets_select = None
    search_columns = None

    def _type_suggest(text):
        """Auto-suggest across all selected datasets' type names, with a
        dataset-aware gray hint (``type · male-cns:v1.0``) so cross-dataset
        entries stay traceable. Type matches come first for string input;
        the range expands to instance/bodyId only when no type matched and
        the search scope is 'auto'."""
        ds = datasets_select.value if datasets_select is not None else []
        scope = search_columns.value if search_columns is not None else "auto"
        # Keep the complete candidate pool for local continuation filtering;
        # the input menu, not the backend matcher, limits visible rows.
        return dataset_aware_suggestions(text, ds, scope, limit=None)

    form_col, results_col = tool_page(
        "Cross-Dataset Comparison",
        "Analyze one dataset across thresholds or compare connectivity across datasets.",
        icon="sync_alt",
        doc="cross_dataset.md",
    )

    with form_col:
        with ui.card().classes("w-full drocat-card").props('id="card-interdataset-datasets"'):
            section_header("Datasets", "storage")
            datasets_select = dataset_multi_selector(
                label="Datasets to compare (one dataset with multiple thresholds is also supported)",
            )
            # Round 2 entrance: the standalone type-mapping preview. The
            # button stays disabled until >= 2 selected datasets have
            # cached neuron indexes; the popup composes the mapping across
            # every ordered pair of the selection (informational only).
            from ..components.type_mapping_panel import create_type_mapping_entry

            with ui.row().classes("items-center gap-4 flex-wrap"):
                type_mapping_button = create_type_mapping_entry(
                    lambda: list(datasets_select.value or []))

            def _sync_type_mapping_state(_e=None):
                type_mapping_button.refresh_state()

            datasets_select.on_value_change(_sync_type_mapping_state)
            _sync_type_mapping_state()
            output_dir = dir_input(scope="inter_dataset")

        with ui.card().classes("w-full drocat-card").props('id="card-interdataset-neurons"'):
            section_header("Neuron Selection", "hub")
            source_input = neuron_list_input(
                label="Source Neurons",
                placeholder="Type or upload CSV/TSV/Excel with neuron types/bodyIds",
                hint="Source neurons for pathfinding. Type one query per chip or upload a CSV/TSV/Excel file (first column).",
                suggestions=_type_suggest,
                available_neurons=lambda: datasets_select.value if datasets_select is not None else [],
                # History rows tag the selected datasets they were recorded
                # for, so cross-dataset entries stay traceable.
                show_history_datasets=True,
            ).classes("drocat-fixed-neuron-input")
            target_input = neuron_list_input(
                label="Target Neurons",
                placeholder="Type or upload CSV/TSV/Excel with neuron types/bodyIds",
                hint="Target neurons for pathfinding. Type one query per chip or upload a CSV/TSV/Excel file (first column).",
                suggestions=_type_suggest,
                available_neurons=lambda: datasets_select.value if datasets_select is not None else [],
                show_history_datasets=True,
            ).classes("drocat-fixed-neuron-input")
            mapping_select, _grouper_card, resolve_grouping = custom_grouping_block(
                label="Custom Mapping",
                datasets_provider=lambda: list(datasets_select.value or []),
                require_names=True,
                tab_key="inter_dataset",
                watch_elements=[datasets_select],
                query_inputs={"source": source_input, "target": target_input},
            )

        with ui.card().classes("w-full drocat-card").props('id="card-interdataset-core"'):
            section_header("Core Parameters", "tune")
            with param_grid(3):
                comparison_mode = select_input(
                    "Mode", COMPARISON_MODES, "path",
                    hint="'path': discover edges via path traversal. 'edge': compare edges independently by weight.",
                )
                path_mode = select_input(
                    "Path Enumeration", PATH_MODES, "all",
                    hint="'all': every path within the layer limit (FindAllPath). "
                         "'shortest': only per-pair minimum-hop paths (FindShortestPath) — "
                         "Max Layers is an EXACT depth bound (8 by default; high values "
                         "like 99 give an effectively unlimited search).",
                )
                max_interlayer = number_input(
                    "Max Intermediate Layers", 2, 0, 100,
                    hint="Maximum hops between source and target. In shortest mode this "
                         "is an EXACT depth bound: 0 = direct connections only, 8 = default, "
                         "and a high unreachable number (e.g. 99) gives an effectively "
                         "unlimited search.",
                )
            thresholds_input = neuron_list_input(
                label="Synapse Thresholds",
                initial=[3, 5, 10],
                unit_label="threshold",
                show_filter=False,
                show_upload=False,
                hint="List of min synapse thresholds to analyze. "
                     "Type one threshold per chip (e.g. 3, 5, 10), or keep the defaults.",
            ).classes("w-full drocat-full-row-control")
            find_reciprocal = checkbox_input(
                "Find Reciprocal Connections", False,
                hint="Build reciprocal graphs and include them in reports.",
            )

            # --- Advanced Settings (collapsed) ---
            with ui.expansion("Advanced Settings", icon="settings_suggest").classes("w-full"):
                edge_limit_bodyid_hint = None
                with param_grid(2):
                    pathfinding = select_input(
                        "Pathfinding Algorithm", PATHFINDING_ALGORITHMS, get_user_default("pathfinding"),
                        hint="MemoizedDFS: recommended default (fastest measured at all depths, no graph copy). DFS: backward memoized, best with few targets. MeetInMiddle: shallow queries. DP: robust. Bidirectional: shortest-first but high memory.",
                        help_doc="pathfinding_algorithms.html",
                    )
                    top_edges = number_input(
                        "Top Edges in Analysis Reports", 500, 10, 5000,
                        hint="Limits top-edge comparison/overlap results and edge/path "
                             "presence-matrix rows and the path-presence data used by "
                             "comparison summary plots. It does not trim the pathfinding "
                             "graph or set the per-visualization drawn-edge cap: use "
                             "Edge Limit – BodyIds for graph trimming and Visualization "
                             "Edge Limit for plotted edges.",
                    )
                search_columns = select_input(
                    "Search Columns", SEARCH_COLUMNS, get_user_default("search_columns"),
                    hint="Which columns to search when resolving neuron names in every dataset. "
                         "'auto': all columns (bodyId -> type -> instance -> flywireType/others). "
                         "Use 'type'/'instance'/'bodyId' to restrict the search.",
                )
                with ui.row().classes("gap-4"):
                    skip_bodyid = checkbox_input("Skip BodyId Level", get_user_default("skip_bodyId"), hint="Skip bodyId-level results for speed.")
                    cache_only = checkbox_input("Cache Only (Offline)", get_user_default("cache_only"), hint="Use only local cache, no server connection.")
                    auto_type_mapping = checkbox_input(
                        "Auto Type Mapping", get_user_default("auto_type_mapping"),
                        hint="Auto-map type names across datasets via the male-cns v1.0 "
                             "neuron info (e.g. FAFB MTe07 <-> male-cns MeVPLo2) when no "
                             "custom LabelMapper preset is selected.",
                    )
                with param_grid(3):
                    min_ratio = number_input(
                        "Min Connection Ratio", get_user_default("min_ratio"), 0, 1, 0.01,
                        hint="Minimum weight/post ratio for an edge to be kept.",
                    )
                    min_prob = number_input(
                        "Min Traversal Prob.", get_user_default("min_traversal_probability"), 0, 1, 0.01,
                        hint="Minimum traversal probability for an edge to be kept.",
                    )
                    output_format = select_input(
                        "Output Format", ["csv", "xlsx"], get_user_default("output_format"),
                        hint="Format for exported data tables.",
                    )
                with param_grid(2):
                    parallel = checkbox_input(
                        "Parallel Processing", True,
                        hint="Run per-dataset work in parallel where possible.",
                    )
                    max_workers = number_input(
                        "Max Workers", 4, 1, 16,
                        hint="Number of parallel workers (only used when Parallel Processing is on).",
                    )
                with param_grid(2):
                    edge_limit_bodyid = number_input(
                        "Edge Limit – BodyIds", 1000000, 0, 1000000000,
                        hint="Top-N strongest non-reserved edges kept in the bodyId-level "
                             "graph of the FindAllPath runs (source/target edges are always "
                             "kept in addition). Applied only when Layers ≥ 3 (deep searches); "
                             "shallow runs keep the complete graph. 0 = unlimited.",
                    )
                    edge_limit_viz = number_input(
                        "Visualization Edge Limit", get_user_default("edgeN_limit"), 10, 5000,
                        hint="Maximum edges drawn per visualization (network / Sankey / "
                             "heatmap) in the FindAllPath runs. Limits memory usage for "
                             "highly connected neurons. Same default as the Complete Paths tab.",
                    )
                    # the bodyId edge limit only applies to deep searches
                    # ('all' mode); in shortest mode it is an explicit opt-in
                    # (default off - trimming can inflate shortest distances)
                    def _sync_bodyid_edge_limit():
                        enabled = (
                            path_mode.value == 'shortest'
                            or (max_interlayer.value or 0) >= 3
                        )
                        edge_limit_bodyid.set_enabled(enabled)
                        if edge_limit_bodyid_hint is not None:
                            edge_limit_bodyid_hint.set_visibility(not enabled)

                    _sync_bodyid_edge_limit()
                    max_interlayer.on_value_change(lambda _e: _sync_bodyid_edge_limit())

                edge_limit_bodyid_hint = ui.label(
                    "Unavailable for shallow searches (Max Intermediate Layers 0–2); "
                    "set Max Intermediate Layers to 3+ to enable BodyId edge trimming."
                ).classes("text-caption drocat-muted").set_visibility(
                    path_mode.value != 'shortest' and (max_interlayer.value or 0) < 3
                )

                def _apply_path_mode_defaults(notify=False):
                    """A mode switch resets the mode-specific defaults:
                    shortest -> Max Layers 8, Edge Limit – BodyIds 0 (off);
                    all -> Max Layers 2, Edge Limit – BodyIds 1M (deep
                    searches). The user is warned their values were reset."""
                    if path_mode.value == 'shortest':
                        pathfinding.disable()
                        edge_limit_bodyid.set_enabled(True)
                        edge_limit_bodyid.value = 0
                        max_interlayer.value = 8
                    else:
                        pathfinding.enable()
                        edge_limit_bodyid.value = 1000000
                        max_interlayer.value = 2
                    _sync_bodyid_edge_limit()
                    if notify:
                        ui.notify(
                            f"Path Enumeration switched to '{path_mode.value}': "
                            "Max Layers and Edge Limit – BodyIds were reset to the "
                            "mode defaults — re-enter custom values if needed.",
                            type="warning",
                        )
                path_mode.on_value_change(lambda _e: _apply_path_mode_defaults(notify=True))
                _apply_path_mode_defaults()

        with ui.card().classes("w-full drocat-card").props('id="card-interdataset-hemisphere"'):
            section_header("Hemisphere Analysis", "sync_alt")
            with ui.row().classes("items-center gap-4 flex-wrap"):
                separate_hemi = checkbox_input(
                    "Hemisphere-aware", False,
                    hint="Split type/group aggregation into _L/_R/_U hemisphere labels.",
                ).props('id=checkbox-separate-hemi')
            with ui.row().classes("items-center gap-4 flex-wrap"):
                symmetry_analysis = checkbox_input(
                    "Symmetry Analysis", True,
                    hint="Generate per-dataset hemisphere symmetry summaries (auto-enabled with Hemisphere-aware).",
                ).props('id=checkbox-symmetry')
            with ui.row().classes("items-center gap-4 flex-wrap"):
                keep_hemi_conserved = checkbox_input(
                    "Keep Only Hemisphere-Conserved Edges", False,
                    hint="Keep only edges conserved between hemispheres (requires Hemisphere-aware).",
                ).props('id=checkbox-hemi-conserved')
            def _sync_hemisphere_options():
                if separate_hemi.value:
                    keep_hemi_conserved.enable()
                    symmetry_analysis.enable()
                    # auto-enabled with Hemisphere-aware (per the hint)
                    symmetry_analysis.value = True
                else:
                    # uncheck + disable the hemisphere-dependent options so a
                    # greyed-out True is never passed to the backend
                    keep_hemi_conserved.disable()
                    keep_hemi_conserved.value = False
                    symmetry_analysis.disable()
                    symmetry_analysis.value = False
            separate_hemi.on_value_change(lambda _e: _sync_hemisphere_options())
            _sync_hemisphere_options()

    with results_col:
        output_panel.create(run_label="Run Comparison", run_icon="play_arrow")

    async def run_comparison():
        src_mode, src_neurons = source_input.get_value()
        tgt_mode, tgt_neurons = target_input.get_value()
        sources = apply_filter_mode(src_neurons, src_mode)
        targets = apply_filter_mode(tgt_neurons, tgt_mode)

        if not sources:
            ui.notify("Please provide at least one source neuron", type="warning")
            return

        datasets = datasets_select.value or []
        if not datasets:
            ui.notify("Please add at least 1 dataset to analyze", type="warning")
            return

        # Parse thresholds (chip values are already normalized to integers;
        # split comma-joined chips defensively in case a list was typed into
        # one chip before the run).
        try:
            thresholds = [
                int(v)
                for item in thresholds_input.get_value()[1]
                for v in str(item).replace(' ', '').split(',')
                if v
            ]
        except (TypeError, ValueError):
            ui.notify("Invalid thresholds format. Use comma-separated integers.", type="negative")
            return
        if not thresholds:
            ui.notify("Please enter at least one synapse threshold", type="warning")
            return

        # Resolve custom grouping (preset or inline); inline group labels are
        # compulsory for cross-dataset comparisons and validated here.
        mapping_path, mapping_ok = resolve_grouping()
        if not mapping_ok:
            return

        output_panel.clear()
        output_panel.set_running(True)

        constructor_params = {
            "datasets": datasets,
            "source_neurons": sources,
            "target_neurons": targets,
            "output_folder": output_dir.value,
            "comparison_mode": comparison_mode.value,
            "path_mode": path_mode.value,
            "max_interlayer": int(max_interlayer.value),
            "thresholds": thresholds,
            "top_edges": int(top_edges.value),
            "graph_edge_limit_bodyid": int(edge_limit_bodyid.value),
            "edgeN_limit": int(edge_limit_viz.value),
            "pathfinding": pathfinding.value,
            "search_columns": search_columns.value,
            "skip_bodyId": skip_bodyid.value,
            "cache_only": cache_only.value,
            "auto_type_mapping": auto_type_mapping.value,
            "_min_ratio": float(min_ratio.value),
            "_min_prob": float(min_prob.value),
            "_output_format": output_format.value,
            "parallel": parallel.value,
            "max_workers": int(max_workers.value) if parallel.value else None,
            "separate_hemispheres": separate_hemi.value,
            "keep_only_hemisphere_conserved_connections": keep_hemi_conserved.value,
            "symmetry_analysis": symmetry_analysis.value,
            "find_reciprocal": find_reciprocal.value,
        }
        if mapping_path:
            constructor_params["overall_mapping_json"] = mapping_path

        result = await output_panel.run(runner, "inter_dataset", constructor_params, "run",
                                        output_dir=output_dir.value)

        # Each dataset-level path analysis initializes its neuron sets before
        # comparing them. Record only when at least one source/target pair was
        # resolved to real neurons in that process.
        match_info = result.get("neuron_match") or {}
        if match_info.get("any_pair"):
            from ..history_store import record as _record_history
            _record_history(
                [str(v) for v in sources + targets],
                datasets=list(datasets_select.value or []),
            )

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], result.get("output_folder") or output_dir.value)

    output_panel.run_button.on_click(run_comparison)
    output_panel.cancel_button.on_click(runner.cancel)
