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
            # Feature B: static rough-range hint (whole-dataset
            # connection-pair density per neuron). Informational only —
            # query-specific alignment is exported per run in the
            # threshold_alignment files.
            ui.label(
                "Rough whole-dataset hint (connection-pair density per neuron): "
                "BANC ≥ 3 ≈ FAFB ≥ ~6–7 ≈ male-cns ≥ ~7–9; BANC ≥ 5 ≈ FAFB ≥ ~9–11 ≈ "
                "male-cns ≥ ~12–15 (FAFB v783, male-cns v1.0; BANC v626/v888). "
                "Query-specific alignment differs — check the threshold_alignment files "
                "in your run output."
            ).classes("text-xs opacity-60 w-full").style("line-height:1.35")
            find_reciprocal = checkbox_input(
                "Find Reciprocal Connections", False,
                hint="Build reciprocal graphs and include them in reports.",
            )

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

        # --- Advanced Settings (kept at the bottom, in its own card) ---
        with ui.card().classes("w-full drocat-card").props('id="card-interdataset-advanced"'):
            with ui.expansion(
                "Advanced Settings", icon="settings_suggest",
            ).classes("w-full drocat-section-expansion"):
                with param_grid(2):
                    # F1: StrongestFirst is the only 'all'-mode algorithm —
                    # the selector was removed; the payload sends the
                    # constant.
                    max_paths_bodyid = number_input(
                        "Max Paths (BodyId)", get_user_default("max_paths_bodyid"), 0, 100000000,
                        hint="Path budget for StrongestFirst enumeration: when the search "
                             "exceeds it, ALL paths above the achieved strength cutoff "
                             "(tau) are kept and tau is reported in the run notes. "
                             "0 = auto (StrongestFirst: 1M budget).",
                    )
                    edge_budget = number_input(
                        "Edge Budget", get_user_default("graph_edge_limit_bodyid"), 0, 100000000,
                        hint="After the lossless prunes, discovery cones exceeding this "
                             "many bodyId edges are floored just above the N-th strongest "
                             "edge's weight (w0 = w1 + 1) — exactly equivalent to raising "
                             "the threshold; the applied floor is reported as "
                             "edge_weight_floor. 0 = off. Shortest mode never floors.",
                    )
                    top_edges = number_input(
                        "Top Edges in Analysis Reports", 500, 10, 5000,
                        hint="Limits top-edge comparison/overlap results and edge/path "
                             "presence-matrix rows and the path-presence data used by "
                             "comparison summary plots. It does not trim the pathfinding "
                             "graph or set the per-visualization drawn-edge cap: see "
                             "Max Paths (BodyId) for the path budget and Visualization "
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
                # Feature F: single enumeration + per-threshold replay.
                replay_paths = checkbox_input(
                    "Replay Paths (single enumeration)", get_user_default("replay_paths"),
                    hint="Path mode 'all': enumerate ONCE at the lowest threshold and "
                         "materialize every higher threshold from the bottleneck-annotated "
                         "path set — identical outputs, no re-enumeration. Disable to "
                         "force legacy per-threshold enumeration. Shortest mode is never "
                         "replayed (min-hop sets are not nested across thresholds).",
                )
                auto_extend_thresholds = checkbox_input(
                    "Auto-extend Collapsed Thresholds", False,
                    hint="F7: when a run's effective tau collapses asked thresholds, extend "
                         "each dataset with k × τ_ref points (τ_ref = max per-dataset tau) "
                         "while ≤ 2× the max asked threshold — the schedule is global, so "
                         "the expanded points stay shared across datasets. Default off; "
                         "suggested by the banner on collapse.",
                )
                drop_untyped = checkbox_input(
                    "Drop Untyped Neurons", get_user_default("drop_untyped"),
                    hint="Remove edges touching untyped neurons (Unknown / bodyId-fallback "
                         "labels) from the cross-dataset results — they can never match "
                         "across datasets. Dropped rows are exported to "
                         "untyped_dropped_records.csv and the dropped-neuron counts are "
                         "appended to user_warning_notes.txt.",
                )

                # Feature E: per-dataset thresholds (vertical comparison).
                # One ascending threshold list per selected dataset; empty =
                # fall back to the global list.
                dataset_thresholds_toggle = checkbox_input(
                    "Per-dataset thresholds", False,
                    hint="Assign a DIFFERENT threshold list per dataset (vertical "
                         "comparison run mode). Each dataset then runs its own ascending "
                         "list; horizontal cross-dataset tables only have content at "
                         "thresholds shared by ≥ 2 datasets, and the threshold_alignment "
                         "files carry the cross-dataset comparison.",
                )
                dataset_thresholds_container = ui.column().classes("w-full gap-1")
                dataset_threshold_inputs: dict = {}

                def _parse_threshold_list(text: str):
                    values = []
                    for part in str(text or '').replace(' ', '').split(','):
                        if not part:
                            continue
                        values.append(int(part))
                    return sorted(set(values))

                def _rebuild_dataset_threshold_rows():
                    dataset_threshold_inputs.clear()
                    dataset_thresholds_container.clear()
                    selected = list(datasets_select.value or [])
                    if not dataset_thresholds_toggle.value or not selected:
                        return
                    global_values = thresholds_input.get_value()[1]
                    with dataset_thresholds_container:
                        for ds in selected:
                            with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                                ui.label(ds).classes("text-xs font-medium min-w-[180px]")
                                initial = ",".join(str(v) for v in global_values)
                                inp = ui.input(
                                    value=initial,
                                    placeholder="e.g. 3, 5, 10 (empty = global list)",
                                    on_change=None,
                                ).classes("flex-1 min-w-[220px]").props("dense outlined")
                                inp.on("blur", lambda e, i=inp: _normalize_row(i))
                                dataset_threshold_inputs[ds] = inp

                    def _normalize_row(inp):
                        try:
                            values = _parse_threshold_list(inp.value)
                            inp.value = ",".join(str(v) for v in values)
                        except (TypeError, ValueError):
                            ui.notify(
                                f"Invalid thresholds for per-dataset editor — use "
                                f"comma-separated integers.",
                                type="negative",
                            )

                def _collect_dataset_thresholds():
                    if not dataset_thresholds_toggle.value:
                        return None
                    overrides = {}
                    for ds, inp in dataset_threshold_inputs.items():
                        text = (inp.value or "").strip()
                        if not text:
                            continue  # empty = fall back to global
                        try:
                            values = _parse_threshold_list(text)
                        except (TypeError, ValueError):
                            raise ValueError(
                                f"Invalid per-dataset thresholds for {ds}: '{text}' "
                                f"(use comma-separated integers)")
                        if values:
                            overrides[ds] = values
                    return overrides or None

                dataset_thresholds_toggle.on_value_change(
                    lambda _e: _rebuild_dataset_threshold_rows())
                datasets_select.on_value_change(
                    lambda _e: _rebuild_dataset_threshold_rows())
                with param_grid(3):
                    # F9: ratio/probability filters are disabled (ratio is a
                    # readout column now). The entrances stay in the code,
                    # hidden, for the future ratio-weighted mode.
                    min_ratio = number_input(
                        "Min Connection Ratio", 0, 0, 1, 0.01,
                        hint="Disabled: connection_ratio is a readout column "
                             "(weight / all-post incoming weight) — it no longer "
                             "filters.",
                    ).set_visibility(False)
                    min_prob = number_input(
                        "Min Traversal Prob.", 0, 0, 1, 0.01,
                        hint="Disabled: traversal_probability is a readout column "
                             "(ratio/0.3, capped at 1.0) — it no longer filters.",
                    ).set_visibility(False)
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
                    # Fix C: the lossy bodyId edge limit was removed — the
                    # StrongestFirst path budget above is the single knob.
                    edge_limit_viz = number_input(
                        "Visualization Edge Limit", get_user_default("edgeN_limit"), 10, 5000,
                        hint="Maximum edges drawn per visualization (network / Sankey / "
                             "heatmap) in the FindAllPath runs. Limits memory usage for "
                             "highly connected neurons. Same default as the Complete Paths tab.",
                    )

                def _apply_path_mode_defaults(notify=False):
                    """A mode switch resets the mode-specific defaults:
                    shortest -> Max Layers 8, Edge Budget 0 (off, never
                    floors); all -> Max Layers 2, Edge Budget 1M (deep
                    searches). The user is warned their values were reset."""
                    if path_mode.value == 'shortest':
                        max_interlayer.value = 8
                        edge_budget.value = 0
                        edge_budget.disable()
                    else:
                        max_interlayer.value = 2
                        edge_budget.enable()
                        edge_budget.value = get_user_default("graph_edge_limit_bodyid") or 1000000
                    if notify:
                        ui.notify(
                            f"Path Enumeration switched to '{path_mode.value}': "
                            "Max Layers and Edge Budget were reset to the "
                            "mode defaults — re-enter custom values if needed.",
                            type="warning",
                        )
                path_mode.on_value_change(lambda _e: _apply_path_mode_defaults(notify=True))
                _apply_path_mode_defaults()

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

        # Feature E: per-dataset threshold overrides (vertical comparison)
        try:
            dataset_thresholds = _collect_dataset_thresholds()
        except ValueError as exc:
            ui.notify(str(exc), type="negative")
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
            "dataset_thresholds": dataset_thresholds,
            "replay_paths": replay_paths.value,
            "auto_extend_thresholds": auto_extend_thresholds.value,
            "drop_untyped": drop_untyped.value,
            "top_edges": int(top_edges.value),
            # Fix D: the Edge Budget (lossy floor above the N-th strongest
            # edge, w0 = w1 + 1). 0 = off; shortest mode never floors.
            "graph_edge_limit_bodyid": int(edge_budget.value),
            "max_paths_bodyid": int(max_paths_bodyid.value) or None,
            "edgeN_limit": int(edge_limit_viz.value),
            # F1: StrongestFirst is the only 'all'-mode algorithm; the
            # selector was removed.
            "pathfinding": "StrongestFirst",
            "search_columns": search_columns.value,
            "skip_bodyId": skip_bodyid.value,
            "cache_only": cache_only.value,
            "auto_type_mapping": auto_type_mapping.value,
            # F9: ratio/probability filters are disabled — hidden UI,
            # metadata-only keys, sent 0.
            "_min_ratio": 0.0,
            "_min_prob": 0.0,
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

        # F6: persistent effective-threshold banner — the analyzer writes
        # effective_thresholds.json when a τ collapse happened; surface it
        # as a durable notice above the log.
        try:
            notice_path = os.path.join(output_dir.value or "",
                                       "effective_thresholds.json")
            if os.path.isfile(notice_path):
                with open(notice_path, "r", encoding="utf-8") as nf:
                    notice = json.load(nf)
                banner = notice.get("banner")
                if banner:
                    output_panel.set_notice(banner)
                else:
                    output_panel.clear_notice()
            else:
                output_panel.clear_notice()
        except Exception:
            pass

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
