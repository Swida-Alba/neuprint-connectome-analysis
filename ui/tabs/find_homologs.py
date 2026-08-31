"""FindHomologs Tab - Cross-dataset homolog finding."""

from nicegui import ui
from ..config import SIMILARITY_METRICS, get_user_default
from ..components.common import (
    dataset_selector, neuron_list_input, number_input, select_input,
    checkbox_input, dir_input, section_header, param_grid, tool_page,
)
from ..components.output_panel import OutputPanel
from ..components.skeleton_visualization_settings import skeleton_visualization_settings
from ..runner import ScriptRunner
from ..type_suggestions import dataset_suggestions


def create_find_homologs_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("Homolog Output")
    source_dataset = None

    def _type_suggest(text):
        dataset_name = source_dataset.value if source_dataset is not None else ""
        return dataset_suggestions(text, dataset_name, limit=None)

    form_col, results_col = tool_page(
        "Homolog Finding",
        "Find potential homologs across datasets via connectivity profile similarity.",
        icon="compare",
        doc="find_homologs.md",
    )

    with form_col:
        with ui.card().classes("w-full drocat-card").props('id="card-findhomologs-datasets"'):
            section_header("Datasets", "storage")
            with param_grid(2):
                source_dataset = dataset_selector(
                    label="Source Dataset",
                    hint="Dataset where the source neuron lives.",
                )
                target_dataset = dataset_selector(
                    label="Target Dataset",
                    default=get_user_default("default_target_dataset"),
                    hint="Dataset to search for homologs in.",
                )
            output_dir = dir_input(scope="find_homologs")

        with ui.card().classes("w-full drocat-card").props('id="card-findhomologs-neurons"'):
            section_header("Source Neuron", "search")
            source_input = neuron_list_input(
                label="Source Neuron(s) (type or bodyId)",
                show_filter=False,
                show_upload=True,
                suggestions=_type_suggest,
                available_neurons=lambda: source_dataset.value
                if source_dataset is not None else "",
                hint="Enter one or more neuron types, bodyIds, or higher-category "
                     "labels (e.g. cell class). All inputs are expanded per type "
                     "and aggregated into one grouped output folder.",
            ).classes("drocat-fixed-neuron-input")

        with ui.card().classes("w-full drocat-card"):
            section_header("Search Parameters", "tune")
            with param_grid(3):
                top_n = number_input("Top N Candidates", get_user_default("top_n"), 5, 100, hint="Number of top homolog candidates to return.")
                top_k = number_input("Top K Partners", get_user_default("top_k"), 5, 50, hint="Top K partners per direction for profile construction.")
                top_m = number_input("Min Types (M)", get_user_default("top_m"), 3, 20, hint="Minimum unique partner types in profile.")
            with param_grid(2):
                similarity_metric = select_input(
                    "Sort By", SIMILARITY_METRICS, get_user_default("similarity_metric"),
                    hint="Metric used ONLY for ordering the candidate list (top-N cut). "
                         "Selectable: jaccard (default, conservative — bounded [0,1], "
                         "never degenerate), rank_union, cosine. Every metric is still "
                         "computed by the backend; rank_corr is hidden because it is "
                         "unreliable at bodyId level (see BENCHMARK_RESULTS.md §6c).",
                )
            with ui.row().classes("gap-4"):
                use_fast = checkbox_input("Fast Search", get_user_default("fast_search"), hint="Use adjacency expansion for faster candidate discovery.")
                vector_prefilter = checkbox_input("Vector Pre-filtering", get_user_default("vector_prefilter"), hint="Pre-filter candidates using vector cosine similarity.")
                expand_2hop = checkbox_input("2-Hop Expansion", get_user_default("expand_2hop"), hint="Include 2-hop typed partners for untyped 1-hop neurons.")
            with param_grid(3):
                min_synapse_threshold = number_input(
                    "Min Synapse Threshold", get_user_default("min_synapse_num"), 1, 100,
                    hint="Minimum synapse count for a connection to enter a profile.",
                )
                use_cache = checkbox_input(
                    "Use Cache", get_user_default("use_cache"),
                    hint="Cache profiles and connections locally for faster repeat searches.",
                )
                use_auto_type_mapping = checkbox_input(
                    "Auto Type Mapping", get_user_default("auto_type_mapping"),
                    hint="Standardize partner type names to canonical (male-cns) names "
                         "before cross-dataset comparison.",
                )
            with ui.row().classes("w-full items-center gap-4"):
                visualize = checkbox_input(
                    "Visualize Candidates",
                    True,
                    hint="Generate 3D skeleton visualizations of the top matches.",
                )
                visualization_settings = skeleton_visualization_settings(
                    default_top_n=5,
                    top_n_label="Visualize Top N Candidates",
                    top_n_hint="Number of top homolog candidates to render as 3D skeletons.",
                    default_visualize_by="type",
                    show_high_quality_warning=True,
                    # Homolog searches render many candidates at once; do not
                    # pop the figure open unless the user asks for it here.
                    default_show_fig=False,
                    dataset_provider=lambda: [
                        source_dataset.value,
                        target_dataset.value,
                    ],
                    dataset_watchers=[source_dataset, target_dataset],
                )
            saveas = ui.input(
                label="Save Folder Name (optional)",
                placeholder="e.g., aMe12_homologs",
            ).classes("w-full drocat-input").tooltip(
                "Custom output folder name. Leave empty for the unified auto name "
                "(homologs_<source_ds>_to_<target_ds>_<query>_<timestamp>)."
            )
            full_cache = checkbox_input(
                "Pre-build Full Dataset Cache", False,
                hint="Fetch connections for EVERY uncached neuron before searching. "
                     "Very slow on first use (can take hours); leave off to fetch only "
                     "the connections the search needs.",
            )

    with results_col:
        output_panel.create(run_label="Find Homologs", run_icon="play_arrow")

    async def run_homologs():
        source_vals = source_input.get_value()[1]
        sources = []
        seen = set()
        for value in source_vals or []:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            sources.append(value)
        if not sources:
            ui.notify("Please enter at least one source neuron", type="warning")
            return

        output_panel.clear()
        output_panel.set_running(True)
        visualization_values = visualization_settings.values()
        if visualize.value:
            visualization_settings.warn_empty_custom_palettes()

        # Single combined run: pass the full list of source neurons so the
        # backend resolves each into its real types (coarse labels / other
        # string-column queries expand per type) and writes one output folder
        # grouped by query type.
        constructor_params = {
            "source": sources,
            "source_dataset": source_dataset.value,
            "target_dataset": target_dataset.value,
            "output_dir": output_dir.value,
            "top_n": int(top_n.value),
            "top_k": int(top_k.value),
            "top_m": int(top_m.value),
            "similarity_metric": similarity_metric.value,
            "vector_prefiltering": vector_prefilter.value,
            "include_untyped_partners": expand_2hop.value,
            "visualize_skeleton": visualize.value,
            "visualize_top_n": (
                visualization_values["visualize_top_n"]
                if visualize.value else 0
            ),
            "visualization_settings": visualization_values,
            "min_synapse_threshold": int(min_synapse_threshold.value),
            "use_cache": use_cache.value,
            "saveas": saveas.value.strip() or "",
            "use_auto_type_mapping": use_auto_type_mapping.value,
            "ensure_cache_complete": full_cache.value,
        }
        method_params = {"use_fast": use_fast.value}

        try:
            result = await output_panel.run(
                runner, "find_homologs", constructor_params,
                "find_homologs_multi", method_params=method_params,
                output_dir=output_dir.value,
            )
            if result.get("cancelled"):
                output_panel.set_status("Cancelled", "red")
                return
            succeeded = result.get("returncode") == 0
            output_panel.set_status(
                "Completed" if succeeded else "Failed",
                "green" if succeeded else "red",
            )
            if succeeded:
                from ..history_store import record as _record_history
                _record_history(
                    [str(s) for s in sources],
                    datasets=[source_dataset.value]
                    if source_dataset.value else [],
                )
            files = result.get("files", [])
            if files:
                output_panel.show_files(
                    list(files),
                    result.get("output_folder") or output_dir.value,
                )
        finally:
            output_panel.set_running(False)

    output_panel.run_button.on_click(run_homologs)
    output_panel.cancel_button.on_click(runner.cancel)
