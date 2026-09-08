# comparison — Cross-Dataset Comparison

Cross-dataset comparison module (`src/comparison/`). The primary entry points are
`ComparisonParameters` (a dataclass holding all settings) and
`ComparisonAnalyzer` (the orchestrator). `quick_compare` is the one-liner;
`CrossDatasetTypeMapper` and `LabelMapper` handle naming differences between
datasets.

## ComparisonParameters + ComparisonAnalyzer

```python
from comparison import ComparisonParameters, ComparisonAnalyzer, quick_compare

params = ComparisonParameters(
    datasets=["hemibrain:v1.2.1", "male-cns:v0.9"],
    source_neurons=["aMe12"],           # shared across all datasets
    target_neurons=["PPL101"],
    output_folder="/abs/output/comparison",
    comparison_mode="path",             # or "edge"
    path_mode="all",                     # "all" | "shortest"
    max_interlayer=2,
    thresholds=[1, 3, 5, 10],
    threshold_mode="standard",          # "standard" | "combinations"
    # Combination mode example (replace thresholds above):
    # threshold_dataset_order=["hemibrain:v1.2.1", "male-cns:v0.9"],
    # threshold_combinations=[
    #     {"id": "combo_001", "label": "density match",
    #      "thresholds": {"hemibrain:v1.2.1": 3, "male-cns:v0.9": 8}},
    # ],
    top_edges=500,
    graph_edge_limit_bodyid=0,          # Edge Budget off; set ~1_000_000 to cap the discovery cone
    edgeN_limit=500,
    pathfinding="StrongestFirst",       # built-in default; budgeted by max_paths_bodyid (tau on bite)
    max_paths_bodyid=0,                 # auto -> internal 1,000,000 path budget
    search_columns="auto",              # "auto" | "type" | "instance" | "bodyId"
    skip_bodyId=True,
    cache_only=False,
    auto_type_mapping=True,
    _min_ratio=0.0,
    _min_prob=0.0,
    _output_format="csv",
    parallel=True,
    max_workers=None,
    separate_hemispheres=False,
    keep_only_hemisphere_conserved_connections=False,
    symmetry_analysis=False,
    find_reciprocal=False,
    drop_untyped=True,                  # Drop Untyped Neurons: applied post label-mapping
    overall_mapping_json=None,          # custom cross-dataset mapping
)

analyzer = ComparisonAnalyzer(params, verbose=True)
analyzer.run_comparison()               # returns result dict; runs path/edge analysis
analyzer.export_results()               # writes tables/reports
analyzer.generate_report()              # summary report
```

Quick one-liner:

```python
results = quick_compare(
    datasets=["hemibrain:v1.2.1", "male-cns:v0.9"],
    source_neurons=["MBON14.*_R"],
    target_neurons=["KCg-d.*_R"],
)
```

## Key methods on ComparisonAnalyzer

| Method | Purpose |
| --- | --- |
| `run_path_analysis(...)` | Path-based per-dataset analysis. |
| `run_edge_analysis(...)` | Edge-based (strong direct edges) per-dataset analysis. |
| `run_all_analyses(skip_existing=True)` | Run every configured analysis, skipping completed ones. |
| `run_comparison(skip_existing=True)` | Main orchestrator. |
| `generate_report(output_path=None)` | Summary report. |
| `export_results(output_dir=None)` | Write per-dataset tables, summaries, conserved-path HTML. |
| `generate_html_report(output_path=None)` | HTML report. |

## Type mapping across datasets

```python
from comparison import CrossDatasetTypeMapper, LabelMapper

mapper = CrossDatasetTypeMapper()      # auto-map type names across datasets
labeler = LabelMapper()                # standardize labels if names differ
```

`ComparisonParameters.auto_type_mapping=True` (plus `overall_mapping_json`) is the
usual way to resolve differing type names; `LabelMapper` is the manual override.

## Untyped-neuron drop (drop_untyped)

`drop_untyped=True` (default) removes edges touching untyped neurons from the
cross-dataset results. The predicate is the shared
`utils.label_utils.is_untyped_type_label` (empty label, Unknown/None/NaN
sentinel, all-digit bodyId-fallback label), but the analyzer applies it AFTER
standardized cross-dataset label mapping — the later, authoritative timing.
The delegated per-dataset `FindNeuronConnection` runs therefore execute with
`drop_untyped=False`, so only the comparison-level filter fires and no
per-dataset `data_details/` records are written.

Outputs, only when rows were dropped:

- `comparison_results/untyped_dropped_records.csv` — dropped rows with
  `dataset`, `threshold`, the connection columns, and `untyped_side`
  (`pre` / `post` / `pre+post`).
- Per-run counts appended to the run root's `user_warning_notes.txt`.

## Threshold query model

`threshold_mode="standard"` expands each scalar in `thresholds` into one
same-threshold query for every selected dataset. Use
`threshold_mode="combinations"` with `threshold_combinations` when a query
needs different thresholds per dataset. Custom combination mode requires at
least two selected datasets. Each row must contain exactly one positive
threshold for every selected dataset; it is a comparison identity, not a
per-dataset schedule. The union of cell values is only the deduplicated
raw-run/cache schedule.

Combination rows retain stable `id`/`label` values. Alignment, similarity,
reports, and presence matrices are keyed by that query row and never infer a
scalar from the union. Raw `(dataset, threshold)` jobs shared by multiple
rows run once and are referenced from each row.

## Pathfinding threshold/bottleneck provenance

Every delegated pathfinding threshold folder writes the shared provenance
block to `parameters.txt`, `all_attributes.json`, and
`data_details/parameters.csv`: `requested_threshold`, canonical
`applied_threshold`, `applied_threshold_source`, the effective
`strongest_first_budget` and bite flag, landing `tau`, `tau_canonical`, `w2`
(`strongest_dropped_bottleneck`), Edge Budget `edge_budget`/`w0`/`w1`,
`strongest_retained_bottleneck` (`W*`), and `paths_complete`.

The comparison root additionally exports:

- `effective_thresholds.json` — the UI/run-guide notice with one `runs` row
  per dataset and requested threshold, plus `queries`/`combinations` with
  requested threshold maps and applied per-dataset provenance.
- `comparison_results/pathfinding_provenance.csv` — the complete machine-
  readable row set. Use `applied_threshold` for the canonical equivalent Min
  Synapse Count; `tau` is the StrongestFirst landing/collapse bound, not
  necessarily the minimal applied threshold.
- `comparison_results/threshold_combinations.csv` — the canonical query
  manifest. It joins `query_id`, dataset, requested/applied threshold,
  StrongestFirst budget/tau, Edge Budget `w0`/`w1`, `w2`, `W*`, completeness,
  and raw-run/alias provenance.
- `comparison_results/threshold_sensitivity.csv` and
  `comparison_results/unified_summary.csv` — summary tables that retain the
  same provenance fields, including `skipped`/`duplicate_of` for tau-collapsed
  thresholds and untyped-drop counts where applicable. In combination mode,
  sensitivity rows carry `query_id`/`query_label`; adjacent-threshold
  retention is not inferred across unrelated query rows.
- `similarity_matrices/similarity_query_{query_id}.csv` and
  `similarity_matrices/similarity_by_query.csv` — query-keyed similarity
  exports for advanced combinations.

The Edge Budget is a lossy graph floor in `all` mode only. Shortest mode can
be bounded by the StrongestFirst path budget and report tau, but its Edge
Budget is ignored and `edge_weight_floor` remains empty.

## Notes

- `comparison_mode="path"` uses the pathfinding engine (FindAllPath /
  FindShortestPath); `comparison_mode="edge"` preserves strong direct edges.
- `path_mode` selects per-pair minimum-hop vs all-paths behavior.
- `parallel=True` with a bounded `max_workers` speeds many datasets; start with
  `skip_bodyId=True` and `max_interlayer=2`.
