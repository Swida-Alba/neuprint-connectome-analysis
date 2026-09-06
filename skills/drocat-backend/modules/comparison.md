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

## Notes

- `comparison_mode="path"` uses the pathfinding engine (FindAllPath /
  FindShortestPath); `comparison_mode="edge"` preserves strong direct edges.
- `path_mode` selects per-pair minimum-hop vs all-paths behavior.
- `parallel=True` with a bounded `max_workers` speeds many datasets; start with
  `skip_bodyId=True` and `max_interlayer=2`.
