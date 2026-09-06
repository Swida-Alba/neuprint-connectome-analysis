# Cross-Dataset Comparison (inter_dataset)

Reproduce the **Cross-Dataset Comparison** UI tab as a direct backend call. Runs
`ComparisonAnalyzer` over N datasets with shared source/target queries.

## Backend contract

- **tool_key:** `inter_dataset`
- **import:** `from comparison import ComparisonParameters, ComparisonAnalyzer`
- **wrap:** the UI builds a `ComparisonParameters` object, then
  `ComparisonAnalyzer(params, verbose=True)`, then
  `analyzer.run_comparison()` and `analyzer.export_results()`.
- **class:** `ComparisonAnalyzer` (var `analyzer`)

## Parameters the UI builds (ComparisonParameters)

```python
from comparison import ComparisonParameters, ComparisonAnalyzer

params = ComparisonParameters(
    datasets=["hemibrain:v1.2.1", "male-cns:v0.9"],
    source_neurons=["aMe12"],           # shared across all datasets
    target_neurons=["PPL101"],
    output_folder="/absolute/output/comparison",
    comparison_mode="path",             # or "edge" to preserve strong direct edges
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
    max_workers=None,                   # None disables parallelism
    separate_hemispheres=False,
    keep_only_hemisphere_conserved_connections=False,
    symmetry_analysis=False,
    find_reciprocal=False,
)
# optional: params = ComparisonParameters(..., overall_mapping_json="/path/to/mapping.json")

analyzer = ComparisonAnalyzer(params, verbose=True)
analyzer.run_comparison()
analyzer.export_results()
```

## Run

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_Compare_<date>.py
```

## Outputs

- Per-dataset comparison tables (CSV/XLSX), threshold summaries, report, and
  conserved-path HTML views.

## Notes

- `comparison_mode="path"` uses the pathfinding engine (FindAllPath/FindShortestPath);
  `comparison_mode="edge"` preserves strong direct edges. The `path_mode` selects
  per-pair minimum-hop vs all-paths behavior.
- Use `auto_type_mapping=True` (and `overall_mapping_json`) when type names differ
  between datasets.
- Use `parallel=True` with a bounded `max_workers` for many datasets; start with
  `skip_bodyId=True` and `max_interlayer=2`.
