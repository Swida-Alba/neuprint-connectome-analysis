# Connectivity · Comparison (connectivity_profiling)

Reproduce the **Connectivity tab → Comparison sub-tab** as a direct backend
call. Builds connectivity profiles for a query set and compares them across
one or more datasets.

## Backend contract

- **tool_key:** `connectivity_profiling`
- **import:** `from comparison.profile_comparator import ConnectivityProfileComparer`
- **class:** `ConnectivityProfileComparer` (var `comparer`)
- **method:** `comparer.run()`

## Parameters the UI builds

```python
from comparison.profile_comparator import ConnectivityProfileComparer

comparer = ConnectivityProfileComparer(
    query=["aMe12", "aMe10"],           # or a single query for a custom group
    datasets=["male-cns:v0.9", "hemibrain:v1.2.1"],
    output_dir="/absolute/output/profiles",
    top_k=15,
    top_m=5,
    min_synapse_threshold=3,
    direction="both",                   # input, output, or both
    generate_heatmaps=True,
    show_figures=False,
    verbose=True,
    use_cache=True,
    aggregation_level="type",           # "type" | "bodyid" | "custom"
    skip_bodyId_level=False,
    ensure_cache_complete=False,
)
# for aggregation_level="custom", add: comparer = ConnectivityProfileComparer(..., custom_mapping_file="/path/to/mapping.json")

results = comparer.run()
```

## Run

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_Profile_<date>.py
```

## Outputs

- Similarity matrices, heatmaps (when `generate_heatmaps=True`), and profile
  comparison tables under the output folder.

## Notes

- `aggregation_level="custom"` requires `custom_mapping_file`.
- `ensure_cache_complete=True` forces a full cache; use it deliberately.
- The profiling resolver resolves the query before comparing; a completed run
  means the queried chips resolved in the dataset.
