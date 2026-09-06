# Complete Paths (find_path)

Reproduce the **Find Path** UI tab as a direct backend call. Multi-hop
pathfinding between two neuron groups.

## Backend contract

- **tool_key:** `find_path`
- **import:** `from coana import FindNeuronConnection`
- **class:** `FindNeuronConnection` (var `fc`)
- **init:** `fc.InitializeNeuronInfo()`
- **method:** `fc.FindAllPath(forward_only=True, find_reciprocal=fc.find_reciprocal)`
  (the UI uses `find_all_path`; `fc.FindPath()` is the single-strategy variant)

## Parameters the UI builds

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    dataset="male-cns:v0.9",
    sourceNeurons=["aMe12"],          # types/bodyIds/instances/regex
    targetNeurons=["PPL101"],
    output_dir="/absolute/output/paths",
    min_synapse_num=3,
    min_ratio=0.0,
    min_traversal_probability=0.0,
    max_interlayer=2,
    filter_by="bodyId",               # or "type"
    pathfinding="StrongestFirst",     # built-in default; DP/MemoizedDFS/DFS = unbounded complete runs (API)
    max_paths_bodyid=0,               # auto -> internal 1,000,000 path budget (tau reported when it bites)
    graph_edge_limit_bodyid=0,        # Edge Budget off; set ~1_000_000 to cap the discovery cone
    visualize_before_reconstruct=False,
    search_columns="auto",              # "auto" | "type" | "instance" | "bodyId"
    network_layout="distributed",
    use_cache=True,
    edgeN_limit=500,
    output_format="csv",              # or "xlsx"
    skip_bodyId=True,                 # faster type-level first pass
    showfig=False,
    custom_source_name="",
    custom_target_name="",
    keyword_in_path_to_remove=["None"],
    cache_only=False,
    saveas="",
    separate_hemispheres=False,
    hemisphere_filter="all",
    keep_only_hemisphere_conserved_connections=False,
    symmetry_analysis=False,
    find_reciprocal=False,
)
# optional: fc constructor param custom_mapping_file for a custom grouping/mapping JSON

fc.InitializeNeuronInfo()
fc.FindAllPath(forward_only=True, find_reciprocal=fc.find_reciprocal)
```

## Run

Save the above as a focused script (e.g. `archive/scripts_local/agent_FindPath_<date>.py`)
and run through the launcher:

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 \
  --script archive/scripts_local/agent_FindPath_<date>.py
```

Or run the template `scripts/FindPath.py` (edit its config block):

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script scripts/FindPath.py
```

## Outputs

- CSV/XLSX type-level (`path_type`) and bodyId-level (`path_bodyId`) path tables,
  path summaries, and optional network/heatmap HTML.
- Use `visualize_before_reconstruct=False` (the UI default) for headless runs and
  hand off to PlotPath for the interactive network.

## Notes

- `max_interlayer=0` means "no limit" only when a source/target set is marked as
  all neurons; otherwise limit hops, then increase after a smaller run completes.
- Empty `sourceNeurons`/`targetNeurons` have special meaning in the backend; state
  the intended scope before using them.
- Keep the UI closed and `showfig=False` until the output is validated.
