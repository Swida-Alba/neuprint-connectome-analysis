# Shortest Paths (find_shortest)

Reproduce the **Shortest Paths** UI tab as a direct backend call. Minimum
hop-count pathfinding between neuron groups (shares `FindNeuronConnection` with
Complete Paths).

## Backend contract

- **tool_key:** `find_shortest`
- **import:** `from coana import FindNeuronConnection`
- **class:** `FindNeuronConnection` (var `fc`)
- **init:** `fc.InitializeNeuronInfo()`
- **method:** `fc.FindShortestPath(forward_only=True, find_reciprocal=fc.find_reciprocal)`

## Parameters the UI builds

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    dataset="male-cns:v0.9",
    sourceNeurons=["aMe12"],
    targetNeurons=["PPL101"],
    output_dir="/absolute/output/shortest",
    min_synapse_num=3,
    min_ratio=0.0,
    min_traversal_probability=0.0,
    max_interlayer=2,
    filter_by="bodyId",
    graph_edge_limit_bodyid=0,     # shortest mode is never floored (trimming distorts hop distances)
    max_paths_bodyid=0,            # auto -> 1M StrongestFirst budget on min-hop paths (tau reported when it bites)
    visualize_before_reconstruct=False,
    search_columns="auto",              # "auto" | "type" | "instance" | "bodyId"
    network_layout="distributed",
    use_cache=True,
    edgeN_limit=500,
    output_format="csv",
    skip_bodyId=True,
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
fc.FindShortestPath(forward_only=True, find_reciprocal=fc.find_reciprocal)
```

## Run

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 \
  --script archive/scripts_local/agent_FindShortest_<date>.py
```

## Outputs

- Per-pair minimum-hop path tables (CSV/XLSX) and summaries.

## Notes

- The UI keeps the early network preview disabled (`visualize_before_reconstruct=False`).
- Start with `max_interlayer=2` and `skip_bodyId=True`; verify counts before
  enabling bodyId-level output.
- This is also the engine behind `inter_dataset`'s `path_mode="shortest"`.
