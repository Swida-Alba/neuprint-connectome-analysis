# Direct-analysis workflow recipes

These recipes are intentionally small. Replace names, datasets, thresholds,
and output paths with the user's request; do not run them with placeholder
tokens or placeholder input files.

## Pathfinding → network visualization

1. Run a type-level, CSV-first FindPath pass:

   ```python
   from coana import FindNeuronConnection

   output = "/absolute/output/aMe12_to_PPL101"
   fc = FindNeuronConnection(
       dataset="male-cns:v0.9",
       sourceNeurons=["aMe12"],
       targetNeurons=["PPL101"],
       output_dir=output,
       max_interlayer=2,
       min_synapse_num=3,
       skip_bodyId=True,
       output_format="csv",
       use_cache=True,
       showfig=False,
   )
   fc.InitializeNeuronInfo()
   fc.FindAllPath(forward_only=True)
   ```

2. Find the newest path table without guessing a filename:

   ```bash
   find /absolute/output/aMe12_to_PPL101 -type f \
     \( -name '*allpaths*.csv' -o -name '*paths*.csv' -o -name '*.xlsx' \) \
     -print | sort
   ```

3. Pass the verified file to `VisualizePath`, set `showfig=False`, and inspect
   the generated network, Sankey, and heatmap HTML. Use `showfig=True` only
   when the user wants the browser opened.

## Empty network canvas

Use this when the user wants to draw a graph directly rather than visualize
FindPath data:

```python
from vispath_pkg import VisualizePath

vp = VisualizePath(
    path_file=None,
    output_folder="/absolute/output/empty_canvas",
    generate_empty_network=True,
    showfig=True,
)
vp.visualize()
```

Report the returned HTML path. The user can enable Edit Mode, add nodes, draw
edges, edit properties, and export from the generated page.

## Fast/slow escalation for large analyses

1. Start with one source and one target, `max_interlayer=1` or `2`, CSV output,
   `skip_bodyId=True`, and `showfig=False`.
2. Verify row counts and that source/target types are present.
3. Increase hop count, add bodyId-level output, or add visual exports one at a
   time.
4. Keep `edgeN_limit` bounded for HTML and use `network_layout="distributed"`
   or `"hierarchical"` for large graphs.
5. If a query is expensive, reuse cache and record whether data came from cache
   or the remote API.

## Debugging a failing script

```bash
python -m py_compile scripts/FindPath.py
PYTHONNOUSERSITE=1 conda run -n drocat-4.5.0 python -c \
  'import inspect; from coana import FindNeuronConnection as C; print(inspect.signature(C))'
```

Then inspect only the error's module and call site with `rg`. Check, in order:

1. correct environment and `PYTHONNOUSERSITE=1`;
2. token availability without printing its value;
3. dataset spelling and local cache/data files;
4. input schema and path file selection;
5. output-folder permissions and disk space;
6. the smallest reproducible query.

Do not patch dependencies or global configuration to hide a project-level
parameter mismatch. If a backend signature changed, update the focused run
script and add a regression test for the corrected call.

## Result triage

- Empty CSV: distinguish “no biological matches” from “query/filter mismatch”
  by checking the resolved source/target selection and thresholds.
- Missing HTML: inspect the preceding log for a failed export or an empty input
  graph; do not infer success from a zero exit code alone.
- Huge output: lower `edgeN_limit`, use CSV, skip bodyId-level work, or narrow
  the query; record the chosen trade-off.
- Misaligned 3D mesh: verify dataset/template coordinates and use the
  dataset-specific guidance before changing transforms.
- FAFB failure: verify local files and CAVE token. BANC failure: verify
  network reachability of the public bucket (no token, no local files
  needed; the pathfinding connection cache rebuilds automatically from the
  per-version merged table). API-only NeuPrint assumptions do not apply.
