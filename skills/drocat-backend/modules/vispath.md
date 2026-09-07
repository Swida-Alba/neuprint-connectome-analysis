# vispath — VisualizePath

Module `vispath-subproject/src/vispath_pkg/vispath.py` (imported as `vispath_pkg`;
not installed into site-packages — add `vispath-subproject/src` to `PATH`).
`VisualizePath` renders interactive Cytoscape network, Sankey, and heatmap HTML
from a path/connection table, and can build a blank editable network canvas.

## Constructor (key params)

```python
from vispath_pkg import VisualizePath

vp = VisualizePath(
    path_file="/abs/output/paths/path_data.csv",    # CSV or XLSX; or None for empty canvas
    output_folder="/abs/output/network",
    network_layout="hierarchical",      # spring, circular, distributed
    source_color="#4A90E2",
    intermediate_color="#50E3C2",
    target_color="#B8E986",
    link_color="rgba(74,144,226,0.3)",
    showfig=False,
    output_format="csv",
    generate_empty_network=False,       # True → blank editable Cytoscape canvas
)
```

## Key methods

| Method | Purpose |
| --- | --- |
| `visualize(plot_heatmap=True, plot_Sankey=True, plot_network=True)` | Render all three HTML views; returns `(connections, graph)`. |
| `visualize_heatmap(custom_row_order=None, custom_col_order=None)` | Heatmap only. |
| `visualize_sankey()` | Sankey only. |
| `visualize_network()` | Network only. |
| `visualized_paths_for_export()` | Connection table for export. |

## Examples

```python
# Standard: render a completed FindPath/comparison CSV
from vispath_pkg import VisualizePath
vp = VisualizePath(path_file="/abs/output/paths/path_data.csv",
                   output_folder="/abs/output/network",
                   network_layout="hierarchical", showfig=False)
connections, graph = vp.visualize()

# Empty editable canvas
vp = VisualizePath(path_file=None, output_folder="/abs/output/empty_network",
                   generate_empty_network=True, network_layout="hierarchical",
                   showfig=True)
vp.visualize()
```

## Edge cap and artifact naming (pathfinding runs)

- `_select_edges_for_plot` is the shared selector behind the Visualization Edge
  Limit (`edgeN_limit`): a drawing-only cap on the number of unique edges
  rendered per HTML view. It never changes fetch, graph, or path outputs, and a
  single complete path may exceed the cap to stay intact. Type-level and
  bodyId-level visualizations share the same cap.
- `_organize_vispath_artifacts` (generalized in `src/coana.py`) fixes the
  canonical naming contract for both levels: `Network_<run>.html`,
  `Heatmap_<run>.html`, `Sankey_<run>.html`, and
  `visualization_data/<run>_data_*.csv` inside `visualization/` and
  `bodyId_visualization/` (raw legacy names such as
  `bodyId_visualization_network.html` no longer remain). Early previews keep
  `network_early/` and `network_early_bodyId/` with `Network_<run>.html` inside.
- When the drawing cap trims, a companion CSV lists what was rendered:
  `visualization/visualization_data/type_paths_visualized.csv` (type level) and
  `bodyId_visualization/visualization_data/bodyId_paths_visualized.csv`
  (bodyId level). Missing artifact types stay absent.

## Notes

- The input may be CSV or Excel; for Excel use `sheet_name="path_type"` /
  `"path_bodyId"` when autodetection is ambiguous.
- Use a completed FindPath/comparison artifact rather than a hand-written file
  unless the schema has been verified.
- `generate_empty_network=True` requires `path_file=None`.
- For large graphs use `network_layout="distributed"` or `"hierarchical"`; keep
  `showfig=False` for headless runs and open the returned `*_network.html` manually.
