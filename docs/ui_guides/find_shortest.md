# Shortest Paths

Find only the **shortest (minimum hop-count) paths** between source and
target neuron groups. Backend: `FindNeuronConnection.FindShortestPath` — the
full FindAllPath pipeline (discovery, enrichment, type-path derivation,
outputs, visualization) with shortest-only enumeration.

## Semantics

- For every reachable (source, target) pair, the minimum-hop paths under
  the search criteria (Min Synapse Count — ratio/probability are readout
  columns since F9) are returned. **All tied shortest paths are kept**, each once.
- Discovery is target-rooted by default: the search starts at each target,
  follows incoming edges, and reconstructs only reverse branches that reach
  an enrolled source bodyId. This avoids building the full fan-out of source
  neurons that cannot contribute to a requested target.
- The bodyId-level minimum is enforced per exact `(source bodyId, target
  bodyId)` pair before any type aggregation. A longer path to a close target
  is therefore removed even when another target has a longer valid shortest
  path; this is not a single global path-length cutoff.
- Those filtered paths are then mapped to neuron types. All distinct
  target-involved type sequences backed by the bodyId shortest paths are
  retained, including longer sequences when they are shortest for a different
  target instance. If multiple bodyId pairs produce the identical type
  sequence, the type table represents that sequence once; enable bodyId output
  to distinguish the individual pairs.
- Distances are hop counts on the threshold-filtered graph (synapse weights
  are edge attributes, not path costs).

## Source-set recommendation

If the source field contains more than one type/name query, the page shows a
warning because each query can expand to many source bodyIds and enlarge the
target-rooted search. Prefer a small source set, ideally one source type or
query. Use explicit bodyId inputs and keep bodyId output enabled when exact
source-target pairs are needed.

## Depth cap and shortest-path guarantee

- **Max Intermediate Layers = 0** means direct connections only.
- For a positive value `M`, paths are capped at `M + 1` hops. Increase `M`
  when a query needs deeper intermediate layers; a high value such as 99 is
  effectively unlimited for practical connectome queries.
- Backward discovery continues for each target until its enrolled source
  bodyIds have been reached or the hop bound is exhausted. A target's first
  appearance is only the minimum distance from *some* source; it is not a
  global cutoff for every source-target pair. Therefore a path longer than
  the nearest target distance can still be the shortest path for a different
  source bodyId.
- When the depth bound truncates discovery, results are shortest only within
  the explored, threshold-filtered graph. A returned path at that boundary is
  a solution under the current graph, not proof of a globally shortest route;
  the run records this caveat in `user_warning_notes.txt`.

## Why there is no algorithm selector

Shortest enumeration is a backward-BFS distance pass plus a guided DFS over
the shortest-path DAG — polynomial in the graph size per pair. The
branching^depth explosion that makes Find All Paths expensive (it explores
non-shortest branches) does not exist here, but the *total* number of
min-hop paths can still grow quickly at depth: tied routes multiply at
every layer of equal-length parallel corridors.

## Path budget (Max Paths, BodyId)

The min-hop enumeration runs through the **StrongestFirst** budget:
paths are emitted strongest-first (descending bottleneck) and, when the
budget (`max_paths_bodyid`, 0 = auto → 1,000,000) is reached, all ties
at the achieved cutoff **tau** are drained and weaker paths dropped —
the output is exactly "all min-hop paths with bottleneck ≥ tau",
reported in the run notes. Unbitten runs are identical to the complete
min-hop set. The Edge Budget floor never applies here.

## Edge Budget: off in shortest mode

Unlike Find All Paths (where the Edge Budget floors an oversized
discovery cone), shortest mode **never floors** — for any Edge Budget
value, including 1M defaults:

- Its purpose (bounding the exponential path count) is unnecessary for
  polynomial shortest enumeration.
- Flooring is a threshold raise: it preserves pair *reachability* but
  not *shortest distance* — a dropped weak edge can silently inflate a
  reported distance.

The tab disables the Edge Budget input in shortest mode; API callers
passing `graph_edge_limit_bodyid` get the full un-floored graph here.
The **Max Paths (BodyId)** input is the knob that bounds this mode.

## Visualization Edge Limit

Visualization Edge Limit only caps the number of edges drawn in each
network, heatmap, or Sankey view. It does not trim the fetched graph and
does not change shortest-path discovery.

## BodyId enrollment and recommended follow-up

With type-level output, multiple exact bodyId pairs can map to the same type
sequence and are intentionally represented by one type-level row. The run
warns that source-target bodyId pairs may be lost in this aggregation. Check
the root-level `source_neurons.csv` (`isInPath`) and `target_neurons.csv`
(`Checked`, `Layer`) for enrollment details.

- To inspect specific source-target bodyId pairs, run Shortest Paths with
  **Skip BodyId** off.
- To enumerate all paths within a specified depth, run **Complete Paths**.

## Everything else matches Find All Paths

Connection cache reuse, keyword exclusion, custom grouping, hemisphere
options (separate L/R, conserved-edge filter, symmetry analysis),
reciprocal enrichment, CSV/XLSX outputs, and the final
network/Sankey/heatmap visualizations.

## All-neurons queries (all_neurons)

Typing the special chip `all_neurons` as the **source or target** loads the
full (typed) neuron set on that side, so the run fetches every adjacent
neuron at the given thresholds (Min Synapse Count):

- `all_neurons` replaces every other chip in the same input.
- Both source and target = `all_neurons` is not allowed.
- An `all_neurons` side forces **Max Intermediate Layers = 0** (direct
  connections only) — the search stays bounded instead of exploring the
  whole graph. With shortest-path semantics this reports every direct edge
  (one hop) from or to the other side.

The backend (`InitializeNeuronInfo`) enforces the same rules, so script and
API callers get identical behavior.

## Output

Same layout as Find All Paths (see the Complete Paths guide):

```
{output_dir}/find-paths-shortest_MCNS_aMe12_to_PPL101_L1w3r0p0_20260815_142717/
  aMe12_to_PPL101_allpaths_type.csv  # shortest paths (same columns as Find All Paths)
  source_neurons.csv, target_neurons.csv  # enrollment/status details
  all_attributes.json
  parameters.txt, user_warning_notes.txt
  data_details/                      # matrices, connection_type.csv, parameters.csv
  visualization/                     # Network / Sankey / Heatmap HTML + visualization_data/ inputs
  hemisphere_symmetry/...            # when Symmetry Analysis is on
  find_reciprocal/...                # when Find Reciprocal Connections is on
```

`Lmax` records the exact `Max Intermediate Layers` bound used for the run;
`L99` is a practical high-bound choice, not a special unlimited value.

With **Skip BodyId** off, bodyId-level tables (`..._allpaths_bodyId.csv`) are
written next to the type-level ones. The full reference is in
`docs/OUTPUT_FILES.md`.
