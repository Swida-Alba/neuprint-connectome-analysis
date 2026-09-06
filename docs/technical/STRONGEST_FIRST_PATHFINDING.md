# StrongestFirst Pathfinding — Technical Report

**Status:** implemented (2026-09-04; revised 2026-09-05). Scope: the
unified path-bounding stack for `FindAllPath` / `FindShortestPath` and
every consumer that ranks or limits paths (cross-dataset comparison,
Complete Paths tab, path visualization).
2026-09-05 revision: **StrongestFirst is the only selectable 'all'-mode
algorithm** (the UI selector was removed; the `pathfinding` parameter
stays for scripts/tests), the retired edge-limit knob was repurposed as
the lossy **Edge Budget** floor (Fix D, w0 = w1 + 1), and complete runs
report their natural tau.
Companion plans: `plan-cross-dataset-pathfinding-optimization.md`
(implementation), `plan-cross-dataset-report-fixes.md` (Fix A cap
policy, Fix C trim unification, §6b/§6c),
`plan-sf-only-edge-budget.md` (this revision).

---

## 1. The conceptual model

Two different knobs bound a pathfinding run, and they are not the same
kind of object:

| | manual threshold (`min_synapse_num`) | StrongestFirst budget (`max_paths_bodyid`) |
|---|---|---|
| what it trims | the bodyId **graph**, a priori | the ranked **path list**, a posteriori |
| when | before/during discovery | during enumeration |
| chosen by | the user | emerges from the data |
| effect | removes edges (and everything behind them) | removes whole weak paths, keeps the strongest |

Because StrongestFirst ranks paths by **bottleneck** (the minimum edge
weight along a path — a route is as strong as its weakest hop) and stops
at the budget with a fully drained tie group, a budgeted run satisfies an
exact equivalence:

> **τ-equivalence.** `StrongestFirst(threshold=t, budget=B)` with
> reported cutoff τ produces exactly the same path set as a *complete*
> run at `min_synapse_num = τ` (same query, same hop bound).
> *Proof:* the emitted set is `{paths : every edge ≥ τ}` by the drain
> semantics; a complete run at threshold τ enumerates precisely that set,
> since every such path is also a valid ≥t path and nothing weaker
> survives. ∎

So a low threshold with capping in a large graph **behaves as a raised
threshold** — never as a blind trim — and the run reports which raised
threshold it corresponds to. τ is data-dependent (not user-chosen);
users who want a specific τ raise the threshold or the budget.

## 2. Pipeline layers (where each mechanism runs)

Everything happens at the **bodyId level**; type-mapped outputs are
downstream projections of the bodyId path set (type-level aggregation
before pathfinding would create phantom type paths — the "bundle
effect"; see TYPE_AGGREGATION_AND_BODYID_DISCOVERY.md).

1. **Discovery** (Phase 1): layer tables fetched from the
   threshold-filtered connection cache — cones are nested across
   thresholds (cone(t=10) ⊆ cone(t=3)).
2. **Lossless hop-budget pruning** (`prune_layers_hop_budget`): drops
   edges with `dist_S(u) + 1 + dist_T(v) > max_interlayer + 1` — they
   cannot lie on any admissible path. Lossless by proof; measured
   39–77% cone reduction on real queries.
3. **Dead-end node pruning**: nodes that cannot reach any target
   (post-pruning) are removed. Also lossless. Both passes report a
   **strongest-retained bottleneck** — the widest-path maximin value
   W\* = max over source→target paths of the min edge weight — which is
   *identical* before and after the passes (that is what lossless
   means), and is printed with the note *"top paths unchanged"*.
4. **StrongestFirst enumeration** (`find_paths_strongest_first`): A*-style
   best-first on the prefix bound `min(running bottleneck,
   W[remaining][node])`. Emits complete intact paths in descending
   bottleneck order; stops at the budget and drains ties at τ.
5. **Enrichment / type-path derivation / saves / visualization**: unchanged
   downstream stages; per-threshold enrichment denominators are computed
   exactly as before.

**Retired then repurposed (Fix C → Fix D):** the lossy bodyId edge trim
(`_trim_edges_with_path_integrity` via `_graph_edge_frames`) no longer
runs in FindAllPath/FindShortestPath — it bounded the graph but not the
paths, was not nested across thresholds, and could silently remove the
weakest hop of a top path. The arbitrary-order `max_paths_bodyid`
truncation for legacy enumerators is likewise gone: any budget routes
through StrongestFirst. **Fix D (2026-09-05)** repurposed the former
knob as the **Edge Budget**: after the lossless prunes, a cone exceeding
the budget is floored at `w0 = (N-th strongest edge weight) + 1`
(`apply_edge_budget_floor`) — see §4b. The old `_trim_bodyid_edges`
top-N trim was deleted; `_trim_edges_with_path_integrity` survives only
for the FindNetwork type-level group trims.

## 3. Guarantees

- **No budget bite ⇒ identical set** to complete enumeration (only the
  order differs). Unit-tested against MemoizedDFS.
- **Budget bite ⇒ `all intact paths with bottleneck ≥ τ`** — ties at τ
  fully drained; never an arbitrary subset. τ and the fact of bounding
  are recorded in `user_warning_notes.txt` and `all_attributes.json`
  (`strongest_first_cutoff`, `trim_policy`).
- **Determinism**: same graph ⇒ same emission sequence (stable tie
  rules).
- **Per-threshold comparability**: discovery cones and unbounded runs
  are nested across thresholds; budgeted runs are compared at equal τ
  (`tau` / `paths_complete` columns in `threshold_sensitivity.csv`).

## 4. Efficiency profile

| mechanism | bounds | measured cost |
|---|---|---|
| lossless hop-budget pruning | graph (exact) | O(E); 39–77% of discovery edges removed on real queries |
| strongest-first budget | paths + search | work ≈ output + frontier above τ |
| retired edge trim | graph (lossy) | bounded edges but NOT paths; non-nested across thresholds |

Benchmarks (BANC-era baseline, since re-verified on male-cns):
complete enumeration 533 s / 3.6 GB → StrongestFirst + pruning +
`skip_bodyId=True` 19.5 s / 11 MB on the same query; sensorimotor
long-path matrix (JO→VNC motor, JO→DN, visual→DN at L4–L5): StrongestFirst
54–186 s where complete MemoizedDFS exceeded 300 s.

Worst case (budget never reached, no ties): cost ≈ complete enumeration
plus one O(cutoff·E) DP — never asymptotically worse.

## 4b. Reading the two numbers: W\* (pruning ceiling) and τ (budget)

A run reports two different strength numbers — do not confuse them:

- **W\* — strongest retained path bottleneck** (pruning note,
  §6b): computed BEFORE enumeration, after the lossless prunes. It is
  the widest-path maximin value of the whole cone — the strength of the
  single best source→target route. It is the **ceiling**: no path can be
  stronger than W\*, and because both prunes are lossless, W\* is
  identical before and after them ("top paths unchanged"). Any manual
  threshold ≤ W\* leaves this best path intact; thresholds above W\*
  would remove the best route.
- **τ — budget cutoff** (StrongestFirst note): where the path budget
  landed, if it landed. The output contains all intact paths with
  bottleneck ≥ τ; τ ≤ W\* always.

Reading them together: W\* = "your best route is this strong, and
pruning preserved it"; τ = "you kept everything down to this strength".
A complete run (budget not bitten) reports its natural τ = the weakest
path's bottleneck with `paths_complete = true` — meaning every
threshold up to that value yields the identical set.

**Implemented (Fix D, 2026-09-05): the Edge Budget floor.** After the
lossless prunes, if the pruned cone still exceeds the Edge Budget
(`graph_edge_limit_bodyid`, default 1M; 0/None = off) the N-th
strongest edge weight w1 is located by quickselect and every edge with
`weight < w0 = w1 + 1` is dropped, followed by a second lossless pass.
The **+1 is the load-bearing part**: edges strictly heavier than w1
number fewer than N by definition of the N-th rank, so the floored cone
always fits the budget — a multi-million-edge tie tier at w1 cannot
blow it up, and on locally truncated products (BANC ≥ 3) a landing at
w1 = 3 floors at 4 instead of being a no-op. Degenerate cones (a single
weight tier, or a budget below the distinct-weight support) revert with
an honest note instead of returning an empty graph. The floor is a pure
threshold raise: **effective cutoff = max(τ_budget, w0)**, reported as
`edge_weight_floor` / `edge_budget_landing` in the run attributes,
`parameters.txt`, and a lossy-floor note. Shortest mode is never
floored.

## 5. API reference

- `FastGraph.find_paths_strongest_first(sources, targets, cutoff,
  budget=None, per_pair_k=None, stats=None, verbose=False)` — budgeted
  strongest-first enumeration. `stats` receives `emitted`, `tau`,
  `budget_bitten`, `last_bound`.
- `FastGraph._widest_path_backward(targets, cutoff)` — delegates to
  `strongest_core.widest_path_backward(adj, targets, cutoff)`.
- `vispath_pkg/strongest_core.py` — the shared core:
  `widest_path_backward`, `path_bottleneck`, `interior_strength`,
  `path_rank_key`, `selection_threshold`, `drain_budget`.
- `coana.prune_layers_hop_budget(conn_layers, sources, targets, bound, …)`
  — the lossless pass; returns `(pruned_tables, stats)` with
  `strongest_retained`.
- `coana.apply_edge_budget_floor(conn_layers, budget, sources, targets,
  bound, …)` — the Fix D lossy floor; returns `(floored_tables, stats)`
  with `applied`, `landing` (w1), `floor` (w0), `dropped`,
  `prune_dropped`, `strongest_retained`. Never mutates its inputs;
  reverts honestly when flooring would empty the graph.
- `ComparisonParameters.max_paths_bodyid` — `None`/0 = auto
  (StrongestFirst: internal 1M budget; legacy enumerators: unbounded);
  `>0` = that budget, always routed through StrongestFirst.
- `ComparisonParameters.graph_edge_limit_bodyid` — the **Edge Budget**
  (default 1M in the UI; `None`/0 = off for API callers). Lossy floor,
  'all' mode only.
- Run metadata: `all_attributes.json` and `parameters.txt` record
  `strongest_first_cutoff` (budget τ or natural τ),
  `strongest_first_budget_bitten`, `edge_weight_floor`,
  `edge_budget_landing`, `applied_tau`, and the `graph_pruning_record`
  (lossless passes + the floor).
- `ComparisonAnalyzer._path_taus[(dataset, threshold)]` — per-run τ for
  the sensitivity export; `_path_run_meta[(dataset, threshold)]` adds
  `applied_folder`, `edge_weight_floor`, `skipped`, `duplicate_of`.

## 6. Verification

- Property tests: strongest-first set ≡ complete set without a budget;
  budgeted set ≡ `{bottleneck ≥ τ}` (brute-force checked on randomized
  graphs); lossless pruning keeps exactly the admissible path universe.
- Golden-master: type-level outputs byte-identical with/without the
  optimizations; per-threshold sets nested (`threshold_sensitivity.csv`
  monotone).
- Real-data matrix (male-cns:v1.0 + FAFB v783 — BANC under
  reconstruction): JO→VNC motor, JO→DN, visual→DN at L4–L5 — see
  `plan-cross-dataset-report-fixes.md` §2b and the sensorimotor harness
  under `local_data/sensorimotor_optim_test/`.

## 7. Decision log

- 2026-09-05: **StrongestFirst-only UI** — the Algorithm selector was
  removed from the Complete Paths and Cross-Dataset tabs (and Settings);
  the `pathfinding` parameter stays honored for scripts/tests so the
  complete-enumeration test coverage keeps working.
- 2026-09-05: **Fix D Edge Budget implemented** with w0 = w1 + 1 (the
  strict floor guarantees kept edges < N even under a massive tie tier;
  the earlier plan's plain top-N landing kept all boundary ties).
  Degenerate guards: single-tier cones and below-support budgets revert
  with a note. Auto-adaptive escalation (raise w0 on a wall-clock guard)
  remains deferred.
- 2026-09-05: **F5 τ-folder discipline** — replayed multi-threshold
  runs materialize only real thresholds (a fresh tau-folder with tau
  denominators + every asked t > tau); collapsed asked thresholds alias
  the tau folder and the intermediate minsyn_{t0} folder is removed.
- 2026-09-05: **F9 ratio/probability filters retired** — both are
  readout columns (connection_ratio now uses the threshold-free ALL-POST
  denominator); the UI entrances are hidden and the pipeline ignores the
  parameters with a notice. Rationale: the threshold-based denominator
  made ratios jump across thresholds and broke cone nesting when the
  filters were active.
- 2026-09-04: budget routing overrides the Algorithm selector when a
  positive budget is set (Fix C) — one trimming mechanism, no arbitrary
  truncation.
- 2026-09-04: lossy edge trim retired from FindAllPath/FindShortestPath
  (legacy `FindPath` keeps its local behavior; the tabs do not use it).
- 2026-09-04: interior-weakest-hop preference kept only as the display
  tie-breaker in the visualization selector; enumeration ranks by the
  full-path bottleneck.
