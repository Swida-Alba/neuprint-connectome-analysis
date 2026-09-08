# Auto Type Mapping — Implementation Report

*Technical reference for the cross-dataset auto type mapping engine as of
2026-09-09. Companion to the user-facing `docs/AUTO_TYPE_MAPPING.md`; this
report documents the implementation: components, the bridge-rule algebra,
the derivation walk, resolution precedence, the shared UI backend, pooling,
and the visualization contract.*

## 1. Components and data flow

```
male-cns v1.0/v0.9 neuron_df (release-aware crosswalk tables)
        │  type / flywireType / hemibrainType / mancType cells
        ▼
CrossDatasetTypeMapper                      src/comparison/cross_dataset_type_mapper.py
  ├─ _build_type_mappings()        stored 1-to-1 mappings + N-to-1 / 1-to-N conflicts
  ├─ _load_flywire_type_tables()   per-release FAFB/BANC labels + additional-name tables
  ├─ _apply_banc_label_overlay()   curated BANC per-dataset label votes
  ├─ _apply_banc_release_overlay() root_626 ↔ root_888 type relation
  ├─ _build_mcns_v09_mappings()    native v0.9 + explicit same-name alias
  ├─ _apply_annotation_bridge_overlay()   same-name identity + annotation-bridge pairs
  ├─ get_type_bridges()            derivation chains (the evidence algebra, §3)
  ├─ get_alias_candidates()        per-dataset alias candidates (rename / same name / one-of-N)
  └─ get_mapped_type()             stored-mapping lookup (production resolution)
        ▼
mapped_type_targets()                       ui/neuron_index.py — THE shared backend (§5)
  ├─ 'See available neurons' viewer mapped view   (enrich_native_type_matches)
  └─ Cross-dataset tab Type Mapping panel summary (type_mapping_panel._compute)
        ▼
mapping_visualization.py                    flows → network / linker graph / Sankey / CSV
```

Every consumer surface resolves a foreign type through
`mapped_type_targets(mapper, foreign_type, foreign_ds, selected_ds)` — the
union of the stored-mapping/alias resolution and the derivation-bridge
ends. One engine, one number.

## 2. Version control — per-release namespaces

Releases are separate mapping namespaces; nothing resolves through another
release's tables:

| release | mapping namespace | tables |
|---|---|---|
| male-cns v1.0 | `male-cns:v1.0` | the crosswalk df |
| male-cns v0.9 | `male-cns:v0.9` (own native table) | v0.9 table; shared names alias explicitly to v1.0 |
| FAFB v783 | `flywire_FAFB_v783` | `flywire_FAFB_v783_allneurons_neuron_df.csv` |
| BANC v626 | `banc_v626` | `banc_v626_allneurons_neuron_df.csv` |
| BANC v888 | `banc_v888` | `banc_v888_allneurons_neuron_df.csv` |

Both BANC releases sit in `DATASET_PRIORITY` (v888 immediately after
v626), so `_detect_type_source` auto-detects a v888-only type name as
`banc_v888` instead of falling through to "unknown" — pinned in
`tests/core/test_dataset_identity_collisions.py`.

Consequences (user directive, 2026-09-07):

- A `banc_v888` selection resolves **only** against the v888 tables —
  v626 names ("via banc v626") and v626 bodyId pools are impossible.
- `male-cns:v0.9` never silently borrows the v1.0 table or body IDs. Shared
  primary names use an explicit `release_alias` hop before downstream v1.0
  mapping; v0.9-only names use only their own native `*Type` fallback.
- BANC v626↔v888 has one narrow `banc_release_crosswalk` type bridge when
  the relation is available, plus exact same-name type identity as a safe
  fallback when it is not. Generic BANC↔BANC annotation/transitive paths
  remain blocked.

The v0.9 release probe records 176,379 common bodyIds, 163,130 same-name
typed rows, 1,354 typed-row disagreements, and 11,597 shared primary names.
`_release_alias_diagnostics["release_alias_disagreement"]` keeps the
actionable source-side subset: 29 shared v0.9 names across 71 joined rows,
with a bounded example list. A disagreement is visible to audit/reporting but
does not rewrite the selected release's bodyId pool.

## 3. The bridge-rule algebra (derivation walk)

The derivation walk (`get_type_bridges`) expands a source type through
licensed evidence edges and records every simple path that lands on a real
primary `type` of the target dataset. Every edge is licensed by
`BRIDGE_SOURCE_MAP` (declarative; verified against the tables at load).

### 3.1 Direct bridge forms (plus same-name identity everywhere)

```
HEMI  --hT--  MCNS                        MANC --mT-- MCNS
MCNS  --fT--aT--  FAFB   |  MCNS --fT-- FAFB  |  MCNS --aT-- FAFB
MCNS  --mct--  BANC       |  FAFB --aT--ACT-- BANC
HEMI  --hemibrain_cell_type-- BANC
MANC  --manc_cell_type-- BANC
BANC v626 --banc_release_crosswalk-- BANC v888
```

`hT/mT/fT` = hemibrainType/mancType/flywireType cells on the male-cns rows;
`mct` = BANC `malecns_cell_type`; `aT` = FAFB `additional_type(s)`;
`ACT` = BANC `Alternative Cell Type(s)`. The BANC direct bridges also use
`fafb_cell_type`, `hemibrain_cell_type`, and `manc_cell_type` for their named
target namespaces. MCNS `flywireType` does not land in BANC.

### 3.2 Connector licenses (`ROUTE_MIDS`)

A chain visits at most ONE intermediate namespace, and only from the
licensed set:

- male-cns is the controlled connector for the neuprint crosswalk families;
- BANC label bridges and the BANC release relation are direct-only;
- **BANC is never a connector** between unrelated endpoint pairs;
- MCNS↔BANC is `malecns_cell_type`, not an fT/FAFB/ACT detour.

This makes the BANC ban structural (a chain cannot even be built through
it) and enforces the two-linker no-flip rule: a flipped `fT/aT` chain
would have to revisit the source namespace, which the route guard forbids.
All bridges are bidirectional — the reverse travel direction reverses the
hop order (the SAME bridge, not a flip).

### 3.3 Preference order — evidence first, same-name LAST

`_name_neighbors` emits evidence edges (crosswalk, annotation,
reverse-crosswalk, cross-namespace landing) **before** same-name
membership edges, so the walk's visited race records the EVIDENCE chain
for a pair. A same-name pair whose crosswalk cell names itself therefore
derives through the pair's registered metadata linker (for example
`malecns_cell_type '<name>'` for MCNS↔BANC), not the
bare name echo:

```
DN1a[FAFB·type] → DN1a[MCNS·flywireType]      (crosswalk-verified)
```

The bare same-name chain is the LAST choice — offered only when no
evidence edge connects the pair (then the UI still says "no metadata
verification — please double check"). The reverse-crosswalk leg no longer
skips the same-name arrival: the verification edge must be walkable.

### 3.4 Arrival gate for registry-less pairs

The post-arrival annotation continuation is a two-linker REGISTRY-standard
privilege (male-cns↔FAFB etc.: crosswalk primary → its annotated
siblings). Registry-less pairs — every BANC pair — stop at the arrival;
without the gate the walk wandered BANC annotation classes
(`APDN3 → R8_unclear → T1`), landing coarse hub names and inflating a
circadian FAFB→BANC run to 3,262 neurons (post-fix: 21/21 types, 242/242
FAFB neurons → 42 v888 targets, 207 unique).

### 3.5 Data hygiene inside the walk

- Untyped labels (`Unknown`, empty, bare numbers) never become bridge
  nodes or targets (the BANC `Unknown` convention hub stayed out).
- FAFB `additional_type(s)` cells listing several names keep only the
  in-use ones (a FAFB primary, a male-cns type name, or a `flywireType`
  cell value) when the cell names any used name — e.g. `AVLP011` keeps
  `aSP8b` and drops the unused `AVLP012`; fully-unused cells stay
  (indistinguishable). The verified `LMTe01, CL125` cell on APDN3 rows
  survives intact.

### 3.6 Terminal evidence hops and the cross-reference rule (2026-09-09)

Two walk guards close the `l-LNv → BM_*` derivation class (a FAFB
circadian type fanning out to BANC's 1,212-neuron `BM_InOm` and friends
with zero row-level support on the reached types):

- **Terminal evidence hops** — `TERMINAL_LINKER_COLUMNS` (the curated
  BANC label columns, `banc_release_crosswalk`, `release_alias`) carry
  the reached type's own row-level evidence. When such a hop ARRIVES in
  the target namespace, the derivation ends there; the walk never
  continues into that namespace's annotation graph. When the hop lands
  in a licensed intermediate instead (BANC `malecns_cell_type 'MDN'` →
  MCNS `MDN` —flywireType→ FAFB `MDN`), the licensed hub leg still
  continues, and an annotation hop directly after a terminal hop is
  refused (`ann_chained`).
- **Cross-reference rule** — the primary→alt emission skips alt tokens
  that are themselves primaries of the same namespace. A primary's
  annotation cell naming another primary is a cross-reference (BANC's
  `Alternative Cell Type(s)` concatenates the other datasets' curated
  labels), not a rename; hopping to it as a name node only fed the
  name-graph wander. Same-name identity stays handled by the membership
  block, and the designed FAFB `aT` → BANC `ACT` two-linker standard is
  untouched (its tokens are not primaries of the landing namespace).
- **Zero-evidence chains are dropped at pooling** — safety net on top of
  the licensing: `ui.neuron_index.chain_is_supported` refuses a chain
  whose target-home linkers ALL pooled zero rows on the reached type
  (a measured zero). Same-name/crosswalk-arrival chains (full-population
  pools) and unmeasured sides pass; the panel and the viewer log the
  drop instead of rendering an unsupported derivation.
- `get_mapping_origins` reports the one-linker descriptor of a 2-hop
  bridge (e.g. `malecns_cell_type via 'MDN'` for a curated label pair)
  instead of dropping it.

## 4. Production resolution precedence

`_build_type_mappings` + the release/BANC overlays +
`_apply_annotation_bridge_overlay` fill the
stored mappings consumed by `get_mapped_type`:

1. **BANC label/release overlays** — direct curated label votes and the
   root relation fill their release-local slots; conflicts are never guessed.
2. **Crosswalk routes** (male-cns anchored, per release) — win when
   present; several `flywireType` names per male-cns type become 1-to-N
   conflicts, never guesses.
3. **Transitive mappings** — each release's male-cns anchor gains the
   anchor's other targets.
4. **Annotation-bridge overlay** — same-name identity, then the
   annotation bridge (exactly one candidate maps; several become a
   1-to-N conflict with `origin` provenance). BANC↔BANC annotation paths
   are skipped.
5. Exports state the derivation: `mapping_origin` in
   `auto_type_mapping.csv`, `origin` in the conflicts CSV.

## 5. Shared backend and surface parity

`mapped_type_targets()` resolves one foreign type as the **union** of
(a) the alias/stored resolution (`get_alias_candidates`: rename, same
name, refused N-to-1 members) and (b) the derivation-bridge ends
(`get_type_bridges`). Both UI surfaces consume it:

- **'See available neurons' viewer** — `enrich_native_type_matches`
  annotates every covered foreign type; the mapped-type view lists the
  selected dataset's neurons of the mapped names.
- **Cross-dataset Type Mapping panel** — `_compute` seeds
  `origin_seeded_flows` with the resolved origins, then derives the
  summary from the same backend.

Parity contract (pinned by
`tests/core/test_type_mapper_real_datasets.py::test_panel_and_viewer_share_mapped_type_backend`):
`circadian_clock` resolves to 21 FAFB types / 242 neurons → 40 male-cns
targets / **219 unique neurons on BOTH surfaces**. The panel no longer
sums per-flow counts — shared targets (`s-LNv`, `5thsLNv_LNd6` hit by two
flows, `SMP227` by three) are counted once; the old Σ-with-multiplicity
reported 243.

## 6. BodyId pooling and the pool fix

`pool_bridge_body_ids` pools, per standardized linker, the bodyIds of the
home side's endpoint rows whose linker column carries the linker value;
an unconstrained side pools the FULL endpoint type (crosswalk-arrival and
bare same-name chains name that side's identity without a per-side cell).
For BANC's curated label bridges, `fafb_match` and `malecns_match` are
optional provenance diagnostics only. A locally observed match may be
counted as verified or flagged as a conflict, but it never filters a
type-label bridge and never joins bodyIds across datasets. Each endpoint
therefore reports its own coverage pool, even when one type covers only a
subset of the other type's population. Multiple linkers union independent
home-side candidates. `mancBodyid` remains a release-local metadata field
for its existing MANC evidence, not a requirement for general type mapping.
The virtual `banc_release_crosswalk` linker is the corresponding exception
for BANC v626↔v888: it reads the complete `root_626`↔`root_888` relation,
filters both sides by their requested type, and preserves repeated roots.
For example, the real `L5` bridge reaches 1,655 of 1,683 BANC v888 rows
while retaining 1,651 v626 roots.

Pool dictionaries used by the Type Mapping panel are keyed by
`(source_dataset, target_dataset, source_type, foreign_type)`. This keeps
same-named types and opposite directions independent; renderers retain a
two-part-key fallback for older programmatic callers and tests.

Pool fix: the crosswalk-arrival linker's value is the **cell content**
(`via`) — the reverse leg's hop value is the arrival (male-cns) type
name while the cell carries the foreign token (MCNS rows typed
`5thsLNv_LNd6` carry `flywireType` cells `s-LNv_a,LNd_a`). Matching the
arrival name emptied the target pool and blanked the coverage of rows
like `5th-LNv → 5thsLNv_LNd6`. Post-fix all 44 circadian pairs pool.

`granularity` ("n to m") remains a compatibility summary, while the pool
now carries independent `source_coverage` and `target_coverage` strings
(`covered <pool> of <endpoint type total> (<pct>)` — the shared
`format_coverage` formatter with thousands separators and a one-decimal
share). The historical `coverage` field remains the target-side alias for
callers that still read it, and an unmeasurable side yields `""`.

Per-side pool STATE is explicit since 2026-09-09: `source_basis` /
`target_basis` distinguish `linker rows` (a linker measured a subset),
`full population` (the unconstrained fallback), `release relation
participants`, and `unmeasured` (the side's coverage index was
unavailable — never rendered as a measured zero). The same numbers are
machine-readable as `source_pool_size` / `source_type_total` /
`target_pool_size` / `target_type_total` (totals `None` when
unmeasured), and per-linker `body_ids` are order-deduplicated so card
counts equal pool counts. Renderers go through `format_pool_side`, which
maps the four states to `FB: 1,655 of 1,683 (98.3%)` / `FB: all 1,683` /
`FB: not measured` / `not pooled` so a subset, a tautological full
population, and an unmeasurable side can never look alike. Panel text
and CSV output make clear that these are coverage counts, not
bodyId-to-bodyId pairings. The pool also carries `source_type_body_ids`
/ `target_type_body_ids` — the FULL population of each mapped type in
its own dataset (sorted, independent of the linker-filtered subsets),
exported as the mapping CSV's `source_body_ids` / `target_body_ids`
columns; listed per type per side, never paired across datasets.

## 7. Visualization contract

- **Panel result presentation** (user 2026-09-07, refreshed 2026-09-09):
  the Type Mapping panel's results are presentation-refined around the
  FAFB `APDN3` ↔ male-cns example.  `build_type_coverage` derives two
  views over the mapped pairs, rendered as a TOP-LEVEL "Type coverage"
  expansion PER dataset pair — placed directly above its pair card and
  named with the pair (`Type coverage — <source> → <target>
  (bidirectional, 1-to-N / N-to-1)`), scoped to
  that pair's flows: the FORWARD view (one row per queried type: its
  neuron count, the target types, `1-to-N`/`1-to-1`, the queried type's
  TOTAL mapped number and the target-side total as `x of y (pct)`
  bodyIds) and the BACKWARD view (one row per receiving type: the source
  types mapping onto it, `N-to-1` when several converge — three FAFB
  circadian types onto male-cns `SMP227` — with both sides' coverage).
  Both tables name their coverage columns by DATASET (`<dataset> side
  (bodyIds)`; user 2026-09-09: "source/target side" read as flipped in
  the backward view, so the dataset name replaces the side words).  A
  side's
  denominator is the population of the endpoint types involved (types
  are disjoint body sets, so a 1-to-N row's target total is the summed
  population of all mapped target types); a side whose pools were
  unmeasurable renders `not measured`, never a fake `0 of n`.  The
  per-pair cards label both sides ("12 FAFB → 4 MCNS"), annotate every
  linker in Map used with its own pooled bodyId count (pooling now
  covers both rendered chains), and show per-side BASIS-AWARE pool
  coverage (`FB: 1,655 of 1,683 (98.3%)` / `FB: all 1,683` /
  `FB: not measured` / `not pooled`); long cells wrap within capped
  column widths so every column stays visible.  The summary strip
  splits received vs issued mapped neurons.
- **One shared per-pair weight** (`pair_flow_weight`, user 2026-09-07):
  the Sankey ribbon, the network pair edge, the linker-path edges and
  the composed-graph edges all draw the SAME number for a mapped pair —
  per side the pooled bodyId count when pooled, else that side's neuron
  count, collapsed by `min`.  Before, the network duplicated the SOURCE
  type's whole count onto every edge (all edges of a 12-neuron type
  showed "12"), the linker graph fell back foreign-first, and only the
  pooled Sankey agreed — the same pair showed a different number in
  each artifact.
- **Pool hover counts are the UNION across pairs** (user 2026-09-07):
  `_endpoint_pool_counts` uniques the pooled bodyIds per type; an N-to-1
  target pools a different disjoint subset per counterpart (male-cns
  CL125/SLP249/PLP080/SLP250 pool 4/4/2/2 of FAFB APDN3's 12 bodyIds),
  and the old max-across-pairs hovered "pool 4 bodyIds" next to
  "12 neurons" — the union reports the 12 the mapping actually reaches.
- **Sankey edge keys are layer-less** (user 2026-09-07): one pair's two
  derivation chains (direct annotation bridge + crosswalk chain) reach
  the shared band at different hop depths; keyed by layer, `create_sankey`
  drew the shared node pair as PARALLEL ribbons with the weight counted
  twice (PLP080 → APDN3 twice).  Keyed by (source, target), same-pair
  links merge with max — one biological connection, one ribbon.
- **Adjustable edge-label size** (user 2026-09-07): the on-edge weight
  labels were pinned to 9px; the network control panel now carries an
  "Edge Label Size" spinner (`edge_label_font_size`, default 9) wired
  through the undo history, independent of the node-label Font Size.
- **Linker network** (`build_bridge_linker_graph`): one column per
  bridge linker COLUMN, in canonical first-appearance order (male-cns↔FAFB:
  `type | flywireType | additional_type(s) | type` = four columns).
  Layering by per-chain hop order (the old behavior) let a 1-linker
  chain's `additional_type(s)` node share the `flywireType` column and
  let a shared linker node's position be overwritten by the last chain.
- **Unified header legends**: dataset chips render `CODE: full (count)`
  (`dataset_legend_meta` carries the node count and optional swatch
  override); the dynamic group chip for a code with a static chip is
  skipped (the duplicated `FAFB (65)` / `FAFB: flywire_FAFB_v783` header
  rows are gone); linker columns join the same header row with their
  `LINKER_COLORS` swatch. The Sankey note uses the same chip shape.
- **Export mapping CSV** (user 2026-09-09, replaces the old "Export
  bridges (CSV)"): buttons `Export mapping` (per pair) and
  `Export mapping — all pairs (CSV)`; filenames `mapping_*.csv`.  One
  FIXED column set for every pair — `source_dataset, source_entry,
  matched_column, source_type, target_dataset, target_type,
  relationship, source_neurons, target_neurons, bridge, bridge_columns,
  mapping_origin, source_pool, source_total, target_pool, target_total,
  pool_coverage, pool_coverage_basis` — so the all-pairs file is a plain
  header + rows concatenation (the old per-pair pivoted
  `bridge-<column>` fields needed a union-of-columns hack to avoid
  ragged rows).  `source_entry` is the value that matched on the source
  side (the type itself when matched via `type`, otherwise the label
  value); `relationship` is derived 1-to-1/1-to-N per source type; the
  numeric pool/total columns make coverage machine-readable while
  `pool_coverage` stays the human-readable `source covered …; target
  …` field.  `source_body_ids` / `target_body_ids` carry the FULL
  per-type populations (every bodyId of the mapped type in its own
  dataset, ';'-joined — user 2026-09-09): listed per type per side,
  never a bodyId-to-bodyId pairing.  The export never represents bodyId
  pairings.

## 8. Testing matrix

| suite | pins |
|---|---|
| `tests/core/test_type_mapper_source_map.py` | declarative licensing vs the tables, per-pair sweeps |
| `tests/core/test_type_mapper_bridge_rules.py` | the algebra: reverse crosswalk legs, connector licenses, BANC ban, no-flip order, untyped exclusion, label-hop terminality + primary-valued-alt refusal (the `l-LNv → BM_*` regression), the designed `aT`→`ACT` standard, real-data acceptance |
| `tests/core/test_type_mapper_annotation_bridge.py` | overlay precedence, exports, release-name resolution |
| `tests/core/test_type_mapper_real_datasets.py` | circadian parity (panel == viewer, 219 unique), linker layout + header legend chips, direct BANC label routes, two-linker cap, APDN3 pair weights == Sankey ribbons, APDN3 pool-union hover, Sankey no parallel links, edge-label size control |
| `tests/core/test_banc_release_and_mcns_version.py` | BANC label votes/verification, auto-label exclusion, duplicated root relation, MCNS v0.9 alias/native fallback |
| `tests/core/test_dataset_release_registry.py` | shared recommendation policy and unavailable-release behavior |
| `tests/ui/test_dataset_release_notice.py` | explicit single/multi selector recommendation action and suppression |
| `tests/core/test_type_mapping_composed.py` | mapping-CSV fixed-width contract (`source_dataset`…`pool_coverage_basis`), `format_coverage` states, `not measured` coverage rows, shared `pair_flow_weight` formula, pool-count union, forward 1-to-N + reverse N-to-1 coverage rows |
| `tests/ui/test_alias_matches.py` | viewer enrichment, mapped-type view, pool granularity |

Probes under `local_data/`: `repro_two_flows.py` (surface parity),
`probe_t2_t3.py` (version control + used-name filter),
`probe_circadian_banc888.py` (FAFB→BANC v888 coverage),
`bridge_rules_probe.py` (the algebra sweep).

## 9. Known limits

- `optic-lobe` and `manc:v1.2.3` are not in the male-cns crosswalk table
  and therefore have no verified mappings (same-name only).
- BANC `fafb_alignment_cell_type` remains search/alignment-only and
  `fanc_cell_type` remains deliberately unlicensed.
- MCNS v0.9 is intentionally name-aliased to v1.0 only for shared primary
  names. Its body IDs remain native, and v0.9-only fallback labels are lower
  tier than the certified v1.0 crosswalk.
- Multi-candidate evidence is never guessed: 1-to-N conflicts are
  exported for manual adjudication.
