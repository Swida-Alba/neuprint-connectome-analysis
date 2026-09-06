# Auto Type Mapping — Implementation Report

*Technical reference for the cross-dataset auto type mapping engine as of
2026-09-07. Companion to the user-facing `docs/AUTO_TYPE_MAPPING.md`; this
report documents the implementation: components, the bridge-rule algebra,
the derivation walk, resolution precedence, the shared UI backend, pooling,
and the visualization contract.*

## 1. Components and data flow

```
male-cns v1.0 neuron_df (crosswalk table)
        │  type / flywireType / hemibrainType / mancType cells
        ▼
CrossDatasetTypeMapper                      src/comparison/cross_dataset_type_mapper.py
  ├─ _build_type_mappings()        stored 1-to-1 mappings + N-to-1 / 1-to-N conflicts
  ├─ _load_flywire_type_tables()   per-release FAFB/BANC primary + additional-name tables
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
| male-cns v0.9 | `male-cns:v0.9` (own, empty) | — the v1.0 crosswalk cannot verify v0.9 names |
| FAFB v783 | `flywire_FAFB_v783` | `flywire_FAFB_v783_allneurons_neuron_df.csv` |
| BANC v626 | `banc_v626` | `banc_v626_allneurons_neuron_df.csv` |
| BANC v888 | `banc_v888` | `banc_v888_allneurons_neuron_df.csv` |

Consequences (user directive, 2026-09-07):

- A `banc_v888` selection resolves **only** against the v888 tables —
  v626 names ("via banc v626") and v626 bodyId pools are impossible.
- `male-cns:v0.9` never silently borrows the v1.0 crosswalk; without its
  own crosswalk it simply has no verified mappings (same-name identity
  still works).
- No BANC↔BANC cross-release mapping: the overlay skips those pairs and
  no licensed bridge form connects the two releases.

## 3. The bridge-rule algebra (derivation walk)

The derivation walk (`get_type_bridges`) expands a source type through
licensed evidence edges and records every simple path that lands on a real
primary `type` of the target dataset. Every edge is licensed by
`BRIDGE_SOURCE_MAP` (declarative; verified against the tables at load).

### 3.1 Direct bridge forms (plus same-name identity everywhere)

```
HEMI  --hT--  MCNS                        MANC --mT-- MCNS
MCNS  --fT--aT--  FAFB   |  MCNS --fT-- FAFB  |  MCNS --aT-- FAFB
MCNS  --(fT)--ACT--  BANC |  MCNS --ACT-- BANC |  MCNS --(fT)-- BANC
FAFB  --aT--ACT--  BANC  |  FAFB --aT-- BANC    |  FAFB --ACT-- BANC
```

`hT/mT/fT` = hemibrainType/mancType/flywireType cells on the male-cns rows;
`aT` = FAFB `additional_type(s)`; `ACT` = BANC `Alternative Cell Type(s)`.

### 3.2 Connector licenses (`ROUTE_MIDS`)

A chain visits at most ONE intermediate namespace, and only from the
licensed set:

- male-cns connects {HEMI, MANC} ↔ {FAFB, BANC} and HEMI ↔ MANC;
- FAFB connects male-cns ↔ BANC;
- **BANC is never a connector** (never between male-cns and FAFB);
- every other pair (male-cns↔FAFB, FAFB↔BANC) is direct-only.

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
derives through `flywireType '<name>'` (metadata verification), not the
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

## 4. Production resolution precedence

`_build_type_mappings` + `_apply_annotation_bridge_overlay` fill the
stored mappings consumed by `get_mapped_type`:

1. **Crosswalk routes** (male-cns anchored, per release) — win when
   present; several `flywireType` names per male-cns type become 1-to-N
   conflicts, never guesses.
2. **Transitive mappings** — each release's male-cns anchor gains the
   anchor's other targets.
3. **Annotation-bridge overlay** — same-name identity, then the
   annotation bridge (exactly one candidate maps; several become a
   1-to-N conflict with `origin` provenance). Overlay pairs are the
   flywire-family pairs with a BANC endpoint; BANC↔BANC is skipped.
4. Exports state the derivation: `mapping_origin` in
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
`mancBodyid` joins male-cns↔manc directly (99.7%).

Pool fix: the crosswalk-arrival linker's value is the **cell content**
(`via`) — the reverse leg's hop value is the arrival (male-cns) type
name while the cell carries the foreign token (MCNS rows typed
`5thsLNv_LNd6` carry `flywireType` cells `s-LNv_a,LNd_a`). Matching the
arrival name emptied the target pool and blanked the coverage of rows
like `5th-LNv → 5thsLNv_LNd6`. Post-fix all 44 circadian pairs pool.

`granularity` ("n to m") and `coverage` ("covered n of m") carried the
same two numbers — surfaces now show ONE `pool_coverage` column
(`covered <target pool> of <target type total>`), filled even when one
side's pool is empty. The pool dict keeps `granularity` for API
compatibility.

## 7. Visualization contract

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
- **Bridges CSV**: one `pool_coverage` column (was `granularity`,
  `coverage`).

## 8. Testing matrix

| suite | pins |
|---|---|
| `tests/core/test_type_mapper_source_map.py` | declarative licensing vs the tables, per-pair sweeps |
| `tests/core/test_type_mapper_bridge_rules.py` | the algebra: reverse crosswalk legs, connector licenses, BANC ban, no-flip order, untyped exclusion, real-data acceptance |
| `tests/core/test_type_mapper_annotation_bridge.py` | overlay precedence, exports, release-name resolution |
| `tests/core/test_type_mapper_real_datasets.py` | circadian parity (panel == viewer, 219 unique), linker layout + header legend chips, DNp50 crosswalk-verified route, two-linker cap |
| `tests/core/test_type_mapping_composed.py` | bridges CSV `pool_coverage` contract, uniform widths |
| `tests/ui/test_alias_matches.py` | viewer enrichment, mapped-type view, pool granularity |

Probes under `local_data/`: `repro_two_flows.py` (surface parity),
`probe_t2_t3.py` (version control + used-name filter),
`probe_circadian_banc888.py` (FAFB→BANC v888 coverage),
`bridge_rules_probe.py` (the algebra sweep).

## 9. Known limits

- `optic-lobe` and `manc:v1.2.3` are not in the male-cns crosswalk table
  and therefore have no verified mappings (same-name only).
- `male-cns:v0.9` has no crosswalk of its own; giving it one is a data
  task, not a code task.
- Multi-candidate evidence is never guessed: 1-to-N conflicts are
  exported for manual adjudication.
