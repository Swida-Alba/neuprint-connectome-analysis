# BANC Data Integration

This project supports the BANC (Brain And Nerve Cord) connectome using its
**public release bucket** — no FlyWire/Codex login, no CAVE token, and no
manual downloads are required. Use the exact dataset identifier that matches
the release: `banc_v626` or `banc_v888`. (Legacy `flywire_BANC_v626` /
`flywire_BANC_v888` spellings are accepted everywhere and resolve to the same
folders and caches.)

## What is supported

| Workflow | Status |
|---|---|
| Connectivity / pathfinding / network analysis | ✅ (local parquet tables) |
| **3D skeleton visualization** (native BANC space) | ✅ (public-bucket SWCs) |
| BANC ROI / neuropil meshes + CNS outline template | ✅ (public `region_outlines` layer) |
| Morphology find-similar / NBLAST-style similarity | ❌ deferred |
| CAVE API queries (`brain_and_nerve_cord`) | ❌ token-gated, not used |

## Data Preparation (automatic)

Selecting `banc_v888` or `banc_v626` in any tool prepares the dataset
automatically from the public bucket (`~134 MB`, one time):

- **Neuron metadata** from `compiled_data/banc_888/banc_888_meta.feather`
  (188,508 neurons; ids, curated `cell_type`, `Alternative Cell Type(s)`,
  classes, neurotransmitters, proofread flags).
- **Connections** from `neuron_connectivity/<version>/synapses_v1_..._
  connectioncountsperneuropil_countthresh3.parquet` (per-neuropil pair
  counts; synapse size >= 3 and connection count >= 3 thresholds are baked
  into the product — matching what the Codex download provided).

The raw products stay in `datasets/<dataset>/downloads/` so re-runs are
offline. The written `metadata.json` records `source: banc_public_gcs`.
Preparation also fills the neuron table's `post` column (post-synaptic
counts) from the merged connections, matching what the manual Codex path
produced.

### Manual downloads are no longer needed

The bucket path fully replaces the old manual Codex downloads
(`neurons.csv.gz` + `connections_princeton.csv.gz`).  The converter keeps a
legacy compatibility path for those files, but no workflow requires them.

### Migrating a pre-rename checkout

Datasets prepared before the `flywire_BANC_*` -> `banc_*` rename are
migrated with:

```bash
python scripts/migrate_banc_dataset_names.py [--dry-run]
```

Old spellings keep working even without the migration (every dataset-name
entry point canonicalizes via `utils.naming_utils.canonical_dataset_name`).

## 3D Skeleton Visualization

BANC renders in **native BANC space** (nanometres). Skeletons are the
official release SWCs, fetched on demand from

```
compiled_data/banc_888/banc_banc_space_swc/<root_888_id>_<resolution>.swc
```

- One file per neuron: `_skeleton.swc` (full resolution, proofread neurons)
  or `_l2.swc` (coarse L2 approximation for everything else). A 404 on the
  preferred resolution transparently falls back to the other.
- `banc_v888` body ids resolve directly; `banc_v626` ids resolve through the
  meta-feather crosswalk (`root_626` -> `banc_888_id`), reaching **99.1%**
  coverage (v888: 99.3%). The remaining ~1% are tiny fragments with no
  skeleton anywhere; they are skipped with a warning.
- Fetched skeletons cache into the shared raw skeleton store
  (`cache/<dataset>/skeletons/raw_skeletons/<id>.swc.zst`, raw level 0 with
  a provenance header that also records which source served the neuron),
  so subsequent renders are instant and offline.
- Every preset view (interactive dropdown, PNG view export, individual
  plots, video) shares one per-dataset camera table, and saved HTML opens on
  the calibrated BANC frontal view (anterior at -Y) on every export path.

### Skeleton source chain (unified)

There is no per-run source selection: every fetch walks one chain and uses
the first source that exists for the neuron —

1. **888 L2** — `compiled_data/banc_888/banc_banc_space_swc/{id}_l2.swc`
   (nanometres; the cache-level product for ~most neurons),
2. **888 full-resolution** — `{id}_skeleton.swc` (proofread neurons; serves
   the few neurons that have no L2 at all, e.g. one aMe12),
3. **v626 pcg-skel** — `neuron_skeletons/swcs-from-pcg-skel/{root_626}.swc`
   (micrometres, scaled ×1000 on fetch; v888 ids are translated through the
   id crosswalk). This covers the neurons absent from the 888 export in
   both resolutions (~1.7 % of the release, e.g. one aMe12 and one l-LNv
   body).

The old **BANC Skeleton Resolution** select (`l2` / `full` / `full_auto`)
was removed from the Skeleton tab — the chain is always L2 → full → pcg.
The deprecated `banc_skeleton_resolution` constructor argument is still
accepted (validated, ignored) for script compatibility.

Rendered tube radii are normalized per neuron (median → a shared 240 nm
target by default; see below). L2 tubes are the cache-level product and
count as ~90% simplified against the full-resolution skeletons: they skip
face decimation unless the asked `skeleton_mesh_simplification` exceeds
that baseline — the excess is scaled onto the L2 density (asked 0.95
removes 50% of L2 tube faces; asked 0.99 removes 90%).
The Skeleton tab's mesh-simplification default stays the FAFB-style 0.90:
it drives the full-resolution sources, and a 4,000-face floor bounds
full-resolution decimation at extreme slider values.

**id namespaces**: `root_626` and `root_888` coincide for ~61 % of the
release (115,737 of 188,508 rows) and differ for the rest; the meta-feather
crosswalk covers 100 % of neurons in both directions, so a v888 body id
always reaches its v626 pcg-skel twin and vice versa.

**Tube radius normalization**: the release products carry inconsistent
radius calibers per neuron (median radii range ~50–172 nm), which shows up
as mismatched tube thicknesses in one scene. The Skeleton tab exposes a
**Normalized Tube Radius** checkbox (default on) with a **Radius Target**
(240 nm median by default): each neuron's radii are rescaled so its median
hits the target, preserving relative branch thickness. Uncheck to use the
raw radii.

## Synapse markers

BANC defaults to skipped synapses. Opting in (any synapse mode other than
`skip`) triggers a one-time ~3.9 GB download of the release per-synapse
table into `datasets/<dataset>/downloads/` (resumable; subsequent renders
are local), then draws **pre-synaptic site markers** at the T-bar
positions — the release publishes only pre-site coordinates, so the
paired cone/sphere markers used for FAFB/NeuPrint are not available. The
UI shows an explicit warning when synapses are enabled for a BANC
dataset. The `pre-post sites` mode is not supported for BANC.

## Pathfinding & connectivity

Connectivity comes from the bucket-prepared, **per-version** local tables —
`banc_v626` and `banc_v888` are distinct datasets with distinct id spaces
(~61 % of ids coincide across releases and are the same neurons; the rest
are remapped; the crosswalk covers 100 % of both directions). FindPath /
FindAllPath read the per-neuron connection cache, which **rebuilds
automatically** from `<dataset>_merged_connections.parquet` whenever the
table is newer — no manual cache step and no CAVE token. Notes:

- BANC is not on the NeuPrint server: pathfinding never contacts it, and
  `use_cache=False` (online-only mode) is unsupported for BANC and raises a
  clear error.
- ROI-based path filters are unavailable (the release ships no primary ROI
  list); `roi_coverage` in the metadata sidecar is marked
  `not_available_for_banc`.
- `metadata.json` statistics are regenerated from the tables on every
  preparation (neuron counts, real type coverage excluding `'Unknown'`,
  synapse totals).

## ROI meshes, brain and VNC outlines

The public `region_outlines` CloudVolume layer provides four aggregates
(`BANC_outline`, `BANC_neuropil`, `BANC_brain_neuropil`, `BANC_vnc_neuropil`),
all in nanometres — the same frame as the skeletons, so they render without
any transform. The release publishes **no named ROI meshes**, so named ROIs
(e.g. `AL(R)`) use the same handling as FAFB: fetched from **male-cns**
(`JRCFIB2022Mraw`) and bridged into BANC space at render time, cached under
`cache/<ds>/meshes_transformed/BANC/`. The offered ROI list is the male-cns
set plus the four native aggregates. `brain_mesh='native'` (or the explicit `brain_mesh='BANC'`) draws the
**brain portion** of the whole-CNS outline and `vnc_mesh=True` the **VNC
portion**: the two are segmented by the neck coordinate
(`y = 470 µm` waist; cut at `y = 350 µm` so the neck stays with the VNC),
mirroring the male-cns geometric extraction. Boundary faces are assigned by
centroid so brain and VNC meet without a crack.

## Usage

```python
from coana import FindNeuronConnection

fnc = FindNeuronConnection()
fnc.dataset = 'banc_v626'
fnc.sourceNeurons = ['720575940596125868']
fnc.targetNeurons = ['720575940597856265']
fnc.InitializeNeuronInfo()
fnc.FindAllPath()
```

```python
from visualize_skeleton import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset='banc_v888',
    neuron_layers=[['l-LNv']],
    brain_mesh='native',
    mesh_roi=['BANC_neuropil'],
    skip_synapse=True,
)
vs.plot_neurons()
```

## Notes

* **Root IDs** are very large integers; DROCAT keeps them as exact strings.
* **`type` coverage**: ~75% of neurons carry a curated `cell_type`; the rest
  are `'Unknown'` (same convention as the Codex tables). The
  `Alternative Cell Type(s)` column aggregates the curated name plus the
  per-dataset cross-fly type matches and remains fully wired into the
  cross-dataset auto type mapper (which reads the CSV form of the table).
* **Synapse coordinates**: BANC per-synapse tables are nanometres; the local
  merged-connections table aggregates weights per neuropil, so precise
  synapse-site rendering for BANC is a future item.
* The v626 metadata carries a `root_626` -> `banc_888_id` crosswalk (cached
  at `cache/<dataset>/banc_id_crosswalk.parquet`); skeletons are always
  fetched by their 888-namespace file name.
