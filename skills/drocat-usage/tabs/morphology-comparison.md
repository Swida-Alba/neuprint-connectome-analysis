# Morphology · Comparison (morphology_comparison)

Reproduce the **Morphology tab → Comparison sub-tab** as a direct backend
call: an intra-dataset N×N morphology comparison of 2+ queried neurons
(types, bodyIds, or patterns). Produces a bodyId-level similarity matrix, a
type-level aggregation (mean over cross-member pairs; diagonal = intra-type
cohesion), heatmaps, and a report.

## Backend contract

- **tool_key:** `morphology_comparison`
- **import:** `from morphology_comparison import MorphologyProfileComparer`
- **class:** `MorphologyProfileComparer` (var `comparer`)
- **method:** `comparer.run()`

## Parameters the UI builds

```python
from morphology_comparison import MorphologyProfileComparer

comparer = MorphologyProfileComparer(
    dataset="male-cns:v1.0",             # ONE dataset (intra-dataset only; no BANC)
    query=["aMe12", "aMe10", "aMe.*"],   # types, bodyIds, or regex patterns
    method="vector_v2",                  # "vector_v2" (default) | "nblast"
    max_members_per_type=25,             # members sampled per type
    max_total_neurons=200,               # safety cap (NBLAST: hard limit 30)
    fetch_online=True,                   # pull missing skeletons via API (NeuPrint SWC / FAFB bundle→CAVE)
    output_dir="/absolute/output/morph_cmp",
    saveas="",
    generate_heatmaps=True,
    show_figures=False,
    verbose=True,
    n_workers=8,
    use_cache=True,
)
results = comparer.run()
```

## Run

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_MorphCompare_<date>.py
```

## Outputs

- `type_level/type_similarity_{method}.csv` — type×type matrix (diagonal =
  intra-type cohesion).
- `bodyid_level/bodyid_similarity_{method}.csv` — every individual pair.
- `members.csv` — resolved population with per-neuron status
  (`compared` / `no vector` / `no dotprops`).
- `visualization/heatmap_*.html`, `report.html`, `parameters.json`,
  `README.txt`.

## Notes

- **Intra-dataset only**: scores live in one dataset's coordinate space
  against that dataset's caches. Cross-dataset comparison belongs to the
  connectivity side (`HomologFinder` / `ConnectivityProfileComparer`).
- `method="nblast"` requires ≤ 30 total neurons and local raw skeletons
  (online fetch only happens through `MorphologyComparer`'s shared
  dotprops pipeline); it excludes contralateral pairs from type means.
- `vector_v2` uses the per-dataset `SkeletonVectorCacheV2`; missing members
  are fetched online by default (`fetch_online=True`) through the same
  skeleton pipeline as Find Similar (NeuPrint raw SWC; FAFB healed zip
  → CAVE/local-fix fallback) and persist into the shared cache. Set
  `fetch_online=False` for a strictly offline comparison.
- Each row is one TYPE: bodyId queries resolve to their type; patterns
  (`aMe.*`) expand against the dataset's type names (exact names always
  win over pattern interpretation).
