# FAFB (FlyWire) Data Integration

This project supports the FAFB (Full Adult Fly Brain) dataset from FlyWire using a local file-based approach for high performance.

## Data Preparation

We provide a dedicated script to prepare the FAFB data for analysis. This script handles file organization, format conversion (to Parquet), and data enrichment.

### 1. Download Data

Download the following files from [FlyWire Codex Downloads](https://codex.flywire.ai/api/download?dataset=fafb):

**Required for local connection analysis:**
*   `classification.csv.gz` (Neuron Classification) - **~1 MB**
*   `connections_princeton_no_threshold.csv.gz` (Connectivity) - **~263 MB**
    *   *Note: `connections_princeton.csv.gz` or `connections.csv.gz` are also accepted as fallbacks.*

**Recommended metadata enrichment (optional):**
*   `consolidated_cell_types.csv.gz` (Consolidated Cell Types) - **~1 MB**
*   `names.csv.gz` (Neuron Names) - **~1 MB**
*   `coordinates.csv.gz` (Soma Coordinates) - **~5 MB**
*   `neurons.csv.gz` (Neurotransmitters) - **~2 MB**
*   `cell_stats.csv.gz` (Cell Statistics) - **~2.5 MB**

The converter warns when enrichment files are missing and continues with
incomplete neuron metadata. It stops only when the classification or
connectivity input is missing.

**Optional Visualization Files:**
*   `sk_lod1_783_healed.zip` (Skeletons - needed for local 3D skeleton visualization) - **~13 GB**
*   `fafb_v783_princeton_synapse_table.csv.gz` (Synapses - needed for local synapse-table visualization) - **~2.5 GB**

**Storage Summary:**
*   **Minimal Analysis:** ~300 MB download → ~200 MB final size
*   **With Synapses:** +2.5 GB download → +1.7 GB final size
*   **With Skeletons:** +13 GB download → +13 GB final size
*   **Full Dataset:** ~16 GB total storage required

### 2. Run the Converter

You can trigger the data preparation in two ways:

**Option A: Run the converter directly (Recommended)**
```bash
# Run from the project root
python src/FAFB_file_converter.py
```

**Option B: Run any analysis script**
Simply selecting `flywire_FAFB_v783` in a tool (or initializing
`FindNeuronConnection` with that dataset) will automatically check for and
convert the data if needed.

The script will check for the required files and guide you if anything is missing.

**What the script does:**
1.  Creates the `datasets/flywire_FAFB_v783` directory structure.
2.  Checks the `datasets/flywire_FAFB_v783/downloads` folder for source files.
3.  If files are missing, it prints a list of what to download and where to put them.
4.  Converts CSVs to optimized **Parquet** files for fast loading.
5.  Merges metadata (names, coordinates, types) into a single neuron dataframe.
6.  Moves the skeleton zip file to the correct location.

### 3. Cleanup (Removable Files)

After the script successfully completes (look for "✓ Conversion complete" messages), you can safely delete the entire `downloads` folder to save space.

**Removable Folder:**
*   `datasets/flywire_FAFB_v783/downloads/` (The entire folder can be deleted)

**Do NOT delete:**
*   The generated `.parquet` files in `datasets/flywire_FAFB_v783/`.
*   The `sk_lod1_783_healed.zip` file in `datasets/flywire_FAFB_v783/` (The converter moves this file out of downloads, so it is safe).

## Usage

### Path Finding

Use `FindNeuronConnection` with the FAFB dataset name.

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from coana import FindNeuronConnection

fnc = FindNeuronConnection(
    dataset='flywire_FAFB_v783',
    sourceNeurons=['l-LNv.*'],  # Regex patterns supported
    targetNeurons=['s-LNv.*'],
    max_interlayer=2,
    min_synapse_num=5,
    verbose_mode='simple',
)

fnc.InitializeNeuronInfo()
fnc.FindAllPath()
```

### Visualization

Visualize skeletons using `VisualizeSkeleton` (or the shorthand `Vis3S`).

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from coana import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    token='',  # Not needed for FAFB local data
    output_dir='/path/to/output',
    neuron_layers=['l-LNv'],  # Neuron types or bodyIds
    skip_synapse=True,
    neuron_alpha=0.2,
    min_synapse_num=3,
    skeleton_mode='tube',
    legend_mode='layer',
    show_fig=True,
    brain_mesh='template',  # Use native FAFB coordinates
    cache_neurons=True,
)

vs.plot_neurons()
```

### CAVE API Fetching (force_API_fetching)

For more up-to-date skeleton data, you can fetch skeletons directly from the CAVE API instead of using the local ZIP file:

```python
from coana import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    output_dir='/path/to/output',
    neuron_layers=['l-LNv'],
    skip_synapse=True,
    neuron_alpha=0.2,
    skeleton_mode='tube',
    legend_mode='layer',
    show_fig=True,
    brain_mesh='template',
    cache_neurons=True,
    force_API_fetching=True,  # Fetch from CAVE API instead of local ZIP
)

vs.plot_neurons()
```

**Key Features:**
- **Source priority (tube mode, caching on)**: prepared CAVE mesh cache → shared raw SWC cache (`.swc.zst`) → healed skeleton bundle → CAVE API. Prepared meshes are CAVE-derived (extrusion-free) and already at the render preparation level.
- **Source priority (line mode)**: SWC-first — raw SWC cache → healed bundle → CAVE API — so every scene renders from reconstruction skeletons; mesh sources would have to be skeletonized in memory with different geometry.
- **force_API_fetching=True**: routes every body straight to the CAVE API (meshes cached in the prepared mesh cache when caching is enabled).
- **Automatic Fallback**: bodies missing from every local source fall through to the CAVE API automatically.
- **Updated Data**: Use `force_API_fetching=True` to ensure you're using the most up-to-date neuron morphologies from the CAVE API.

### Fixing Skeleton Extrusion Issues

The downloaded FAFB skeleton ZIP (`sk_lod1_783_healed.zip`) may contain neurons with extrusion artifacts (mesh errors that appear as spikes or protrusions). These typically occur around the soma (cell body) region when aggressive mesh simplification is applied.

#### Understanding Extrusions

Extrusion artifacts happen because:
1. The soma region has high vertex density in the original mesh
2. When simplification is applied uniformly (e.g., 0.95 = remove 95% of faces), the soma's fine structure collapses
3. This creates "spiky" protrusions extending from the cell body

#### Solution 1: Automatic Extrusion Detection and Fix

Enable `auto_fix_extrusions=True` to automatically detect and replace problematic skeletons:

```python
from coana import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    neuron_layers=['MTe50', 'MTe51', 'MTe54'],
    skeleton_mesh_simplification=0.95,
    auto_fix_extrusions=True,  # Automatically detect and fix extrusions
    cache_neurons=True,
    show_fig=True,
)
vs.plot_neurons()
```

**How auto_fix_extrusions works:**
1. When loading skeletons from ZIP, each is converted to a simplified mesh
2. Edge length analysis detects abnormal "spiky" geometry (edge ratio > 10x median)
3. Problematic neurons are automatically fetched fresh from CAVE API
4. If a CAVE fetch fails, the long parent→child edge is mapped back to the local
   tree and only that child subtree is pruned when the cut is safe
5. **Extrusion check results are cached** in `cache/{dataset}/extrusion_check_results.parquet`
6. On subsequent runs, only new neurons are checked (previously checked neurons use cached results)
7. CAVE replacements are cached; local fallback repairs remain in memory and do not
   overwrite the canonical raw skeleton

**Performance notes:**
- First run may take longer due to mesh analysis for extrusion detection
- Subsequent runs are fast because check results are cached in parquet format
- Set `auto_fix_extrusions=False` if you need faster loading and can tolerate artifacts

#### Solution 2: Soma-Aware Simplification (Built-in)

The visualization system includes **soma-aware simplification** that applies gentler simplification to the soma region:

```python
from coana import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    neuron_layers=[720575940624086675],
    skeleton_mesh_simplification=0.95,     # Aggressive on skeleton branches
    soma_mesh_simplification=0.8,          # Gentler on cell body (default)
    soma_region_radius=20000,              # 20µm radius around soma (default)
    cache_neurons=True,
    show_fig=True,
)
vs.plot_neurons()
```

**Default cache settings:**
- Skeleton simplification: 0.95 (keep 5% of faces)
- Soma simplification: 0.8 (keep 20% of faces) 
- Soma region radius: 20,000nm (20µm)

#### Solution 3: Detect and Fix Extrusions Manually

Use the built-in detection tools to identify problematic neurons:

```python
from coana import VisualizeSkeleton

# Check a specific neuron for extrusions
result = VisualizeSkeleton.check_fafb_skeleton_for_extrusions(
    720575940624086675,
    simplification=0.95,
    verbose=True
)

if result['has_extrusions']:
    print(f"Extrusions detected! Severity: {result['severity']}")
    print(f"Recommendation: {result['recommendation']}")
    
    # Fix it by fetching from CAVE API
    VisualizeSkeleton.fix_fafb_extrusions([720575940624086675])
```

Or use **auto-fix** mode:

```python
# Check AND automatically fix if extrusions found
result = VisualizeSkeleton.check_fafb_skeleton_for_extrusions(
    720575940624086675,
    auto_fix=True,  # Automatically fetch from API if needed
    verbose=True
)
print(f"Auto-fixed: {result['auto_fixed']}")
```

#### Solution 4: Manual API Fetching

For more control, you can manually fetch fresh skeletons via CAVE API:

```python
from coana import VisualizeSkeleton

# Method 1: Fix specific neurons by fetching them via API
# Run with force_API_fetching=True for the problematic neurons only
vs = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    neuron_layers=[720575940596125868, 720575940597856265],  # Problematic bodyIds
    force_API_fetching=True,  # Fetch fresh data from CAVE API
    show_fig=False,  # Just cache the fixed skeletons
    cache_neurons=True,
)
vs.plot_neurons()  # This caches the API-fetched skeletons

# Method 2: Once fixed, run your full visualization
# API-cached skeletons are automatically prioritized over ZIP data
vs2 = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    neuron_layers=['l-LNv', 's-LNv'],  # Mix of neurons
    force_API_fetching=False,  # Use local data, but API cache takes priority
    show_fig=True,
    brain_mesh='template',
)
vs2.plot_neurons()
```

#### Extrusion Detection API

The `detect_mesh_extrusions()` method provides detailed analysis:

```python
from coana import VisualizeSkeleton

# If you have a mesh object already
result = VisualizeSkeleton.detect_mesh_extrusions(
    mesh,
    soma_pos=[100, 200, 300],  # Soma position (optional)
    soma_radius=20000,         # 20µm radius
    verbose=True
)

# Result dictionary contains:
# - 'has_extrusions': bool
# - 'severity': 'none', 'mild', 'moderate', or 'severe'
# - 'extrusion_count': number of problematic vertices
# - 'edge_length_ratio': max/median edge ratio (>3 indicates issues)
# - 'soma_region_issues': bool (extrusions near cell body)
# - 'recommendation': suggested action string
```

**How Extrusion Fixes Work:**
1. CAVE replacement meshes are cached in the prepared mesh cache
   (`FlyWireMeshCache`); the legacy `cache/{dataset}/API_cache/skeletons/`
   pickles are read only as a migration fallback
2. Extrusion check results are cached per neuron in
   `cache/{dataset}/extrusion_check_results.parquet`
3. TreeNeuron sources (bundle / raw SWC cache) run the extrusion check every
   render; flagged neurons are replaced through the CAVE API
4. This allows you to selectively fix problematic neurons without re-downloading the entire 13GB ZIP
5. Fixed neurons persist across sessions via the cache

**Note on force_API_fetching Behavior:**
- **VisualizeSkeleton**: Prioritizes prepared caches even when
  `force_API_fetching=False`; `force_API_fetching=True` bypasses them
- **FindNeuronConnection**: Uses API only when `force_API_fetching=True` (for consistency with local data)

**Requirements:**
- CAVE token for API fetching (obtain from https://codex.flywire.ai/auth_token)
- Set token in `config.json` (or the gitignored `config_local.json` fallback) or as environment variable `CAVE_TOKEN`

**Note:** BANC is a separate public-release source and does not support
`force_API_fetching`; see the [BANC Integration Guide](BANC_INTEGRATION.md).

## Find Similar (Morphology) on FAFB

- **vector_v2** uses the skeleton-vector cache built from the healed bundle
  (`find_similar/morphology/skeleton__vectors_v2.parquet`). The prepared-mesh
  cache is the V1 counterpart and is not used by vector_v2 scoring.
- **Skeleton loading follows the canonical FlyWire chain**
  (`morphology.load_flywire_skeletons_batch`): raw `.swc.zst` cache → healed
  bundle (newly served trees are cached into the raw store) → per-run
  extrusion check with cached results (flagged neurons replaced through
  CAVE) → token-gated CAVE skeletonization. The prepared mesh cache is
  never consulted — scoring is TreeNeuron-native.
- **NBLAST** scores the entire candidate pool from skeletons — the local raw
  store first, healed-bundle fallback — so pool coverage does not depend on
  vector-cache rows.
- **NBLAST is not mirror-invariant**: contralateral same-type pairs score at
  chance. Type-level means therefore aggregate ipsilateral pairs only,
  classified via the neuron index's `hemisphere` column (`somaSide` on
  NeuPrint datasets); `results.csv` keeps every pair row. The vector method
  lateral-normalizes at vectorization and uses both sides.
- **Dotprops are rebuilt per search in memory and never written to disk** —
  only raw skeletons and vector rows persist, so a schema or parameter change
  can never leave stale dotprops behind.
- A vector cache written by an older schema version is detected (schema
  version + lateral-normalization marker) and rebuilt automatically on the
  next search; the ZCA whitener is refit when its fit version differs.

## Notes

*   **Root IDs**: FAFB root IDs are very large integers. The system handles them as strings internally to avoid precision loss, but you can pass them as integers in your scripts.
*   **Caching**: The caching system currently produces warnings for FAFB IDs due to their size, but this does not affect the analysis results.
*   **Skeletons**: Skeleton visualization requires either the `sk_lod1_783_healed.zip` file or CAVE API access (via `force_API_fetching=True`). VisualizeSkeleton always prioritizes API-cached skeletons over ZIP data.
*   **Extrusion Issues**: The downloaded `sk_lod1_783_healed.zip` may contain neurons with extrusion artifacts (mesh errors appearing as spikes). Use `VisualizeSkeleton.fix_fafb_extrusions([bodyId1, bodyId2, ...])` to fetch fresh meshes for problematic neurons. If CAVE is unavailable during automatic repair, the visualizer prunes a safely localized bad node subtree in memory without overwriting the raw ZIP/cache source.
