# Cross-Dataset Comparison Guide

**Version:** 4.0  
**Last Updated:** November 2025

## Overview

The Cross-Dataset Comparison module enables systematic analysis of neural connectivity patterns across multiple connectome datasets. This is essential for understanding evolutionary conservation, identifying dataset-specific circuits, and validating findings across independently reconstructed neural maps.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Comparison Modes](#comparison-modes)
4. [Usage Examples](#usage-examples)
5. [Output Files Reference](#output-files-reference)
6. [Calculated Parameters](#calculated-parameters)
7. [HTML Report Features](#html-report-features)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Example

```python
from comparison import ComparisonParameters, ComparisonAnalyzer

# Define comparison parameters
params = ComparisonParameters(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    source_neurons=['aMe12'],
    target_neurons=['PPL101'],
    max_interlayer=1,
    thresholds=[1, 5, 10],
    output_folder='/path/to/output',
    skip_bodyId=True,  # Optional: Skip bodyId-level data for faster type-level analysis
)

# Run comparison
analyzer = ComparisonAnalyzer(params, verbose=True)
results = analyzer.run_comparison()
```

This produces:
- Interactive HTML report with network visualizations
- CSV files with edge/path presence matrices
- PNG heatmaps and charts
- PDF summary

---

## Core Concepts

### What is Cross-Dataset Comparison?

Cross-dataset comparison analyzes the same neural circuit query across multiple connectome reconstructions to answer:

1. **Conservation**: Which connections are present in all datasets?
2. **Divergence**: Which connections are unique to specific datasets?
3. **Strength Variation**: How do connection weights vary across datasets?
4. **Circuit Structure**: Are multi-hop paths conserved or divergent?

### Key Terminology

| Term             | Definition                                                               |
| ---------------- | ------------------------------------------------------------------------ |
| **Edge**         | A direct connection between two neuron types (e.g., `aMe12 → KCg-d`)     |
| **Path**         | A multi-hop route from source to target (e.g., `aMe12 → KCg-d → PPL101`) |
| **Threshold**    | Minimum synapse count required for an edge to be included                |
| **Conservation** | Presence of an edge/path across multiple datasets                        |
| **Dead-end**     | An edge that doesn't contribute to any complete source→target path       |

### Datasets Supported

- **NeuPrint datasets**: `hemibrain:v1.2.1`, `male-cns:v0.9`, `optic-lobe:v1.0.1`, etc.
- **FAFB**: `flywire_FAFB_v783` (local parquet files)
- **Standalone BANC**: `banc_v626` / `banc_v888` (public-release parquet files)

---

## Comparison Modes

### Path-Based Mode (`comparison_mode='path'`)

**How it works:**
1. Finds all paths from source to target neurons
2. Extracts edges from discovered paths
3. Aggregates by neuron type

**Pros:**
- Only includes edges on functional paths
- Better represents circuit-level connectivity
- No dead-end connections

**Cons:**
- Strong edges may appear absent if they're only on paths with weak intermediate connections
- Path filtering can mask true connectivity

**Warning displayed:**
> ⚠️ Path-Based Filtering Caveat: Strong edges may appear absent if they only exist on paths with weaker intermediate edges that fall below the threshold.

### Edge-Based Mode (`comparison_mode='edge'`)

**How it works:**
1. Finds all paths at the **lowest** threshold first
2. Extracts unique edges from paths
3. Re-queries edge weights directly from the database
4. Filters edges independently by their own weight

**Pros:**
- Direct edge weight comparison
- No path-filtering artifacts
- Strong edges always appear if they meet threshold

**Cons:**
- May include dead-end connections not on complete paths
- Loses path-level circuit structure

**Warnings displayed:**
> ⚠️ Edge-Based Comparison Mode:
> - **Caveat 1 - Dead-ends**: May include dead-end connections that don't contribute to complete source→target paths.
> - **Caveat 2 - Weight Mismatch**: Edge weights in the Edge Presence Matrix represent total synapses between all neuron pairs of the types, while Path Presence Matrix hop weights represent only synapses from neurons actually participating in paths.

### Choosing a Mode

| Use Case                             | Recommended Mode |
| ------------------------------------ | ---------------- |
| Functional circuit analysis          | `path`           |
| Direct connectivity comparison       | `edge`           |
| Identifying all possible connections | `edge`           |
| Studying specific pathways           | `path`           |
| Large-scale conservation analysis    | `edge`           |

---

## Hemisphere-Aware Comparison

Use these settings to split types by hemisphere, compute symmetry statistics, and optionally filter to hemisphere-conserved edges.

### Key Parameters

| Parameter                                    | Type | Default | Description                                                                                                                                                                       |
| -------------------------------------------- | ---- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `separate_hemispheres`                       | bool | False   | When True, type/group labels are suffixed with `_L/_R/_U` using dataset hemisphere annotations. All type-level and custom-group aggregations are split by hemisphere.             |
| `symmetry_analysis`                          | bool | True    | Generates per-dataset, per-threshold hemisphere symmetry summaries (ipsilateral vs contralateral). Auto-enabled when `separate_hemispheres=True`.                                 |
| `keep_only_hemisphere_conserved_connections` | bool | False   | When True, keep only edges that are conserved between hemispheres (e.g., `A_L→B_L` and `A_R→B_R`). Requires `separate_hemispheres=True` and is disabled with a warning otherwise. |

### Notes and Behavior

- When `separate_hemispheres=True`, the comparison report includes a Hemisphere Symmetry section with Jaccard and conserved/union metrics.
- When `separate_hemispheres=False`, the HTML report shows a notice that hemisphere symmetry is unavailable.
- FlyWire FAFB uses reversed hemisphere annotation relative to NeuPrint datasets. The comparison prints a warning when mixing FAFB with NeuPrint.

### Example

```python
params = ComparisonParameters(
    datasets=['male-cns:v0.9', 'flywire_FAFB_v783'],
    source_neurons=['aMe12'],
    target_neurons=['PPL101'],
    thresholds=[1, 3, 5],
    output_folder='/path/to/output',
    separate_hemispheres=True,
    symmetry_analysis=True,  # Auto-enabled when separate_hemispheres=True
    keep_only_hemisphere_conserved_connections=True,
)
```

---

## Reciprocal Analysis (`find_reciprocal`)

When enabled, the path graph is enriched with direct reciprocal connections and additional visualizations are generated.

### Behavior

- `find_reciprocal=True` triggers reciprocal edge discovery in `FindAllPath`.
- Reciprocal graphs and CSVs are saved under each dataset’s `find_reciprocal/` folder.
- The comparison report includes reciprocal visualizations when available.

---

## Usage Examples

### Example 1: Basic Two-Dataset Comparison

```python
from comparison import ComparisonParameters, ComparisonAnalyzer

params = ComparisonParameters(
    # Datasets to compare
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    datasets_nickname=['hemi', 'mcns'],  # Short names for displays
    
    # Query parameters
    source_neurons=['aMe12'],
    target_neurons=['PPL101'],
    max_interlayer=1,  # Allow 1 intermediate layer
    
    # Analysis settings
    thresholds=[1, 3, 5, 10],
    top_edges=50,  # Top-edge/path rows retained in comparison reports
    comparison_mode='path',
    
    # Output
    output_folder='/path/to/output',
)

analyzer = ComparisonAnalyzer(params, verbose=True)
results = analyzer.run_comparison()
```

### Example 2: Three-Dataset with FlyWire

```python
params = ComparisonParameters(
    datasets=[
        'hemibrain:v1.2.1',
        'male-cns:v0.9', 
        'flywire_FAFB_v783'  # Local FlyWire data
    ],
    datasets_nickname=['hemi', 'mcns', 'fafb'],
    
    source_neurons=['aMe12'],
    target_neurons=['PPL101'],
    max_interlayer=1,
    thresholds=[1, 5, 10],
    
    comparison_mode='edge',  # Edge-based for direct comparison
    output_folder='/path/to/output',
)

analyzer = ComparisonAnalyzer(params)
results = analyzer.run_comparison()
```

### Example 3: Using LabelMapper for Dataset-Specific Naming

When neuron names differ across datasets, use `LabelMapper` to define the mapping:

```python
from comparison import ComparisonParameters, ComparisonAnalyzer
from comparison.label_mapper import LabelMapper

# Create mapper with dataset-specific neuron names
mapper = LabelMapper(
    source_mapping_dict={
        'hemibrain:v1.2.1': [['aMe12']],
        'male-cns:v0.9': [['aMe12-like']],  # Different naming convention
    },
    source_labels=['aMe12'],
    target_mapping_dict={
        'hemibrain:v1.2.1': [['PPL101']],
        'male-cns:v0.9': [['PPL1_dopamine']],  # Different naming
    },
    target_labels=['PPL101']
)

params = ComparisonParameters(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    
    # Pass the mapper object - it handles dataset-specific naming
    source_neurons=mapper,
    target_neurons=mapper,
    
    max_interlayer=1,
    thresholds=[1, 5, 10],
    output_folder='/path/to/output',
)

analyzer = ComparisonAnalyzer(params)
results = analyzer.run_comparison()
```

See the [LabelMapper Guide](./LabelMapper_Guide.md) for more details on mapping configuration.

> **UI:** the web UI manages the same mappings as reusable presets —
> **Settings tab → Custom Type Mappings** (table-grid editor, saved in
> `cache/user_mappings.json`) and **Cross-Dataset tab → Custom Type Mapping**
> selector. The chosen preset is passed to the run as `overall_mapping_json`
> and acts as an overlay: explicit source/target queries stay, only matching
> neurons are renamed to their custom groups.

### Example 4: Accessing Results Programmatically

```python
analyzer = ComparisonAnalyzer(params)
results = analyzer.run_comparison()

# Access aligned edge data at threshold 5
aligned_data = analyzer.get_aligned_data(threshold=5)
print(aligned_data.head())
# Output: DataFrame with edges as rows, datasets as columns

# Get key findings
print(results['key_findings'])

# Generate text report
report = analyzer.generate_report()
print(report)
```

### Example 5: Internal Network (Source = Target)

When you want to analyze **internal connectivity within a neuron group**, you can set `source_neurons` and `target_neurons` to the **same neurons**. This is useful for:

- Studying recurrent/reciprocal connections within a neuron type family
- Analyzing intra-group connectivity patterns
- Comparing internal circuit structure across datasets

```python
from comparison import ComparisonParameters, ComparisonAnalyzer

# Define the neuron group to analyze
neuron_group = ['aMe.*']  # All aMe-type neurons (medulla intrinsic)

params = ComparisonParameters(
    datasets=['male-cns:v0.9', 'hemibrain:v1.2.1', 'flywire_FAFB_v783'],
    datasets_nickname=['male-CNS', 'hemibrain', 'FAFB'],
    
    # Same neurons for both source and target
    source_neurons=neuron_group,
    target_neurons=neuron_group,
    
    # Direct connections only (no intermediate layers)
    max_interlayer=0,
    
    # Analysis settings
    thresholds=[5, 10, 20],
    comparison_mode='edge',  # Edge-based to capture all connections
    
    output_folder='/path/to/output',
)

analyzer = ComparisonAnalyzer(params, verbose=True)
results = analyzer.run_comparison()
```

**Key Points for Internal Network Analysis:**

1. **`max_interlayer=0`**: For direct connections only, set to 0. Set to 1+ if you want to include paths through intermediate neurons.

2. **Self-connections**: The analysis will include both:
   - Connections between different neurons of the same type (e.g., aMe12_A → aMe12_B)
   - Connections between different types (e.g., aMe12 → aMe17)

3. **Neuron Counts Comparison**: The HTML report will show how many neurons of each type exist in each dataset (useful for understanding why some connections may be missing).

4. **Recommended Settings**:
   - Use `comparison_mode='edge'` to capture all potential connections
   - Use moderate thresholds (5+) to filter weak/spurious connections
   - Consider using wildcards (`aMe.*`) to include all subtypes

### Example 6: Using LabelMapper for Cross-Dataset Standardization

When neuron names differ across datasets (e.g., `aMe12` in hemibrain vs `aMe12_R` in FAFB), use `LabelMapper` to standardize them.

**Option A: Using a Mapping File (Recommended)**

Create a CSV file `mappings.csv`:
```csv
custom_label,hemibrain:v1.2.1,male-cns:v0.9,flywire_FAFB_v783
aMe12,aMe12,aMe12,720575940610453042
PPL101,PPL101,PPL101,720575940621886666
```

Then use it in your script:

```python
from comparison import ComparisonParameters, ComparisonAnalyzer, LabelMapper

# Initialize LabelMapper with a unified JSON mapping file
# Note: ONLY JSON format is supported for overall_mapping_json
mapper = LabelMapper(overall_mapping_json='mappings.json')

params = ComparisonParameters(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9', 'flywire_FAFB_v783'],
    
    # Pass the mapper to source/target neurons
    # The mapper will look up the standardized labels 'aMe12' and 'PPL101'
    source_neurons=mapper,
    target_neurons=mapper,
    
    # Specify which standardized labels to use
    source_labels=['aMe12'],
    target_labels=['PPL101'],
    
    max_interlayer=1,
    thresholds=[5, 10],
    output_folder='/path/to/output',
)

analyzer = ComparisonAnalyzer(params)
results = analyzer.run_comparison()
```

**Option B: Separate Source and Target Mappings**

If you have separate files for source and target mappings:

```python
# Initialize with separate files
mapper = LabelMapper(
    source_mapping_file='source_mappings.csv',
    target_mapping_file='target_mappings.csv'
)

params = ComparisonParameters(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    
    # Use the mapper
    source_neurons=mapper,
    target_neurons=mapper,
    
    # ... other parameters
)
```

**Option C: Using Dictionaries (No File)**

```python
# Define mappings in code
source_map = {
    'hemibrain:v1.2.1': ['aMe12'],
    'male-cns:v0.9': ['aMe12'],
    'flywire_FAFB_v783': ['720575940610453042']
}

target_map = {
    'hemibrain:v1.2.1': ['PPL101'],
    'male-cns:v0.9': ['PPL101'],
    'flywire_FAFB_v783': ['720575940621886666']
}

# Create mapper from dicts
mapper = LabelMapper(
    source_mapping_dict=source_map,
    target_mapping_dict=target_map,
    source_labels=['aMe12'],  # Optional: name for the group
    target_labels=['PPL101']
)

params = ComparisonParameters(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9', 'flywire_FAFB_v783'],
    source_neurons=mapper,
    target_neurons=mapper,
    # ...
)
```

**Option D: Using `overall_label_mapper`**

Alternatively, you can pass the mapper to `overall_label_mapper` and use simple lists for source/target neurons (referencing standardized labels).

```python
mapper = LabelMapper(overall_mapping_json='mappings.json')

params = ComparisonParameters(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    
    # Pass mapper here
    overall_label_mapper=mapper,
    
    # Use standardized labels directly
    source_neurons=['aMe12'],
    target_neurons=['PPL101'],
    
    # ...
)
```

---

## Output Files Reference

All outputs are saved to a timestamped folder: `comparison_results_YYYYMMDD_HHMMSS/`

### Directory Structure

```
comparison_results_20251127_155513/
├── comparison_report.html          # Interactive HTML report
├── vis_summary.pdf                 # PDF summary of visualizations
├── parameters.json                  # comparison settings + provenance contract
├── effective_thresholds.json        # requested → applied notice per dataset
│
├── comparison_results/             # CSV data files
│   ├── edge_presence_matrix.csv    # Unified edge presence (all thresholds)
│   ├── edge_presence_matrix_minsyn_1.csv
│   ├── edge_presence_matrix_minsyn_3.csv
│   ├── edge_presence_matrix_minsyn_5.csv
│   ├── edge_presence_matrix_minsyn_10.csv
│   ├── path_presence_matrix.csv    # Unified path presence (all thresholds)
│   ├── path_presence_matrix_minsyn_1.csv
│   ├── path_presence_matrix_minsyn_3.csv
│   ├── path_presence_matrix_minsyn_5.csv
│   ├── path_presence_matrix_minsyn_10.csv
│   ├── unified_edge_comparison.csv # Full comparison table
│   ├── unified_summary.csv         # metrics + threshold provenance per run
│   ├── threshold_sensitivity.csv   # threshold changes + provenance
│   ├── pathfinding_provenance.csv  # provenance; query_id in combination mode
│   ├── threshold_combinations.csv  # canonical query/dataset manifest
│   ├── edge_presence_matrix_query_<query_id>.csv
│   ├── path_presence_matrix_query_<query_id>.csv
│   ├── untyped_dropped_records.csv # rows removed by Drop Untyped, if any
│   ├── conserved_strong_connections_minsyn_*.csv
│   └── ...
│
├── comparison_visualizations/      # PNG visualizations
│   ├── path_heatmap_1.png
│   ├── path_heatmap_all_thresholds.png
│   ├── edge_heatmap_1.png
│   ├── edge_heatmap_all_thresholds.png
│   ├── threshold_comparison.png
│   ├── similarity_heatmap_jaccard.png
│   ├── similarity_heatmap_cosine.png
│   ├── by_ratio/
│   │   └── ratio_heatmap_*.png
│   ├── by_probability/
│   │   └── traversal_prob_heatmap_*.png
│   ├── edge_heatmap_query_<query_id>.png  # Custom combination rows
│   ├── path_heatmap_query_<query_id>.png
│   └── visualization_data/
│       └── *.csv                   # Raw data for recreating plots
│
├── comparison_results/             # Comparison analysis outputs
│   ├── neuron_counts_summary.csv   # Total neuron counts per dataset
│   ├── neuron_counts_by_type.csv   # Neuron counts by type per dataset
│   ├── neuron_counts_by_group.csv  # Counts by custom group (if used)
│   └── ...
│
├── dataset_data/                   # Per-dataset raw results
│   ├── hemibrain_v1_2_1/
│   │   ├── minsyn_1/
│   │   ├── minsyn_3/
│   │   └── ...
│   ├── male_cns_v0_9/
│   └── flywire_FAFB_v783/
│
└── comparison_networks/            # Interactive network HTML files
    ├── network_threshold_1.html
    ├── network_threshold_3.html
    └── ...
```

Combination-mode similarity exports are written under
`similarity_matrices/` as `similarity_query_<query_id>.csv` plus the combined
`similarity_by_query.csv`. These files are joined by `query_id`; no advanced
comparison is keyed by the sorted union of raw threshold values.

The combination `comparison_report.html` uses the same full report sections as
Standard mode: summary charts, applied-threshold/bottleneck provenance,
similarity tables, query-keyed networks, edge/path matrices, conservation,
overlap, and statistics. Every section iterates every query row. Ratio and
traversal-probability filtering is disabled for pathfinding comparisons, so
those `by_ratio/` and `by_probability/` folders are not emitted for this
report.

### CSV File Descriptions

#### `neuron_counts_summary.csv`

Shows total neuron counts per dataset for both source and target neurons.

| Column          | Description                          |
| --------------- | ------------------------------------ |
| `dataset`       | Dataset name                         |
| `source_count`  | Number of source neurons found       |
| `target_count`  | Number of target neurons found       |
| `total_neurons` | Total unique neurons                 |
| `source_types`  | Number of unique source neuron types |
| `target_types`  | Number of unique target neuron types |

#### `neuron_counts_by_type.csv`

Shows neuron count per type in each dataset. Useful for understanding which neuron types exist in which datasets.

| Column             | Description                                   |
| ------------------ | --------------------------------------------- |
| `type`             | Neuron type name                              |
| `role`             | Whether this type is used as source or target |
| `{dataset}_source` | Count of this type as source in dataset       |
| `{dataset}_target` | Count of this type as target in dataset       |

#### `edge_presence_matrix_minsyn_{threshold}.csv`

Shows edge presence and weight across all datasets at a specific threshold.

| Column               | Description                              |
| -------------------- | ---------------------------------------- |
| `edge_key`           | Edge identifier (`source → target`)      |
| `source_type`        | Source neuron type                       |
| `target_type`        | Target neuron type                       |
| `conservation_count` | Number of datasets where edge is present |
| `{dataset}`          | `True` if present, `0` if absent         |
| `weight_{dataset}`   | Synapse count in that dataset            |
| `max_weight`         | Maximum weight across datasets           |
| `avg_weight`         | Average weight across present datasets   |
| `weight_cv`          | Coefficient of variation (std/mean)      |

#### `path_presence_matrix_minsyn_{threshold}.csv`

Shows multi-hop path presence across all datasets.

| Column                                  | Description                           |
| --------------------------------------- | ------------------------------------- |
| `path_key`                              | Full path (`source → inter → target`) |
| `source`                                | Starting neuron type                  |
| `target`                                | Ending neuron type                    |
| `hops`                                  | Number of edges in path               |
| `intermediates`                         | Intermediate neuron types             |
| `conservation_count`                    | Number of datasets where path exists  |
| `{dataset}`                             | `True` if present, `0` if absent      |
| `weight_{dataset}`                      | Minimum edge weight along path        |
| `hop_weights_{dataset}`                 | Individual hop weights (`-w1-w2-`)    |
| `max_weight`, `avg_weight`, `weight_cv` | Statistics                            |

#### `unified_edge_comparison.csv`

Comprehensive edge table with all thresholds in a single file.

---

## Calculated Parameters

### Conservation Metrics

#### Conservation Count
Number of datasets where an edge/path is present.
```
conservation_count = Σ (1 if edge present in dataset else 0)
```

#### Conservation Rate
Percentage of edges/paths present in all datasets.
```
conservation_rate = (edges in ALL datasets) / (total unique edges) × 100%
```

### Similarity Metrics

> **📖 Detailed Documentation**: See [Graph Similarity Metrics Documentation](GraphSimilarityMetrics_Documentation.md) for comprehensive explanations of all metrics including Edge Rank Correlation, Cosine Similarity, Spearman Rank, and NaN handling.

#### Jaccard Similarity
Measures overlap of edge sets between two datasets.
```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```
- Range: [0, 1]
- Ignores edge weights, only considers presence

#### Cosine Similarity (NEW)
Measures similarity of edge weight vectors using union of edges.
```
Cosine(A, B) = (A · B) / (||A|| × ||B||)
```
- Range: [0, 1], NaN if both vectors are zero
- Scale-invariant (only considers angle, not magnitude)
- Higher values indicate similar weight distributions

#### Edge Rank Correlation (NEW)
Raw Spearman correlation on union of edges (missing edges = weight 0).
```
EdgeRank(A, B) = Spearman(ranks_A, ranks_B)
```
- Range: [-1, +1], NaN if fewer than 3 non-zero edges
- Positive: similar ranking, Negative: inverse ranking
- Captures both overlap and weight ranking agreement

### Weight Statistics

#### Coefficient of Variation (CV)
Measures relative variability of edge weights across datasets.
```
CV = σ / μ
```
where σ is standard deviation, μ is mean of weights across datasets.

- CV < 0.3: Low variability (highly conserved weight)
- CV 0.3-0.7: Moderate variability
- CV > 0.7: High variability

#### Minimum Path Weight (min_weight)
For a path A → B → C, the minimum weight is:
```
min_weight = min(weight(A→B), weight(B→C))
```
This represents the "bottleneck" of signal transmission along the path.

### Pathfinding threshold and bottleneck provenance

The path-based comparison records both the requested Min Synapse Count and
the threshold that describes the materialized output. `requested_threshold`
is the user input. `applied_threshold` is the canonical equivalent threshold:
it is the requested value for a complete run, `w2 + 1` after a StrongestFirst
budget bite, and includes the Edge Budget floor when that graph budget fires.
`applied_threshold_source` identifies whether the requested threshold,
StrongestFirst budget, Edge Budget, or both determined the result.

The related values are:

- `tau` / `strongest_first_tau`: the StrongestFirst landing/collapse bound;
  for an unbitten run it is the natural weakest emitted-path bottleneck.
- `strongest_first_budget`: the effective path-output budget; `0`/empty uses
  the internal 1,000,000-path auto budget, and
  `strongest_first_budget_bitten` says whether it was reached.
- `w0` / `edge_weight_floor`: the Edge Budget graph floor, and `w1` /
  `edge_budget_landing`: the tier that determined that floor. This applies in
  `all` mode only; Shortest Paths never applies the Edge Budget floor.
- `w2` / `strongest_dropped_bottleneck`: the strongest path excluded by a
  StrongestFirst budget bite; `w2 + 1` is the minimal equivalent threshold.
- `W*` / `strongest_retained_bottleneck`: the widest-path ceiling after
  lossless pruning. A path bottleneck is the minimum edge weight on that path.

`paths_complete` is false when a lossy budget affected the materialized path
set. The complete per-run record is in
`comparison_results/pathfinding_provenance.csv`; the same values are included
in `threshold_sensitivity.csv`, `unified_summary.csv`, each delegated
`dataset_data/.../minsyn_N/parameters.txt` and `all_attributes.json`, and the
root `effective_thresholds.json` notice.

In Custom combination mode, `threshold_combinations.csv` is the
canonical query/dataset join: it adds `query_id`, the requested cell, and the
dataset-specific applied/budget/bottleneck values. A raw dataset/threshold run
may be reused by several query rows, so use the query ID when interpreting
comparison tables and dropped-neuron warnings.

### Drop Untyped Neurons

The Advanced Settings **Drop Untyped Neurons** option is checked by default.
The shared predicate treats empty labels, `Unknown`/`None`/`NaN` sentinels, and
numeric bodyId-fallback labels as untyped. Comparison first resolves the
standardized cross-dataset labels, then removes edges touching an untyped
pre- or post-neuron before aggregation. Dropped rows are written once at
`comparison_results/untyped_dropped_records.csv`, and counts are appended to
`user_warning_notes.txt`; delegated per-dataset path folders intentionally
retain their raw rows so the comparison-level filter is the single source of
truth.

### Traversal Probability

Probability of a signal traversing from source to target along a path.
```
path_prob = Π (edge_weight / total_output_of_source)
```
For a 2-hop path A → B → C:
```
path_prob = (weight_AB / out_A) × (weight_BC / out_B)
```

### Connection Ratio

Ratio of edge weight to total output of the source neuron type.
```
ratio = edge_weight / total_output_weight
```

---

## HTML Report Features

The interactive HTML report includes:

### 1. Summary Section
- Key metrics by threshold
- Edge count bar charts
- Total weight comparison

### 2. Neuron Counts Comparison
- **Summary Table**: Total source/target neuron counts per dataset
- **Bar Chart**: Visual comparison of neuron counts
- **By Type**: Detailed breakdown showing how many neurons of each type exist in each dataset
- **By Custom Group**: If custom grouping was used, shows counts per group

This section is especially useful for internal network analysis (source=target) to understand why some connections may be missing.

### 3. Similarity Matrices
- Interactive Jaccard heatmaps
- Cosine similarity heatmaps
- Tabbed by threshold

### 4. Network Visualizations
- **Node colors by role:**
  - 🔴 Red: Source neurons
  - 🔵 Blue: Intermediate neurons
  - 🟣 Purple: Target neurons
  - ⚪ Gray: Dead-end nodes (no complete path through)
  
- **Edge colors by conservation:**
  - 🟢 Green: Conserved (all datasets)
  - 🟠 Orange: Partial (some datasets)
  - ⚪ Gray: Unique (one dataset)

- **Interactive features:**
  - Drag nodes to rearrange
  - Hover for connection details
  - Toggle between Static and "Duang" physics modes

### 5. Edge Presence Matrices
- **View by Threshold**: Compare datasets at each threshold
- **View by Dataset**: Compare thresholds for each dataset
- Toggle between views with buttons

### 6. Path Presence Matrices
- Same dual-view toggle as edge matrices
- Shows hop weights as `-w1-w2-` with **minimum bolded**
- Conservation badges (3/3, 2/3, 1/3)

### 7. Conservation Analysis
- Pie charts showing conserved vs non-conserved edges/paths
- Per-threshold breakdown

### 8. Statistics Tables
- Per-dataset metrics (edge count, total weight, mean, max)
- Pairwise similarity scores

---

## Best Practices

### 1. Choose Appropriate Thresholds

```python
# Start with a wide range
thresholds=[1, 3, 5, 10, 20]

# Then narrow based on results
thresholds=[3, 5, 8, 12]  # More granular around interesting region
```

### 2. Use Nicknames for Readability

```python
params = ComparisonParameters(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9', 'flywire_FAFB_v783'],
    datasets_nickname=['hemi', 'mcns', 'fafb'],  # Used in visualizations
    ...
)
```

### 3. Compare Modes When Unsure

```python
# Run both modes to understand differences
for mode in ['path', 'edge']:
    params = ComparisonParameters(
        comparison_mode=mode,
        output_folder=f'/output/{mode}_comparison',
        ...
    )
    analyzer = ComparisonAnalyzer(params)
    analyzer.run_comparison()
```

### 4. Check for Dead-Ends in Edge Mode

When using edge mode, check the network visualization:
- Gray nodes are dead-ends
- These edges are real but may not be functionally relevant

### 5. Validate Across Thresholds

If an edge appears at t=3 but disappears at t=5:
- In **path mode**: May be path-filtering artifact
- In **edge mode**: Edge truly doesn't meet threshold

---

## Troubleshooting

### No Paths Found

**Symptoms:** Empty results, "No path data available"

**Solutions:**
1. Increase `max_interlayer` (e.g., from 1 to 2)
2. Lower the minimum threshold
3. Check neuron type names match dataset conventions
4. Verify source/target neurons exist in all datasets

### Weight Mismatch Between Edge and Path Tables

**This is expected behavior** in edge mode:
- Edge weights = total type-to-type synapses
- Path weights = synapses from neurons actually on paths

### Slow Performance

**Solutions:**
1. Enable caching (automatic if cache exists)
2. Use `skip_bodyId=True` to skip expensive bodyId-level processing if only type-level data is needed
3. Reduce number of thresholds
4. Reduce `top_edges` to keep comparison report tables smaller (this does not trim pathfinding)
5. Use smaller `max_interlayer`

### FAFB/BANC Local Release Not Found

**Solutions:**
1. Ensure the selected release's parquet files are in the correct location
2. Check `datasets/flywire_FAFB_v783/` or `datasets/banc_v626/`/`banc_v888/`
   exists with the proper structure
3. See [FAFB Integration Guide](../FAFB_INTEGRATION.md) or
   [BANC Integration Guide](../BANC_INTEGRATION.md)

---

## API Reference

### ComparisonParameters

```python
ComparisonParameters(
    datasets: List[str],                      # Dataset identifiers
    datasets_nickname: List[str],             # Short display names
    source_neurons: Union[List, LabelMapper], # Source neuron patterns (list or mapper)
    target_neurons: Union[List, LabelMapper], # Target neuron patterns (list or mapper)
    max_interlayer: int,                      # Max intermediate layers
    thresholds: List[int],                    # Synapse count thresholds
    top_edges: int = 50,                      # Report-table top-edge/path row cap
    comparison_mode: str = 'path',            # 'path' or 'edge'
    output_folder: str,                       # Base output directory
    saveas: str = None,                       # Custom folder name (auto if None)
    token: str = '',                          # NeuPrint API token
    skip_bodyId: bool = False,                # Skip bodyId-level processing for speed
)
```

### ComparisonAnalyzer

```python
analyzer = ComparisonAnalyzer(params, verbose=True)

# Run full comparison
results = analyzer.run_comparison()

# Access aligned data
df = analyzer.get_aligned_data(threshold=5)

# Generate reports
text_report = analyzer.generate_report()
```

---

---

## Threshold equivalence across datasets

In Standard mode the analysis compares all datasets at the SAME threshold
(horizontal comparison). Custom combination mode intentionally permits one
requested threshold per dataset within a named query. Synapse-count conventions differ strongly between datasets —
the median number of synapses per neuron spans ~7x (BANC v626 ≈ 45 post,
BANC v888 ≈ 55 post, FAFB v783 ≈ 308 post, male-cns v1.0 ≈ 334 post / 490
pre+post; BANC re-measured on the refreshed 2026-09-04 tables) — so "BANC
≥ 3" and "FAFB ≥ 3" still do not cut the connectomes at comparable
sparsities, even though the refreshed BANC tables are much denser.
This section gives a rough, whole-dataset alignment; query rows remain the
canonical comparison unit and the alignment files below are raw-run density
diagnostics rather than replacements for a query row.

### Criterion

**bodyId-level per-neuron connection-pair density**: the number of distinct
(presynaptic, postsynaptic) body pairs with weight ≥ t, divided by the
dataset's total neuron count. Unweighted — edge presence only, synapse
counts (weights) ignored.

Caveats (re-baselined 2026-09-07 on the refreshed 2026-09-04 BANC
bucket tables — banc_v626 8,671,709 pairs / 185,165 neurons,
banc_v888 8,691,309 pairs / 188,508 neurons; pair weight = max across
ROI rows; denominators are each dataset's bundled neuron-index rows):

- BANC local downloads remain pre-truncated at weight ≥ 3 (thresholds 1–2
  are no-ops on the pair counts), but the refreshed tables are far denser
  than the 2026-08 downloads (v888: 8.69M vs the old 3.04M pairs).
- male-cns numbers come from the ~98% coverage connection cache.
- Whole-dataset values are a rough hint only. Real matching is
  query-specific — a given query's best-aligned thresholds can differ from
  the global rule by several units (this is exactly why every run exports
  its own alignment).

### Reference values (whole dataset, pairs per neuron)

| t | BANC v626 | BANC v888 | FAFB v783 | male-cns v1.0 |
|---|---|---|---|---|
| 3 | 46.8 | 46.1 | 47.3 | 60.2 |
| 5 | 13.4 | 13.4 | 26.8 | 35.6 |
| 8 | 5.7 | 5.8 | 15.0 | 20.7 |
| 10 | 4.0 | 4.1 | 11.2 | 15.7 |

Pre-refresh values, for runs against old caches (2026-08 downloads):
BANC v626/v888 @3 were 23.2 / 19.2 — i.e. the refresh roughly DOUBLED
BANC's τ=3 density and moved it to parity with FAFB.

Best threshold matches under the pairs-per-neuron criterion (integer
grid, refreshed tables):

| anchor | → FAFB | → male-cns |
|---|---|---|
| BANC @3 (46.1–46.8) | **3** (47.3) | **5** (35.6) |
| BANC @5 (13.4) | **8** (15.0) | **8** (20.7) |
| BANC @8 (5.7–5.8) | 10–15 (11.2 / 6.3) | 10–15 (15.7 / 9.1) |
| BANC @10 (4.0–4.1) | 15 (6.3) | 15 (9.1) |

Rule of thumb (refreshed tables): at **τ = 3, BANC and FAFB are directly
comparable** (46–47 pairs/neuron; male-cns ≈ 1.3x BANC); from **τ ≥ 5**
the classic multipliers re-emerge and grow with τ — FAFB ≈ 2x BANC at τ=5
rising to ≈ 2.7–2.8x at τ=10, male-cns ≈ 2.7x at τ=5 rising to ≈ 3.8–3.9x
at τ=10. The old single rule ("FAFB ≈ 2.2–2.3x BANC, male-cns ≈ 2.8–3x
BANC at every threshold"; "BANC lowest at every threshold") no longer
holds under τ = 3.

### Where the per-query alignment comes from

Every cross-dataset run exports threshold-alignment files (spec Feature C).
They describe the typed/raw threshold grid used for density diagnostics. In
Custom combination mode they are explicitly marked with
`threshold_scope=raw_run_schedule_diagnostic`; use the query manifest for
the actual comparison rows:

- `comparison_results/threshold_alignment_best_matches.csv` — a bisection
  prober over each dataset's lowest-threshold extract finds the
  best-matching density threshold in every other dataset (extended range,
  not limited to the typed thresholds). Primary metric:
  edge-count distance `|n_a − n_b| / max(n_a, n_b, 1)`; tolerance ≤ 0.10.
- `comparison_results/threshold_alignment_matrix.csv` (+ heatmap) —
  pairwise metrics over the typed/raw thresholds only; it is not a substitute
  for the per-row advanced query map.
- `comparison_results/edge_density_per_threshold.csv` and
  `comparison_visualizations/edge_density_threshold_curves.png` — the
  density curves behind the matching (absolute + per-neuron).

### Threshold query modes

The Cross-Dataset tab's **Core Parameters** threshold editor has two modes:

- **Standard**: enter `3, 5, 10`. Each chip becomes one
  comparison query and the same requested threshold is applied to every
  selected dataset.
- **Custom combination**: requires at least two selected datasets. Each table
  row is one complete query. Dataset names are the columns and every row must
  contain one positive threshold in every selected-dataset column. For example:

  | query | BANC | FAFB | male-cns |
  |---|---:|---:|---:|
  | `combo_001` | 3 | 7 | 8 |
  | `combo_002` | 5 | 11 | 15 |

  These rows intentionally replace the old Advanced Settings
  **Per-dataset thresholds** schedule. They do not give each dataset an
  independent list. The values' sorted union is only the deduplicated raw-run
  schedule; comparison alignment and similarity are always computed from the
  row's threshold map. Dataset columns therefore remain paired within a query,
  even when two rows happen to reuse the same raw threshold.

Every combination row receives a stable `query_id` and optional label. The
comparison export `comparison_results/threshold_combinations.csv` is the
canonical query/provenance manifest: join its `query_id`, `dataset`, and
`requested_threshold` (with `threshold_scope=query_cell`) to
`applied_threshold`, `tau`, StrongestFirst budget,
Edge Budget (`w0`/`w1`), `w2`, `W*`, and `paths_complete`. The same query maps
and run rows are available in `effective_thresholds.json`. This makes a
budget-adjusted output auditable without mistaking the applied threshold for
the value entered in the UI.

### Related run features

- **Duplicate-threshold skipping (Feature G)**: with a path budget, a run
  whose weakest emitted path has bottleneck τ produces the identical set
  for every threshold up to τ; later input thresholds ≤ τ are skipped and
  marked `skipped/duplicate_of` in `threshold_sensitivity.csv` (τ collapse).
- **Replay paths (Feature F)**: in path mode 'all' the path set is
  enumerated once at the lowest threshold; every higher threshold is
  materialized from the bottleneck-annotated path set (identical outputs,
  no re-enumeration). Disable via Advanced Settings ▸ Replay Paths.

---

## See Also

- [Cache System Guide](CacheSystem_Guide.md) - Improve query performance
- [Path Finding Documentation](FindAllPath_Documentation.md) - Understand path algorithms
- [FAFB Integration](../FAFB_INTEGRATION.md) - Set up FAFB datasets
- [Example Script](../../archive/examples/comparison/Example_InterDatasetComparison.py) - Full working example
