# Auto Type Mapping for Cross-Dataset Comparison

## Overview

When comparing neuron connectivity profiles across different *Drosophila* connectome datasets (e.g., hemibrain vs male-cns, FAFB vs BANC), a fundamental challenge arises: **the same biological neuron types may have different names in different datasets**. 

For example:
- `lLN7` (hemibrain) = `ALIN4` (male-cns, flywire)
- `DNp01` (male-cns) = `DNp01` (flywire) = `DNp01` (hemibrain) ← same name, no issue
- `MTe07` (flywire) = `MeVPLo2` (male-cns)

The **Auto Type Mapping** feature automatically standardizes neuron type names across datasets, enabling meaningful cross-dataset comparisons of connectivity profiles.

## How It Works

### 1. Canonical Type Names

We use **male-cns** as the canonical reference dataset because:
- It represents a complete CNS reconstruction (unlike hemibrain's partial brain)
- It includes the most up-to-date neuron type annotations
- It has extensive type metadata including mappings to other datasets

The `CrossDatasetTypeMapper` class builds mappings from the male-cns `neuron_df`, which contains columns linking each neuron's type across datasets.

### 2. Type Mapping Source

The mapper uses the male-cns neuron DataFrame located at:
`datasets/male-cns_v1_0/male-cns_v1_0_allneurons_neuron_df.csv`

Key columns used:
- `type`: male-cns type name
- `flywireType`: corresponding type in flywire (FAFB/BANC)
- `hemibrainType`: corresponding type in hemibrain
- `mancType`: corresponding type in MANC

These crosswalk columns may list several names in one cell, separated by
`,`; each name is used individually (a male-cns type pointing at several
flywire types becomes a 1-to-N conflict instead of a bogus joined name).

#### Renamed FlyWire types (additional Type(S) columns)

FlyWire datasets publish an extra additional-type column that records type
renames between releases:

- FAFB v783: `additional_type(s)` in
  `datasets/flywire_FAFB_v783/flywire_FAFB_v783_allneurons_neuron_df.csv`
- BANC v626: `Alternative Cell Type(s)` in
  `datasets/flywire_BANC_v626/flywire_BANC_v626_allneurons_neuron_df.csv`

When a male-cns `flywireType` value is **no longer a primary `type`** in the
target dataset but appears in that column, the mapping resolves to the
current primary name. Example: male-cns `SLP249` has `flywireType = SLP249`,
but in FAFB v783 those neurons are typed `APDN3` and `SLP249` survives only
in `additional_type(s)` — so the auto mapping resolves
`male-cns SLP249 → FAFB APDN3`.

Resolution rules:

- The old name resolves to the **single** primary type listing it (rename).
- If **several** primary types list the old name (a split, e.g. FAFB
  `AOTU008` → `AOTU008a/b/c/d`), no automatic mapping is made and a
  `TypeMappingConflict` is recorded instead.
- Names that are already a primary type pass through unchanged.
- FAFB and BANC resolve independently: a name FAFB renamed may still be
  primary in BANC (e.g. `MDN` → `DNp50` in FAFB, still `MDN` in BANC), so
  each keeps its own mapping namespace.
- Missing dataset tables only disable the rename resolution; the mapper
  still works from the male-cns crosswalk alone.

### 3. User Warnings and Double-Check Recommendation

Auto type mapping is applied automatically, so runs surface what it changed
in two places — **please double check them before interpreting
cross-dataset results**:

1. **Console summary**: when source/target neurons are resolved, the run
   prints the auto-mapped names plus explicit `N-to-1` / `1-to-N` warnings,
   and ends with a reminder to double check the automatic mappings.
2. **`user_warning_notes.txt`** in the run folder root (rendered in the run
   guide's Warnings section). It is written when auto type mapping:
   - **expanded** a queried type name to a different name in a target
     dataset (e.g. `SLP249` → `APDN3` in FAFB, `MeVPLo2` → `MTe07`),
   - hit an **N-to-1** mapping (several types share one name across
     datasets — they are *not* merged to avoid wrong aggregation), or
   - hit a **1-to-N** mapping (a type splits into several names in the
     other dataset — no automatic mapping is made).

   Example:

   ```
   - Auto type mapping expanded queried type 'SLP249' to 'APDN3'
     (FlyWire FAFB v783); the queried name may not exist there.
   - N-to-1 type mapping: 'SLP249' is one of 4 types (CL125, PLP080,
     SLP249, SLP250) that all correspond to 'APDN3' in FlyWire FAFB v783;
     they were NOT merged to avoid wrong aggregation.
   - These name mappings were applied automatically - please double check
     them (against the datasets' type annotations or the exported mapping
     files) before interpreting cross-dataset results.
   ```

For cross-dataset comparisons the full applied mapping is additionally
exported to `auto_type_mapping.csv` (and conflicts to
`auto_type_mapping_conflicts.csv`) in the output folder.

3. **See available neurons viewer (expanded search)**: when a search finds
   no rows in the selected dataset, the viewer probes two expansions and
   shows a clearly separated panel:

   - **Type-name matches (native, mapper-free)**: the search text is
     matched as a case-insensitive substring against the `type` column and
     the taxonomy label columns (`class`/`cell_class`/`cell_type`/… ) of
     every other *cached* dataset's index. Matches are name-similar entries
     — **not necessarily the same type** — listed with neuron counts
     (exact matches first, then by count; capped per dataset). This tier
     works even when the auto type mapping knows nothing about the query.
   - **Auto type mapping** (the tier described above): renamed types,
     splits, and N-to-1 groups for the query name.

   Every matched foreign type is additionally annotated with what it is
   called in the selected dataset (unique rename, same name, or the members
   of a refused N-to-1 aggregation); types without a counterpart stay
   visible unannotated so the user is led to inspect them in the other
   dataset. Cross-dataset rows are informational only — they are never
   merged into the selected dataset's table or selection (bodyIds from
   different datasets must not be mixed) — and the panel repeats the
   double-check recommendation.

   Each dataset block also has a **"Show mapped types here (N types)"**
   button. It runs the equivalent search in the selected dataset: the main
   table switches to a *mapped-type view* listing the current dataset's
   neurons of the mapped type names, with a prominent warning banner
   ("automatic mapping — please double check") and a "Back to normal
   search" exit. Three provenance columns are appended per row: the
   **foreign type(s)** that mapped to it, the **map used column** (`type`,
   `additional_type(s) · via '<old name>'`, or the `flywireType`
   crosswalk), and the **matched column** that triggered the expansion
   (e.g. `cell_type · circadian_clock`). Because the view queries the
   *selected* dataset's index only, selection and add-to-query keep
   working normally; any new query leaves the mapped view and returns to
   the regular workflow.

### 4. Standardization Process

When comparing profiles from different datasets:

1. **Query type standardization**: Input type names are mapped to canonical names for profile retrieval
2. **Partner type standardization**: When computing similarity, the partner types in each profile are standardized to canonical names
3. **Similarity computation**: Jaccard, cosine, and rank correlation are computed on standardized partner type sets

## Usage

### Enabling Auto Type Mapping in ComparisonAnalyzer

```python
from comparison import ComparisonAnalyzer, ComparisonParameters

params = ComparisonParameters(
    datasets=['male-cns:v0.9', 'flywire_FAFB_v783'],
    source_neurons=['MeVPLo2', 'MeVPaMe1'],
    target_neurons=['aMe.*'],
    auto_type_mapping=True,  # Enable type mapping (default: True)
    # ... other parameters
)

analyzer = ComparisonAnalyzer(params)
results = analyzer.run_comparison()
analyzer.export_results()  # Exports filtered auto_type_mapping.csv
```

### In ConnectivityProfileComparer (Cross-Dataset Mode)

```python
from comparison.profile_comparator import ConnectivityProfileComparer

# Cross-dataset comparison with dict query format
comparer = ConnectivityProfileComparer(
    query={
        'hemibrain:v1.2.1': ['MBON-γ2α\'1', 'MBON-γ5β\'2a', 'PPL1-γ1pedc'],
        'male-cns:v0.9': ['MBON-γ2α\'1', 'MBON-γ5β\'2a', 'PPL1-γ1pedc']
    },
    dataset=None,  # Must be None for cross-dataset (warning if not)
    use_auto_type_mapping=True  # Enable partner type standardization
)

# Generates N×M similarity matrix comparing types across datasets
results = comparer.run()
```

## API Reference

### CrossDatasetTypeMapper

```python
from comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper

# Create mapper instance
mapper = CrossDatasetTypeMapper(
    workspace_path='/path/to/workspace',  # Optional
    neuron_df_path=None,  # Optional, uses default location if None
    verbose=True,
    # Optional: explicit FAFB/BANC neuron tables for rename resolution;
    # a None value disables it for that namespace
    flywire_neuron_df_paths=None,
)

# Load mappings (called automatically when needed)
mapper.load()

# Get equivalent type in target dataset
mapped = mapper.get_mapped_type(
    type_name='lLN7',
    source_dataset='hemibrain:v1.2.1', 
    target_dataset='male-cns:v0.9'
)
# Returns: 'ALIN4'

# Resolve type across multiple datasets
mappings = mapper.resolve_type_across_datasets(
    type_name='MeVPLo2',
    datasets=['male-cns:v0.9', 'flywire_FAFB_v783', 'hemibrain:v1.2.1']
)
# Returns: {'male-cns:v0.9': 'MeVPLo2', 'flywire_FAFB_v783': 'MTe07', ...}

# Get canonical (male-cns) type name
canonical = mapper.get_canonical_type('MTe07', source_dataset='flywire_FAFB_v783')
# Returns: 'MeVPLo2'

# Standardize partner types for cross-dataset comparison
standardized = mapper.standardize_partner_types(
    partner_types={'lLN7': 10.5, 'KC-γm': 5.2},
    source_dataset='hemibrain:v1.2.1'
)
# Returns: {'ALIN4': 10.5, 'KC-γm': 5.2} (lLN7 mapped to male-cns name)

# Get display name with cross-dataset mappings
display = mapper.get_display_name(
    type_name='MeVPLo2',
    datasets=['male-cns:v0.9', 'flywire_FAFB_v783'],
    source_dataset='male-cns:v0.9'
)
# Returns: 'MeVPLo2(MTe07)' if names differ across datasets
```

### Export Methods

```python
# Export mappings (filtered to result types)
mapper.export_mapping(
    output_path='auto_type_mapping.csv',
    filter_types={'MeVPLo2', 'ALIN4', 'DNp01'}  # Only include these types
)

# Export mapping conflicts (1-to-N or N-to-1 relationships)
mapper.export_conflicts(
    output_path='auto_type_mapping_conflicts.csv',
    filter_types={'WED092'}  # Optional filtering
)
```

### Disabling Auto Type Mapping

If you want to compare connectivity profiles without type name standardization (e.g., to see raw differences in naming conventions):

```python
# In ComparisonParameters
params = ComparisonParameters(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    auto_type_mapping=False,  # Disable standardization
    # ...
)

# In ConnectivityProfileComparer
comparer = ConnectivityProfileComparer(
    query={'hemibrain:v1.2.1': [...], 'male-cns:v0.9': [...]},
    use_auto_type_mapping=False  # Disable standardization
)
```

## Implementation Details

### Partner Type Standardization

When computing connectivity profile similarity across datasets, the key insight is that **partner types** must also be standardized. For example:

**Profile A (hemibrain: ALIN4 → lLN7)**:
- Upstream: `PPL1-γ1pedc` (20%), `lLN7` (15%), `KC-γm` (10%), ...
- Downstream: `FB2B` (25%), ...

**Profile B (male-cns: ALIN4)**:
- Upstream: `PPL1-γ1pedc` (18%), `ALIN4` (16%), `KC-γm` (12%), ...
- Downstream: `FB2B` (22%), ...

Without standardization, `lLN7` and `ALIN4` would be treated as different types, reducing similarity. With standardization to male-cns names:

**Profile A (standardized)**:
- Upstream: `PPL1-γ1pedc` (20%), `ALIN4` (15%), `KC-γm` (10%), ...

**Profile B (standardized)**:
- Upstream: `PPL1-γ1pedc` (18%), `ALIN4` (16%), `KC-γm` (12%), ...

Now the Jaccard and cosine similarities correctly identify these as highly similar profiles.

### Supported Datasets

Type mappings are available for:
- `male-cns:v0.9` (canonical reference)
- `flywire_FAFB_v783`
- `flywire_BANC_v626`
- `hemibrain:v1.2.1`
- `manc:v1.0` / `manc:v1.2.1`

### Output Files

When running `ComparisonAnalyzer.export_results()` with `auto_type_mapping=True`, two files are generated:

1. **auto_type_mapping.csv**: Type mappings for neurons in results only
   ```csv
   male-cns:v0.9,flywire_FAFB_v783,flywire_BANC_v626,hemibrain:v1.2.1,manc:v1.0,manc:v1.2.1
   ALIN4,ALIN4,ALIN4,lLN7,,
   DNp01,DNp01,DNp01,DNp01,,
   MeVPLo2,MTe07,MTe07,,,
   ...
   ```

2. **auto_type_mapping_conflicts.csv**: 1-to-N or N-to-1 mapping conflicts
   ```csv
   source_dataset,source_type,target_dataset,target_types,relationship
   male-cns:v0.9,WED092,flywire_FAFB_v783,"WED092b, WED092c, WED092d",1-to-N
   ```

### Conflict Handling

- **1-to-N mappings**: One male-cns type maps to multiple types in another dataset. These are logged as warnings but the mapping proceeds using all variations.
- **N-to-1 mappings**: Multiple types from different datasets map to the same canonical name. These types should NOT be aggregated incorrectly.

Use `mapper.is_n_to_1_type(type_name, dataset)` to check if a type is involved in a conflict.

## Performance Considerations

- Type mapping lookups are O(1) hash table operations
- Mapping tables are loaded lazily on first use
- For large-scale batch comparisons, the mapper is passed once to avoid repeated loading
- Partner standardization adds minimal overhead (dict comprehension)

## Troubleshooting

### "Type mapper not loaded" warning

This means the neuron_df file wasn't found. Check:
1. File exists at `datasets/male-cns_v0_9/male-cns_v0_9_allneurons_neuron_df.csv`
2. The workspace path is correctly configured

### Type not found in mapping

If a type has no cross-dataset mapping:
- The original type name is preserved (identity mapping)
- This is expected for types unique to one dataset

### Unexpected similarity scores

If cross-dataset similarity seems too low:
1. Check if key partner types have mappings
2. Consider that some connectivity differences are real biological variation
3. Use `use_auto_type_mapping=False` to see raw (unmapped) comparison

## See Also

- [Cross-Dataset Comparison](./core-features/CrossDatasetComparison_Guide.md) - Overview of cross-dataset analysis
- [Connectivity Profiling](./CONNECTIVITY_PROFILING.md) - Connectivity profile computation
