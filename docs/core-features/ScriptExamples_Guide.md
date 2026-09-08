# Script Examples Guide

Copy-paste examples for the DROCAT backend scripts. Run them from the repository root with the `drocat-4.5.0` conda environment activated:

```bash
conda activate drocat-4.5.0
```

API tokens are loaded automatically from `config_local.json` / `config.json` (see [docs/INSTALLATION.md](../INSTALLATION.md#4-authentication)). Output is written to `local_data/` by default.

---

## 1. Quick Example — Pathfinding

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['KC.*'],       # Regex patterns supported
    targetNeurons=['MBON03'],     # Searches: bodyId → type → instance
    min_synapse_num=10,
)

fc.InitializeNeuronInfo()
fc.FindDirectConnections()  # Direct connections
# or
fc.FindAllPath()           # Multi-hop pathways
```

### Available Methods

| Method | Description | Use Case |
| --- | --- | --- |
| `FindDirectConnections()` | Direct synaptic connections | One-hop analysis |
| `FindAllPath()` | Multi-hop pathways | Circuit tracing |
| `FetchNeuronsOnly()` | Get neuron metadata only | Data exploration |

📖 [Basic Usage Guide](BasicUsage_Guide.md) — detailed parameters · [Score Calculation Guide](ScoreCalculation_Guide.md) — `connection_ratio`, `traversal_probability`, and path metrics · [PathFinding Methods](PathFinding_Methods.md) — algorithm selection for `FindAllPath`

---

## 2. Hemisphere-Aware Cross-Dataset Comparison

```python
from comparison import ComparisonParameters, ComparisonAnalyzer

params = ComparisonParameters(
    datasets=['male-cns:v0.9', 'flywire_FAFB_v783'],
    source_neurons=['aMe12'],
    target_neurons=['PPL101'],
    thresholds=[1, 3, 5],
    output_folder='/path/to/output',
    separate_hemispheres=True,  # Adds _L/_R/_U suffixes at type/group level
    keep_only_hemisphere_conserved_connections=True,  # Keep only L/R-conserved edges
    symmetry_analysis=True,     # Auto-enabled when separate_hemispheres=True
    find_reciprocal=True,       # Build reciprocal graphs and reports
)

analyzer = ComparisonAnalyzer(params, verbose=True)
analyzer.run_comparison()
analyzer.generate_report()
```

📖 [Cross-Dataset Comparison Guide](CrossDatasetComparison_Guide.md)

---

## 3. NeuronBridge Integration (EM→LM Mapping)

Find GAL4/Split-GAL4 driver lines matching your EM neurons:

```python
from src.neuronbridge_finder import NeuronBridgeFinder

finder = NeuronBridgeFinder(
    verbose=True,
    separate_splitgal4=True  # Separate GAL4 from Split-GAL4
)

# Find lines for a neuron type across multiple datasets
results = finder.find_lines_batch(
    queries='aMe12',
    dataset=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    output_dir='./results',
    download_img_for_top_n_lines=20
)
```

Key features:

- **Weighted score ranking**: lines ranked by `weighted_score = avg_score × coverage_ratio`
- **Multi-dataset search**: hemibrain, male-cns, FAFB, and standalone BANC
- **Co-labeling analysis**: specificity and overlap between driver lines
- **Automatic image download**: FlyLight imagery for top candidates

📖 [NeuronBridge Workflow Guide](NeuronBridge_Workflow.md)

---

## 4. Agent-Assisted Direct Analysis (no UI)

Ask your AI agent to use the [`drocat-usage`](../../skills/drocat-usage/SKILL.md) skill — it runs the backend scripts directly, keeps their relative paths correct, and inspects generated artifacts. Copy-paste example prompt:

```text
Use the DROCAT v4.5.0 direct-analysis skill. Run a cached FindPath analysis
from aMe12 to PPL101 in male-cns:v0.9, with max_interlayer=2, CSV output, and
save everything under local_data/agent_runs/aMe12_to_PPL101. Inspect the files,
summarize row counts and warnings, and finish by reporting the validated
artifacts. Do not open the UI or stop at a plan.
```

The bundled launcher can also be used directly:

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 \
  --script scripts/FindPath.py \
  --dry-run        # remove --dry-run to execute
```

Use `scripts/FindDirect.py` for one-hop edges, `scripts/PlotPath.py` for path HTML, `scripts/plot3dSkeleton.py` for morphology, and `scripts/InterDatasetComparator.py` for cross-dataset comparisons. The skill includes recipes for NeuronBridge, FlyLight, homologs, profiles, and the empty editable network canvas.

---

## 5. Output Examples

### NeuronBridge FindLines Output

```
NB-find-lines_aMe12_20241230/
├── line_summary.csv           # Ranked by weighted_score
├── gal4_lexa_summary.csv      # GAL4/LexA lines
├── split_gal4_summary.csv     # Split-GAL4 lines
├── all_lines.csv              # All matches
└── images/                    # Downloaded FlyLight images
```

### Pathfinding Output

```
connection_data/
├── network.html               # Interactive network
├── sankey.html                # Flow diagram
├── heatmap.html               # Connection matrix
└── paths.csv                  # Path data
```

📖 [Output Files Reference](../OUTPUT_FILES.md)

---

## 6. Example Scripts

Complete working examples live in [`archive/examples/`](../../archive/examples/):

| Example | Description |
| --- | --- |
| `basic/` | Basic pathfinding and visualization |
| `comparison/` | Cross-dataset comparison |
| `performance/` | Benchmarking |
| `visualization/` | Advanced visualization options |

The `archive/scripts_local/` folder holds local development variants of the main scripts (debug/one-off analysis); the canonical scripts live in `scripts/`.
