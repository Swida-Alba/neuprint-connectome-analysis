# *Drosophila* Connectome Analysis Toolkit (DROCAT) v4.5.0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/downloads/)

DROCAT is a Python toolkit for analyzing and visualizing connectome data from **all NeuPrint databases and FlyWire datasets** — type-based pathfinding, interactive network visualizations with neurotransmitter grouping, 3D neuron morphology rendering, cross-dataset comparison, and EM↔LM driver line mapping (NeuronBridge). Everything is available both through a web UI and as standalone scripts.

> [!TIP]
> 🤖 **Agent-assisted:** ask your AI agent to run the bundled
> [`drocat-install`](skills/drocat-install/SKILL.md) skill — it installs all
> dependencies, configures tokens, verifies the installation, and launches
> the web UI for you. Installing the project also installs the analysis skills,
> so the agent already has them afterward (no fetch is needed):
> [`drocat-usage`](skills/drocat-usage/SKILL.md) for tab-matched script analyses
> (a recipe for every analysis panel), and [`drocat-backend`](skills/drocat-backend/SKILL.md)
> for flexible composition of backend modules. New to agents? Start with the
> [agent-assisted install section](docs/INSTALLATION.md#5-agent-assisted-install).

---

## Table of Contents

- [*Drosophila* Connectome Analysis Toolkit (DROCAT) v4.5.0](#drosophila-connectome-analysis-toolkit-drocat-v450)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Quick Start](#quick-start)
  - [Documentation](#documentation)
  - [Supported Datasets](#supported-datasets)
    - [FlyWire / Codex (3, local files required)](#flywire--codex-3-local-files-required)
  - [What's New in v4.5.0](#whats-new-in-v450)
  - [Contributing](#contributing)
  - [License](#license)
  - [Support](#support)

---

## Key Features

| Feature | Details |
| --- | --- |
| **Dataset Support** | NeuPrint (hemibrain, male-cns, optic-lobe, manc) + FlyWire (FAFB, BANC), inter-dataset analysis |
| **EM↔LM Mapping** | NeuronBridge integration for GAL4/Split-GAL4 driver line discovery |
| **Visualization** | 3D skeletons, interactive networks, Sankey diagrams, heatmaps |
| **Similarity Tabs** | Connectivity (find similar + comparison, cross-dataset capable) and Morphology (find similar, intra-dataset) with connectivity-expanded candidates, ROI filtering, full-morphology downloads |
| **Analysis** | Multi-hop pathfinding, cross-dataset comparison, hemisphere-aware analysis |
| **Performance** | 10-100x speedup with local caching, Polars acceleration |

---

## Quick Start

**Requirements:** conda (auto-installed if missing) and internet access on first run.

**Option 1 — One-click install & launch.** After cloning the repository, double-click to open the launcher:

| Platform | Command |
| --- | --- |
| macOS / Linux | `mac_DROCAT.command` |
| Windows | `windows_DROCAT.bat` |

On first run it creates the versioned `drocat-4.5.0` Python 3.11 environment (via the bundled installer in `archive/install/`), installs the pinned dependencies, runs `pip check`, verifies the installation, and opens the web UI at **http://127.0.0.1:8080**. Later runs are self-healing: a missing or inconsistent environment is repaired automatically before starting. If the port is busy, the launcher offers a new one interactively.

**Option 2 — Agent-assisted install.** Copy the following prompt to your AI agent and let it finish cloning the repo, installing, verifying, and launching DROCAT:

> Fetch https://raw.githubusercontent.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/v4.5.0/skills/drocat-install/SKILL.md and follow it to finish cloning the repo, installing, verifying, and launching DROCAT on this machine.

For script analysis without the UI, the agent uses the checked-in analysis skills
(with the UI closed): [`drocat-usage`](skills/drocat-usage/SKILL.md) for
one-tab analyses and [`drocat-backend`](skills/drocat-backend/SKILL.md) for
flexible backend composition. They are part of the repository, so an installed
agent has them — no fetch is required.

**Manual launch** (after installation) — double-click `mac_DROCAT.command` (macOS / Linux) or `windows_DROCAT.bat` (Windows), or from a terminal:

```bash
conda activate drocat-4.5.0 && python ui/app.py
```

Every UI panel links to its own instruction guide (see [docs/ui_guides/README.html](docs/ui_guides/README.html)).

📖 **[Full Installation Guide](docs/INSTALLATION.md)** — installer details, manual setup, environment policy, token configuration (NeuPrint / CAVE), and agent-assisted install.

---

## Documentation

**Start here:**

| Guide | Description |
| --- | --- |
| **[Quick Start](docs/QUICK_START.md)** | First-time setup and basic examples |
| **[Installation](docs/INSTALLATION.md)** | One-click, agent-assisted, and manual install + token configuration |
| **[Script Examples](docs/core-features/ScriptExamples_Guide.md)** | Copy-paste code for pathfinding, comparison, NeuronBridge |
| **[Troubleshooting](docs/TROUBLESHOOTING.md)** | Common issues and solutions |
| **[Documentation Hub](docs/README.md)** | Full documentation index |

**Feature guides:**

| Feature | Guide | Script |
| --- | --- | --- |
| **Basic Usage** | [Basic Usage Guide](docs/core-features/BasicUsage_Guide.md) | `FindDirect.py`, `FindPath.py` |
| **Score Calculations** | [Score Calculation Guide](docs/core-features/ScoreCalculation_Guide.md) | All pathfinding scripts |
| **EM↔LM Mapping** | [NeuronBridge Guide](docs/core-features/NeuronBridge_Guide.md) | `NeuronBridge_FindLines.py` |
| **Cross-Dataset** | [Comparison Guide](docs/core-features/CrossDatasetComparison_Guide.md) | `InterDatasetComparator.py` |
| **Homolog Finding** | [Homolog Guide](docs/core-features/HomologFinding_Guide.md) | `FindHomologs.py` (Connectivity tab → Find Similar) |
| **3D Visualization** | [3D Skeleton Guide](docs/visualizations/3D_Skeleton_Guide.md) | `plot3dSkeleton.py` |
| **Path Visualization** | [Interaction Guide](docs/visualizations/VisualizePath_Interaction_Guide.md) | `PlotPath.py` |
| **Web UI Panels** | [docs/ui_guides/README.html](docs/ui_guides/README.html) | All web UI panels |
| **Output Files** | [Output Files Reference](docs/OUTPUT_FILES.md) | File formats |

---

## Supported Datasets

All NeuPrint server datasets are supported (verified against `api.neuprint.janelia.org`), plus the FlyWire/Codex datasets. NeuPrint datasets are fetched automatically; FlyWire datasets use local files (see the Settings tab).

<details>
<summary><b>NeuPrint (11 datasets)</b> — male-cns, hemibrain, optic-lobe, manc, fib19, mushroombody</summary>

| Dataset | Description (from NeuPrint server) |
| --- | --- |
| `male-cns:v1.0` | Complete MaleCNS connectome (Janelia FlyEM + Cambridge) — latest |
| `male-cns:v0.9` | Complete MaleCNS connectome |
| `hemibrain:v1.2.1` | Adult female brain reconstruction (central complex + surrounding neuropils) |
| `hemibrain:v1.1` | Older hemibrain release |
| `optic-lobe:v1.1` | Drosophila optic lobe (right lobe, ~50k neurons, subset of MaleCNS) |
| `optic-lobe:v1.0.1` | Fly optic lobe reconstruction (~50k neurons) |
| `manc:v1.2.3` | MANC connectome — latest |
| `manc:v1.2.1` | MANC connectome |
| `manc:v1.0` | MANC connectome (original) |
| `fib19:v1.0` | Partial reconstruction of the fly medulla / lobula / lobula plate |
| `mushroombody` | Fly alpha lobe in the mushroom body (983 neurons) |

</details>

### FlyWire / Codex (3, local files required)

| Dataset | Description |
| --- | --- |
| `flywire_FAFB_v783` | Female Adult Fly Brain (FAFB v783, 139,255 neurons) |
| `flywire_BANC_v888` | Brain and Nerve Cord (BANC v888, 158,262 neurons) |
| `flywire_BANC_v626` | Brain and Nerve Cord, older (BANC v626, 115,151 neurons) |

> BANC (Brain And Nerve Cord) is served via FlyWire/Codex as `flywire_BANC_v888` (local data files). The NeuPrint server metadata also lists a hidden `banc:v888` entry, but it is not queryable through the NeuPrint API and is therefore not supported.

📖 **[FlyWire Setup Guide](docs/FLYWIRE_USAGE.md)** · **[Available ROI Meshes](docs/AVAILABLE_ROIS.md)**

---

## What's New in v4.5.0

- **Script-first analysis with coding agents** — run pathfinding, comparison, NeuronBridge, FlyLight, homolog, profile, PlotPath, and 3D skeleton scripts without the UI, via the [`drocat-usage`](skills/drocat-usage/SKILL.md) skill and its `run_direct.py` launcher.
- **Local FAFB/BANC dataset support** — local-first caching for 10-100x faster FlyWire access ([FAFB Integration](docs/FAFB_INTEGRATION.md)).
- **NT visualization & grouping** — neurotransmitter edge groups, custom groups, export/import ([Network Features](docs/visualizations/VisualizePath_Network_Features.md)).
- **Similarity tab reorganization** — the Similarity group is now two main tabs, each with Find Similar / Comparison sub-tabs: **Connectivity** (find similar = homolog search across datasets or within one dataset via Target = Source; comparison = multi-dataset connectivity profiling) and **Morphology** (find similar = intra-dataset vector/NBLAST search; comparison = intra-dataset N×N morphology comparison with type-level + bodyId-level matrices, heatmaps, and a report — `vector_v2` or NBLAST scoring, NBLAST capped at 30 total neurons). The old "Connectivity similarity" mode is folded into Find Similar — it was intra-dataset homolog finding under another name. Morphological find-similar details: multiple queries run independently, connectivity-expanded candidates read directly from the connection cache (top-N×3 similar *types* expanded to all their members), ROI filtering, intra-type reference data, dual result tables (bodyId-level `results.csv` + type-level `type_summary.csv`), and query-plus-top-N 3D skeleton visualizations. NBLAST type means use ipsilateral pairs only (mirror scores are unreliable); the vector method lateral-normalizes and uses both sides. A full-morphology mode downloads every skeleton with a resumable progress/ETA pull and compares against the whole local population ([Connectivity guide](docs/ui_guides/connectivity.html), [Morphology guide](docs/ui_guides/morphology.html), [Comparison guide](docs/ui_guides/morphology_comparison.html)).
- **Palette editor** — drag-and-drop reordering of discrete palette colors, a range slider applied directly to the displayed palette, a reset button beside the preview, and lateral range labels.
- **3D Skeleton reorganization** — independent card blocks for general appearance, neuron colors, synapse colors, and brain-region ROIs, with hemisphere-aware options.

📖 **[Full changelog](docs/README.md#recent-updates)** · **[Agent-assisted install](docs/INSTALLATION.md#5-agent-assisted-install)**

---

## Contributing

Contributions are welcome — please open an issue or PR on [GitHub](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis).

## License

MIT License — see [LICENSE](LICENSE).

## Support

- **[Documentation](docs/README.md)** — full index
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** — common issues
- **[Issues](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/issues)** — bug reports and feature requests

When reporting an issue, include: Python version and OS, the full error traceback, and minimal code to reproduce.
