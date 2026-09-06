# FlyWire/FAFB/BANC Dataset Usage Guide

This toolkit supports analysis of FlyWire (FAFB) and BANC datasets using a
**local file-based workflow**. Download the raw files from the Codex/FlyWire
portal and place them in the exact `datasets/<dataset>/downloads/` directory;
the toolkit then converts them to Parquet for analysis. FAFB also has an
optional CAVE API path for workflows that explicitly request remote fetching;
BANC remains local-file only.

## 1. Data Preparation

### Step 1: Download Data Files
You need to download the required CSV files from the Codex Download Page (or equivalent source).

**For FAFB (flywire_FAFB_v783):**
- See [FAFB Integration Guide](FAFB_INTEGRATION.md) for the full file list.

**For BANC (`banc_v626` or `banc_v888`):**
- See [BANC Integration Guide](BANC_INTEGRATION.md).
- **Preparation is automatic:** metadata + connections download from the
  public BANC release bucket (no login, no token), and 3D skeleton
  visualization works in native BANC space with skeletons fetched from the
  same bucket.

### Step 2: Place Files in Directory
Create the directory structure in your project folder and place the downloaded files there.

**For FAFB:**
```
datasets/
  └── flywire_FAFB_v783/
      └── downloads/
            ├── classification.csv.gz
            ├── names.csv.gz
            ├── coordinates.csv.gz
            ├── neurons.csv.gz
            ├── cell_stats.csv.gz
            ├── consolidated_cell_types.csv.gz
            ├── connections_princeton_no_threshold.csv.gz
            ├── fafb_v783_princeton_synapse_table.csv.gz  (optional)
            └── sk_lod1_783_healed.zip                    (optional)
```

**For BANC:**
```
datasets/
  └── banc_v626/
      └── downloads/
            ├── neurons.csv.gz
            ├── connections_princeton.csv.gz
```

### Step 3: Run Conversion
You can run the conversion script manually, or it will run automatically the first time you try to use the dataset.

**Manual Conversion (FAFB):**
```bash
python src/FAFB_file_converter.py
```

**Manual Conversion (BANC v626 default):**
```bash
python src/BANC_file_converter.py
```

For v888, pass the selected dataset name to the converter function:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from BANC_file_converter import ensure_banc_data; d='banc_v888'; ensure_banc_data(d, 'datasets/' + d)"
```

This script will:
1.  Read the CSV files from the `downloads` folder.
2.  Merge and enrich the neuron metadata.
3.  Convert the data into optimized Parquet files (`.parquet`) for fast loading.
4.  Save the processed files in the dataset folder.

## 2. Using FlyWire/BANC Data in Analysis

Once the data is prepared, you can use it just like any other NeuPrint dataset.

### Example: Finding Connections

```python
from coana import FindNeuronConnection

# Initialize connection finder
fc = FindNeuronConnection(
    token='dummy_token',  # Token is ignored for local files
    dataset='banc_v626', # or 'flywire_FAFB_v783'
    sourceNeurons=['720575940621039145'],  # Use Root IDs
    targetNeurons=['720575940619419758'],
    min_synapse_num=5
)

# Run analysis
fc.InitializeNeuronInfo()
fc.FindDirectConnections()
```

### Example Script

Select the prepared dataset in the UI or in a direct `FindNeuronConnection`
script. The first run automatically checks the matching `downloads/` folder
and converts the raw files if the generated tables are absent.

### Example: Visualizing Skeletons (FAFB Only)

If you downloaded the `sk_lod1_783_healed.zip` file for FAFB, you can visualize 3D skeletons.

```python
from coana import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    neuron_layers=['720575940621039145'],
    brain_mesh='template',  # Uses FAFB template
    FAFB_template_correction=True # Default: True. Corrects the slight tilt of the FAFB template.
)

vs.plot_neurons()
```

### FAFB Tilt Correction
The FAFB/FlyWire template mesh has a slight tilt relative to the standard view axes. By default (`FAFB_template_correction=True`), `VisualizeSkeleton` applies a rotation correction to align the brain:
- **Z-axis rotation**: -4 degrees (corrects left-right tilt in front view)
- **Y-axis rotation**: -3 degrees (corrects tilt in top view)

This ensures that the brain appears straight in standard views (Front, Top, etc.). If you need the original raw coordinates (e.g., for alignment with other raw FAFB data), you can set `FAFB_template_correction=False`.

**Note:** FlyWire-based BANC dataset does not currently support skeleton visualization.

## 3. Important Notes

-   **Storage**: The converted Parquet files are much smaller and faster than the raw CSVs, but the initial raw files can be large. Ensure you have enough disk space.
-   **Updates**: If you want to update the data, simply delete the files in the dataset folder and place new CSV files in the `downloads` folder. The converter will run again.
-   **IDs**: FlyWire/BANC use long integer Root IDs (e.g., `720575940...`). Ensure you use these IDs in your queries.
