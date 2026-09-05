# VisualizePath Updates - January 2026

## Summary of Changes

This document summarizes the recent enhancements to the `VisualizePath` network visualization, including neurotransmitter (NT) edge grouping, enhanced hover information, and improved group management.

---

## 1. Neurotransmitter (NT) Edge Groups ✨

### Overview
The network visualization now automatically creates independent edge groups based on neurotransmitter types. This allows you to select, color, and manage edges by their neurotransmitter identity.

### Supported NT Types
| NT Type | Full Name     | Color            |
| ------- | ------------- | ---------------- |
| ACH     | Acetylcholine | #F39C12 (Orange) |
| GABA    | GABA          | #27AE60 (Green)  |
| GLUT    | Glutamate     | #E74C3C (Red)    |
| DA      | Dopamine      | #9B59B6 (Purple) |
| SER     | Serotonin     | #3498DB (Blue)   |
| OCT     | Octopamine    | #1ABC9C (Teal)   |

### How It Works
1. **Automatic Detection**: NT types are read from the `nt_type` column in your data
2. **Dynamic Grouping**: NT groups appear in the Group Selector dropdown
3. **Color-Coded Display**: Each NT type has a distinct color for easy identification

### Group Selector Dropdown
The dropdown now includes three categories:
```
┌─ Node Types ──────────────────┐
│  ○ All Source Nodes           │
│  ○ All Intermediate Nodes     │
│  ○ All Target Nodes           │
├─ Edge Types ──────────────────┤
│  ○ All Positive Edges         │
│  ○ All Negative Edges         │
├─ NT Edge Groups ──────────────┤
│  ○ ACH Edges                  │
│  ○ GABA Edges                 │
│  ○ GLUT Edges                 │
│  ...                          │
└───────────────────────────────┘
```

### Input Data Format
To enable NT edge grouping, your input data should include an `nt_type` column:

```python
import pandas as pd

# Path-based format with NT types
paths_df = pd.DataFrame({
    'path_block': ['A -> B -> C', 'D -> E -> F'],
    'weights': [[100, 50], [80, 40]],
    'nt_types': [['ACH', 'GABA'], ['GLUT', 'ACH']]  # NT for each connection
})

# Edge-list format with NT type
edges_df = pd.DataFrame({
    'source': ['A', 'B', 'C'],
    'target': ['B', 'C', 'D'],
    'weight': [100, 50, 30],
    'nt_type': ['ACH', 'GABA', 'GLUT']  # NT for each edge
})
```

### Using NT Edge Groups
1. **Open the network HTML** in a browser
2. **Expand the "Group Selection" panel** on the left sidebar
3. **Select an NT group** from the dropdown (e.g., "ACH Edges")
4. **Adjust color/opacity** using the sliders
5. **Click "Apply to Group"** to update all edges of that NT type

---

## 2. Enhanced Hover Information 🔍

### Edge Hover Details
Hovering over an edge now displays comprehensive information including NT type:

```
┌─────────────────────────────────────┐
│ Connection: A → B                   │
│ Weight: 150 synapses ⬅ Current      │
│ Ratio: 0.2500                       │
│ Probability: 0.8500                 │
│ NT: ACH                             │  ← NEW!
│ ─────────────────                   │
│ Custom Label 1: value               │
│ Custom Label 2: value               │
└─────────────────────────────────────┘
```

### Color-Coded NT Display
The NT type is displayed with its characteristic color:
- **ACH**: Orange text
- **GABA**: Green text
- **GLUT**: Red text
- **DA**: Purple text
- **SER**: Blue text
- **OCT**: Teal text

---

## 3. Improved Group Management 🎛️

### Group Selection Features
- **Select any group** from the dropdown to apply colors/opacity
- **NT groups** are dynamically generated based on your data
- **Mixed selection** - select nodes or edges independently

### Color Application
1. Select a group (e.g., "GABA Edges")
2. Choose a color using the color picker
3. Adjust opacity with the slider
4. Click "Apply to Group" to update all elements in that group

### Keyboard Shortcuts
| Key | Action              |
| --- | ------------------- |
| H   | Hide selected nodes |
| E   | Hide selected edges |
| ESC | Clear selection     |

### Right-Click Context Actions
- **Right-click on node**: Hide node and connected edges
- **Right-click on edge**: Hide single edge
- **In Edit Mode**: Right-click deletes instead of hides

---

## 4. Edit Mode Improvements 🔧

### Fixed Issues
- **Event handler conflicts**: Fixed issue where disabling edit mode would break click handlers
- **Namespace isolation**: Edit mode handlers now use namespaced events to avoid conflicts

### Edit Mode Features
When enabled:
- **Drag nodes** to reposition
- **Double-click node** to edit label/type
- **Double-click edge** to edit properties
- **Right-click** to delete (instead of hide)
- **Draw new edges** between nodes

---

## 5. Input File Format Updates 📄

### NT Types Column
For path-based input files, add `nt_types` column:

```csv
path_block,weights,nt_types
"A -> B -> C","[100, 50]","['ACH', 'GABA']"
"D -> E -> F","[80, 40]","['GLUT', 'ACH']"
```

### Edge-List with NT
For edge-list input files, add `nt_type` column:

```csv
source,target,weight,nt_type
A,B,100,ACH
B,C,50,GABA
C,D,30,GLUT
```

### Automatic NT Detection
When using FindPath.py or similar scripts with FAFB/FlyWire data:
- NT types are automatically fetched from the database
- Added to connections via `nt_type` column
- Passed through to visualizations

---

## 6. Code Changes Summary 💻

### vispath.py Changes
```python
# NT edge groups automatically generated from data
nt_edge_group_options = ""
if nt_types_in_data:
    for nt_type in unique_nt_types:
        nt_edge_group_options += f'<option value="nt_{nt_type.lower()}">{nt_type} Edges</option>'

# Group selector includes NT options
<select id="groupSelector">
    <optgroup label="Node Types">...</optgroup>
    <optgroup label="Edge Types">...</optgroup>
    <optgroup label="NT Edge Groups">{nt_edge_group_options}</optgroup>
</select>
```

### JavaScript Functions Updated
- `getNTColor()` - Returns color for NT type (supports full and abbreviated names)
- `updateGroupControls()` - Handles NT group selection
- `applyGroupColor()` - Applies colors to NT edge groups
- `selectGroup()` - Selects all edges of a specific NT type

---

## 7. Example Workflow 🔄

### Analyzing Neurotransmitter Distribution
```python
from vispath_pkg import VisualizePath

# Load path data with NT types
vp = VisualizePath(
    'pathways_with_nt.xlsx',
    output_folder='./nt_analysis'
)
vp.visualize()

# Open the generated HTML
# 1. Expand "Group Selection" panel
# 2. Select "ACH Edges" - see all cholinergic connections
# 3. Change color to highlight
# 4. Select "GABA Edges" - compare with GABAergic connections
```

### Highlighting Inhibitory vs Excitatory
```python
# In the browser:
# 1. Select "GABA Edges" (inhibitory)
# 2. Set color to green, opacity to 100%
# 3. Apply to group
# 4. Select "ACH Edges" (excitatory)  
# 5. Set color to red, opacity to 100%
# 6. Apply to group
# Now you have clear visual separation of inhibitory vs excitatory pathways!
# (GLUT is treated as inhibitory in the NT group coloring, alongside GABA.)
```

---

## 8. Troubleshooting 🔧

### NT Groups Not Appearing
- Ensure your data has `nt_type` or `nt_types` column
- Check that NT values are non-empty strings
- Verify NT types match expected values (ACH, GABA, GLUT, DA, SER, OCT)

### Group Color Not Applying
- Make sure you click "Apply to Group" after selecting color
- Check that the correct group is selected in the dropdown
- Verify edges exist in that group

### Edit Mode Issues
- If click handlers seem broken, refresh the page
- Edit mode handlers now use namespaced events to avoid conflicts
- Disabling edit mode preserves other click functionality

---

## Summary

### Key Features Added
✅ NT edge groups in group selector dropdown  
✅ NT type display in edge hover tooltip  
✅ Color-coded NT visualization  
✅ Fixed edit mode event handler conflicts  
✅ Improved group management UI  
✅ Updated input file format documentation  

### Benefits
- **Better Analysis**: Visualize connections by neurotransmitter type
- **Easy Comparison**: Quickly compare ACH vs GABA vs GLUT pathways
- **Publication Ready**: Color-code edges by NT for figures
- **Robust Interaction**: Fixed click/edit mode conflicts

---

**Version**: January 2026  
**Status**: All features tested and documented ✓
