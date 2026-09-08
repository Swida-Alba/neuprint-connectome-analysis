# Available ROI Meshes by Dataset

This document lists all available Region of Interest (ROI) meshes for each NeuPrint dataset.
Use these ROI names with the `mesh_roi` parameter in `VisualizeSkeleton`.

## Usage

```python
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    mesh_roi=['EB', 'LH', 'AL'],  # ROIs will auto-expand to bilateral variants
    # ...
)
```

**Note:** ROIs without (L)/(R) suffix will automatically expand to both hemispheres if available.
For example, `'LH'` expands to `['LH(L)', 'LH(R)']`.

---

## hemibrain:v1.2.1

**Total ROIs:** 230

### Common Brain Regions (Bilateral)

| Region | Description                                              |
| ------ | -------------------------------------------------------- |
| `LH`   | Lateral Horn → LH(L), LH(R)                              |
| `AL`   | Antennal Lobe → AL(L), AL(R)                             |
| `MB`   | Mushroom Body components: CA, PED, aL, a'L, bL, b'L, gL  |
| `SLP`  | Superior Lateral Protocerebrum → SLP(L), SLP(R)          |
| `SMP`  | Superior Medial Protocerebrum → SMP(L), SMP(R)           |
| `SIP`  | Superior Intermediate Protocerebrum → SIP(L), SIP(R)     |
| `AOTU` | Anterior Optic Tubercle → AOTU(L), AOTU(R)               |
| `AVLP` | Anterior Ventrolateral Protocerebrum → AVLP(L), AVLP(R)  |
| `PVLP` | Posterior Ventrolateral Protocerebrum → PVLP(L), PVLP(R) |
| `PLP`  | Posteriorlateral Protocerebrum → PLP(L), PLP(R)          |
| `WED`  | Wedge → WED(L), WED(R)                                   |
| `LAL`  | Lateral Accessory Lobe → LAL(L), LAL(R)                  |
| `CRE`  | Crepine → CRE(L), CRE(R)                                 |
| `SCL`  | Superior Clamp → SCL(L), SCL(R)                          |
| `ICL`  | Inferior Clamp → ICL(L), ICL(R)                          |
| `IB`   | Inferior Bridge (unpaired)                               |
| `ATL`  | Antler → ATL(L), ATL(R)                                  |
| `AB`   | Asymmetrical Body → AB(L), AB(R)                         |

### Central Complex (Unpaired)

| Region | Description          |
| ------ | -------------------- |
| `EB`   | Ellipsoid Body       |
| `FB`   | Fan-shaped Body      |
| `PB`   | Protocerebral Bridge |
| `NO`   | Nodulus              |

### Antennal Lobe Glomeruli

Detailed glomerulus subdivisions available, e.g., `AL-DA1`, `AL-DA2`, `AL-DM1`, etc.
Most have bilateral variants: `AL-DA2(L)`, `AL-DA2(R)`

---

## male-cns:v0.9

**Total ROIs:** 5412

### Common Brain Regions (Bilateral)

| Region | Description                                              |
| ------ | -------------------------------------------------------- |
| `LH`   | Lateral Horn → LH(L), LH(R)                              |
| `AL`   | Antennal Lobe → AL(L), AL(R)                             |
| `EB`   | Ellipsoid Body (unpaired)                                |
| `FB`   | Fan-shaped Body (unpaired)                               |
| `CA`   | Calyx → CA(L), CA(R)                                     |
| `PED`  | Peduncle → PED(L), PED(R)                                |
| `SLP`  | Superior Lateral Protocerebrum → SLP(L), SLP(R)          |
| `SMP`  | Superior Medial Protocerebrum → SMP(L), SMP(R)           |
| `SIP`  | Superior Intermediate Protocerebrum → SIP(L), SIP(R)     |
| `AOTU` | Anterior Optic Tubercle → AOTU(L), AOTU(R)               |
| `AVLP` | Anterior Ventrolateral Protocerebrum → AVLP(L), AVLP(R)  |
| `PVLP` | Posterior Ventrolateral Protocerebrum → PVLP(L), PVLP(R) |
| `PLP`  | Posteriorlateral Protocerebrum → PLP(L), PLP(R)          |
| `LAL`  | Lateral Accessory Lobe → LAL(L), LAL(R)                  |
| `CRE`  | Crepine → CRE(L), CRE(R)                                 |
| `AMMC` | Antennal Mechanosensory Motor Center → AMMC(L), AMMC(R)  |
| `GOR`  | Gorget → GOR(L), GOR(R)                                  |

### Optic Lobe Regions

| Region | Description                        |
| ------ | ---------------------------------- |
| `ME`   | Medulla → ME(L), ME(R)             |
| `LO`   | Lobula → LO(L), LO(R)              |
| `LOP`  | Lobula Plate → LOP(L), LOP(R)      |
| `LA`   | Lamina → LA(L), LA(R)              |
| `AME`  | Accessory Medulla → AME(L), AME(R) |

### VNC Regions (Bilateral)

| Region      | Description                                    |
| ----------- | ---------------------------------------------- |
| `LegNp(T1)` | T1 Leg Neuropil → LegNp(T1)(L), LegNp(T1)(R)   |
| `LegNp(T2)` | T2 Leg Neuropil → LegNp(T2)(L), LegNp(T2)(R)   |
| `LegNp(T3)` | T3 Leg Neuropil → LegNp(T3)(L), LegNp(T3)(R)   |
| `mVAC(T1)`  | T1 Mesothoracic VAC → mVAC(T1)(L), mVAC(T1)(R) |
| `mVAC(T2)`  | T2 Mesothoracic VAC → mVAC(T2)(L), mVAC(T2)(R) |
| `mVAC(T3)`  | T3 Mesothoracic VAC → mVAC(T3)(L), mVAC(T3)(R) |
| `Ov`        | Ovoid → Ov(L), Ov(R)                           |

### Detailed Column/Layer ROIs

Male-CNS has thousands of detailed ME/LO/LOP column ROIs like:
- `ME_L_col_10_15`, `ME_R_col_20_25`
- `LO_L_col_15_20`, `LOP_R_col_25_30`
- Layer ROIs: `ME_L_layer_01` through `ME_L_layer_10`

---

## manc:v1.0

**Total ROIs:** 61

### VNC Regions (Bilateral)

| Region      | Description                                               |
| ----------- | --------------------------------------------------------- |
| `LegNp(T1)` | T1 Leg Neuropil → LegNp(T1)(L), LegNp(T1)(R)              |
| `LegNp(T2)` | T2 Leg Neuropil → LegNp(T2)(L), LegNp(T2)(R)              |
| `LegNp(T3)` | T3 Leg Neuropil → LegNp(T3)(L), LegNp(T3)(R)              |
| `mVAC(T1)`  | T1 Mesothoracic VAC → mVAC(T1)(L), mVAC(T1)(R)            |
| `mVAC(T2)`  | T2 Mesothoracic VAC → mVAC(T2)(L), mVAC(T2)(R)            |
| `mVAC(T3)`  | T3 Mesothoracic VAC → mVAC(T3)(L), mVAC(T3)(R)            |
| `Ov`        | Ovoid → Ov(L), Ov(R)                                      |
| `GF`        | Giant Fiber → GF(L), GF(R)                                |
| `ADMN`      | Anterior Dorsal Mesothoracic Neuropil → ADMN(L), ADMN(R)  |
| `PDMN`      | Posterior Dorsal Mesothoracic Neuropil → PDMN(L), PDMN(R) |

### Nerves

| Region   | Description                                         |
| -------- | --------------------------------------------------- |
| `AbN1-4` | Abdominal Nerves → AbN1(L), AbN1(R), etc.           |
| `CvN`    | Cervical Nerve → CvN(L), CvN(R)                     |
| `MesoAN` | Mesothoracic Accessory Nerve → MesoAN(L), MesoAN(R) |
| `MesoLN` | Mesothoracic Leg Nerve → MesoLN(L), MesoLN(R)       |
| `MetaLN` | Metathoracic Leg Nerve → MetaLN(L), MetaLN(R)       |
| `ProAN`  | Prothoracic Accessory Nerve → ProAN(L), ProAN(R)    |
| `ProCN`  | Prothoracic Cervical Nerve → ProCN(L), ProCN(R)     |
| `ProLN`  | Prothoracic Leg Nerve → ProLN(L), ProLN(R)          |

### Unpaired Regions

`ANm`, `AbNT`, `CV`, `IntTct`, `LTct`

---

## optic-lobe:v1.1

**Total ROIs:** 2690

### Brain Regions (Bilateral)

Same as male-cns:v0.9 brain regions including `LH`, `AL`, `EB`, `FB`, etc.

### Optic Lobe Regions

| Region | Description                        |
| ------ | ---------------------------------- |
| `ME`   | Medulla → ME(L), ME(R)             |
| `LO`   | Lobula → LO(L), LO(R)              |
| `LOP`  | Lobula Plate → LOP(L), LOP(R)      |
| `LA`   | Lamina → LA(R)                     |
| `AME`  | Accessory Medulla → AME(L), AME(R) |

### Detailed Column/Layer ROIs

Extensive column ROIs for right hemisphere optic lobe:
- `ME_R_col_XX_YY` - Medulla columns
- `LO_R_col_XX_YY` - Lobula columns  
- `LOP_R_col_XX_YY` - Lobula Plate columns
- Layer ROIs: `ME_R_layer_01` through `ME_R_layer_10`

---

## flywire_FAFB_v783 (FAFB local release)

**Note:** FAFB does not have native ROI meshes in NeuPrint. ROI meshes are fetched from **male-cns:v0.9** and transformed to FLYWIRE coordinates.

**Available ROIs:** Same as male-cns:v0.9 (see above)

**Usage:**
```python
vs = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    mesh_roi=['EB', 'LH(R)', 'AL(R)'],  # Use male-cns ROI names
    brain_mesh='template',  # Uses native FLYWIRE coordinates
)
```

---

## Notes

1. **ROI Expansion:** When you specify a base ROI name like `'LH'`, it automatically expands to `['LH(L)', 'LH(R)']` if both exist in the dataset.

2. **Caching:** ROI meshes are cached locally after first download to speed up subsequent visualizations.

3. **Coordinate Transforms:** For FAFB, ROI meshes from male-cns are automatically transformed to FLYWIRE coordinates.

4. **Primary ROIs:** Some datasets have "primary ROIs" which are the main anatomical regions, as opposed to subdivisions.

5. **Full ROI Lists:** The complete ROI lists are stored as JSON files in the `cache/` directory:
   - `cache/hemibrain_v1_2_1/available_rois.json`
   - `cache/male-cns_v0_9/available_rois.json`
   - `cache/manc_v1_0/available_rois.json`
   - `cache/optic-lobe_v1_1/available_rois.json`
   - FlyWire FAFB uses ROIs transformed from the male-cns:v0.9 dataset.
