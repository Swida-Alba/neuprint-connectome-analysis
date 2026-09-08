# Cross-module composition recipes

Build custom workflows by composing backend modules. Each recipe passes the
output artifact of one module into the next. The full method/parameter reference
is in [`module-index.md`](module-index.md).

## Pathfind → analyze → visualize (coana → comparison → vispath)

```python
# 1. coana: find paths and write a CSV
from coana import FindNeuronConnection
fc = FindNeuronConnection(dataset="male-cns:v0.9", sourceNeurons=["aMe12"],
                          targetNeurons=["PPL101"], output_dir="/abs/output/paths",
                          min_synapse_num=3, max_interlayer=2, filter_by="bodyId",
                          skip_bodyId=True, output_format="csv", use_cache=True, showfig=False)
fc.InitializeNeuronInfo()
fc.FindAllPath(forward_only=True, find_reciprocal=False)

# 2. vispath: render it
from vispath_pkg import VisualizePath
vp = VisualizePath(path_file="/abs/output/paths/path_data.csv",
                   output_folder="/abs/output/network", network_layout="hierarchical",
                   showfig=False)
connections, graph = vp.visualize()
```

## Cache → morphology → connectivity find-similar → skeleton (the Similarity pipeline)

```python
# 1. coana: ensure the connection cache for the dataset
from coana import FindNeuronConnection
fc = FindNeuronConnection(dataset="male-cns:v0.9", use_cache=True, cache_only=False)
fc.InitializeNeuronInfo()
fc.build_connection_cache(neuron_types=["aMe12"])

# 2. morphology: screen by morphology
from morphology import MorphologyComparer
comparer = MorphologyComparer(query="aMe12", dataset="male-cns:v0.9",
                              candidate_source="auto", candidate_cap=200,
                              output_dir="/abs/output/similar_morph", use_cache=True)
results = comparer.find_similar()

# 3. connectivity find-similar (HomologFinder, Target = Source for intra-dataset)
from comparison.profile_comparator import HomologFinder
finder = HomologFinder(source="aMe12", source_dataset="male-cns:v0.9",
                       target_dataset="male-cns:v0.9", output_dir="/abs/output/similar_profile",
                       use_cache=True, verbose=True)
finder.find_homologs_fast()

# 4. skeleton: render top candidates
from visualize_skeleton import VisualizeSkeleton
vs = VisualizeSkeleton(dataset="male-cns:v0.9", neuron_layers=[["aMe12"], ["aMe10", "aMe9"]],
                       output_dir="/abs/output/skeleton", skeleton_mode="line",
                       show_fig=False, skip_synapse=True)
vs.plot_neurons()
```

## Comparison (multi-dataset) → conserved-path visualization

```python
from comparison import ComparisonParameters, ComparisonAnalyzer
params = ComparisonParameters(datasets=["hemibrain:v1.2.1", "male-cns:v0.9"],
                              source_neurons=["aMe12"], target_neurons=["PPL101"],
                              output_folder="/abs/output/comparison",
                              comparison_mode="edge", max_interlayer=2,
                              thresholds=[1, 3, 5, 10], skip_bodyId=True, auto_type_mapping=True)
analyzer = ComparisonAnalyzer(params, verbose=True)
analyzer.run_comparison()
analyzer.export_results()

from vispath_pkg import VisualizePath
vp = VisualizePath(path_file="/abs/output/comparison/<conserved>.csv",
                   output_folder="/abs/output/comparison/net",
                   network_layout="distributed", showfig=False)
vp.visualize()
```

## Driver lines → FlyLight images → summary

```python
from neuronbridge_finder import NeuronBridgeFinder
finder = NeuronBridgeFinder(verbose=True)
finder.find_lines_batch(queries=["aMe12"], dataset="male-cns:v0.9",
                        output_dir="/abs/output/nb_lines", match_type="both",
                        download_images="flylight", download_img_for_top_n_lines=3,
                        max_download_images_per_line=5)

from flylight_downloader import FlyLightDownloader
dl = FlyLightDownloader(output_dir="/abs/output/flylight", formats=["png"],
                        image_types=["mip"], max_workers=4, verbose="pbar")
dl.download(line_name=["SS00001"], output_dir="/abs/output/flylight",
            max_files=20, generate_summary=["pdf"])
```

## Homologs (cross-dataset) → skeleton of hits

```python
from comparison.profile_comparator import HomologFinder
finder = HomologFinder(source="aMe12", source_dataset="male-cns:v0.9",
                       target_dataset="hemibrain:v1.2.1", output_dir="/abs/output/homologs",
                       top_n=20, similarity_metric="rank_union", vector_prefiltering=True,
                       visualize_skeleton=True, visualize_top_n=5,
                       visualization_settings={}, use_auto_type_mapping=True, verbose=True)
finder.find_homologs_fast()

from visualize_skeleton import VisualizeSkeleton
vs = VisualizeSkeleton(dataset="hemibrain:v1.2.1",
                       neuron_layers=[["aMe12"], ["upstream/hemibrain types"]],
                       output_dir="/abs/output/skeleton", skeleton_mode="line",
                       show_fig=False, skip_synapse=True)
vs.plot_neurons()
```

## General guidance

- Chain by passing files — find the actual produced path, don't guess filenames.
- Distinct resources: NeuPrint caches vs FAFB local files vs BANC public-release
  files vs the shared skeleton cache;
  one does not replace the other.
- Keep `showfig=False` and bound graph size until artifacts are validated.
- When mixing datasets, keep `use_auto_type_mapping`/`LabelMapper` consistent to
  avoid cross-dataset label mismatches.
