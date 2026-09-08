"""Plan 2026-09-07 Phase E: exported UserGuide pathfinding content.

Verifies that all three pathfinding tool specs (find_path, find_shortest,
inter_dataset) carry the reusable explanation block, that the block renders
in every output format, that the new file patterns match the artifacts the
backend now writes, and that the glossary documents the new metadata keys.
"""

import json

from pathlib import Path

import ui.output_guide as og


def _make_run_folder(tmp_path):
    run = tmp_path / "find-paths-complete_TEST_src_to_tgt_L2w3"
    run.mkdir()
    (run / "parameters.txt").write_text("requested_threshold: 3\n")
    (run / "data_details").mkdir()
    (run / "data_details" / "untyped_dropped_records.csv").write_text("a\n")
    viz = run / "visualization" / "visualization_data"
    viz.mkdir(parents=True)
    (run / "visualization" / "Network_run.html").write_text("x")
    (viz / "run_data_connections.csv").write_text("a\n")
    (viz / "type_paths_visualized.csv").write_text("p\n")
    bviz = run / "bodyId_visualization" / "visualization_data"
    bviz.mkdir(parents=True)
    (run / "bodyId_visualization" / "Network_run.html").write_text("x")
    (run / "bodyId_visualization" / "Heatmap_run.html").write_text("x")
    (run / "bodyId_visualization" / "Sankey_run.html").write_text("x")
    (bviz / "run_data_connections.csv").write_text("a\n")
    (bviz / "bodyId_paths_visualized.csv").write_text("p\n")
    return run


def test_all_three_pathfinding_specs_carry_explanation():
    for tool in ("find_path", "find_shortest", "inter_dataset"):
        spec = og.TOOL_GUIDE_SPECS[tool]
        assert spec.get("explanation"), tool
        headings = [s["heading"] for s in spec["explanation"]]
        assert any("path set was produced" in h for h in headings), tool
        assert any("vocabulary" in h for h in headings), tool


def test_explanation_pipeline_order():
    sections = og.TOOL_GUIDE_SPECS["find_path"]["explanation"]
    pipeline = next(s for s in sections if s.get("pipeline"))["pipeline"]
    text = "\n".join(pipeline)
    assert "requested threshold" in text
    assert "lossless hop/dead-end pruning" in text
    assert "Edge Budget floor w0" in text
    assert "StrongestFirst path-budget ordering and tau" in text
    assert "applied threshold / retained path set" in text
    assert "visualization-only edge limit" in text
    # reader order: threshold before floor before budget before drawing cap
    assert text.index("lossless") < text.index("Edge Budget") \
        < text.index("StrongestFirst") < text.index("visualization-only")


def test_explanation_vocabulary_table():
    sections = og.TOOL_GUIDE_SPECS["find_shortest"]["explanation"]
    table = next(s for s in sections if s.get("table"))["table"]
    flat = " ".join(str(cell) for row in table for cell in row)
    for token in ("W*", "tau", "w2", "w1", "w0", "applied_threshold",
                  "paths_complete", "pruned", "bottleneck"):
        assert token in flat, token


def test_new_file_patterns_match_artifacts(tmp_path):
    run = _make_run_folder(tmp_path)
    content = og.assemble_run_content(run, "find_path", {})
    matched = {entry["pattern"]: entry["matched"]
               for entry in content["entries"] if entry["matched"]}
    assert matched["data_details/untyped_dropped_records.csv"]
    assert matched["visualization/visualization_data/"
                   "type_paths_visualized.csv"]
    assert matched["bodyId_visualization/Network_*.html"]
    assert matched["bodyId_visualization/Heatmap_*.html"]
    assert matched["bodyId_visualization/Sankey_*.html"]
    assert matched["bodyId_visualization/visualization_data/"
                   "*_data_connections.csv"]
    assert matched["bodyId_visualization/visualization_data/"
                   "bodyId_paths_visualized.csv"]


def test_explanation_renders_in_all_formats(tmp_path):
    run = _make_run_folder(tmp_path)
    content = og.assemble_run_content(run, "find_shortest", {})
    assert content["explanation"]

    html = og.render_html(content)
    assert "Pathfinding model" in html
    assert "How the path set was produced" in html
    assert "Drop Untyped Neurons" in html

    md = og.render_markdown(content)
    assert "## Pathfinding model" in md
    assert "requested threshold" in md
    assert "| W* |" in md

    txt = og.render_txt(content)
    assert "PATHFINDING MODEL" in txt
    assert "-> requested threshold" in txt
    assert "W*" in txt


def test_write_run_guide_contains_model(tmp_path):
    run = _make_run_folder(tmp_path)
    path = og.write_run_guide(run, "find_path", {}, fmt="markdown")
    text = path.read_text(encoding="utf-8")
    assert "Pathfinding model" in text
    assert "applied_threshold" in text
    assert "untyped_dropped_records" in text


def test_glossary_documents_new_keys():
    for key in ("edge_budget_landing", "strongest_retained_bottleneck",
                "strongest_dropped_bottleneck", "applied_threshold_source",
                "strongest_first_budget", "tau_canonical",
                "requested_threshold", "drop_untyped", "untyped_side",
                "untyped_dropped_rows", "untyped_dropped_neurons",
                "edge_budget", "edge_budget_applied",
                "strongest_first_tau"):
        description, _range = og.glossary_entry(key)
        assert description != og.glossary_entry("__missing__")[0], key


def test_inter_dataset_nested_outputs_point_to_explanation():
    spec = og.TOOL_GUIDE_SPECS["inter_dataset"]
    dataset_entry = next(e for e in spec["files"]
                         if e["pattern"] == "dataset_data/**")
    assert "pathfinding" in dataset_entry["description"]
    assert spec.get("explanation")


# ---------------------------------------------------------------------------
# Applied threshold / bottleneck values in the exported guide
# ---------------------------------------------------------------------------

_PROVENANCE = {
    "requested_threshold": 3,
    "applied_threshold": 8,
    "applied_threshold_source": "strongest_first_budget",
    "strongest_first_budget": 1000000,
    "strongest_first_budget_bitten": True,
    "strongest_first_tau": 12.0,
    "tau_canonical": 8,
    "strongest_dropped_bottleneck": 7.0,
    "edge_budget": 1000000,
    "edge_budget_applied": False,
    "edge_budget_landing": None,
    "edge_weight_floor": None,
    "strongest_retained_bottleneck": 15.0,
    "paths_complete": False,
}


def _add_provenance(run, provenance=None, use_json=True):
    provenance = provenance or dict(_PROVENANCE)
    if use_json:
        (run / "all_attributes.json").write_text(
            json.dumps(provenance), encoding="utf-8")
    else:
        lines = ["Parameters for processing src to tgt:"]
        for key, value in provenance.items():
            if value is None:
                value = "not applied" if "floor" in key else "not reached"
            lines.append(f"{key}: {value}")
        (run / "parameters.txt").write_text("\n".join(lines) + "\n",
                                            encoding="utf-8")


def test_applied_state_read_from_all_attributes(tmp_path):
    run = _make_run_folder(tmp_path)
    _add_provenance(run)
    state = og._read_applied_state(run)
    assert state["applied_threshold"] == 8
    assert state["strongest_first_tau"] == 12.0
    assert state["paths_complete"] is False


def test_applied_state_fallback_parses_parameters_txt(tmp_path):
    run = _make_run_folder(tmp_path)
    _add_provenance(run, use_json=False)
    state = og._read_applied_state(run)
    assert state["applied_threshold"] == 8
    # the legacy 'applied_tau (min path bottleneck)' alias maps to tau
    assert state["strongest_first_tau"] == 12.0
    assert state["edge_weight_floor"] is None
    assert state["paths_complete"] is False


def test_applied_block_renders_under_parameters_in_all_formats(tmp_path):
    run = _make_run_folder(tmp_path)
    _add_provenance(run)
    content = og.assemble_run_content(run, "find_shortest", {})

    assert content["applied"]["applied_threshold"] == 8
    headline = og._applied_headline(content["applied"])
    assert "EXACTLY a complete run at Min Synapse Count = 8" in headline
    assert "strongest_first_budget" in headline

    html = og.render_html(content)
    assert "Applied threshold (this run)" in html
    assert "EXACTLY a complete run at Min Synapse Count = 8" in html
    assert "<code>12</code>" in html          # tau row

    md = og.render_markdown(content)
    assert "## Applied threshold (this run)" in md
    assert "applied threshold (equivalent Min Synapse Count)" in md
    assert "w0 (Edge Budget floor)" in md

    txt = og.render_txt(content)
    assert "APPLIED THRESHOLD (THIS RUN)" in txt
    assert "EXACTLY a complete run at Min Synapse Count = 8" in txt
    assert "w2 (strongest dropped path bottleneck): 7" in txt


def test_vocabulary_table_shows_this_run_values(tmp_path):
    run = _make_run_folder(tmp_path)
    _add_provenance(run)
    content = og.assemble_run_content(run, "find_path", {})

    html = og.render_html(content)
    assert "<th>This run</th>" in html
    # tau row shows this run's landing tau; w2 row shows the dropped value
    assert "<td>StrongestFirst landing" in html
    assert "<code>7</code>" in html
    # 'pruned' maps onto edge_budget_applied for the run state
    assert "<code>False</code>" in html

    txt = og.render_txt(content)
    assert "This run" in txt
    md = og.render_markdown(content)
    assert "| This run |" in md


def test_no_provenance_omits_applied_block(tmp_path):
    run = _make_run_folder(tmp_path)  # parameters.txt lacks applied_threshold
    content = og.assemble_run_content(run, "find_path", {})
    assert content["applied"] is None
    assert content["applied_by_dataset"] is None

    html = og.render_html(content)
    assert "Applied threshold (this run)" not in html
    assert "<th>This run</th>" not in html
    assert "Pathfinding model" in html  # model section still renders


def test_by_dataset_banner_renders_for_comparison_runs(tmp_path):
    run = tmp_path / "cross-dataset_TEST"
    run.mkdir()
    (run / "effective_thresholds.json").write_text(json.dumps({
        "banner": ["hemibrain: asked [1, 3, 5] -> applied [1, 9] (τ=9)"],
        "datasets": {
            "hemibrain:v1.2.1": {
                "input": [1, 3, 5],
                "effective": [1, 9],
                "skipped": [3, 5],
                "applied_folder": {"3": 9, "5": 9},
                "tau": 9.0,
            },
        },
    }), encoding="utf-8")
    content = og.assemble_run_content(run, "inter_dataset", {})

    assert content["applied_by_dataset"]["hemibrain:v1.2.1"]["tau"] == 9.0
    html = og.render_html(content)
    assert "Per-dataset" in html
    assert "hemibrain:v1.2.1" in html
    assert "3\u21929" in html  # collapsed 3 -> 9 with a literal arrow

    txt = og.render_txt(content)
    assert "asked [1, 3, 5] -> applied [1, 9]" in txt
    md = og.render_markdown(content)
    assert "3->9" in md
