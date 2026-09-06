"""
HTML Report Generator for Cross-Dataset Comparison.

Generates a static HTML report organized by content type (not by threshold).
Structure:
- Overview & Key Findings
- Edge Count Charts (all thresholds)
- Similarity Matrices (all thresholds)  
- Networks (side-by-side per threshold)
- Edge Presence Matrices (per threshold)
- Path Presence Matrices (per threshold)
- Conservation Analysis (per threshold)
- Statistics Tables (per threshold)
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List


def _make_link(path: str, base_dir: str) -> str:
    """Render an HTML link to a result file relative to the run's output folder."""
    if not path or not os.path.exists(path):
        return '-'
    rel_path = os.path.relpath(path, base_dir).replace(os.sep, '/')
    return f'<a href="{rel_path}" target="_blank">Open</a>'


def get_canonical_name(display_name: str) -> str:
    """Strip a merged-display suffix ('MBON14 (merged)') from a label."""
    if '(' in display_name:
        return display_name.split('(')[0].strip()
    return display_name


def get_base_name(name: str) -> str:
    """Strip a hemisphere suffix ('_L'/'_R'/'_U') from a label."""
    if name.endswith('_L') or name.endswith('_R') or name.endswith('_U'):
        return name[:-2]
    return name


def matches_patterns(label: str, patterns: set) -> bool:
    """Match a label against source/target patterns (regex, glob, or exact)."""
    import re
    # Get canonical name for matching (handles merged display names)
    canonical = get_canonical_name(label)

    # Get base name without hemisphere suffix (for separate_hemispheres mode)
    base_name = get_base_name(canonical)

    # Check label, canonical name, and base name (without suffix)
    names_to_check = list(set([label, canonical, base_name]))

    for name in names_to_check:
        for pattern in patterns:
            # Handle regex patterns (containing .* or other regex chars)
            # vs simple glob patterns (containing just *)
            if '.*' in pattern:
                # Already a regex pattern (e.g., "aMe.*"), use directly
                regex_pattern = pattern
            elif '*' in pattern:
                # Simple glob pattern (e.g., "aMe*"), convert * to .*
                # Escape special regex chars except *
                regex_pattern = re.escape(pattern).replace(r'\*', '.*')
            else:
                # Exact match pattern, escape for regex
                regex_pattern = re.escape(pattern)

            if re.match(f'^{regex_pattern}$', name, re.IGNORECASE):
                return True
    return False


def generate_html_report(
    analyzer,
    dataset_names: List[str],
    thresholds: List[int],
    mode_specific_note: str,
    path_count_data: List[Dict],
    key_findings_per_threshold: Dict[int, Dict]
) -> str:
    """
    Generate a static HTML report organized by content type.
    """
    html_parts = []
    
    # Get nicknames for display (shorter names)
    nicknames = analyzer.parameters.get_dataset_nicknames()
    nickname_map = {d: nicknames[i] for i, d in enumerate(dataset_names)}
    
    # HTML Header with CSS
    html_parts.append(_generate_html_header())
    
    # Report header
    html_parts.append(_generate_report_header(analyzer, dataset_names, thresholds, mode_specific_note, nickname_map))

    # Concern 2 (report-fixes): in-page banner — applied minimal
    # thresholds per dataset (Feature G τ collapse + lossy floor marker)
    html_parts.append(_generate_applied_threshold_banner(analyzer, dataset_names))

    # Table of Contents
    html_parts.append(_generate_toc(thresholds))
    
    # 1. Summary Section
    html_parts.append(_generate_summary_section(
        analyzer, dataset_names, thresholds, path_count_data, 
        key_findings_per_threshold, nickname_map
    ))
    
    # 1.5. Neuron Counts Section
    html_parts.append(_generate_neuron_counts_section(analyzer, dataset_names, nickname_map))

    # 1.6. Type Mapping Section (§4 of the report-fixes plan)
    html_parts.append(_generate_type_mapping_section(analyzer, dataset_names))

    # 1.75. Hemisphere Symmetry Section
    html_parts.append(_generate_hemisphere_symmetry_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # 2. Similarity Matrices Section
    html_parts.append(_generate_similarity_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # 3. Networks Section
    html_parts.append(_generate_networks_section(analyzer, dataset_names, thresholds, nickname_map))

    # 3.5. Reciprocal Visualizations Section (REMOVED - per-dataset reciprocal links are now in Networks section)
    # html_parts.append(_generate_reciprocal_visualizations_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # 4. Edge Presence Matrices Section
    html_parts.append(_generate_edge_matrices_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # 5. Path Presence Matrices Section
    html_parts.append(_generate_path_matrices_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # 6. Conservation Analysis Section
    html_parts.append(_generate_conservation_section(analyzer, dataset_names, thresholds, 
                                                      key_findings_per_threshold, nickname_map))
    
    # 6.5. Dataset Overlap Matrices Section
    html_parts.append(_generate_overlap_matrices_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # 7. Statistics Section
    html_parts.append(_generate_statistics_section(analyzer, dataset_names, thresholds, nickname_map))
    
    # Footer
    html_parts.append(_generate_footer())
    
    return ''.join(html_parts)


def _generate_html_header() -> str:
    """Generate HTML head with CSS styles."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cross-Dataset Comparison Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script>
        // CDN fallback guard: show a clear banner instead of a silent blank canvas
        if (typeof vis === 'undefined') {
            window.addEventListener('DOMContentLoaded', function() {
                const section = document.getElementById('networks');
                if (section && !section.querySelector('.cdn-error')) {
                    const div = document.createElement('div');
                    div.className = 'cdn-error';
                    div.style.cssText = 'background:#fffbeb;border:1px solid #f59e0b;border-radius:8px;padding:12px 14px;color:#92400e;font-size:13px;margin-bottom:15px;';
                    div.textContent = '⚠️ vis-network failed to load (CDN unreachable). Check your internet connection and reload this page.';
                    section.prepend(div);
                }
            });
        }
    </script>
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #64748b;
            --success-color: #22c55e;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --card-bg: #ffffff;
            --border-color: #e2e8f0;
            --conserved-color: #22c55e;
            --partial-color: #f59e0b;
            --unique-color: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-color);
            color: #1e293b;
            line-height: 1.6;
        }
        .container { max-width: 1600px; margin: 0 auto; padding: 20px; }
        header {
            background: linear-gradient(135deg, var(--primary-color), #1d4ed8);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
        }
        header h1 { font-size: 2rem; margin-bottom: 10px; }
        header p { opacity: 0.9; }
        .section { margin: 40px 0; }
        .section-header {
            background: linear-gradient(90deg, var(--primary-color), #60a5fa);
            color: white;
            padding: 15px 25px;
            border-radius: 12px 12px 0 0;
            font-size: 1.3rem;
            font-weight: 600;
        }
        .section-content {
            background: var(--card-bg);
            border-radius: 0 0 12px 12px;
            padding: 24px;
            border: 1px solid var(--border-color);
            border-top: none;
        }
        .card {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--border-color);
        }
        .card h3 {
            color: var(--primary-color);
            margin-bottom: 16px;
            font-size: 1.1rem;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }
        .card h4 {
            color: var(--secondary-color);
            margin: 12px 0 8px 0;
            font-size: 1rem;
        }
        .grid { display: grid; gap: 20px; }
        .grid-2 { grid-template-columns: repeat(2, 1fr); }
        .grid-3 { grid-template-columns: repeat(3, 1fr); }
        .stat-box {
            background: var(--bg-color);
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-box .value { font-size: 1.8rem; font-weight: bold; color: var(--primary-color); }
        .stat-box .label { color: var(--secondary-color); font-size: 0.85rem; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 0.85rem;
        }
        th, td {
            padding: 8px 10px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        th { background: var(--bg-color); font-weight: 600; }
        tr:hover { background: var(--bg-color); }
        /* Sticky header for scrollable tables */
        .sticky-table-container {
            max-height: 500px;
            overflow-y: auto;
            position: relative;
        }
        .sticky-table-container thead th {
            position: sticky;
            top: 0;
            background: #f1f5f9;
            z-index: 10;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .badge {
            padding: 3px 8px;
            border-radius: 20px;
            font-size: 0.7rem;
            font-weight: 600;
        }
        .badge-success { background: #dcfce7; color: #166534; }
        .badge-warning { background: #fef3c7; color: #92400e; }
        .badge-danger { background: #fee2e2; color: #991b1b; }
        .presence-check { color: var(--success-color); }
        .presence-cross { color: var(--danger-color); }
        .toc {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            border: 1px solid var(--border-color);
        }
        .toc h2 { margin-bottom: 15px; color: var(--primary-color); }
        .toc ul { list-style: none; display: flex; flex-wrap: wrap; gap: 10px; }
        .toc a {
            display: inline-block;
            padding: 8px 16px;
            background: var(--bg-color);
            color: var(--primary-color);
            text-decoration: none;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .toc a:hover { background: var(--primary-color); color: white; }
        .print-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 12px 24px;
            background: var(--primary-color);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 1000;
        }
        .print-btn:hover { background: #1d4ed8; }
        .tabs { margin-top: 15px; }
        .tab-buttons { display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 15px; }
        .tab-btn {
            padding: 8px 16px;
            border: 1px solid var(--border-color);
            background: var(--bg-color);
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        .tab-btn:hover { background: #e2e8f0; }
        .tab-btn.active { background: var(--primary-color); color: white; border-color: var(--primary-color); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .network-legend {
            display: flex;
            gap: 20px;
            margin-bottom: 10px;
            font-size: 0.9rem;
        }
        .legend-item { display: flex; align-items: center; gap: 5px; }
        .legend-color {
            width: 20px;
            height: 4px;
            border-radius: 2px;
        }
        .chart-container { min-height: 350px; }
        @media print {
            .print-btn { display: none; }
            .card { break-inside: avoid; }
            .section { break-before: page; }
            .tab-content { display: block !important; }
            .tab-buttons { display: none; }
        }
        @media (max-width: 768px) {
            .grid-2, .grid-3 { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
"""


def _generate_applied_threshold_banner(analyzer, dataset_names: List[str]) -> str:
    """Concern 2: in-page banner stating the APPLIED minimal synapse
    threshold per dataset (Feature G τ collapse). Shown only when at
    least one dataset was budget-bitten — otherwise the asked thresholds
    applied unchanged and no banner is warranted."""
    applied = {}
    meta_map = getattr(analyzer, '_path_run_meta', {}) or {}
    for (ds, _t), meta in meta_map.items():
        if ds not in dataset_names:
            continue
        tau = meta.get('tau')
        if meta.get('budget_bitten') and tau is not None:
            applied[ds] = max(applied.get(ds, 0), float(tau))
    if not applied:
        return ''
    rows = ''.join(
        f'<tr><td>{ds}</td><td style="text-align:center;">{applied[ds]:g}</td></tr>'
        for ds in dataset_names if ds in applied)
    return (
        '<div style="margin:14px 0; padding:10px 14px; border:1px solid #d9a441;'
        ' background:#fdf6e3; border-radius:6px; font-size:0.95em;">'
        '<strong>Applied minimal thresholds (StrongestFirst budget reached):</strong>'
        '<table style="margin-top:6px; border-collapse:collapse;">'
        f'{rows}</table>'
        '<div style="margin-top:6px; color:#666;">Each dataset\'s output contains '
        'exactly the intact paths whose weakest hop is at or above its applied '
        'threshold — identical to a complete run at that threshold. Lower input '
        'thresholds were skipped as τ-collapse duplicates of these runs.</div>'
        '</div>')


def _generate_report_header(analyzer, dataset_names: List[str], thresholds: List[int],
                            mode_specific_note: str, nickname_map: Dict[str, str]) -> str:
    """Generate the report header section."""
    from datetime import datetime
    
    params = analyzer.parameters
    thresholds_str = ', '.join(str(t) for t in thresholds)
    datasets_display = ', '.join(nickname_map[d] for d in dataset_names)
    
    # Path enumeration display: in shortest mode a Max Interlayer of 0 means
    # unlimited search depth (discovery stops when all targets are found).
    path_mode_display = getattr(params, 'path_mode', 'all')
    if path_mode_display == 'shortest' and params.max_interlayer <= 0:
        max_interlayer_display = 'unlimited (shortest mode)'
    else:
        max_interlayer_display = str(params.max_interlayer)
    shortest_note = (
        '<p><em>Shortest path mode: results contain only the per-(source, target) '
        'minimum-hop paths (all ties kept); longer alternative routes are excluded.</em></p>'
        if path_mode_display == 'shortest' else ''
    )
    
    return f"""
        <header>
            <h1>📊 Cross-Dataset Comparison Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Datasets: {datasets_display}</p>
            <p>Thresholds: {thresholds_str} | Mode: <strong>{params.comparison_mode}</strong> | Path Enumeration: <strong>{path_mode_display}</strong> | Max Interlayer: {max_interlayer_display}</p>
            {shortest_note}
        </header>
        {mode_specific_note}
"""


def _generate_toc(thresholds: List[int]) -> str:
    """Generate table of contents."""
    return """
        <div class="toc">
            <h2>📑 Quick Navigation</h2>
            <ul>
                <li><a href="#summary">📋 Summary & Key Findings</a></li>
                <li><a href="#neuron-counts">🧬 Neuron Counts Comparison</a></li>
                <li><a href="#hemisphere-symmetry">🪞 Hemisphere Symmetry</a></li>
                <li><a href="#similarity">🔢 Similarity Matrices</a></li>
                <li><a href="#networks">🕸️ Network Visualizations</a></li>
                <li><a href="#edge-matrices">🔗 Edge Presence Matrices</a></li>
                <li><a href="#path-matrices">🛤️ Path Presence Matrices</a></li>
                <li><a href="#conservation">🏆 Conservation Analysis</a></li>
                <li><a href="#statistics">📉 Statistics</a></li>
            </ul>
        </div>
"""


def _generate_summary_section(analyzer, dataset_names: List[str], thresholds: List[int],
                               path_count_data: List[Dict], key_findings: Dict,
                               nickname_map: Dict[str, str]) -> str:
    """Generate summary section with overview and charts."""
    html_parts = []
    
    html_parts.append("""
        <div id="summary" class="section">
            <div class="section-header">📋 Summary & Key Findings</div>
            <div class="section-content">
    """)
    html_parts.append("""
                <div class="card">
                    <h3>Key Findings by Threshold</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Threshold</th>
                                <th>Total Edges</th>
                                <th>Common Edges</th>
                                <th>Edge Conservation</th>
                                <th>Total Paths</th>
                                <th>Common Paths</th>
                                <th>Path Conservation</th>
                            </tr>
                        </thead>
                        <tbody>
    """)
    for t in thresholds:
        kf = key_findings.get(t, {})
        total_edges = kf.get('total_edges', 0)
        common_edges = kf.get('common_edges', 0)
        edge_rate = (common_edges / total_edges * 100) if total_edges > 0 else 0
        total_paths = kf.get('total_paths', 0)
        common_paths = kf.get('common_paths', 0)
        path_rate = (common_paths / total_paths * 100) if total_paths > 0 else 0
        
        html_parts.append(f'<tr><td><strong>t={t}</strong></td><td>{total_edges}</td><td>{common_edges}</td><td>{edge_rate:.1f}%</td><td>{total_paths}</td><td>{common_paths}</td><td>{path_rate:.1f}%</td></tr>')
    
    html_parts.append('</tbody></table></div>')
    
    # Edge counts chart
    corrected_data = []
    for d in dataset_names:
        nick = nickname_map[d]
        for t in thresholds:
            aligned = analyzer.get_aligned_data(t)
            count = int((aligned[d] > 0).sum()) if not aligned.empty and d in aligned.columns else 0
            corrected_data.append({'dataset': nick, 'threshold': t, 'count': count})
    
    data_json = json.dumps(corrected_data)
    nicknames_json = json.dumps([nickname_map[d] for d in dataset_names])
    thresholds_json = json.dumps(thresholds)
    
    # Create folder to save used data for debugging/verification
    used_data_dir = os.path.join(analyzer.parameters.full_output_path, 'comparison_report_used_data')
    os.makedirs(used_data_dir, exist_ok=True)
    
    # Calculate total weights, avg edge ratio (connection_ratio), and avg traversal probability
    total_weight_data = []
    avg_ratio_data = []
    avg_prob_data = []
    
    # Pre-load and save ratio data for each threshold
    ratio_data_cache = {}
    for t in thresholds:
        try:
            ratio_data = analyzer._get_edge_ratio_data_for_threshold(t)
            if not ratio_data.empty:
                ratio_data.to_csv(os.path.join(used_data_dir, f'ratio_data_t{t}.csv'))
                ratio_data_cache[t] = ratio_data
        except Exception as e:
            print(f"[HTML Report] Warning: Failed to get ratio data for threshold {t}: {e}")
    
    for d in dataset_names:
        nick = nickname_map[d]
        for t in thresholds:
            aligned = analyzer.get_aligned_data(t)
            total_w = int(aligned[d].sum()) if not aligned.empty and d in aligned.columns else 0
            total_weight_data.append({'dataset': nick, 'threshold': t, 'weight': total_w})
            
            # Get connection_ratio data (edge-level ratio = weight / post-synaptic sites)
            # Only average over edges that actually exist in this dataset
            try:
                ratio_data = ratio_data_cache.get(t, pd.DataFrame())
                if not ratio_data.empty and d in ratio_data.columns:
                    dataset_ratios = ratio_data[d]
                    non_zero_ratios = dataset_ratios[dataset_ratios > 0]
                    if len(non_zero_ratios) > 0:
                        avg_ratio = float(non_zero_ratios.mean())
                    else:
                        avg_ratio = 0.0
                    if pd.isna(avg_ratio):
                        avg_ratio = 0.0
                else:
                    avg_ratio = 0.0
            except Exception:
                avg_ratio = 0.0
            avg_ratio_data.append({'dataset': nick, 'threshold': t, 'ratio': avg_ratio})
            
            # Get traversal probability data if available
            # Only average over paths that actually exist in this dataset (ignore 0s from other datasets' paths)
            try:
                prob_data = analyzer._get_prob_data_for_threshold(t)
                if not prob_data.empty and d in prob_data.columns:
                    # Only include paths where this dataset has non-zero probability
                    # This ensures we don't dilute the average with 0s from paths in other datasets
                    dataset_probs = prob_data[d]
                    non_zero_probs = dataset_probs[dataset_probs > 0]
                    if len(non_zero_probs) > 0:
                        avg_prob = float(non_zero_probs.mean())
                    else:
                        avg_prob = 0.0
                    if pd.isna(avg_prob):
                        avg_prob = 0.0
                else:
                    avg_prob = 0.0
            except:
                avg_prob = 0.0
            avg_prob_data.append({'dataset': nick, 'threshold': t, 'prob': avg_prob})
    
    # Save all chart data to files for verification
    pd.DataFrame(corrected_data).to_csv(os.path.join(used_data_dir, 'edge_count_data.csv'), index=False)
    pd.DataFrame(total_weight_data).to_csv(os.path.join(used_data_dir, 'total_weight_data.csv'), index=False)
    pd.DataFrame(avg_ratio_data).to_csv(os.path.join(used_data_dir, 'avg_ratio_data.csv'), index=False)
    pd.DataFrame(avg_prob_data).to_csv(os.path.join(used_data_dir, 'avg_prob_data.csv'), index=False)
    
    weight_data_json = json.dumps(total_weight_data)
    ratio_data_json = json.dumps(avg_ratio_data)
    prob_data_json = json.dumps(avg_prob_data)
    
    html_parts.append(f"""
                <div class="card">
                    <h3>Edge Counts Across All Thresholds</h3>
                    <div id="edgeCountChart" class="chart-container"></div>
                </div>
                <script>
                    (function() {{
                        const data = {data_json};
                        const datasets = {nicknames_json};
                        const thresholds = {thresholds_json};
                        const traces = datasets.map(ds => {{
                            const yVals = thresholds.map(t => {{
                                const item = data.find(d => d.dataset === ds && d.threshold === t);
                                return item ? item.count : 0;
                            }});
                            return {{
                                name: ds,
                                x: thresholds.map(t => 'T=' + t),
                                y: yVals,
                                type: 'bar',
                                text: yVals.map(v => v.toString()),
                                textposition: 'outside'
                            }};
                        }});
                        const allYVals = traces.flatMap(t => t.y);
                        const maxY = Math.max(...allYVals);
                        Plotly.newPlot('edgeCountChart', traces, {{
                            barmode: 'group',
                            xaxis: {{ title: 'Threshold' }},
                            yaxis: {{ title: 'Edge Count', range: [0, maxY * 1.15] }},
                            legend: {{ orientation: 'h', y: -0.15 }}
                        }}, {{responsive: true}});
                    }})();
                </script>
                
                <div class="card">
                    <h3>Total Edge Weight Across All Thresholds</h3>
                    <div id="totalWeightChart" class="chart-container"></div>
                </div>
                <script>
                    (function() {{
                        const data = {weight_data_json};
                        const datasets = {nicknames_json};
                        const thresholds = {thresholds_json};
                        const traces = datasets.map(ds => {{
                            const yVals = thresholds.map(t => {{
                                const item = data.find(d => d.dataset === ds && d.threshold === t);
                                return item ? item.weight : 0;
                            }});
                            return {{
                                name: ds,
                                x: thresholds.map(t => 'T=' + t),
                                y: yVals,
                                type: 'bar',
                                text: yVals.map(v => v.toString()),
                                textposition: 'outside'
                            }};
                        }});
                        const allYVals = traces.flatMap(t => t.y);
                        const maxY = Math.max(...allYVals);
                        Plotly.newPlot('totalWeightChart', traces, {{
                            barmode: 'group',
                            xaxis: {{ title: 'Threshold' }},
                            yaxis: {{ title: 'Total Edge Weight (Synapse Count)', range: [0, maxY * 1.15] }},
                            legend: {{ orientation: 'h', y: -0.15 }}
                        }}, {{responsive: true}});
                    }})();
                </script>
                
                <div class="card">
                    <h3>Average Connection Ratio Across All Thresholds</h3>
                    <div id="avgRatioChart" class="chart-container"></div>
                </div>
                <script>
                    (function() {{
                        const data = {ratio_data_json};
                        const datasets = {nicknames_json};
                        const thresholds = {thresholds_json};
                        const traces = datasets.map(ds => {{
                            const yVals = thresholds.map(t => {{
                                const item = data.find(d => d.dataset === ds && d.threshold === t);
                                return item ? item.ratio : 0;
                            }});
                            return {{
                                name: ds,
                                x: thresholds.map(t => 'T=' + t),
                                y: yVals,
                                type: 'bar',
                                text: yVals.map(v => v.toFixed(4)),
                                textposition: 'outside'
                            }};
                        }});
                        const allYVals = traces.flatMap(t => t.y);
                        const maxY = Math.max(...allYVals);
                        Plotly.newPlot('avgRatioChart', traces, {{
                            barmode: 'group',
                            xaxis: {{ title: 'Threshold' }},
                            yaxis: {{ title: 'Avg Connection Ratio (w_ij / W_j)', range: [0, maxY * 1.15] }},
                            legend: {{ orientation: 'h', y: -0.15 }}
                        }}, {{responsive: true}});
                    }})();
                </script>
                
                <div class="card">
                    <h3>Average Path Traversal Probability</h3>
                    <div id="avgProbChart" class="chart-container"></div>
                </div>
                <script>
                    (function() {{
                        const data = {prob_data_json};
                        const datasets = {nicknames_json};
                        const thresholds = {thresholds_json};
                        const traces = datasets.map(ds => {{
                            const yVals = thresholds.map(t => {{
                                const item = data.find(d => d.dataset === ds && d.threshold === t);
                                return item ? item.prob : 0;
                            }});
                            return {{
                                name: ds,
                                x: thresholds.map(t => 'T=' + t),
                                y: yVals,
                                type: 'bar',
                                text: yVals.map(v => v.toFixed(4)),
                                textposition: 'outside'
                            }};
                        }});
                        const allYVals = traces.flatMap(t => t.y);
                        const maxY = Math.max(...allYVals);
                        Plotly.newPlot('avgProbChart', traces, {{
                            barmode: 'group',
                            xaxis: {{ title: 'Threshold' }},
                            yaxis: {{ title: 'Avg Traversal Probability', range: [0, maxY * 1.15] }},
                            legend: {{ orientation: 'h', y: -0.15 }}
                        }}, {{responsive: true}});
                    }})();
                </script>
""")
    
    html_parts.append('</div></div>')
    return ''.join(html_parts)


def _generate_neuron_counts_section(analyzer, dataset_names: List[str], 
                                     nickname_map: Dict[str, str]) -> str:
    """Generate neuron counts comparison section with type mapping aggregation."""
    # Try to import type mapper for cross-dataset type name merging
    try:
        from .cross_dataset_type_mapper import get_type_mapper
        type_mapper = get_type_mapper()
        type_mapper.load()
        has_type_mapper = True
    except Exception:
        has_type_mapper = False
        type_mapper = None
    
    html_parts = []
    
    html_parts.append("""
        <div id="neuron-counts" class="section">
            <div class="section-header">🧬 Neuron Counts Comparison</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Comparison of source/target neuron existence across datasets. Shows which neuron types are present in each dataset.
                </p>
""")
    
    # Get neuron count data from analyzer
    summary_df = getattr(analyzer, '_neuron_counts_summary', None)
    type_df = getattr(analyzer, '_neuron_type_counts', None)
    group_df = getattr(analyzer, '_neuron_group_counts', None)
    
    # Summary table with bar chart
    if summary_df is not None and not summary_df.empty:
        html_parts.append('<div class="card"><h3>Summary: Total Neuron Counts</h3>')
        html_parts.append('<table><thead><tr><th>Dataset</th><th>Source Neurons</th><th>Target Neurons</th><th>Source Types</th><th>Target Types</th></tr></thead><tbody>')
        
        for _, row in summary_df.iterrows():
            html_parts.append(f'''<tr>
                <td><strong>{row.get("dataset", "")}</strong></td>
                <td>{row.get("source_count", 0)}</td>
                <td>{row.get("target_count", 0)}</td>
                <td>{row.get("source_types", 0)}</td>
                <td>{row.get("target_types", 0)}</td>
            </tr>''')
        
        html_parts.append('</tbody></table>')
        
        # Add bar chart for neuron counts - Grouped by Source/Target
        chart_data = []
        for _, row in summary_df.iterrows():
            chart_data.append({
                'dataset': row.get('dataset', ''),
                'source': row.get('source_count', 0),
                'target': row.get('target_count', 0)
            })
        
        chart_json = json.dumps(chart_data)
        html_parts.append(f'''
            <div id="neuronCountChart" class="chart-container" style="height: 300px;"></div>
            <script>
                (function() {{
                    const data = {chart_json};
                    
                    // Group by Source/Target
                    // Traces are datasets
                    const traces = [];
                    data.forEach(d => {{
                        traces.push({{
                            name: d.dataset,
                            x: ['Source Neurons', 'Target Neurons'],
                            y: [d.source, d.target],
                            type: 'bar'
                        }});
                    }});
                    
                    Plotly.newPlot('neuronCountChart', traces, {{
                        barmode: 'group',
                        xaxis: {{ title: 'Role' }},
                        yaxis: {{ title: 'Neuron Count' }},
                        legend: {{ orientation: 'v' }}
                    }}, {{responsive: true}});
                }})();
            </script>
        ''')
        html_parts.append('</div>')
    
    # Combined Type counts table and chart
    if type_df is not None and not type_df.empty:
        # Get all relevant columns
        source_cols = [c for c in type_df.columns if c not in ['type', 'role'] and 'source' in c.lower()]
        target_cols = [c for c in type_df.columns if c not in ['type', 'role'] and 'target' in c.lower()]
        all_cols = source_cols + target_cols
        
        # Helper function to get canonical type name using type mapper
        def get_canonical_type(type_name: str) -> str:
            if not has_type_mapper or type_mapper is None:
                return type_name
            try:
                # Get display name with all equivalent names
                display_name = type_mapper.get_display_name(type_name, dataset_names)
                # Extract just the canonical base (before parentheses)
                if '(' in display_name:
                    return display_name.split('(')[0].strip()
                return display_name
            except Exception:
                return type_name
        
        # Helper function to get grouped display name with alternatives
        def get_display_type_name(type_name: str) -> str:
            if not has_type_mapper or type_mapper is None:
                return type_name
            try:
                return type_mapper.get_display_name(type_name, dataset_names)
            except Exception:
                return type_name
        
        # Aggregate type counts by canonical name
        canonical_counts = {}  # {canonical_type: {col: total_count}}
        dataset_specific_types = {}  # {canonical_type: {col: set(original_types)}}
        canonical_to_display = {}  # {canonical_type: display_name_with_alternatives}
        
        for _, row in type_df.iterrows():
            orig_type = row.get('type', '')
            if not orig_type:
                continue
            
            # Get canonical type (for grouping)
            canonical = get_canonical_type(orig_type)
            dataset_display = get_display_type_name(orig_type)
            # Use the display name that contains the most info (alternatives)
            if canonical not in canonical_to_display or len(dataset_display) > len(canonical_to_display[canonical]):
                canonical_to_display[canonical] = dataset_display
            
            # Initialize canonical entry if needed
            if canonical not in canonical_counts:
                canonical_counts[canonical] = {col: 0 for col in all_cols}
                dataset_specific_types[canonical] = {col: set() for col in all_cols}
            
            # Aggregate counts
            for col in all_cols:
                val = row.get(col, 0)
                if pd.notna(val) and val > 0:
                    canonical_counts[canonical][col] += int(val)
                    dataset_specific_types[canonical][col].add(orig_type)
        
        # Filter out entries with zero total count
        canonical_counts = {
            k: v for k, v in canonical_counts.items() 
            if sum(v.values()) > 0
        }
        
        if canonical_counts:
            html_parts.append('<div class="card"><h3>Neuron Counts by Type (Source & Target)</h3>')
            html_parts.append('<p style="color: var(--secondary-color); font-size: 0.9em; margin-bottom: 10px;">Neuron counts grouped by canonical type name. Dataset-specific type names shown in parentheses if they differ.</p>')
            
            # Sort by total count
            sorted_types = sorted(
                canonical_counts.keys(),
                key=lambda t: sum(canonical_counts[t].values()),
                reverse=True
            )
            # Take top 50 for chart, but show all in table
            top_types = sorted_types[:50]
            
            # Build combined chart data
            chart_traces = []
            
            # Color palette
            colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#14b8a6', '#3b82f6', '#8b5cf6', '#ec4899']
            
            # Make sure we have traces for Source and Target for each dataset
            # Columns are like 'male-cns:v0.9_source', 'flywire_FAFB_v783_target'
            # We want to group them nicely.
            
            # Sort columns to alternate datasets or group source/target?
            # Let's just iterate through source_cols then target_cols for the chart
            
            trace_idx = 0
            for col in source_cols:
                display_col = col.replace('_source', ' (Source)').replace('_v', ':v').replace('_', ' ')
                chart_traces.append({
                    'name': display_col,
                    'x': [canonical_to_display.get(t, t) for t in top_types],
                    'y': [canonical_counts[t].get(col, 0) for t in top_types],
                    'type': 'bar',
                    'marker': {'color': colors[trace_idx % len(colors)]}
                })
                trace_idx += 1
            
            for col in target_cols:
                display_col = col.replace('_target', ' (Target)').replace('_v', ':v').replace('_', ' ')
                chart_traces.append({
                    'name': display_col,
                    'x': [canonical_to_display.get(t, t) for t in top_types],
                    'y': [canonical_counts[t].get(col, 0) for t in top_types],
                    'type': 'bar',
                    'marker': {'color': colors[trace_idx % len(colors)]}
                })
                trace_idx += 1
            
            chart_json = json.dumps(chart_traces)
            html_parts.append(f'''
                <div id="combinedTypeChart" class="chart-container" style="height: 500px;"></div>
                <script>
                    (function() {{
                        const traces = {chart_json};
                        Plotly.newPlot('combinedTypeChart', traces, {{
                            barmode: 'group',
                            xaxis: {{ 
                                title: 'Neuron Type',
                                tickangle: -45,
                                automargin: true
                            }},
                            yaxis: {{ title: 'Count' }},
                            legend: {{ orientation: 'h', y: 1.1 }},
                            margin: {{ b: 150 }}
                        }}, {{responsive: true}});
                    }})();
                </script>
            ''')
            
            # Combined Table (Fully Expanded)
            html_parts.append('<div class="sticky-table-container" style="overflow-x: auto; max-height: none;"><table class="presence-table"><thead><tr><th>Type</th>')
            
            # Table Headers
            table_cols = []
            for col in source_cols:
                display = col.replace('_source', '<br>(Source)').replace('_v', ':v').replace('_', ' ')
                table_cols.append((col, display))
            for col in target_cols:
                display = col.replace('_target', '<br>(Target)').replace('_v', ':v').replace('_', ' ')
                table_cols.append((col, display))
                
            for _, display in table_cols:
                html_parts.append(f'<th>{display}</th>')
            html_parts.append('</tr></thead><tbody>')
            
            for canonical in sorted_types:
                display_name = canonical_to_display.get(canonical, canonical)
                html_parts.append(f'<tr><td><strong>{display_name}</strong></td>')
                
                for col, _ in table_cols:
                    val = canonical_counts[canonical].get(col, 0)
                    if val == 0:
                        html_parts.append('<td class="absent">-</td>')
                    else:
                        # Check for dataset-specific name
                        orig_types = dataset_specific_types[canonical].get(col, set())
                        cell_content = f"{int(val)}"
                        
                        # Logic to append specific type name
                        # If the specific type used is different from the canonical name's base part
                        if len(orig_types) == 1:
                            specific_type = list(orig_types)[0]
                            # Remove hemisphere suffix from check
                            specific_base = specific_type.replace('_L','').replace('_R','')
                            canonical_base = canonical.replace('_L','').replace('_R','')
                            
                            # Simple check: if specific type is NOT contained in the display name (which has parens), OR
                            # if it IS different from the canonical base and we want to be explicit.
                            # User example: MeVPLo2 (canonical) vs MTe07 (specific)
                            if specific_base != canonical_base:
                                cell_content += f" <span style='font-size:0.8em; color:gray'>({specific_type})</span>"
                        
                        html_parts.append(f'<td class="present">{cell_content}</td>')
                html_parts.append('</tr>')
            
            html_parts.append('</tbody></table></div>')
            html_parts.append('</div>')
    
    # Group counts table (if custom groups exist)
    if group_df is not None and not group_df.empty:
        html_parts.append('<div class="card"><h3>Neuron Counts by Custom Group</h3>')
        html_parts.append('<div style="overflow-x: auto;"><table class="presence-table"><thead><tr><th>Custom Group</th>')
        
        data_cols = [c for c in group_df.columns if c not in ['custom_group', 'role']]
        for col in data_cols:
            display_col = col.replace('_v', ':v').replace('_', ' ')
            html_parts.append(f'<th>{display_col}</th>')
        html_parts.append('</tr></thead><tbody>')
        
        for _, row in group_df.iterrows():
            html_parts.append(f'<tr><td><strong>{row.get("custom_group", "")}</strong></td>')
            for col in data_cols:
                val = row.get(col, 0)
                if pd.isna(val) or val == 0:
                    html_parts.append('<td class="absent">-</td>')
                else:
                    html_parts.append(f'<td class="present">{int(val)}</td>')
            html_parts.append('</tr>')
        
        html_parts.append('</tbody></table></div></div>')
    
    # If no data available, show message
    if (summary_df is None or summary_df.empty) and (type_df is None or type_df.empty):
        html_parts.append('''
            <div class="card">
                <p style="color: var(--secondary-color);">
                    ℹ️ Neuron count data not available. Run export_results() to generate this comparison.
                </p>
            </div>
        ''')
    
    html_parts.append('</div></div>')
    return ''.join(html_parts)


def _generate_similarity_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                  nickname_map: Dict[str, str]) -> str:
    """Generate similarity matrices section with square cell heatmaps for all metrics."""
    from .metrics import ComparisonMetrics
    metrics = ComparisonMetrics()
    
    html_parts = []
    html_parts.append("""
        <div id="similarity" class="section">
            <div class="section-header">🔢 Similarity Matrices</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Pairwise similarity between datasets at each threshold.<br>
                    <strong>ALL-EDGE METRICS</strong> (compare all edges, assign 0 to missing edges):<br>
                    &nbsp;&nbsp;• <strong>Edge Rank</strong> [-1, 1]: Spearman correlation comparing how edges are <em>ranked by weight</em> across datasets<br>
                    &nbsp;&nbsp;• <strong>Cosine</strong> [0, 1]: Measures <em>directional similarity</em> of weight vectors (scale-invariant, ignores magnitude)<br>
                    <strong>SET-BASED METRICS</strong>:<br>
                    &nbsp;&nbsp;• <strong>Jaccard</strong> [0, 1]: Binary edge <em>overlap ratio</em> |A∩B|/|A∪B| (ignores weights)<br>
                    &nbsp;&nbsp;• <strong>Spearman (shared)</strong> [-1, 1]: Rank correlation on <em>shared edges only</em> (N/A if &lt;3 shared edges)
                </p>
""")
    
    # Note: Connectivity Profile Similarity is shown in the separate connectivity_profile_comparison.html
    # report since it uses different methodology (graph-based rank correlation vs threshold-sensitive edge comparison)
    
    for threshold in thresholds:
        aligned = analyzer.get_aligned_data(threshold)
        
        # Always include ALL datasets, even if they have no connections at this threshold
        # This ensures consistent matrix dimensions across thresholds
        available = dataset_names  # Use all datasets, not just those with data
        
        if len(available) < 2:
            continue
        
        labels = [nickname_map[d] for d in available]
        
        # Ensure aligned data has columns for all datasets (fill missing with 0)
        for d in dataset_names:
            if d not in aligned.columns:
                aligned[d] = 0
        
        # Get path data for this threshold (for path rank correlation)
        path_data = None
        if hasattr(analyzer, '_get_path_data_for_threshold'):
            try:
                path_data = analyzer._get_path_data_for_threshold(threshold)
            except Exception:
                pass
        
        similarities = metrics.calculate_all_pairwise_similarities(
            aligned, dataset_names, threshold=1, include_advanced_metrics=True, path_data=path_data
        )
        
        # Save similarity data to CSV (inside comparison_results folder)
        try:
            full_output_path = getattr(analyzer.parameters, 'full_output_path', None)
            if full_output_path:
                csv_dir = os.path.join(full_output_path, 'similarity_matrices')
                os.makedirs(csv_dir, exist_ok=True)
                # Add threshold to the DataFrame before saving
                similarities_with_threshold = similarities.copy()
                similarities_with_threshold['threshold'] = threshold
                similarities_with_threshold.to_csv(os.path.join(csv_dir, f'similarity_threshold_{threshold}.csv'), index=False)
        except Exception:
            pass
        
        n = len(available)
        # Initialize similarity matrices for 4 key metrics
        # Diagonal is always 1.0 for self-comparison
        jaccard = [[1.0 if i == j else None for j in range(n)] for i in range(n)]
        spearman_sim = [[1.0 if i == j else None for j in range(n)] for i in range(n)]
        edge_rank_sim = [[1.0 if i == j else None for j in range(n)] for i in range(n)]
        cosine_sim = [[1.0 if i == j else None for j in range(n)] for i in range(n)]
        
        for _, row in similarities.iterrows():
            d1, d2 = row['dataset_1'], row['dataset_2']
            if d1 in available and d2 in available:
                i1, i2 = available.index(d1), available.index(d2)
                jac = row.get('jaccard_similarity', None)
                # Spearman returns raw correlation in [-1, 1], NaN for undefined
                spearman_val = row.get('spearman_rank_correlation', None)
                # Edge rank now returns raw correlation in [-1, 1], NaN for undefined
                edge_rank_val = row.get('edge_rank_correlation', None)
                cosine_val = row.get('cosine_similarity', None)
                # Handle NaN values - use None to show as "N/A"
                if pd.isna(jac): jac = None
                if pd.isna(edge_rank_val): edge_rank_val = None
                if pd.isna(cosine_val): cosine_val = None
                if pd.isna(spearman_val): spearman_val = None
                # Fill symmetric matrices
                jaccard[i1][i2] = jaccard[i2][i1] = jac
                spearman_sim[i1][i2] = spearman_sim[i2][i1] = spearman_val
                edge_rank_sim[i1][i2] = edge_rank_sim[i2][i1] = edge_rank_val
                cosine_sim[i1][i2] = cosine_sim[i2][i1] = cosine_val
        
        # Calculate cell size for square cells - smaller to fit 4 in a row
        cell_size = 50
        # Chart size scales with number of datasets but caps for 4-in-row layout
        chart_size = min(n * cell_size + 80, 200)
        
        # Always display 4 metrics: Edge Rank, Cosine, Jaccard, Spearman
        num_metrics = 4
        max_width_pct = f"{100 // num_metrics}%" if num_metrics <= 4 else "25%"
        
        # Display metrics in a single row with responsive layout
        html_parts.append(f"""
                <div class="card">
                    <h3>Threshold = {threshold}</h3>
                    
                    <!-- Metrics in one row -->
                    <div style="display: flex; flex-wrap: nowrap; gap: 10px; overflow-x: auto; padding: 8px 0;">
                        <!-- Edge Rank Correlation (union) -->
                        <div style="flex: 1; min-width: {chart_size}px; max-width: {max_width_pct}; background: #eff6ff; border-radius: 6px; padding: 8px;">
                            <h5 style="font-size: 10px; margin: 0 0 4px 0; color: #1e40af; text-align: center;">🔷 Edge Rank</h5>
                            <div id="edge_rank_{threshold}" style="width: 100%; height: {chart_size}px;"></div>
                        </div>
                        <!-- Cosine Similarity (union) -->
                        <div style="flex: 1; min-width: {chart_size}px; max-width: {max_width_pct}; background: #eff6ff; border-radius: 6px; padding: 8px;">
                            <h5 style="font-size: 10px; margin: 0 0 4px 0; color: #1e40af; text-align: center;">🔷 Cosine</h5>
                            <div id="cosine_{threshold}" style="width: 100%; height: {chart_size}px;"></div>
                        </div>
                        <!-- Jaccard -->
                        <div style="flex: 1; min-width: {chart_size}px; max-width: {max_width_pct}; background: #fef3c7; border-radius: 6px; padding: 8px;">
                            <h5 style="font-size: 10px; margin: 0 0 4px 0; color: #92400e; text-align: center;">🔶 Jaccard</h5>
                            <div id="jaccard_{threshold}" style="width: 100%; height: {chart_size}px;"></div>
                        </div>
                        <!-- Spearman (shared) -->
                        <div style="flex: 1; min-width: {chart_size}px; max-width: {max_width_pct}; background: #fef3c7; border-radius: 6px; padding: 8px;">
                            <h5 style="font-size: 10px; margin: 0 0 4px 0; color: #92400e; text-align: center;">🔶 Spearman</h5>
                            <div id="spearman_{threshold}" style="width: 100%; height: {chart_size}px;"></div>
                        </div>
                    </div>
                    <p style="font-size: 0.75em; color: #64748b; margin-top: 8px; text-align: center;">
                        🔷 All-edge (compare all edges, 0 for missing) | 🔶 Set-based (shared edges only)
                    </p>
                </div>
                <script>
                    (function() {{
                        const labels = {json.dumps(labels)};
                        const jaccard = {json.dumps(jaccard)};
                        const edgeRankSim = {json.dumps(edge_rank_sim)};
                        const cosineSim = {json.dumps(cosine_sim)};
                        const spearmanSim = {json.dumps(spearman_sim)};
                        const layout = {{
                            margin: {{ l: 45, r: 10, t: 10, b: 45 }},
                            xaxis: {{ tickangle: -45, scaleanchor: 'y', constrain: 'domain', tickfont: {{size: 8}} }},
                            yaxis: {{ autorange: 'reversed', constrain: 'domain', tickfont: {{size: 8}} }}
                        }};
                        // Annotation function for [0, 1] metrics
                        const makeAnnotations = (data, labels) => data.flatMap((row, i) => 
                            row.map((val, j) => ({{
                                x: labels[j], y: labels[i], 
                                text: val === null ? 'N/A' : val.toFixed(2), 
                                showarrow: false,
                                font: {{ color: (val === null || val > 0.5) ? 'white' : 'black', size: 10 }}
                            }})));
                        // Annotation function for [-1, 1] range (Edge Rank, Spearman)
                        const makeDivergingAnnotations = (data, labels) => data.flatMap((row, i) => 
                            row.map((val, j) => ({{
                                x: labels[j], y: labels[i], 
                                text: val === null ? 'N/A' : val.toFixed(2), 
                                showarrow: false,
                                font: {{ color: (val === null || val > 0) ? 'white' : 'black', size: 10 }}
                            }})));
                        // Use consistent green colorscale: higher value = darker green
                        const greenScale = [[0, '#ffffff'], [0.3, '#c6efce'], [0.6, '#22c55e'], [1, '#166534']];
                        // Diverging colorscale for [-1, 1]: red (negative) -> white (0) -> green (positive)
                        const divergingScale = [[0, '#dc2626'], [0.5, '#ffffff'], [1, '#166534']];
                        // Edge Rank uses diverging scale [-1, 1] (all-edge, blue background)
                        Plotly.newPlot('edge_rank_{threshold}', [{{
                            z: edgeRankSim, x: labels, y: labels, type: 'heatmap',
                            colorscale: divergingScale, zmin: -1, zmax: 1, showscale: false
                        }}], {{...layout, annotations: makeDivergingAnnotations(edgeRankSim, labels)}}, {{responsive: true}});
                        // Cosine uses [0, 1] scale (all-edge, blue background)
                        Plotly.newPlot('cosine_{threshold}', [{{
                            z: cosineSim, x: labels, y: labels, type: 'heatmap',
                            colorscale: greenScale, zmin: 0, zmax: 1, showscale: false
                        }}], {{...layout, annotations: makeAnnotations(cosineSim, labels)}}, {{responsive: true}});
                        // Jaccard uses [0, 1] scale (set-based, yellow background)
                        Plotly.newPlot('jaccard_{threshold}', [{{
                            z: jaccard, x: labels, y: labels, type: 'heatmap',
                            colorscale: greenScale, zmin: 0, zmax: 1, showscale: false
                        }}], {{...layout, annotations: makeAnnotations(jaccard, labels)}}, {{responsive: true}});
                        // Spearman uses diverging scale [-1, 1] (set-based, yellow background)
                        Plotly.newPlot('spearman_{threshold}', [{{
                            z: spearmanSim, x: labels, y: labels, type: 'heatmap',
                            colorscale: divergingScale, zmin: -1, zmax: 1, showscale: false
                        }}], {{...layout, annotations: makeDivergingAnnotations(spearmanSim, labels)}}, {{responsive: true}});
                    }})();
                </script>
""")
    
    html_parts.append('</div></div>')
    return ''.join(html_parts)


def _generate_hemisphere_symmetry_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                          nickname_map: Dict[str, str]) -> str:
    """Generate hemisphere symmetry summary section."""
    html_parts = []
    
    # Check if separate_hemispheres is enabled
    separate_hemispheres = bool(getattr(getattr(analyzer, 'parameters', None), 'separate_hemispheres', False))
    
    summaries = analyzer.get_hemisphere_symmetry_summaries()

    html_parts.append("""
        <div id="hemisphere-symmetry" class="section">
            <div class="section-header">🪞 Hemisphere Symmetry</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Hemisphere symmetry summaries per dataset and threshold (ipsilateral vs contralateral).
                </p>
    """)

    # Show notice if separate_hemispheres=False
    if not separate_hemispheres:
        html_parts.append('''
            <div class="card" style="background: #fef3c7; border: 1px solid #f59e0b;">
                <p style="color: #92400e; margin: 0;">
                    <strong>⚠️ Hemisphere analysis unavailable:</strong> 
                    <code>separate_hemispheres=False</code> in this comparison. 
                    To enable hemisphere symmetry analysis, set <code>separate_hemispheres=True</code> 
                    in ComparisonParameters. This adds _L/_R/_U suffixes to neuron type labels, 
                    allowing comparison of left vs right hemisphere connectivity.
                </p>
            </div>
        ''')
        html_parts.append("</div></div>")
        return ''.join(html_parts)

    if not summaries:
        html_parts.append('<div class="card"><p style="color:#999; text-align:center;">No hemisphere symmetry summaries found.</p></div>')
        html_parts.append("</div></div>")
        return ''.join(html_parts)

    html_parts.append('<div class="tabs"><div class="tab-buttons">')
    for i, t in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(f'<button class="tab-btn {active}" onclick="showSymTab({t})">t = {t}</button>')
    html_parts.append('</div>')

    for i, threshold in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(f'<div id="sym_tab_{threshold}" class="tab-content {active}">')
        html_parts.append('<div class="card">')
        html_parts.append(f'<h3>Hemisphere Symmetry at Threshold = {threshold}</h3>')
        html_parts.append('<table><thead><tr>'
                          '<th>Dataset</th>'
                          '<th>Ipsi Jaccard</th>'
                          '<th>Contra Jaccard</th>'
                          '<th>Ipsi Conserved/Union</th>'
                          '<th>Contra Conserved/Union</th>'
                          '<th>Types Conserved/Union</th>'
                          '<th>Counts L/R</th>'
                          '</tr></thead><tbody>')

        for dataset in dataset_names:
            summary = summaries.get(threshold, {}).get(dataset)
            if not summary:
                continue
            ipsi = summary.get('ipsi', {})
            contra = summary.get('contra', {})
            types = summary.get('neuron_types', {})
            counts = summary.get('hemisphere_counts', {}).get('total', {})
            ipsi_j = ipsi.get('jaccard', 0)
            contra_j = contra.get('jaccard', 0)
            ipsi_cons = f"{ipsi.get('conserved', 0)}/{ipsi.get('union', 0)}"
            contra_cons = f"{contra.get('conserved', 0)}/{contra.get('union', 0)}"
            types_cons = f"{types.get('types_conserved', 0)}/{types.get('types_union', 0)}"
            lr_counts = f"{counts.get('L', 0)}/{counts.get('R', 0)}"
            ds_label = nickname_map.get(dataset, dataset)
            html_parts.append(
                f'<tr><td><strong>{ds_label}</strong></td>'
                f'<td>{ipsi_j:.3f}</td>'
                f'<td>{contra_j:.3f}</td>'
                f'<td>{ipsi_cons}</td>'
                f'<td>{contra_cons}</td>'
                f'<td>{types_cons}</td>'
                f'<td>{lr_counts}</td></tr>'
            )

        html_parts.append('</tbody></table></div></div>')

    html_parts.append("""
                </div>
                <script>
                    function showSymTab(threshold) {
                        document.querySelectorAll('#hemisphere-symmetry .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#hemisphere-symmetry .tab-btn').forEach(el => el.classList.remove('active'));
                        document.getElementById('sym_tab_' + threshold).classList.add('active');
                        event.target.classList.add('active');
                    }
                </script>
            </div>
        </div>
    """)

    return ''.join(html_parts)


def _generate_networks_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                nickname_map: Dict[str, str]) -> str:
    """Generate networks section with conservation-colored edges and role-colored nodes."""
    thresholds_json = json.dumps(thresholds)
    num_networks = len(thresholds)
    # Responsive grid: 1 col on small, 2 cols if 2+ networks
    grid_cols = min(num_networks, 2)
    separate_hemispheres = bool(getattr(getattr(analyzer, 'parameters', None), 'separate_hemispheres', False))
    mirror_disabled_attr = '' if separate_hemispheres else 'disabled'
    mirror_btn_style = (
        'padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border-color); '
        'background: #64748b; color: white; font-size: 12px; white-space: nowrap; '
        + ('cursor: pointer;' if separate_hemispheres else 'cursor: not-allowed; opacity: 0.5;')
    )
    mirror_btn_title = '' if separate_hemispheres else 'Hemisphere mirroring requires separate_hemispheres=True'
    
    # Check if source == target (self-edge scenario)
    is_source_equals_target = False
    self_edge_warning = ""
    try:
        if hasattr(analyzer, 'label_mapper') and analyzer.label_mapper:
            src_set = set(analyzer.label_mapper.get_all_std_labels('source'))
            tgt_set = set(analyzer.label_mapper.get_all_std_labels('target'))
        else:
            src_list = analyzer.parameters._ensure_flat_list(analyzer.parameters.source_neurons)
            tgt_list = analyzer.parameters._ensure_flat_list(analyzer.parameters.target_neurons)
            src_set = set(src_list)
            tgt_set = set(tgt_list)
            
        if src_set == tgt_set:
            is_source_equals_target = True
            # Count self-edges across all thresholds
            self_edge_count = 0
            for threshold in thresholds:
                aligned = analyzer.get_aligned_data_for_network(threshold)
                if not aligned.empty:
                    for edge_key in aligned.index:
                        if ' -> ' in str(edge_key):
                            parts = str(edge_key).split(' -> ')
                            if len(parts) == 2 and parts[0] == parts[1]:
                                self_edge_count += 1
                    break  # Only count once (same edges at all thresholds)
            if self_edge_count > 0:
                self_edge_warning = f'''
                <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                    <span style="color: #92400e;">⚠️ <strong>Self-edges detected:</strong> Source and target neuron lists are identical. 
                    <strong>{self_edge_count}</strong> self-edges (type→same type, e.g., aMe1→aMe1) exist in the edge matrix 
                    but are excluded from path analysis (FindAllPath requires source≠target for direct connections).</span>
                </div>'''
    except:
        pass
    
    html_parts = []
    html_parts.append(f"""
        <div id="networks" class="section">
            <div class="section-header">🕸️ Network Visualizations</div>
            <div class="section-content">
                {self_edge_warning}
                <div style="margin-bottom: 15px;">
                    <p style="color: var(--secondary-color); margin: 0 0 8px 0;">
                        <strong>Edge conservation:</strong> 
                        <span style="color: #22c55e;">● Conserved (all)</span> 
                        <span style="color: #f59e0b; margin-left: 12px;">● Partial (some)</span>
                        <span style="color: #94a3b8; margin-left: 12px;">● Unique (one)</span>
                    </p>
                    <p style="color: var(--secondary-color); margin: 0;">
                        <strong>Node role:</strong> 
                        <span style="color: #ef4444;">● Source</span> 
                        <span style="color: #3b82f6; margin-left: 12px;">● Intermediate</span>
                        <span style="color: #8b5cf6; margin-left: 12px;">● Target</span>
                        <span style="color: #9ca3af; margin-left: 12px;">● Dead-end (no path through)</span>
                    </p>
                </div>
                <script>
                    // Global network mode state (default: static)
                    // Filter mode: 0 = Show All, 1 = Conserved Only, 2 = No Unique
                    window.networkFilterMode = {{}}; // Per-threshold filter mode
                    window.networkPhysicsEnabled = {{}}; // Per-threshold physics state
                    window.hideDeadEndNodes = {{}}; // Per-threshold dead-end state
                    window.allNetworks = {{}};
                    window.allThresholds = {thresholds_json};
                    
                    // Initialize per-threshold states
                    window.allThresholds.forEach(t => {{
                        window.networkFilterMode[t] = 0;  // 0=All, 1=Conserved, 2=NoUnique
                        window.networkPhysicsEnabled[t] = false;
                        window.hideDeadEndNodes[t] = false;
                    }});
                    
                    // Per-network toggle functions
                    function toggleNetworkFilter(threshold) {{
                        // Cycle through modes: 0 (All) -> 1 (Conserved) -> 2 (No Unique) -> 0
                        window.networkFilterMode[threshold] = (window.networkFilterMode[threshold] + 1) % 3;
                        const mode = window.networkFilterMode[threshold];
                        const btn = document.getElementById('filter_btn_' + threshold);
                        
                        const modeLabels = ['🌐 Show All', '✅ Conserved Only', '🚫 No Unique'];
                        const modeColors = ['var(--secondary-color)', '#22c55e', '#f59e0b'];
                        btn.innerHTML = modeLabels[mode];
                        btn.style.background = modeColors[mode];
                        
                        updateNetworkDisplay(threshold);
                    }}
                    
                    function toggleDeadEndNodes(threshold) {{
                        window.hideDeadEndNodes[threshold] = !window.hideDeadEndNodes[threshold];
                        const btn = document.getElementById('deadend_btn_' + threshold);
                        
                        if (window.hideDeadEndNodes[threshold]) {{
                            btn.innerHTML = '🚫 Hide Dead-ends';
                            btn.style.background = '#f59e0b';
                        }} else {{
                            btn.innerHTML = '👁️ Show Dead-ends';
                            btn.style.background = 'var(--secondary-color)';
                        }}
                        
                        updateNetworkDisplay(threshold);
                    }}
                    
                    function toggleNetworkPhysics(threshold) {{
                        window.networkPhysicsEnabled[threshold] = !window.networkPhysicsEnabled[threshold];
                        const btn = document.getElementById('physics_btn_' + threshold);
                        
                        if (window.networkPhysicsEnabled[threshold]) {{
                            btn.innerHTML = '💥 Duang Mode';
                            btn.style.background = 'var(--primary-color)';
                        }} else {{
                            btn.innerHTML = '📌 Static Mode';
                            btn.style.background = 'var(--secondary-color)';
                        }}
                        
                        applyNetworkPhysics(threshold);
                    }}
                    
                    // Hemisphere mirroring toggle
                    window.hemisphereMirrorEnabled = {{}};  // Per-threshold mirror state
                    window.allThresholds.forEach(t => {{
                        // Mirror auto-enables when Separate Hemispheres (L/R) is checked
                        window.hemisphereMirrorEnabled[t] = {'true' if separate_hemispheres else 'false'};
                    }});
                    
                    function toggleHemisphereMirror(threshold) {{
                        window.hemisphereMirrorEnabled[threshold] = !window.hemisphereMirrorEnabled[threshold];
                        const btn = document.getElementById('mirror_btn_' + threshold);
                        
                        if (window.hemisphereMirrorEnabled[threshold]) {{
                            btn.innerHTML = '🪞 Mirrored';
                            btn.style.background = '#0ea5e9';
                            applyHemisphereMirror(threshold);
                        }} else {{
                            btn.innerHTML = '🪞 Mirror Hemispheres';
                            btn.style.background = '#64748b';
                            resetHemisphereMirror(threshold);
                        }}
                    }}
                    
                    function getBaseName(label) {{
                        // Extract base name without hemisphere suffix (handles parentheses)
                        const base = label.split('(')[0].trim();
                        if (base.endsWith('_L') || base.endsWith('_R') || base.endsWith('_U')) {{
                            return base.slice(0, -2);
                        }}
                        return base;
                    }}
                    
                    function getHemisphere(label) {{
                        // Get hemisphere from label suffix (handles parentheses)
                        const base = label.split('(')[0].trim();
                        if (base.endsWith('_L')) return 'L';
                        if (base.endsWith('_R')) return 'R';
                        if (base.endsWith('_U')) return 'U';
                        return 'U';  // Unknown
                    }}
                    
                    function applyHemisphereMirror(threshold) {{
                        if (!window.allNetworks[threshold]) return;
                        const netData = window.allNetworks[threshold];
                        const net = netData.network;
                        const nodes = netData.nodes;
                        
                        // Get current visible node IDs from the DataSet
                        const visibleNodeIds = new Set(nodes.getIds());
                        
                        // Store original positions if not already stored
                        if (!netData.originalPositions) {{
                            netData.originalPositions = {{}};
                            const positions = net.getPositions();
                            for (const [id, pos] of Object.entries(positions)) {{
                                netData.originalPositions[id] = {{ x: pos.x, y: pos.y }};
                            }}
                        }}
                        
                        // Update original positions with current visible positions
                        // (hierarchical layout may have changed since last mirror toggle)
                        const currentPositions = net.getPositions();
                        for (const [id, pos] of Object.entries(currentPositions)) {{
                            netData.originalPositions[id] = {{ x: pos.x, y: pos.y }};
                        }}
                        
                        // Use current positions as the layout baseline
                        const positions = currentPositions;
                        const nodeIds = Object.keys(positions);
                        if (nodeIds.length === 0) return;
                        
                        // Calculate bounds of current visible layout
                        let minX = Infinity, maxX = -Infinity;
                        for (const id of nodeIds) {{
                            minX = Math.min(minX, positions[id].x);
                            maxX = Math.max(maxX, positions[id].x);
                        }}
                        const centerX = (minX + maxX) / 2;
                        
                        // Build base name -> hemisphere node mapping (only for visible nodes)
                        const baseNodeMap = {{}};
                        const allNodes = netData.allNodes;
                        allNodes.forEach(node => {{
                            // Only include nodes that are currently visible
                            if (!visibleNodeIds.has(node.id)) return;
                            
                            const baseName = getBaseName(node.label);
                            const hemi = getHemisphere(node.label);
                            if (!baseNodeMap[baseName]) baseNodeMap[baseName] = {{}};
                            baseNodeMap[baseName][hemi] = node;
                        }});
                        
                        // Apply mirrored layout: enforce symmetric X for L/R pairs
                        const updates = [];
                        const minOffset = 60;
                        for (const [baseName, hemiNodes] of Object.entries(baseNodeMap)) {{
                            const nodeL = hemiNodes['L'];
                            const nodeR = hemiNodes['R'];
                            const nodeU = hemiNodes['U'];
                            const candidates = [nodeL, nodeR, nodeU].filter(Boolean);
                            if (candidates.length === 0) continue;
                            
                            let sumY = 0;
                            let countY = 0;
                            candidates.forEach(n => {{
                                if (positions[n.id]) {{
                                    sumY += positions[n.id].y;
                                    countY++;
                                }}
                            }});
                            const baseY = countY > 0 ? sumY / countY : 0;
                            
                            if (nodeL && nodeR && positions[nodeL.id] && positions[nodeR.id]) {{
                                const leftPos = positions[nodeL.id];
                                const rightPos = positions[nodeR.id];
                                let offset = (Math.abs(leftPos.x - centerX) + Math.abs(rightPos.x - centerX)) / 2;
                                if (!isFinite(offset) || offset < minOffset) offset = minOffset;
                                updates.push({{ id: nodeL.id, x: centerX + offset, y: baseY }});
                                updates.push({{ id: nodeR.id, x: centerX - offset, y: baseY }});
                            }} else if (nodeL && positions[nodeL.id]) {{
                                let offset = Math.abs(positions[nodeL.id].x - centerX);
                                if (!isFinite(offset) || offset < minOffset) offset = minOffset;
                                updates.push({{ id: nodeL.id, x: centerX + offset, y: baseY }});
                            }} else if (nodeR && positions[nodeR.id]) {{
                                let offset = Math.abs(positions[nodeR.id].x - centerX);
                                if (!isFinite(offset) || offset < minOffset) offset = minOffset;
                                updates.push({{ id: nodeR.id, x: centerX - offset, y: baseY }});
                            }}
                            
                            if (nodeU && positions[nodeU.id]) {{
                                updates.push({{ id: nodeU.id, x: centerX, y: baseY }});
                            }}
                        }}
                        
                        // Apply updates (only to visible nodes)
                        updates.forEach(u => {{
                            nodes.update({{ id: u.id, x: u.x, y: u.y }});
                        }});
                        
                        // Fit the view
                        net.fit({{ animation: true }});
                    }}
                    
                    function resetHemisphereMirror(threshold) {{
                        if (!window.allNetworks[threshold]) return;
                        const netData = window.allNetworks[threshold];
                        const net = netData.network;
                        const nodes = netData.nodes;
                        
                        // Restore original positions if available
                        if (netData.originalPositions) {{
                            const updates = [];
                            for (const [id, pos] of Object.entries(netData.originalPositions)) {{
                                updates.push({{ id: parseInt(id), x: pos.x, y: pos.y }});
                            }}
                            updates.forEach(u => {{
                                nodes.update(u);
                            }});
                            net.fit({{ animation: true }});
                        }}
                    }}
                    
                    function updateNetworkDisplay(threshold) {{
                        if (!window.allNetworks[threshold]) return;
                        
                        const netData = window.allNetworks[threshold];
                        const net = netData.network;
                        const edges = netData.edges;
                        const nodes = netData.nodes;
                        const allEdges = netData.allEdges;
                        const allNodes = netData.allNodes;
                        const conservedEdgeIds = netData.conservedEdgeIds;
                        const uniqueEdgeIds = netData.uniqueEdgeIds || new Set();
                        const nodeRoles = netData.nodeRoles || {{}};
                        
                        const mode = window.networkFilterMode[threshold];
                        
                        // Filter edges based on mode
                        let filteredEdges = allEdges;
                        if (mode === 1) {{
                            // Conserved Only
                            filteredEdges = allEdges.filter(e => conservedEdgeIds.has(e.id));
                        }} else if (mode === 2) {{
                            // No Unique (show conserved + partial, hide unique)
                            filteredEdges = allEdges.filter(e => !uniqueEdgeIds.has(e.id));
                        }}
                        
                        // Get connected nodes from filtered edges
                        let connectedNodeIds = new Set();
                        filteredEdges.forEach(e => {{
                            connectedNodeIds.add(e.from);
                            connectedNodeIds.add(e.to);
                        }});
                        
                        // Filter nodes based on mode
                        let filteredNodes;
                        if (mode === 0) {{
                            filteredNodes = allNodes;
                        }} else {{
                            filteredNodes = allNodes.filter(n => connectedNodeIds.has(n.id));
                        }}
                        
                        // Recalculate dead-ends based on CURRENT filtered edges
                        // Dead-end = intermediate node with only incoming OR only outgoing edges in current view
                        // OR node that only connects to dead-ends (recursive)
                        // IMPORTANT: Source and target nodes are NEVER dead-ends
                        let dynamicDeadEndNodeIds = new Set();
                        
                        if (filteredEdges.length > 0) {{
                            // Build adjacency for current filtered graph
                            let adjOut = {{}};
                            let adjIn = {{}};
                            let activeNodes = new Set();
                            
                            filteredEdges.forEach(e => {{
                                if (!adjOut[e.from]) adjOut[e.from] = [];
                                adjOut[e.from].push(e.to);
                                
                                if (!adjIn[e.to]) adjIn[e.to] = [];
                                adjIn[e.to].push(e.from);
                                
                                activeNodes.add(e.from);
                                activeNodes.add(e.to);
                            }});
                            
                            // Iterative dead-end detection
                            let changed = true;
                            while (changed) {{
                                changed = false;
                                activeNodes.forEach(nodeId => {{
                                    if (dynamicDeadEndNodeIds.has(nodeId)) return;
                                    
                                    const role = nodeRoles[nodeId] || 'intermediate';
                                    if (role !== 'intermediate') return;
                                    
                                    const outNeighbors = adjOut[nodeId] || [];
                                    const inNeighbors = adjIn[nodeId] || [];
                                    
                                    // Condition 1: All outgoing paths lead to dead-ends (or no outgoing)
                                    // every() returns True for empty sequence
                                    const leadsToDeadEnd = outNeighbors.every(n => dynamicDeadEndNodeIds.has(n));
                                    
                                    // Condition 2: All incoming paths come from dead-ends (or no incoming)
                                    const comesFromDeadEnd = inNeighbors.every(n => dynamicDeadEndNodeIds.has(n));
                                    
                                    if (leadsToDeadEnd || comesFromDeadEnd) {{
                                        dynamicDeadEndNodeIds.add(nodeId);
                                        changed = true;
                                    }}
                                }});
                            }}
                        }}
                        
                        // Update node colors to reflect current dead-end status
                        // Source and target always keep their colors, never become dead-end
                        const roleColors = {{
                            'source': {{ background: '#ef4444', border: '#b91c1c' }},
                            'intermediate': {{ background: '#3b82f6', border: '#1d4ed8' }},
                            'target': {{ background: '#8b5cf6', border: '#6d28d9' }},
                            'dead-end': {{ background: '#9ca3af', border: '#6b7280' }}
                        }};
                        
                        filteredNodes = filteredNodes.map(n => {{
                            const role = nodeRoles[n.id] || 'intermediate';
                            // Source and target are NEVER dead-ends - they keep their role
                            const isDeadEnd = (role === 'intermediate') && dynamicDeadEndNodeIds.has(n.id);
                            const displayRole = isDeadEnd ? 'dead-end' : role;
                            const colors = roleColors[displayRole];
                            return {{
                                ...n,
                                color: colors,
                                title: n.label + ' (' + displayRole + ')'
                            }};
                        }});
                        
                        // Apply dead-end filter if enabled
                        // NEVER hide source or target nodes, even if dead-end toggle is on
                        if (window.hideDeadEndNodes[threshold]) {{
                            filteredNodes = filteredNodes.filter(n => {{
                                const role = nodeRoles[n.id] || 'intermediate';
                                // Keep source and target nodes always
                                if (role === 'source' || role === 'target') return true;
                                // Filter out dead-end intermediates
                                return !dynamicDeadEndNodeIds.has(n.id);
                            }});
                            filteredEdges = filteredEdges.filter(e => {{
                                const fromRole = nodeRoles[e.from] || 'intermediate';
                                const toRole = nodeRoles[e.to] || 'intermediate';
                                // Keep edges to/from source or target
                                if (fromRole === 'source' || fromRole === 'target') return true;
                                if (toRole === 'source' || toRole === 'target') return true;
                                // Filter out edges involving dead-end intermediates
                                return !dynamicDeadEndNodeIds.has(e.from) && !dynamicDeadEndNodeIds.has(e.to);
                            }});
                        }}
                        
                        // Apply to network
                        edges.clear();
                        edges.add(filteredEdges);
                        nodes.clear();
                        nodes.add(filteredNodes);
                        
                        // Reinitialize layout
                        net.setOptions({{
                            nodes: {{ size: 18, font: {{ size: 11 }} }},
                            edges: {{ smooth: {{ type: 'curvedCW', roundness: 0.1 }} }},
                            layout: {{
                                hierarchical: {{
                                    enabled: true,
                                    direction: 'UD',
                                    sortMethod: 'directed',
                                    levelSeparation: 120,
                                    nodeSpacing: 80,
                                    treeSpacing: 100
                                }}
                            }},
                            physics: {{ enabled: false }}
                        }});
                        setTimeout(() => {{
                            net.setOptions({{ layout: {{ hierarchical: false }} }});
                            // Re-apply mirror positions if mirror mode is enabled
                            if (window.hemisphereMirrorEnabled[threshold]) {{
                                applyHemisphereMirror(threshold);
                            }}
                            net.fit({{ animation: true }});
                        }}, 200);
                    }}
                    
                    function applyNetworkPhysics(threshold) {{
                        if (!window.allNetworks[threshold]) return;
                        
                        const net = window.allNetworks[threshold].network;
                        
                        if (window.networkPhysicsEnabled[threshold]) {{
                            // Duang mode
                            net.setOptions({{
                                nodes: {{ size: 20, font: {{ size: 12 }} }},
                                edges: {{ smooth: {{ type: 'continuous', roundness: 0.2 }} }},
                                layout: {{ hierarchical: false }},
                                physics: {{
                                    enabled: true,
                                    solver: 'forceAtlas2Based',
                                    forceAtlas2Based: {{
                                        gravitationalConstant: -80,
                                        centralGravity: 0.005,
                                        springLength: 120,
                                        springConstant: 0.06,
                                        damping: 0.5,
                                        avoidOverlap: 0.8
                                    }},
                                    stabilization: {{ enabled: true, iterations: 100, updateInterval: 25 }},
                                    minVelocity: 0.5
                                }}
                            }});
                            net.once('stabilized', () => {{ net.fit({{ animation: true }}); }});
                        }} else {{
                            // Static mode
                            net.setOptions({{
                                nodes: {{ size: 18, font: {{ size: 11 }} }},
                                edges: {{ smooth: {{ type: 'curvedCW', roundness: 0.1 }} }},
                                layout: {{
                                    hierarchical: {{
                                        enabled: true,
                                        direction: 'UD',
                                        sortMethod: 'directed',
                                        levelSeparation: 120,
                                        nodeSpacing: 80,
                                        treeSpacing: 100
                                    }}
                                }},
                                physics: {{ enabled: false }}
                            }});
                            setTimeout(() => {{
                                net.setOptions({{ layout: {{ hierarchical: false }} }});
                                net.fit({{ animation: true }});
                            }}, 200);
                        }}
                    }}
                </script>
""")
    
    # Mode toggle (Threshold vs Dataset)
    nicknames = [nickname_map[d] for d in dataset_names]
    nicknames_json = json.dumps(nicknames)
    
    html_parts.append(f'''
                <!-- Toggle Mode Selector -->
                <div style="margin-bottom: 15px;">
                    <span style="font-weight: 600; margin-right: 10px;">View by:</span>
                    <button class="tab-btn active" id="network_mode_threshold" onclick="switchNetworkMode('threshold')">Threshold</button>
                    <button class="tab-btn" id="network_mode_dataset" onclick="switchNetworkMode('dataset')">Dataset</button>
                </div>
''')
    
    # By Threshold View
    html_parts.append('<div id="network_by_threshold" class="tabs"><div class="tab-buttons">')
    for i, t in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(f'<button class="tab-btn {active}" onclick="showNetworkTab({t})">t = {t}</button>')
    html_parts.append('</div>')
    
    # Network containers (only first visible initially)
    for i, threshold in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(f'<div id="network_tab_{threshold}" class="tab-content {active}">')
        html_parts.append(_generate_conservation_network(analyzer, dataset_names, threshold, nickname_map))
        html_parts.append('</div>')
    
    html_parts.append('</div>')  # Close network_by_threshold tabs div
    
    # By Dataset View
    html_parts.append('<div id="network_by_dataset" class="tabs" style="display: none;"><div class="tab-buttons">')
    for i, d in enumerate(dataset_names):
        active = 'active' if i == 0 else ''
        nick = nickname_map[d]
        html_parts.append(f'<button class="tab-btn {active}" onclick="showNetworkDatasetTab(\'{nick}\')">{nick}</button>')
    html_parts.append('</div>')
    
    # Dataset-centric network containers (showing all thresholds for one dataset)
    for i, d in enumerate(dataset_names):
        nick = nickname_map[d]
        active = 'active' if i == 0 else ''
        html_parts.append(f'<div id="network_dataset_tab_{nick}" class="tab-content {active}">')
        html_parts.append(_generate_dataset_network(analyzer, d, thresholds, nickname_map))
        html_parts.append('</div>')
    
    html_parts.append('</div>')  # Close network_by_dataset tabs div
    
    # JavaScript for tab switching
    html_parts.append("""
                <script>
                    function switchNetworkMode(mode) {
                        document.getElementById('network_mode_threshold').classList.toggle('active', mode === 'threshold');
                        document.getElementById('network_mode_dataset').classList.toggle('active', mode === 'dataset');
                        document.getElementById('network_by_threshold').style.display = mode === 'threshold' ? 'block' : 'none';
                        document.getElementById('network_by_dataset').style.display = mode === 'dataset' ? 'block' : 'none';
                        
                        // Re-fit ONLY VISIBLE networks (must call redraw() first to recalculate canvas dimensions)
                        // vis.js cannot render in hidden containers
                        setTimeout(function() {
                            if (mode === 'threshold') {
                                // Only redraw the active threshold network
                                const activeTab = document.querySelector('#network_by_threshold .tab-content.active');
                                if (activeTab) {
                                    const threshold = activeTab.id.replace('network_tab_', '');
                                    if (window.allNetworks && window.allNetworks[threshold]) {
                                        window.allNetworks[threshold].network.redraw();
                                        window.allNetworks[threshold].network.fit({ animation: true });
                                    }
                                }
                            } else {
                                // Only redraw the active dataset networks
                                const activeTab = document.querySelector('#network_by_dataset .tab-content.active');
                                if (activeTab) {
                                    const dataset = activeTab.id.replace('network_dataset_tab_', '');
                                    Object.keys(window.allNetworks || {}).forEach(function(key) {
                                        if (key.startsWith(dataset + '_')) {
                                            window.allNetworks[key].network.redraw();
                                            window.allNetworks[key].network.fit({ animation: true });
                                        }
                                    });
                                }
                            }
                        }, 100);
                    }
                    
                    // Re-draw and fit networks after page load to handle any initial rendering issues
                    // IMPORTANT: Only redraw networks that are in VISIBLE containers
                    // vis.js cannot calculate dimensions for hidden containers (display:none)
                    window.addEventListener('load', function() {
                        setTimeout(function() {
                            // Only redraw the first/active threshold network on load
                            // Tab switching handles hidden networks when they become visible
                            const firstThreshold = window.allThresholds[0];
                            if (window.allNetworks && window.allNetworks[firstThreshold]) {
                                const netData = window.allNetworks[firstThreshold];
                                if (netData && netData.network) {
                                    netData.network.redraw();
                                    netData.network.fit({ animation: true });
                                }
                            }
                        }, 300);
                    });
                    
                    function showNetworkTab(threshold) {
                        // Hide all network tabs in threshold view
                        document.querySelectorAll('#network_by_threshold .tab-content').forEach(el => el.classList.remove('active'));
                        // Remove active from all buttons in threshold view
                        document.querySelectorAll('#network_by_threshold .tab-btn').forEach(el => el.classList.remove('active'));
                        // Show selected tab
                        document.getElementById('network_tab_' + threshold).classList.add('active');
                        // Mark button as active
                        event.target.classList.add('active');
                        
                        // Re-fit the network since it may have been hidden
                        // Must call redraw() first to recalculate canvas dimensions
                        if (window.allNetworks && window.allNetworks[threshold]) {
                            setTimeout(function() {
                                const netData = window.allNetworks[threshold];
                                netData.network.redraw();
                                netData.network.fit({ animation: true });
                            }, 100);
                        }
                    }
                    
                    function showNetworkDatasetTab(dataset) {
                        // Hide all network tabs in dataset view
                        document.querySelectorAll('#network_by_dataset .tab-content').forEach(el => el.classList.remove('active'));
                        // Remove active from all buttons in dataset view
                        document.querySelectorAll('#network_by_dataset .tab-btn').forEach(el => el.classList.remove('active'));
                        // Show selected tab
                        document.getElementById('network_dataset_tab_' + dataset).classList.add('active');
                        // Mark button as active
                        event.target.classList.add('active');
                        
                        // Re-fit the network since it may have been hidden
                        // Must call redraw() first to recalculate canvas dimensions
                        setTimeout(function() {
                            Object.keys(window.allNetworks || {}).forEach(function(key) {
                                if (key.startsWith(dataset + '_')) {
                                    window.allNetworks[key].network.redraw();
                                    window.allNetworks[key].network.fit({ animation: true });
                                }
                            });
                        }, 100);
                    }
                </script>
            </div>
        </div>
""")
    
    return ''.join(html_parts)


def _generate_reciprocal_visualizations_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                                nickname_map: Dict[str, str]) -> str:
    """Generate reciprocal visualizations section with links to VisPath outputs."""
    html_parts = []

    find_reciprocal = bool(getattr(getattr(analyzer, 'parameters', None), 'find_reciprocal', False))

    html_parts.append("""
        <div id="reciprocal-visualizations" class="section">
            <div class="section-header">🔁 Reciprocal Visualizations</div>
            <div class="section-content">
    """)

    if not find_reciprocal:
        html_parts.append("""
            <div class="card">
                <h3>Reciprocal analysis not enabled</h3>
                <p>This report was generated without <strong>find_reciprocal</strong>. Enable it to generate reciprocal network and heatmap visualizations.</p>
            </div>
        """)
        html_parts.append("</div></div>")
        return ''.join(html_parts)

    for t in thresholds:
        html_parts.append(f'<div class="card"><h3>Threshold t = {t}</h3>')
        html_parts.append('<table><thead><tr>'
                  '<th>Dataset</th>'
                  '<th>Type Network</th>'
                  '<th>Type Heatmap</th>'
                  '<th>Group Network</th>'
                  '<th>Group Heatmap</th>'
                  '<th>BodyId Network</th>'
                  '<th>BodyId Heatmap</th>'
                  '<th>Type CSV</th>'
                  '<th>Group CSV</th>'
                  '<th>BodyId CSV</th>'
                  '</tr></thead><tbody>')

        for d in dataset_names:
            nick = nickname_map.get(d, d)
            base_dir = os.path.join(
                analyzer.parameters.full_output_path,
                'dataset_data',
                d,
                f'minsyn_{t}',
                'find_reciprocal',
                'visualizations'
            )

            type_network = _make_link(os.path.join(base_dir, 'reciprocal_type_network.html'), analyzer.parameters.full_output_path)
            type_heatmap = _make_link(os.path.join(base_dir, 'reciprocal_type_heatmap.html'), analyzer.parameters.full_output_path)
            group_network = _make_link(os.path.join(base_dir, 'reciprocal_groups_network.html'), analyzer.parameters.full_output_path)
            group_heatmap = _make_link(os.path.join(base_dir, 'reciprocal_groups_heatmap.html'), analyzer.parameters.full_output_path)
            body_network = _make_link(os.path.join(base_dir, 'reciprocal_bodyId_network.html'), analyzer.parameters.full_output_path)
            body_heatmap = _make_link(os.path.join(base_dir, 'reciprocal_bodyId_heatmap.html'), analyzer.parameters.full_output_path)

            type_csv = _make_link(os.path.join(base_dir, 'reciprocal_connection_type.csv'), analyzer.parameters.full_output_path)
            group_csv = _make_link(os.path.join(base_dir, 'reciprocal_connection_custom_groups.csv'), analyzer.parameters.full_output_path)
            body_csv = _make_link(os.path.join(base_dir, 'reciprocal_connection_bodyId.csv'), analyzer.parameters.full_output_path)

            html_parts.append('<tr>'
                              f'<td><strong>{nick}</strong></td>'
                              f'<td>{type_network}</td>'
                              f'<td>{type_heatmap}</td>'
                              f'<td>{group_network}</td>'
                              f'<td>{group_heatmap}</td>'
                              f'<td>{body_network}</td>'
                              f'<td>{body_heatmap}</td>'
                              f'<td>{type_csv}</td>'
                              f'<td>{group_csv}</td>'
                              f'<td>{body_csv}</td>'
                              '</tr>')

        html_parts.append('</tbody></table></div>')

    html_parts.append('</div></div>')
    return ''.join(html_parts)


def _extract_edges_from_paths(path_data: pd.DataFrame, dataset_names: List[str], max_paths: int = 100) -> set:
    """
    Extract edges from top-M paths ranked by total min_weight across datasets.
    
    This builds a set of edges from the top paths, which ensures complete path
    connectivity rather than selecting disconnected top edges.
    
    Args:
        path_data: DataFrame with path strings as index and dataset columns containing min_weight
        dataset_names: List of dataset names
        max_paths: Maximum number of top paths to include
        
    Returns:
        Set of edge keys (e.g., "A -> B") including BOTH canonical names AND display names
        to ensure matching works regardless of which format the aligned data uses.
    """
    if path_data is None or path_data.empty:
        return set()
    
    # Calculate total weight across datasets for each path
    available_cols = [d for d in dataset_names if d in path_data.columns]
    if not available_cols:
        return set()
    
    # Helper to extract canonical name from display name
    # Handles format: "GNG588(CB0038)" -> "GNG588"
    path_data = path_data.copy()
    path_data['_total'] = path_data[available_cols].sum(axis=1)
    
    # Get top paths by total weight
    top_paths_df = path_data.nlargest(max_paths, '_total')
    
    # Extract edges from each path
    edges = set()
    for path_str in top_paths_df.index:
        path_str = str(path_str)
        # Parse path nodes: "A -> B -> C" or "A → B → C"
        if ' -> ' in path_str:
            nodes = [n.strip() for n in path_str.split(' -> ')]
        elif ' → ' in path_str:
            nodes = [n.strip() for n in path_str.split(' → ')]
        else:
            continue
        
        # Create edge keys for consecutive node pairs
        # Add BOTH display format AND canonical format to ensure matching
        # regardless of which format aligned data uses
        for i in range(len(nodes) - 1):
            src_display = nodes[i]
            dst_display = nodes[i+1]
            src_canonical = get_canonical_name(src_display)
            dst_canonical = get_canonical_name(dst_display)
            
            # Add display name format (e.g., "GNG588(CB0038) -> GNG458(CB0890)")
            display_edge_key = f"{src_display} -> {dst_display}"
            edges.add(display_edge_key)
            
            # Add canonical name format (e.g., "GNG588 -> GNG458")
            canonical_edge_key = f"{src_canonical} -> {dst_canonical}"
            edges.add(canonical_edge_key)
    
    return edges


def _filter_aligned_by_paths(aligned: pd.DataFrame, path_data: pd.DataFrame, 
                              dataset_names: List[str], max_edges: int = 500,
                              max_paths_multiplier: float = 2.0) -> pd.DataFrame:
    """
    Filter aligned edge data to include edges from top paths.
    
    Strategy: Start with max_paths = max_edges * multiplier, extract edges from those paths.
    If not enough edges, expand the number of paths considered.
    
    Args:
        aligned: DataFrame with edge keys as index and dataset columns
        path_data: DataFrame with path keys as index and dataset columns containing min_weight
        dataset_names: List of dataset names
        max_edges: Target number of edges to include
        max_paths_multiplier: Initial multiplier for number of paths to consider
        
    Returns:
        Filtered aligned DataFrame
    """
    if path_data is None or path_data.empty or aligned.empty:
        # Fallback to original top-edges approach if no path data
        if len(aligned) > max_edges:
            available_cols = [d for d in dataset_names if d in aligned.columns]
            if available_cols:
                aligned = aligned.copy()
                aligned['_total'] = aligned[available_cols].sum(axis=1)
                aligned = aligned.nlargest(max_edges, '_total').drop(columns=['_total'])
        return aligned
    
    # Start with initial number of paths
    max_paths = int(max_edges * max_paths_multiplier)
    edges_from_paths = _extract_edges_from_paths(path_data, dataset_names, max_paths)
    
    # Expand if needed (paths share edges, so may need more paths for enough edges)
    attempts = 0
    while len(edges_from_paths) < max_edges and max_paths < len(path_data) and attempts < 5:
        max_paths = min(int(max_paths * 1.5), len(path_data))
        edges_from_paths = _extract_edges_from_paths(path_data, dataset_names, max_paths)
        attempts += 1
    
    # Filter aligned to only include edges from paths
    # Note: We keep all edges from selected paths even if slightly over max_edges
    # This preserves complete path connectivity
    if edges_from_paths:
        aligned = aligned[aligned.index.isin(edges_from_paths)]
    
    return aligned


def _generate_conservation_network(analyzer, dataset_names: List[str], threshold: int,
                                    nickname_map: Dict[str, str], max_edges: int = 500) -> str:
    """Generate network with conservation-based edge coloring and role-based node coloring.
    
    Args:
        max_edges: Maximum number of edges to show in the network (default 500).
                   Edges are selected from top paths to preserve path connectivity.
    """
    aligned = analyzer.get_aligned_data_for_network(threshold)
    
    # Get path data for this threshold to filter by top paths
    path_data = None
    try:
        path_data = analyzer._get_path_data_for_threshold(threshold)
    except Exception:
        pass
    
    # Filter aligned data using path-based approach
    aligned = _filter_aligned_by_paths(aligned, path_data, dataset_names, max_edges)
    
    # Always include ALL datasets, even if they have no connections at this threshold
    # This ensures consistent behavior and proper conservation coloring
    available = dataset_names  # Use all datasets, not just those with data
    
    # Ensure aligned data has columns for all datasets (fill missing with 0)
    for d in dataset_names:
        if d not in aligned.columns:
            aligned[d] = 0
    
    num_datasets = len(available)
    nicknames = [nickname_map[d] for d in available]
    
    # Get source and target neurons (including all mapped type names across datasets)
    source_neurons = set()
    target_neurons = set()
    
    # Always start with parameters (these are the user's intent)
    try:
        src_list = analyzer.parameters._ensure_flat_list(analyzer.parameters.source_neurons)
        tgt_list = analyzer.parameters._ensure_flat_list(analyzer.parameters.target_neurons)
        source_neurons.update(src_list)
        target_neurons.update(tgt_list)
    except:
        pass
    
    # If label mapper is available, ALSO include the mapped keys
    if hasattr(analyzer, 'label_mapper') and analyzer.label_mapper:
        source_neurons.update(analyzer.label_mapper.get_all_std_labels('source'))
        target_neurons.update(analyzer.label_mapper.get_all_std_labels('target'))
    
    # CRITICAL: If auto_type_mapping is enabled, ALSO include the resolved type names
    # for EACH dataset. This is necessary because edge data uses mapped/canonical names
    # (e.g., 'GNG588') not the original query types (e.g., 'CB0038').
    if hasattr(analyzer, 'parameters') and analyzer.parameters.auto_type_mapping:
        for dataset in dataset_names:
            # Get resolved source neurons for this dataset
            src_resolved = analyzer.parameters.get_source_neurons_for_dataset(dataset)
            source_neurons.update(src_resolved)
            # Get resolved target neurons for this dataset
            tgt_resolved = analyzer.parameters.get_target_neurons_for_dataset(dataset)
            target_neurons.update(tgt_resolved)
    
    # Build display name map for cross-dataset type names
    # Format: {canonical_name: "mcns_name (F:fafb_name/H:hemi_name)"}
    display_name_map = {}
    dataset_legend = {}  # {short_code: full_dataset_name}
    type_mapper = None
    if hasattr(analyzer, 'parameters') and analyzer.parameters.auto_type_mapping:
        type_mapper = analyzer.parameters._auto_type_mapper
        if type_mapper:
            dataset_legend = type_mapper.get_all_dataset_short_codes(dataset_names)
    
    # Build nodes and edges with conservation coloring
    nodes = []
    edges = []
    node_ids = {}
    node_roles = {}  # Track role for each node label
    node_counter = 0
    edge_data = {}  # {(source, target): {dataset: weight}}

    # Build adjacency maps for recursive dead-end detection
    outgoing_map = {}  # label -> list of target labels
    incoming_map = {}  # label -> list of source labels
    
    # Collect edge data per dataset
    for edge_key, row in aligned.iterrows():
        if ' -> ' not in str(edge_key):
            continue
        parts = str(edge_key).split(' -> ')
        source, target = parts[0], parts[1] if len(parts) > 1 else ''
        
        edge_tuple = (source, target)
        if edge_tuple not in edge_data:
            edge_data[edge_tuple] = {}
        
        # Update adjacency maps
        if source not in outgoing_map: outgoing_map[source] = []
        outgoing_map[source].append(target)
        
        if target not in incoming_map: incoming_map[target] = []
        incoming_map[target].append(source)
        
        for dataset in available:
            weight = row.get(dataset, 0)
            if weight > 0:
                edge_data[edge_tuple][dataset] = int(weight)
    
    if not edge_data:
        return '<div class="card"><p>No connections at this threshold.</p></div>'
    
    # Detect nodes with display names (format: "Name(alt1/alt2)") to enable legend
    # Note: aligned data from metrics.py already has display names applied
    import re
    # New format: "MeVPLo2(MTe07)" - canonical name followed by (alternatives) without space
    display_name_pattern = re.compile(r'^[^\(]+\([^\)]+\)$')  # Matches "Name(X)" or "Name(X/Y)" pattern
    
    # Store dataset mappings for hover labels: node -> {dataset_code: name_in_that_dataset}
    node_dataset_mappings = {}
    
    # Collect all unique node names
    all_nodes_raw = set()
    for (src, tgt) in edge_data.keys():
        all_nodes_raw.add(src)
        all_nodes_raw.add(tgt)
    
    # Detect nodes that have display names (to enable legend display)
    for node in all_nodes_raw:
        if display_name_pattern.match(node):
            # Node already has display name format - mark it for legend
            canonical = node.split('(')[0]  # Extract canonical name (no space before paren)
            display_name_map[canonical] = node  # Store mapping for legend trigger
    
    # Apply additional display name transformation if type_mapper available
    # This handles any nodes that might not have been transformed by metrics.py
    if type_mapper:
        for node in all_nodes_raw:
            # Skip nodes that already have display names
            if display_name_pattern.match(node):
                # Already has display name, but get dataset mappings for hover
                display_name, dataset_info = type_mapper.get_display_name_with_dataset_info(node.split('(')[0], dataset_names)
                node_dataset_mappings[node] = dataset_info
                continue
            display_name, dataset_info = type_mapper.get_display_name_with_dataset_info(node, dataset_names)
            if display_name != node:
                display_name_map[node] = display_name
            if dataset_info:
                node_dataset_mappings[display_name] = dataset_info
        
        # Transform edge_data keys to use display names (only for non-display nodes)
        if display_name_map:
            new_edge_data = {}
            new_outgoing_map = {}
            new_incoming_map = {}
            
            for (src, tgt), weights in edge_data.items():
                new_src = display_name_map.get(src, src)
                new_tgt = display_name_map.get(tgt, tgt)
                new_edge_data[(new_src, new_tgt)] = weights
                
                # Update adjacency maps with display names
                if new_src not in new_outgoing_map:
                    new_outgoing_map[new_src] = []
                new_outgoing_map[new_src].append(new_tgt)
                
                if new_tgt not in new_incoming_map:
                    new_incoming_map[new_tgt] = []
                new_incoming_map[new_tgt].append(new_src)
            
            edge_data = new_edge_data
            outgoing_map = new_outgoing_map
            incoming_map = new_incoming_map
    
    # Helper to extract canonical name from display name
    # Handles format: "MeVPaMe1(MTe46)" -> "MeVPaMe1"
    # Helper function to check if a node label matches any pattern
    # For merged display names like "MeVPaMe1(MTe46)", also check the canonical part
    # Also handles hemisphere suffixes (_L/_R/_U) for separate_hemispheres mode
    # First pass: collect all nodes and their initial roles (intermediate)
    all_node_labels = set()
    for (source, target) in edge_data.keys():
        all_node_labels.add(source)
        all_node_labels.add(target)
    
    # Initialize all nodes as intermediate
    for label in all_node_labels:
        node_roles[label] = 'intermediate'
    
    # Override with source/target roles (priority: source > target > intermediate)
    for label in all_node_labels:
        if matches_patterns(label, target_neurons):
            node_roles[label] = 'target'
    for label in all_node_labels:
        if matches_patterns(label, source_neurons):
            node_roles[label] = 'source'
    
    # Detect dead-end nodes: intermediate nodes that only have incoming OR only outgoing edges
    # (i.e., they're not on a complete path from source to target)
    # Recursive detection: also include nodes that only connect to dead-ends
    dead_end_nodes = set()
    
    # Iterative dead-end detection
    changed = True
    while changed:
        changed = False
        for label in all_node_labels:
            if label in dead_end_nodes:
                continue
                
            role = node_roles.get(label, 'intermediate')
            if role != 'intermediate':
                continue
            
            # Get neighbors
            out_neighbors = outgoing_map.get(label, [])
            in_neighbors = incoming_map.get(label, [])
            
            # Condition 1: All outgoing paths lead to dead-ends (or no outgoing)
            # every() returns True for empty sequence
            leads_to_dead_end = all(n in dead_end_nodes for n in out_neighbors)
            
            # Condition 2: All incoming paths come from dead-ends (or no incoming)
            comes_from_dead_end = all(n in dead_end_nodes for n in in_neighbors)
            
            if leads_to_dead_end or comes_from_dead_end:
                dead_end_nodes.add(label)
                changed = True
    
    # Node colors by role
    role_colors = {
        'source': {'background': '#ef4444', 'border': '#b91c1c'},      # Red
        'intermediate': {'background': '#3b82f6', 'border': '#1d4ed8'}, # Blue
        'target': {'background': '#8b5cf6', 'border': '#6d28d9'},      # Purple
        'dead-end': {'background': '#9ca3af', 'border': '#6b7280'}     # Gray for dead-ends
    }
    
    # Create nodes and edges
    edge_id = 0
    conserved_edge_ids = []  # Track IDs of conserved edges
    unique_edge_ids = []  # Track IDs of unique edges (only in 1 dataset)
    conserved_node_ids = set()  # Track node IDs that are part of conserved edges
    
    for (source, target), weights in edge_data.items():
        # Helper to build node hover title with dataset mapping info
        def build_node_title(node_label: str, role: str) -> str:
            lines = [f"{node_label}", f"Role: {role}"]
            if node_label in node_dataset_mappings:
                ds_info = node_dataset_mappings[node_label]
                if ds_info:
                    lines.append("Names by dataset:")
                    for code, name in sorted(ds_info.items()):
                        lines.append(f"  {code}: {name}")
            return '\n'.join(lines)
        
        # Add nodes with role-based coloring
        if source not in node_ids:
            node_ids[source] = node_counter
            role = node_roles.get(source, 'intermediate')
            is_dead_end = source in dead_end_nodes
            if is_dead_end:
                colors = role_colors['dead-end']
                display_role = 'dead-end'
            else:
                colors = role_colors[role]
                display_role = role
            nodes.append({
                'id': node_counter, 
                'label': source, 
                'title': build_node_title(source, display_role),
                'color': colors
            })
            node_counter += 1
        if target not in node_ids:
            node_ids[target] = node_counter
            role = node_roles.get(target, 'intermediate')
            is_dead_end = target in dead_end_nodes
            if is_dead_end:
                colors = role_colors['dead-end']
                display_role = 'dead-end'
            else:
                colors = role_colors[role]
                display_role = role
            nodes.append({
                'id': node_counter, 
                'label': target, 
                'title': build_node_title(target, display_role),
                'color': colors
            })
            node_counter += 1
        
        # Determine conservation level
        present_count = len(weights)
        is_conserved = (present_count == num_datasets)
        is_unique = (present_count == 1)
        if is_conserved:
            color = '#22c55e'  # Conserved - green
            conservation = 'Conserved'
            conserved_edge_ids.append(edge_id)
            # Track nodes that are part of conserved edges
            conserved_node_ids.add(node_ids[source])
            conserved_node_ids.add(node_ids[target])
        elif present_count > 1:
            color = '#f59e0b'  # Partial - orange
            conservation = 'Partial'
        else:
            color = '#94a3b8'  # Unique - gray
            conservation = 'Unique'
            unique_edge_ids.append(edge_id)
        
        # Build hover label with all weights
        hover_lines = [f"{source} → {target}", f"Conservation: {conservation} ({present_count}/{num_datasets})"]
        for i, d in enumerate(available):
            w = weights.get(d, 0)
            status = f"{int(w)}" if w > 0 else "—"
            hover_lines.append(f"{nicknames[i]}: {status}")
        
        edges.append({
            'id': edge_id,
            'from': node_ids[source],
            'to': node_ids[target],
            'color': {'color': color, 'highlight': color},
            'width': 2 + min(present_count, 3),
            'title': '\n'.join(hover_lines)  # Real newline for vis.js tooltip
        })
        edge_id += 1
    
    # Ensure ALL source and target neurons are present in the network (even if isolated)
    # This fixes the issue where target neurons drop out if they have no connections
    # BUT: only add exact neuron names, NOT patterns with wildcards (*, .*)
    # ALSO: Don't add if the neuron is already represented in a merged node (e.g., MeVPLo2 in MeVPLo2(MTe07))
    
    def is_represented_in_nodes(label: str, node_ids: dict) -> bool:
        """Check if a label is already represented in existing nodes.
        
        Returns True if:
        - label is directly in node_ids
        - label with hemisphere suffix (_L/_R/_U) is in node_ids (e.g., aMe12 represented by aMe12_L)
        - label appears at the start of a display name like 'MeVPLo2 (F:MTe07)'
        - label appears inside parentheses of a display name (dataset-prefixed like F:MTe07)
        """
        if label in node_ids:
            return True
        # Check for hemisphere-suffixed versions (e.g., aMe12 is represented by aMe12_L or aMe12_R)
        for suffix in ['_L', '_R', '_U']:
            if (label + suffix) in node_ids:
                return True
        # Check for display names like 'MeVPLo2 (F:MTe07)' where label could be MeVPLo2 or MTe07
        for existing_label in node_ids.keys():
            # Check if label is the base name (before space and parentheses)
            # New format: "MeVPLo2 (F:MTe07)"
            if existing_label.startswith(label + ' ('):
                return True
            # Legacy format: "MeVPLo2(MTe07)"
            if existing_label.startswith(label + '('):
                return True
            # Check for hemisphere suffix before parentheses (e.g., 'aMe12_L (F:aMe12)')
            for suffix in ['_L', '_R', '_U']:
                if existing_label.startswith(label + suffix + ' (') or existing_label.startswith(label + suffix + '('):
                    return True
            # Check if label is inside parentheses (with or without dataset code)
            if '(' in existing_label and ')' in existing_label:
                paren_start = existing_label.index('(')
                paren_end = existing_label.index(')')
                inner = existing_label[paren_start + 1:paren_end]
                # Handle multiple dataset-prefixed names separated by /
                # Format: "F:MTe07/H:MTe07_variant"
                inner_names = [n.strip() for n in inner.split('/')]
                for name in inner_names:
                    # Remove dataset prefix if present (e.g., "F:MTe07" -> "MTe07")
                    if ':' in name:
                        unprefixed = name.split(':', 1)[1]
                        if label == unprefixed:
                            return True
                    elif label == name:
                        return True
        return False
    
    # Check if separate_hemispheres is enabled
    separate_hemispheres = getattr(getattr(analyzer, 'parameters', None), 'separate_hemispheres', False)
    
    def has_hemisphere_suffix(name: str) -> bool:
        """Check if a name already has a hemisphere suffix (_L/_R/_U)."""
        return name.endswith('_L') or name.endswith('_R') or name.endswith('_U')
    
    # Track which canonical names have been added as isolated nodes to avoid duplicates
    # This handles the case where CB0038 (FAFB/BANC) and GNG588 (MCNS) are the same type
    added_canonical_sources = set()
    added_canonical_targets = set()
    
    for label in source_neurons:
        # Skip patterns - they're not actual neuron names
        if '*' in label or '.*' in label:
            continue
        # When separate_hemispheres is True, skip adding isolated nodes for base names
        # (without _L/_R/_U suffix). The actual hemisphere-suffixed nodes will be added
        # from edge data if they have connections.
        if separate_hemispheres and not has_hemisphere_suffix(label):
            continue
        if not is_represented_in_nodes(label, node_ids):
            # Get the merged display name if type_mapper is available
            # This ensures CB0038 and GNG588 become one node: GNG588(CB0038)
            display_label = label
            dataset_info = {}
            if type_mapper:
                display_label, dataset_info = type_mapper.get_display_name_with_dataset_info(label, dataset_names)
            
            # Extract canonical name to avoid adding duplicates
            canonical = display_label.split('(')[0] if '(' in display_label else display_label
            if canonical in added_canonical_sources:
                continue  # Already added this type with merged display name
            added_canonical_sources.add(canonical)
            
            # Also check if display_label is already in node_ids (may have been added by another name)
            if display_label in node_ids:
                continue
            
            # Build hover title with dataset info
            title_lines = [f"{display_label}", "Role: source (isolated)"]
            if dataset_info:
                title_lines.append("Names by dataset:")
                for code, name in sorted(dataset_info.items()):
                    title_lines.append(f"  {code}: {name}")
                node_dataset_mappings[display_label] = dataset_info
            
            node_ids[display_label] = node_counter
            node_roles[display_label] = 'source'
            nodes.append({
                'id': node_counter,
                'label': display_label,
                'title': '\n'.join(title_lines),
                'color': role_colors['source']
            })
            node_counter += 1
            
    for label in target_neurons:
        # Skip patterns - they're not actual neuron names
        if '*' in label or '.*' in label:
            continue
        # When separate_hemispheres is True, skip adding isolated nodes for base names
        # (without _L/_R/_U suffix). The actual hemisphere-suffixed nodes will be added
        # from edge data if they have connections.
        if separate_hemispheres and not has_hemisphere_suffix(label):
            continue
        if not is_represented_in_nodes(label, node_ids):
            # Get the merged display name if type_mapper is available
            display_label = label
            dataset_info = {}
            if type_mapper:
                display_label, dataset_info = type_mapper.get_display_name_with_dataset_info(label, dataset_names)
            
            # Extract canonical name to avoid adding duplicates
            canonical = display_label.split('(')[0] if '(' in display_label else display_label
            if canonical in added_canonical_targets:
                continue  # Already added this type with merged display name
            added_canonical_targets.add(canonical)
            
            # Also check if display_label is already in node_ids
            if display_label in node_ids:
                continue
            
            # Build hover title with dataset info
            title_lines = [f"{display_label}", "Role: target (isolated)"]
            if dataset_info:
                title_lines.append("Names by dataset:")
                for code, name in sorted(dataset_info.items()):
                    title_lines.append(f"  {code}: {name}")
                node_dataset_mappings[display_label] = dataset_info
            
            node_ids[display_label] = node_counter
            node_roles[display_label] = 'target'
            nodes.append({
                'id': node_counter,
                'label': display_label,
                'title': '\n'.join(title_lines),
                'color': role_colors['target']
            })
            node_counter += 1
    
    div_id = f"network_{threshold}"
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)
    conserved_ids_json = json.dumps(conserved_edge_ids)
    unique_ids_json = json.dumps(unique_edge_ids)
    conserved_node_ids_json = json.dumps(list(conserved_node_ids))
    dead_end_node_ids = [node_ids[label] for label in dead_end_nodes if label in node_ids]
    dead_end_node_ids_json = json.dumps(dead_end_node_ids)
    # Map node IDs to roles for dynamic dead-end calculation
    node_roles_by_id = {node_ids[label]: role for label, role in node_roles.items() if label in node_ids}
    node_roles_json = json.dumps(node_roles_by_id)
    
    # Count dead-end nodes and conserved edges for display
    dead_end_count = len(dead_end_nodes)
    dead_end_info = f' | <span style="color: #9ca3af;"><strong>{dead_end_count}</strong> dead-end</span>' if dead_end_count > 0 else ''
    conserved_count = len(conserved_edge_ids)
    unique_count = len(unique_edge_ids)
    conserved_info = f' | <span style="color: #22c55e;"><strong>{conserved_count}</strong> conserved</span>'
    unique_info = f' | <span style="color: #94a3b8;"><strong>{unique_count}</strong> unique</span>' if unique_count > 0 else ''
    
    # Generate dataset legend HTML for cross-dataset type name display
    dataset_legend_html = ""
    if dataset_legend and display_name_map:
        legend_parts = [f'<span title="{full_name}" style="cursor: help;"><b>{code}</b>={full_name}</span>' 
                        for code, full_name in sorted(dataset_legend.items())]
        if legend_parts:
            dataset_legend_html = f'''
            <div style="margin-top: 8px; padding: 8px; background: #f8f9fa; border-radius: 6px; font-size: 11px;">
                <span style="color: #666;">📝 Dataset codes in node names:</span> {" | ".join(legend_parts)}
            </div>
            '''

    mirror_disabled_attr = '' if separate_hemispheres else 'disabled'
    mirror_btn_style = (
        'padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border-color); '
        'background: #64748b; color: white; font-size: 12px; white-space: nowrap; '
        + ('cursor: pointer;' if separate_hemispheres else 'cursor: not-allowed; opacity: 0.5;')
    )
    mirror_btn_title = '' if separate_hemispheres else 'Hemisphere mirroring requires separate_hemispheres=True'
    
    return f'''
        <div class="card">
            <h3>Network at Threshold = {threshold}</h3>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div style="color: var(--secondary-color);">
                    <strong>{len(nodes)}</strong> neurons | <strong>{len(edges)}</strong> edges{conserved_info}{unique_info}{dead_end_info}
                </div>
                <div style="display: flex; gap: 8px;">
                    <button id="filter_btn_{threshold}" onclick="toggleNetworkFilter({threshold})" 
                        style="padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border-color); 
                               background: var(--secondary-color); color: white; cursor: pointer; font-size: 12px; white-space: nowrap;">
                        🌐 Show All
                    </button>
                    <button id="deadend_btn_{threshold}" onclick="toggleDeadEndNodes({threshold})" 
                        style="padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border-color); 
                               background: var(--secondary-color); color: white; cursor: pointer; font-size: 12px; white-space: nowrap;">
                        👁️ Show Dead-ends
                    </button>
                    <button id="physics_btn_{threshold}" onclick="toggleNetworkPhysics({threshold})" 
                        style="padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border-color); 
                               background: var(--secondary-color); color: white; cursor: pointer; font-size: 12px; white-space: nowrap;">
                        📌 Static Mode
                    </button>
                    <button id="mirror_btn_{threshold}" onclick="toggleHemisphereMirror({threshold})" {mirror_disabled_attr}
                        title="{mirror_btn_title}"
                        style="{mirror_btn_style}">
                        🪞 Mirror Hemispheres
                    </button>
                </div>
            </div>
            {dataset_legend_html}
            <div id="{div_id}" style="height: 900px; border: 1px solid var(--border-color); border-radius: 8px;"></div>
        </div>
        <script>
            (function() {{
                const allNodes = {nodes_json};
                const allEdges = {edges_json};
                const conservedEdgeIds = new Set({conserved_ids_json});
                const uniqueEdgeIds = new Set({unique_ids_json});
                const conservedNodeIds = new Set({conserved_node_ids_json});
                const deadEndNodeIds = new Set({dead_end_node_ids_json});
                const nodeRoles = {node_roles_json};
                
                const nodes = new vis.DataSet(allNodes);
                const edges = new vis.DataSet(allEdges);
                const container = document.getElementById('{div_id}');
                const data = {{ nodes: nodes, edges: edges }};
                
                // Initialize with hierarchical layout for proper layer-like positioning
                // Direction 'UD' = top-to-bottom flow (rotated 90° clockwise from LR)
                const options = {{
                    nodes: {{
                        shape: 'dot',
                        size: 18,
                        font: {{ size: 11 }},
                        borderWidth: 2
                    }},
                    edges: {{
                        arrows: {{ to: {{ enabled: true, scaleFactor: 0.7 }} }},
                        smooth: {{ type: 'curvedCW', roundness: 0.1 }}
                    }},
                    layout: {{
                        hierarchical: {{
                            enabled: true,
                            direction: 'UD',
                            sortMethod: 'directed',
                            levelSeparation: 120,
                            nodeSpacing: 100,
                            treeSpacing: 120
                        }}
                    }},
                    physics: {{
                        enabled: false
                    }},
                    interaction: {{
                        hover: true,
                        tooltipDelay: 50,
                        multiselect: true,
                        dragNodes: true
                    }}
                }};
                
                const network = new vis.Network(container, data, options);
                
                // After initial layout, disable hierarchical for free movement
                setTimeout(function() {{
                    network.setOptions({{ layout: {{ hierarchical: false }} }});
                    network.fit({{ animation: true }});
                }}, 200);
                
                // Register with global toggle (store original data for filtering)
                window.allNetworks[{threshold}] = {{
                    network: network,
                    nodes: nodes,
                    edges: edges,
                    allNodes: allNodes,
                    allEdges: allEdges,
                    conservedEdgeIds: conservedEdgeIds,
                    uniqueEdgeIds: uniqueEdgeIds,
                    conservedNodeIds: conservedNodeIds,
                    deadEndNodeIds: deadEndNodeIds,
                    nodeRoles: nodeRoles
                }};
                
                // Auto-enable the mirror layout when Separate Hemispheres (L/R) is on
                if (window.hemisphereMirrorEnabled[{threshold}]) {{
                    applyHemisphereMirror({threshold});
                }}
            }})();
        </script>
'''


def _generate_dataset_network(analyzer, dataset: str, thresholds: List[int],
                               nickname_map: Dict[str, str], max_edges: int = 500) -> str:
    """Generate network visualization for a single dataset across all thresholds.
    
    Shows unique edges with conservation-style coloring based on how many thresholds
    the edge appears at. Hover shows weights at all threshold levels.
    
    Args:
        max_edges: Maximum number of edges to show (default 500).
                   Edges are selected from top paths to preserve path connectivity.
    """
    nick = nickname_map[dataset]
    num_thresholds = len(thresholds)
    
    # Get path data for filtering (use first threshold with data)
    all_path_edges = set()
    for threshold in thresholds:
        try:
            path_data = analyzer._get_path_data_for_threshold(threshold)
            if path_data is not None and not path_data.empty:
                # Get edges from top paths for this threshold
                edges_from_paths = _extract_edges_from_paths(path_data, [dataset], max_paths=int(max_edges * 2))
                all_path_edges.update(edges_from_paths)
        except Exception:
            pass
    
    # Collect edges: {(source, target): {threshold: weight}}
    edge_weights = {}  # {(source, target): {t: weight}}
    all_nodes = set()
    
    # First pass: get total weights per edge for prioritization
    edge_total_weights = {}
    
    for threshold in thresholds:
        aligned = analyzer.get_aligned_data(threshold)
        if dataset not in aligned.columns:
            continue
        
        # Vectorized extraction of edges
        valid_mask = aligned[dataset] > 0
        for edge_key in aligned.loc[valid_mask].index:
            if ' -> ' not in str(edge_key):
                continue
            
            # If we have path edges, filter to only those
            if all_path_edges and edge_key not in all_path_edges:
                continue
                
            parts = str(edge_key).split(' -> ')
            source, target = parts[0], parts[1] if len(parts) > 1 else ''
            weight = aligned.loc[edge_key, dataset]
            # Handle Series (duplicate indices)
            if hasattr(weight, 'iloc'):
                weight = weight.iloc[0]
            weight = int(weight)
            if weight > 0:
                edge_tuple = (source, target)
                if edge_tuple not in edge_weights:
                    edge_weights[edge_tuple] = {}
                    edge_total_weights[edge_tuple] = 0
                edge_weights[edge_tuple][threshold] = weight
                edge_total_weights[edge_tuple] += weight
    
    # Limit to top edges if still too many
    if len(edge_weights) > max_edges:
        top_edges = sorted(edge_total_weights.items(), key=lambda x: -x[1])[:max_edges]
        top_edge_set = set(e[0] for e in top_edges)
        edge_weights = {k: v for k, v in edge_weights.items() if k in top_edge_set}
    
    # Build node set
    for (source, target) in edge_weights:
        all_nodes.add(source)
        all_nodes.add(target)
    
    if not all_nodes:
        return f'<div class="card"><p>No connections for {nick} at any threshold.</p></div>'
    
    # Get source and target neurons for role coloring
    source_neurons = set()
    target_neurons = set()
    
    # If label mapper is available, use standardized labels
    if hasattr(analyzer, 'label_mapper') and analyzer.label_mapper:
        source_neurons = set(analyzer.label_mapper.get_all_std_labels('source'))
        target_neurons = set(analyzer.label_mapper.get_all_std_labels('target'))
    else:
        # Otherwise use parameters
        try:
            src_list = analyzer.parameters._ensure_flat_list(analyzer.parameters.source_neurons)
            tgt_list = analyzer.parameters._ensure_flat_list(analyzer.parameters.target_neurons)
            source_neurons = set(src_list)
            target_neurons = set(tgt_list)
        except:
            pass
    
    # Helper to extract canonical name from display name like "MeVPaMe1(MTe46)" -> "MeVPaMe1"
    # Determine node roles
    node_roles = {}
    for label in all_nodes:
        node_roles[label] = 'intermediate'
    for label in all_nodes:
        if matches_patterns(label, target_neurons):
            node_roles[label] = 'target'
    for label in all_nodes:
        if matches_patterns(label, source_neurons):
            node_roles[label] = 'source'
    
    # Track incoming/outgoing for dead-end detection
    has_outgoing = set()
    has_incoming = set()
    for (source, target) in edge_weights.keys():
        has_outgoing.add(source)
        has_incoming.add(target)
    
    # Detect dead-end nodes
    dead_end_nodes = set()
    for label in all_nodes:
        role = node_roles.get(label, 'intermediate')
        if role == 'intermediate':
            only_incoming = label in has_incoming and label not in has_outgoing
            only_outgoing = label in has_outgoing and label not in has_incoming
            if only_incoming or only_outgoing:
                dead_end_nodes.add(label)
    
    # Node colors by role
    role_colors = {
        'source': {'background': '#ef4444', 'border': '#b91c1c'},
        'intermediate': {'background': '#3b82f6', 'border': '#1d4ed8'},
        'target': {'background': '#8b5cf6', 'border': '#6d28d9'},
        'dead-end': {'background': '#9ca3af', 'border': '#6b7280'}
    }
    
    # Create nodes
    nodes = []
    node_ids = {}
    node_counter = 0
    for label in sorted(all_nodes):
        node_ids[label] = node_counter
        role = node_roles.get(label, 'intermediate')
        is_dead_end = label in dead_end_nodes
        display_role = 'dead-end' if is_dead_end else role
        colors = role_colors[display_role]
        nodes.append({
            'id': node_counter,
            'label': label,
            'title': f"{label} ({display_role})",
            'color': colors
        })
        node_counter += 1
    
    # Create unique edges with conservation-style coloring
    # gray (1 threshold) -> orange (some) -> green (all thresholds)
    edges = []
    edge_id = 0
    edges_by_threshold = {t: [] for t in thresholds}  # Track which edges appear at each threshold
    conserved_edge_ids = []  # Edges at ALL thresholds
    unique_edge_ids = []  # Edges at only 1 threshold
    
    for (source, target), t_weights in edge_weights.items():
        thresholds_present = len(t_weights)
        
        # Conservation-style coloring
        if thresholds_present == num_thresholds:
            color = '#22c55e'  # All thresholds - green
            conservation = 'All thresholds'
            conserved_edge_ids.append(edge_id)
        elif thresholds_present > 1:
            color = '#f59e0b'  # Partial - orange
            conservation = f'{thresholds_present}/{num_thresholds} thresholds'
        else:
            color = '#94a3b8'  # Unique - gray
            conservation = '1 threshold only'
            unique_edge_ids.append(edge_id)
        
        # Build hover with all threshold weights
        hover_lines = [f"{source} → {target}", f"Conservation: {conservation}"]
        for t in thresholds:
            w = t_weights.get(t, 0)
            status = f"{int(w)}" if w > 0 else "—"
            hover_lines.append(f"t={t}: {status}")
        
        edges.append({
            'id': edge_id,
            'from': node_ids[source],
            'to': node_ids[target],
            'color': {'color': color, 'highlight': color},
            'width': 2 + min(thresholds_present, 3),
            'title': '\n'.join(hover_lines),
            'thresholds': list(t_weights.keys())  # Store which thresholds this edge appears at
        })
        
        # Track edges by threshold for filtering
        for t in t_weights.keys():
            edges_by_threshold[t].append(edge_id)
        
        edge_id += 1
    
    div_id = f"network_{nick}_dataset"
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)
    edges_by_threshold_json = json.dumps({str(t): ids for t, ids in edges_by_threshold.items()})
    thresholds_json = json.dumps(thresholds)
    conserved_ids_json = json.dumps(conserved_edge_ids)
    unique_ids_json = json.dumps(unique_edge_ids)
    dead_end_node_ids = [node_ids[label] for label in dead_end_nodes if label in node_ids]
    dead_end_node_ids_json = json.dumps(dead_end_node_ids)
    node_roles_by_id = {node_ids[label]: role for label, role in node_roles.items() if label in node_ids}
    node_roles_json = json.dumps(node_roles_by_id)
    
    # Count statistics
    total_edges = len(edges)
    conserved_count = len(conserved_edge_ids)
    unique_count = len(unique_edge_ids)
    dead_end_count = len(dead_end_nodes)
    
    conserved_info = f'<span style="color: #22c55e;"><strong>{conserved_count}</strong> all-t</span>'
    partial_count = total_edges - conserved_count - unique_count
    partial_info = f'<span style="color: #f59e0b;"><strong>{partial_count}</strong> partial</span>' if partial_count > 0 else ''
    unique_info = f'<span style="color: #94a3b8;"><strong>{unique_count}</strong> single-t</span>' if unique_count > 0 else ''
    dead_end_info = f'<span style="color: #9ca3af;"><strong>{dead_end_count}</strong> dead-end</span>' if dead_end_count > 0 else ''
    
    stats_parts = [s for s in [conserved_info, partial_info, unique_info, dead_end_info] if s]
    stats_str = ' | '.join(stats_parts)
    
    # Build threshold filter buttons
    threshold_buttons = []
    for t in thresholds:
        count = len(edges_by_threshold[t])
        btn_html = f'<button id="t_btn_{nick}_{t}" class="threshold-filter-btn active" onclick="toggleDatasetThreshold(\'{nick}\', {t})" style="padding: 4px 8px; border-radius: 4px; border: 1px solid var(--border-color); background: #22c55e; color: white; cursor: pointer; font-size: 11px; margin-right: 4px;">t={t} ({count})</button>'
        threshold_buttons.append(btn_html)
    
    return f'''
        <div class="card">
            <h3>{nick}: Cross-Threshold Network</h3>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div style="color: var(--secondary-color);">
                    <strong>{len(nodes)}</strong> neurons | <strong>{total_edges}</strong> edges | {stats_str}
                </div>
                <div style="display: flex; gap: 8px;">
                    <button id="deadend_btn_{nick}" onclick="toggleDatasetDeadEnd('{nick}')" 
                        style="padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border-color); 
                               background: var(--secondary-color); color: white; cursor: pointer; font-size: 12px;">
                        👁️ Show Dead-ends
                    </button>
                    <button id="physics_btn_{nick}" onclick="toggleDatasetNetworkPhysics('{nick}')" 
                        style="padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border-color); 
                               background: var(--secondary-color); color: white; cursor: pointer; font-size: 12px;">
                        📌 Static Mode
                    </button>
                </div>
            </div>
            <div style="margin-bottom: 10px;">
                <strong>Show thresholds:</strong> {''.join(threshold_buttons)}
                <span style="margin-left: 10px; color: var(--secondary-color);">
                    <span style="color: #22c55e;">● All</span>
                    <span style="color: #f59e0b; margin-left: 8px;">● Partial</span>
                    <span style="color: #94a3b8; margin-left: 8px;">● Single</span>
                </span>
            </div>
            <div id="{div_id}" style="height: 900px; border: 1px solid var(--border-color); border-radius: 8px;"></div>
        </div>
        <script>
            (function() {{
                const allNodes = {nodes_json};
                const allEdges = {edges_json};
                const edgesByThreshold = {edges_by_threshold_json};
                const thresholds = {thresholds_json};
                const conservedEdgeIds = new Set({conserved_ids_json});
                const uniqueEdgeIds = new Set({unique_ids_json});
                const deadEndNodeIds = new Set({dead_end_node_ids_json});
                const nodeRoles = {node_roles_json};
                
                const nodes = new vis.DataSet(allNodes);
                const edges = new vis.DataSet(allEdges);
                const container = document.getElementById('{div_id}');
                const data = {{ nodes: nodes, edges: edges }};
                
                const options = {{
                    nodes: {{
                        shape: 'dot',
                        size: 18,
                        font: {{ size: 11 }},
                        borderWidth: 2
                    }},
                    edges: {{
                        arrows: {{ to: {{ enabled: true, scaleFactor: 0.7 }} }},
                        smooth: {{ type: 'curvedCW', roundness: 0.1 }}
                    }},
                    layout: {{
                        hierarchical: {{
                            enabled: true,
                            direction: 'UD',
                            sortMethod: 'directed',
                            levelSeparation: 120,
                            nodeSpacing: 100,
                            treeSpacing: 120
                        }}
                    }},
                    physics: {{ enabled: false }},
                    interaction: {{
                        hover: true,
                        tooltipDelay: 50,
                        multiselect: true,
                        dragNodes: true
                    }}
                }};
                
                const network = new vis.Network(container, data, options);
                
                setTimeout(function() {{
                    network.setOptions({{ layout: {{ hierarchical: false }} }});
                    network.fit({{ animation: true }});
                }}, 200);
                
                // Register with global toggle
                window.allNetworks['{nick}_dataset'] = {{
                    network: network,
                    nodes: nodes,
                    edges: edges,
                    allNodes: allNodes,
                    allEdges: allEdges,
                    edgesByThreshold: edgesByThreshold,
                    thresholds: thresholds,
                    conservedEdgeIds: conservedEdgeIds,
                    uniqueEdgeIds: uniqueEdgeIds,
                    deadEndNodeIds: deadEndNodeIds,
                    nodeRoles: nodeRoles,
                    activeThresholds: new Set(thresholds.map(String)),
                    hideDeadEnds: false
                }};
            }})();
            
            // Dataset network state
            if (!window.datasetNetworkPhysicsEnabled) {{
                window.datasetNetworkPhysicsEnabled = {{}};
            }}
            window.datasetNetworkPhysicsEnabled['{nick}'] = false;
            
            function toggleDatasetThreshold(dataset, threshold) {{
                const netData = window.allNetworks[dataset + '_dataset'];
                if (!netData) return;
                
                const tStr = String(threshold);
                const btn = document.getElementById('t_btn_' + dataset + '_' + threshold);
                
                if (netData.activeThresholds.has(tStr)) {{
                    netData.activeThresholds.delete(tStr);
                    btn.style.background = '#94a3b8';
                }} else {{
                    netData.activeThresholds.add(tStr);
                    btn.style.background = '#22c55e';
                }}
                
                updateDatasetNetworkDisplay(dataset);
            }}
            
            function toggleDatasetDeadEnd(dataset) {{
                const netData = window.allNetworks[dataset + '_dataset'];
                if (!netData) return;
                
                netData.hideDeadEnds = !netData.hideDeadEnds;
                const btn = document.getElementById('deadend_btn_' + dataset);
                
                if (netData.hideDeadEnds) {{
                    btn.innerHTML = '🚫 Hide Dead-ends';
                    btn.style.background = '#f59e0b';
                }} else {{
                    btn.innerHTML = '👁️ Show Dead-ends';
                    btn.style.background = 'var(--secondary-color)';
                }}
                
                updateDatasetNetworkDisplay(dataset);
            }}
            
            function updateDatasetNetworkDisplay(dataset) {{
                const netData = window.allNetworks[dataset + '_dataset'];
                if (!netData) return;
                
                const net = netData.network;
                const edges = netData.edges;
                const nodes = netData.nodes;
                const allEdges = netData.allEdges;
                const allNodes = netData.allNodes;
                const edgesByThreshold = netData.edgesByThreshold;
                const activeThresholds = netData.activeThresholds;
                const nodeRoles = netData.nodeRoles;
                
                // Get edge IDs that appear in at least one active threshold
                let activeEdgeIds = new Set();
                activeThresholds.forEach(t => {{
                    (edgesByThreshold[t] || []).forEach(id => activeEdgeIds.add(id));
                }});
                
                // Filter edges to only those with at least one active threshold
                let filteredEdges = allEdges.filter(e => {{
                    const edgeThresholds = e.thresholds || [];
                    return edgeThresholds.some(t => activeThresholds.has(String(t)));
                }});
                
                // Get connected nodes
                let connectedNodeIds = new Set();
                filteredEdges.forEach(e => {{
                    connectedNodeIds.add(e.from);
                    connectedNodeIds.add(e.to);
                }});
                
                let filteredNodes = allNodes.filter(n => connectedNodeIds.has(n.id));
                
                // Recalculate dead-ends based on filtered edges
                // IMPORTANT: Source and target nodes are NEVER dead-ends
                let hasOutgoing = new Set();
                let hasIncoming = new Set();
                filteredEdges.forEach(e => {{
                    hasOutgoing.add(e.from);
                    hasIncoming.add(e.to);
                }});
                
                let dynamicDeadEndNodeIds = new Set();
                filteredNodes.forEach(n => {{
                    const role = nodeRoles[n.id] || 'intermediate';
                    // Only intermediate nodes can be dead-ends
                    // Source and target nodes are NEVER dead-ends
                    if (role === 'intermediate') {{
                        const onlyIn = hasIncoming.has(n.id) && !hasOutgoing.has(n.id);
                        const onlyOut = hasOutgoing.has(n.id) && !hasIncoming.has(n.id);
                        if (onlyIn || onlyOut) {{
                            dynamicDeadEndNodeIds.add(n.id);
                        }}
                    }}
                }});
                
                // Update node colors
                // Source and target always keep their colors, never become dead-end
                const roleColors = {{
                    'source': {{ background: '#ef4444', border: '#b91c1c' }},
                    'intermediate': {{ background: '#3b82f6', border: '#1d4ed8' }},
                    'target': {{ background: '#8b5cf6', border: '#6d28d9' }},
                    'dead-end': {{ background: '#9ca3af', border: '#6b7280' }}
                }};
                
                filteredNodes = filteredNodes.map(n => {{
                    const role = nodeRoles[n.id] || 'intermediate';
                    // Source and target are NEVER dead-ends - they keep their role
                    const isDeadEnd = (role === 'intermediate') && dynamicDeadEndNodeIds.has(n.id);
                    const displayRole = isDeadEnd ? 'dead-end' : role;
                    const colors = roleColors[displayRole];
                    return {{
                        ...n,
                        color: colors,
                        title: n.label + ' (' + displayRole + ')'
                    }};
                }});
                
                // Apply dead-end filter if enabled
                // NEVER hide source or target nodes, even if dead-end toggle is on
                if (netData.hideDeadEnds) {{
                    filteredNodes = filteredNodes.filter(n => {{
                        const role = nodeRoles[n.id] || 'intermediate';
                        // Keep source and target nodes always
                        if (role === 'source' || role === 'target') return true;
                        // Filter out dead-end intermediates
                        return !dynamicDeadEndNodeIds.has(n.id);
                    }});
                    filteredEdges = filteredEdges.filter(e => {{
                        const fromRole = nodeRoles[e.from] || 'intermediate';
                        const toRole = nodeRoles[e.to] || 'intermediate';
                        // Keep edges to/from source or target
                        if (fromRole === 'source' || fromRole === 'target') return true;
                        if (toRole === 'source' || toRole === 'target') return true;
                        // Filter out edges involving dead-end intermediates
                        return !dynamicDeadEndNodeIds.has(e.from) && !dynamicDeadEndNodeIds.has(e.to);
                    }});
                }}
                
                // Apply to network
                edges.clear();
                edges.add(filteredEdges);
                nodes.clear();
                nodes.add(filteredNodes);
                
                // Reinitialize layout
                net.setOptions({{
                    nodes: {{ size: 18, font: {{ size: 11 }} }},
                    edges: {{ smooth: {{ type: 'curvedCW', roundness: 0.1 }} }},
                    layout: {{
                        hierarchical: {{
                            enabled: true,
                            direction: 'UD',
                            sortMethod: 'directed',
                            levelSeparation: 120,
                            nodeSpacing: 80,
                            treeSpacing: 100
                        }}
                    }},
                    physics: {{ enabled: false }}
                }});
                setTimeout(() => {{
                    net.setOptions({{ layout: {{ hierarchical: false }} }});
                    net.fit({{ animation: true }});
                }}, 200);
            }}
            
            function toggleDatasetNetworkPhysics(dataset) {{
                window.datasetNetworkPhysicsEnabled[dataset] = !window.datasetNetworkPhysicsEnabled[dataset];
                const btn = document.getElementById('physics_btn_' + dataset);
                const netData = window.allNetworks[dataset + '_dataset'];
                
                if (!netData) return;
                
                const net = netData.network;
                
                if (window.datasetNetworkPhysicsEnabled[dataset]) {{
                    btn.innerHTML = '💥 Duang Mode';
                    btn.style.background = 'var(--primary-color)';
                    net.setOptions({{
                        nodes: {{ size: 20, font: {{ size: 12 }} }},
                        edges: {{ smooth: {{ type: 'continuous', roundness: 0.2 }} }},
                        layout: {{ hierarchical: false }},
                        physics: {{
                            enabled: true,
                            solver: 'forceAtlas2Based',
                            forceAtlas2Based: {{
                                gravitationalConstant: -80,
                                centralGravity: 0.005,
                                springLength: 120,
                                springConstant: 0.06,
                                damping: 0.5,
                                avoidOverlap: 0.8
                            }},
                            stabilization: {{ enabled: true, iterations: 100, updateInterval: 25 }},
                            minVelocity: 0.5
                        }}
                    }});
                    net.once('stabilized', () => {{ net.fit({{ animation: true }}); }});
                }} else {{
                    btn.innerHTML = '📌 Static Mode';
                    btn.style.background = 'var(--secondary-color)';
                    net.setOptions({{
                        nodes: {{ size: 18, font: {{ size: 11 }} }},
                        edges: {{ smooth: {{ type: 'curvedCW', roundness: 0.1 }} }},
                        layout: {{
                            hierarchical: {{
                                enabled: true,
                                direction: 'UD',
                                sortMethod: 'directed',
                                levelSeparation: 120,
                                nodeSpacing: 80,
                                treeSpacing: 100
                            }}
                        }},
                        physics: {{ enabled: false }}
                    }});
                    setTimeout(() => {{
                        net.setOptions({{ layout: {{ hierarchical: false }} }});
                        net.fit({{ animation: true }});
                    }}, 200);
                }}
            }}
        </script>
'''


def _generate_edge_matrices_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                     nickname_map: Dict[str, str]) -> str:
    """Generate edge presence matrices section with dual toggle (by threshold and by dataset)."""
    html_parts = []
    nicknames = [nickname_map[d] for d in dataset_names]
    nicknames_json = json.dumps(nicknames)
    thresholds_json = json.dumps(thresholds)
    
    html_parts.append(f"""
        <div id="edge-matrices" class="section">
            <div class="section-header">🔗 Edge Presence Matrices</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Edge presence across datasets. ✔️ = present with weight, ❌ = absent.
                </p>
                
                <!-- Toggle Mode Selector -->
                <div style="margin-bottom: 15px;">
                    <span style="font-weight: 600; margin-right: 10px;">View by:</span>
                    <button class="tab-btn active" id="edge_mode_threshold" onclick="switchEdgeMode('threshold')">Threshold</button>
                    <button class="tab-btn" id="edge_mode_dataset" onclick="switchEdgeMode('dataset')">Dataset</button>
                </div>
                
                <!-- By Threshold View -->
                <div id="edge_by_threshold" class="tabs">
                    <div class="tab-buttons">
""")
    
    for i, t in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(f'<button class="tab-btn {active}" onclick="showEdgeTab({t})">t = {t}</button>')
    html_parts.append('</div>')
    
    for i, threshold in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(f'<div id="edge_tab_{threshold}" class="tab-content {active}">')
        html_parts.append(_generate_presence_table(analyzer.get_aligned_data(threshold), dataset_names, nickname_map, threshold))
        html_parts.append('</div>')
    
    html_parts.append('</div>')  # Close edge_by_threshold
    
    # By Dataset View
    html_parts.append(f'''
                <!-- By Dataset View -->
                <div id="edge_by_dataset" class="tabs" style="display: none;">
                    <div class="tab-buttons">
''')
    
    for i, d in enumerate(dataset_names):
        active = 'active' if i == 0 else ''
        nick = nickname_map[d]
        html_parts.append(f'<button class="tab-btn {active}" onclick="showEdgeDatasetTab(\'{nick}\')">{nick}</button>')
    html_parts.append('</div>')
    
    # Generate dataset-centric tables (showing all thresholds for one dataset)
    for i, d in enumerate(dataset_names):
        nick = nickname_map[d]
        active = 'active' if i == 0 else ''
        html_parts.append(f'<div id="edge_dataset_tab_{nick}" class="tab-content {active}">')
        html_parts.append(_generate_edge_dataset_table(analyzer, d, thresholds, nickname_map))
        html_parts.append('</div>')
    
    html_parts.append(f"""
                </div>
                
                <script>
                    function switchEdgeMode(mode) {{
                        document.getElementById('edge_mode_threshold').classList.toggle('active', mode === 'threshold');
                        document.getElementById('edge_mode_dataset').classList.toggle('active', mode === 'dataset');
                        document.getElementById('edge_by_threshold').style.display = mode === 'threshold' ? 'block' : 'none';
                        document.getElementById('edge_by_dataset').style.display = mode === 'dataset' ? 'block' : 'none';
                    }}
                    
                    function showEdgeTab(threshold) {{
                        document.querySelectorAll('#edge_by_threshold .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#edge_by_threshold .tab-btn').forEach(el => el.classList.remove('active'));
                        document.getElementById('edge_tab_' + threshold).classList.add('active');
                        event.target.classList.add('active');
                    }}
                    
                    function showEdgeDatasetTab(dataset) {{
                        document.querySelectorAll('#edge_by_dataset .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#edge_by_dataset .tab-btn').forEach(el => el.classList.remove('active'));
                        document.getElementById('edge_dataset_tab_' + dataset).classList.add('active');
                        event.target.classList.add('active');
                    }}
                </script>
            </div>
        </div>
""")
    
    return ''.join(html_parts)


def _generate_edge_dataset_table(analyzer, dataset: str, thresholds: List[int], 
                                  nickname_map: Dict[str, str]) -> str:
    """Generate edge presence table for a single dataset across all thresholds."""
    nick = nickname_map[dataset]
    max_edges_to_show = 50
    
    # First, identify top edges from the first threshold to limit iteration
    top_edges_set = set()
    first_threshold = thresholds[0] if thresholds else None
    if first_threshold is not None:
        aligned_first = analyzer.get_aligned_data(first_threshold)
        if not aligned_first.empty and dataset in aligned_first.columns:
            top_indices = aligned_first.nlargest(max_edges_to_show * 2, dataset).index
            top_edges_set = set(top_indices)
    
    # Collect data only for top edges across thresholds
    edge_data = {}  # edge_key -> {threshold: weight}
    
    for threshold in thresholds:
        aligned = analyzer.get_aligned_data(threshold)
        if aligned.empty or dataset not in aligned.columns:
            continue
        
        # Only iterate over top edges
        if top_edges_set:
            edges_to_process = [e for e in top_edges_set if e in aligned.index]
        else:
            edges_to_process = list(aligned.index[:max_edges_to_show * 2])
        
        for edge_key in edges_to_process:
            weight = aligned.loc[edge_key, dataset]
            # Handle Series (duplicate indices)
            if hasattr(weight, 'iloc'):
                weight = weight.iloc[0]
            if edge_key not in edge_data:
                edge_data[edge_key] = {}
            edge_data[edge_key][threshold] = weight
    
    if not edge_data:
        return '<p>No data available.</p>'
    
    # Sort edges by lowest threshold weight
    sorted_edges = sorted(
        edge_data.items(), 
        key=lambda x: (-x[1].get(thresholds[0], 0), str(x[0]))
    )[:50]
    
    html = [f'<div style="margin-bottom: 8px; color: var(--primary-color); font-weight: 600;">Dataset: {nick} (all thresholds)</div>']
    html.append('<div style="overflow-x: auto;"><table><thead><tr><th>Edge</th>')
    for t in thresholds:
        html.append(f'<th>t={t}</th>')
    html.append('</tr></thead><tbody>')
    
    for edge_key, weights in sorted_edges:
        html.append(f'<tr><td><strong>{edge_key}</strong></td>')
        for t in thresholds:
            w = weights.get(t, 0)
            if w > 0:
                html.append(f'<td><span class="presence-check">✔️</span> {int(w)}</td>')
            else:
                html.append('<td><span class="presence-cross">❌</span></td>')
        html.append('</tr>')
    
    html.append('</tbody></table></div>')
    return ''.join(html)


def _generate_type_mapping_section(analyzer, dataset_names: List[str]) -> str:
    """§4: auto type mapping section — canonical type → per-dataset names
    for the run's result types (capped at 100 rows), plus a conflicts
    summary scoped to the run's datasets. Rendered only when
    auto_type_mapping is enabled."""
    params = getattr(analyzer, 'parameters', None)
    mapper = getattr(params, '_auto_type_mapper', None)
    if params is None or not getattr(params, 'auto_type_mapping', False) \
            or mapper is None:
        return ''
    try:
        result_types = list(analyzer._collect_result_types())
    except Exception:
        return ''
    if not result_types:
        return ''

    cap = 100
    shown = result_types[:cap]
    rows = []
    for t in shown:
        try:
            mappings = mapper.resolve_type_across_datasets(t, dataset_names)
        except Exception:
            mappings = {ds: t for ds in dataset_names}
        rows.append((t, mappings))

    conflicts_total = len(getattr(mapper, '_conflicts', []) or [])
    html = ['<div id="type-mapping" class="section">',
            '<div class="section-header">🔗 Type Mapping (auto)</div>',
            '<div class="section-content">',
            '<p style="color: var(--secondary-color);">',
            'Canonical (male-cns) type name → equivalent type in each dataset, '
            'for the types involved in this run.</p>',
            '<div style="overflow-x: auto;"><table><thead><tr><th>Canonical type</th>']
    for d in dataset_names:
        html.append(f'<th>{d}</th>')
    html.append('</tr></thead><tbody>')
    for t, mappings in rows:
        html.append(f'<tr><td><strong>{t}</strong></td>')
        for d in dataset_names:
            val = mappings.get(d)
            html.append(f'<td>{val if val else "—"}</td>')
        html.append('</tr>')
    html.append('</tbody></table></div>')
    html.append(f'<p style="color:#666; font-size:0.9em;">Showing {len(rows)} '
                f'of {len(result_types)} involved types'
                + (f'; {conflicts_total} mapping conflicts recorded — see '
                   f'auto_type_mapping_conflicts.csv' if conflicts_total else '; no mapping conflicts')
                + '.</p></div></div>')
    return ''.join(html)


def _normalize_path_key(key) -> str:
    """§3: canonical display-key normalization for hop-weight lookups.

    Unifies separator variants ('->' vs ' → ') and strips display suffixes
    ('Name(OtherName)') so the path-presence table rows and the hop-weight
    dict — built through slightly different key forms — always meet."""
    s = str(key).replace(' → ', ' -> ').replace('->', ' -> ')
    nodes = []
    for node in s.split(' -> '):
        node = node.strip()
        if '(' in node:
            node = node.split('(', 1)[0].strip()
        nodes.append(node)
    return ' -> '.join(nodes)


def _generate_path_matrices_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                     nickname_map: Dict[str, str]) -> str:
    """Generate path presence matrices section with dual toggle (by threshold and by dataset)."""
    html_parts = []
    nicknames = [nickname_map[d] for d in dataset_names]
    
    html_parts.append(f"""
        <div id="path-matrices" class="section">
            <div class="section-header">🛤️ Path Presence Matrices</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Path presence across datasets. Weight = minimum edge weight along path.<br>
                    <em>Hop weights shown as <code>-w₁-w₂-</code> with <strong>min weight bolded</strong>.</em>
                </p>
                
                <!-- Toggle Mode Selector -->
                <div style="margin-bottom: 15px;">
                    <span style="font-weight: 600; margin-right: 10px;">View by:</span>
                    <button class="tab-btn active" id="path_mode_threshold" onclick="switchPathMode('threshold')">Threshold</button>
                    <button class="tab-btn" id="path_mode_dataset" onclick="switchPathMode('dataset')">Dataset</button>
                </div>
                
                <!-- By Threshold View -->
                <div id="path_by_threshold" class="tabs">
                    <div class="tab-buttons">
""")
    
    for i, t in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(f'<button class="tab-btn {active}" onclick="showPathTab({t})">t = {t}</button>')
    html_parts.append('</div>')
    
    for i, threshold in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(f'<div id="path_tab_{threshold}" class="tab-content {active}">')
        path_data = analyzer._get_path_data_for_threshold(threshold)
        if path_data is not None and not path_data.empty:
            html_parts.append(_generate_path_presence_table(analyzer, path_data, dataset_names, nickname_map, threshold))
        else:
            html_parts.append('<p>No path data available.</p>')
        html_parts.append('</div>')
    
    html_parts.append('</div>')  # Close path_by_threshold
    
    # By Dataset View
    html_parts.append(f'''
                <!-- By Dataset View -->
                <div id="path_by_dataset" class="tabs" style="display: none;">
                    <div class="tab-buttons">
''')
    
    for i, d in enumerate(dataset_names):
        active = 'active' if i == 0 else ''
        nick = nickname_map[d]
        html_parts.append(f'<button class="tab-btn {active}" onclick="showPathDatasetTab(\'{nick}\')">{nick}</button>')
    html_parts.append('</div>')
    
    # Generate dataset-centric tables (showing all thresholds for one dataset)
    for i, d in enumerate(dataset_names):
        nick = nickname_map[d]
        active = 'active' if i == 0 else ''
        html_parts.append(f'<div id="path_dataset_tab_{nick}" class="tab-content {active}">')
        html_parts.append(_generate_path_dataset_table(analyzer, d, thresholds, nickname_map))
        html_parts.append('</div>')
    
    html_parts.append(f"""
                </div>
                
                <script>
                    function switchPathMode(mode) {{
                        document.getElementById('path_mode_threshold').classList.toggle('active', mode === 'threshold');
                        document.getElementById('path_mode_dataset').classList.toggle('active', mode === 'dataset');
                        document.getElementById('path_by_threshold').style.display = mode === 'threshold' ? 'block' : 'none';
                        document.getElementById('path_by_dataset').style.display = mode === 'dataset' ? 'block' : 'none';
                    }}
                    
                    function showPathTab(threshold) {{
                        document.querySelectorAll('#path_by_threshold .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#path_by_threshold .tab-btn').forEach(el => el.classList.remove('active'));
                        document.getElementById('path_tab_' + threshold).classList.add('active');
                        event.target.classList.add('active');
                    }}
                    
                    function showPathDatasetTab(dataset) {{
                        document.querySelectorAll('#path_by_dataset .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#path_by_dataset .tab-btn').forEach(el => el.classList.remove('active'));
                        document.getElementById('path_dataset_tab_' + dataset).classList.add('active');
                        event.target.classList.add('active');
                    }}
                </script>
            </div>
        </div>
""")
    
    return ''.join(html_parts)


def _generate_path_presence_table(analyzer, data: pd.DataFrame, dataset_names: List[str], 
                                   nickname_map: Dict[str, str], threshold: int = None) -> str:
    """Generate path presence table with hop weights shown as -w1-w2- with min bolded."""
    if data is None or data.empty:
        return '<p>No data available.</p>'
    
    available = [d for d in dataset_names if d in data.columns]
    if not available:
        return '<p>No datasets available.</p>'
    
    # Get path hop weights from analyzer
    path_hop_weights = analyzer._get_path_hop_weights_for_threshold(threshold) if hasattr(analyzer, '_get_path_hop_weights_for_threshold') else {}
    # §3: normalized lookup so display/canonical key variants meet
    hop_by_norm = {_normalize_path_key(k): v for k, v in path_hop_weights.items()}

    # Add threshold caption if provided
    threshold_caption = f'<div style="margin-bottom: 8px; color: var(--primary-color); font-weight: 600;">Threshold = {threshold}</div>' if threshold is not None else ''

    html = [f'{threshold_caption}<div style="overflow-x: auto;"><table><thead><tr><th>Item</th><th>Len</th>']
    for d in available:
        html.append(f'<th>{nickname_map[d]}</th>')
    html.append('<th>Conservation</th></tr></thead><tbody>')

    # Sort by conservation (count of datasets present), then by total weight
    data_copy = data.copy()
    data_copy['_conservation'] = (data_copy[available] > 0).sum(axis=1)
    data_copy['_total'] = data_copy[available].sum(axis=1)
    data_copy = data_copy.sort_values(['_conservation', '_total'], ascending=[False, False])
    top = data_copy.head(min(50, len(data_copy)))

    for key, row in top.iterrows():
        count = sum(1 for d in available if row.get(d, 0) > 0)
        badge = 'badge-success' if count == len(available) else 'badge-warning' if count > 1 else 'badge-danger'

        # §3: path length = hop count (nodes - 1); prefer the hop-weight
        # list length, fall back to the key's arrow count.
        _norm = _normalize_path_key(key)
        _hop_hits = [hop_by_norm.get(_norm)]
        if any(_hop_hits):
            length = max(len(h) for h in _hop_hits if h)
        else:
            length = _norm.count(' -> ') + 1

        html.append(f'<tr><td><strong>{key}</strong></td><td>{length}</td>')
        for d in available:
            w = row.get(d, 0)
            safe_name = analyzer.parameters._sanitize_name(d)
            if w > 0:
                # Get hop weights if available (normalized lookup, §3)
                hop_weights_str = ''
                hop_weights = hop_by_norm.get(_norm, {}).get(safe_name)
                if hop_weights:
                        min_w = min(hop_weights)
                        # Format with bolded minimum
                        formatted = []
                        for hw in hop_weights:
                            if hw == min_w:
                                formatted.append(f'<strong>{int(hw)}</strong>')
                            else:
                                formatted.append(str(int(hw)))
                        hop_weights_str = f'<br><span style="font-size: 0.8em; color: #666;">-{"-".join(formatted)}-</span>'
                
                html.append(f'<td><span class="presence-check">✔️</span> {int(w)}{hop_weights_str}</td>')
            else:
                html.append('<td><span class="presence-cross">❌</span></td>')
        html.append(f'<td><span class="badge {badge}">{count}/{len(available)}</span></td></tr>')
    
    html.append('</tbody></table></div>')
    return ''.join(html)


def _generate_path_dataset_table(analyzer, dataset: str, thresholds: List[int], 
                                  nickname_map: Dict[str, str]) -> str:
    """Generate path presence table for a single dataset across all thresholds."""
    nick = nickname_map[dataset]
    safe_name = analyzer.parameters._sanitize_name(dataset)
    
    # First, identify top paths from the first threshold to limit iteration
    max_paths_to_show = 50
    top_paths_set = set()
    
    first_threshold = thresholds[0] if thresholds else None
    if first_threshold is not None:
        pdata_first = analyzer._get_path_data_for_threshold(first_threshold)
        if pdata_first is not None and not pdata_first.empty and dataset in pdata_first.columns:
            # Get top paths by weight
            top_indices = pdata_first.nlargest(max_paths_to_show * 2, dataset).index
            top_paths_set = set(top_indices)
    
    # Collect data only for top paths across thresholds
    path_data = {}  # path_key -> {threshold: (weight, hop_weights)}
    
    for threshold in thresholds:
        pdata = analyzer._get_path_data_for_threshold(threshold)
        if pdata is None or pdata.empty or dataset not in pdata.columns:
            continue
        
        # Get hop weights for this threshold
        hop_weights_dict = analyzer._get_path_hop_weights_for_threshold(threshold) if hasattr(analyzer, '_get_path_hop_weights_for_threshold') else {}
        hop_weights_norm = {_normalize_path_key(k): v for k, v in hop_weights_dict.items()}
        
        # Only iterate over top paths (using vectorized access)
        if top_paths_set:
            paths_to_process = [p for p in top_paths_set if p in pdata.index]
        else:
            paths_to_process = list(pdata.index[:max_paths_to_show * 2])
        
        for path_key in paths_to_process:
            weight = pdata.loc[path_key, dataset]
            # Handle Series (duplicate indices)
            if hasattr(weight, 'iloc'):
                weight = weight.iloc[0]
            if path_key not in path_data:
                path_data[path_key] = {}
            
            hop_weights = hop_weights_dict.get(path_key, {}).get(safe_name, []) if hop_weights_dict else []
            if not hop_weights and hop_weights_dict:
                # §3: normalized lookup — display/canonical key variants
                _norm = _normalize_path_key(path_key)
                hop_weights = hop_weights_norm.get(_norm, {}).get(safe_name, [])
            path_data[path_key][threshold] = (weight, hop_weights)
    
    if not path_data:
        return '<p>No data available.</p>'
    
    # Sort paths by lowest threshold weight
    sorted_paths = sorted(
        path_data.items(), 
        key=lambda x: (-x[1].get(thresholds[0], (0, []))[0], str(x[0]))
    )[:50]
    
    html = [f'<div style="margin-bottom: 8px; color: var(--primary-color); font-weight: 600;">Dataset: {nick} (all thresholds)</div>']
    html.append('<div style="overflow-x: auto;"><table><thead><tr><th>Path</th><th>Len</th>')
    for t in thresholds:
        html.append(f'<th>t={t}</th>')
    html.append('</tr></thead><tbody>')
    
    for path_key, weights in sorted_paths:
        # §3: path length = max hop-list length across thresholds, else
        # derived from the key's hop count.
        _lens = [len(h) for (_w, h) in weights.values() if h]
        _norm_key = _normalize_path_key(path_key)
        length = max(_lens) if _lens else (_norm_key.count(' -> ') + 1)
        html.append(f'<tr><td><strong>{path_key}</strong></td><td>{length}</td>')
        for t in thresholds:
            w, hop_weights = weights.get(t, (0, []))
            if w > 0:
                hop_weights_str = ''
                if hop_weights:
                    min_w = min(hop_weights)
                    formatted = []
                    for hw in hop_weights:
                        if hw == min_w:
                            formatted.append(f'<strong>{int(hw)}</strong>')
                        else:
                            formatted.append(str(int(hw)))
                    hop_weights_str = f'<br><span style="font-size: 0.8em; color: #666;">-{"-".join(formatted)}-</span>'
                html.append(f'<td><span class="presence-check">✔️</span> {int(w)}{hop_weights_str}</td>')
            else:
                html.append('<td><span class="presence-cross">❌</span></td>')
        html.append('</tr>')
    
    html.append('</tbody></table></div>')
    return ''.join(html)


def _generate_presence_table(data: pd.DataFrame, dataset_names: List[str], 
                              nickname_map: Dict[str, str], threshold: int = None) -> str:
    """Generate a presence matrix table with optional threshold indication."""
    if data is None or data.empty:
        return '<p>No data available.</p>'
    
    available = [d for d in dataset_names if d in data.columns]
    if not available:
        return '<p>No datasets available.</p>'
    
    # Add threshold caption if provided
    threshold_caption = f'<div style="margin-bottom: 8px; color: var(--primary-color); font-weight: 600;">Threshold = {threshold}</div>' if threshold is not None else ''
    
    html = [f'{threshold_caption}<div style="overflow-x: auto;"><table><thead><tr><th>Item</th>']
    for d in available:
        html.append(f'<th>{nickname_map[d]}</th>')
    html.append('<th>Conservation</th></tr></thead><tbody>')
    
    # Sort by conservation (count of datasets present), then by total weight
    data_copy = data.copy()
    data_copy['_conservation'] = (data_copy[available] > 0).sum(axis=1)
    data_copy['_total'] = data_copy[available].sum(axis=1)
    # Sort by conservation descending, then by total descending
    data_copy = data_copy.sort_values(['_conservation', '_total'], ascending=[False, False])
    top = data_copy.head(min(50, len(data_copy)))
    
    for key, row in top.iterrows():
        count = sum(1 for d in available if row.get(d, 0) > 0)
        badge = 'badge-success' if count == len(available) else 'badge-warning' if count > 1 else 'badge-danger'
        
        html.append(f'<tr><td><strong>{key}</strong></td>')
        for d in available:
            w = row.get(d, 0)
            if w > 0:
                html.append(f'<td><span class="presence-check">✔️</span> {int(w)}</td>')
            else:
                html.append('<td><span class="presence-cross">❌</span></td>')
        html.append(f'<td><span class="badge {badge}">{count}/{len(available)}</span></td></tr>')
    
    html.append('</tbody></table></div>')
    return ''.join(html)


def _generate_conservation_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                    key_findings: Dict, nickname_map: Dict[str, str]) -> str:
    """Generate conservation analysis section with distribution by dataset count."""
    num_datasets = len(dataset_names)
    
    # Get path_presence_matrix from comparison_report
    path_presence_matrix = None
    if hasattr(analyzer, 'comparison_report') and analyzer.comparison_report:
        path_presence_matrix = analyzer.comparison_report.get('path_presence_matrix', None)
    
    # Generate Plotly JSON
    plotly_json = "null"
    try:
        from .visualizations import ComparisonVisualizer
        vis = ComparisonVisualizer()
        # Use type-mapped results for proper cross-dataset comparison
        mapped_results = analyzer.get_mapped_results()
        plotly_json = vis.plot_conservation_across_thresholds_plotly(
            mapped_results,
            thresholds,
            align_func=analyzer.get_aligned_data,
            nickname_map=nickname_map,
            path_presence_matrix=path_presence_matrix
        )
    except Exception as e:
        print(f"Warning: Could not generate Plotly chart: {e}")

    html_parts = []
    html_parts.append(f"""
        <div id="conservation" class="section">
            <div class="section-header">🏆 Conservation Analysis</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Edge and path conservation across all {num_datasets} datasets. Shows distribution by how many datasets each edge/path appears in.
                </p>
                
                <!-- Conservation Across Thresholds Plot (Plotly) -->
                <div id="conservation_plot" style="width:100%; height:950px; margin-bottom: 50px; border: 1px solid var(--border-color); border-radius: 8px; overflow: hidden;"></div>
                <script>
                    try {{
                        var plotData = {plotly_json};
                        Plotly.newPlot('conservation_plot', plotData.data, plotData.layout);
                    }} catch (e) {{
                        console.error("Failed to render Plotly chart:", e);
                        document.getElementById('conservation_plot').innerHTML = '<p style="text-align:center; padding:20px;">Interactive chart failed to load. See static image below.</p>';
                    }}
                </script>
                
                <!-- Fallback Static Image -->
                <div style="text-align: center; margin-bottom: 30px; display: none;" id="conservation_static">
                    <img src="comparison_visualizations/conservation_across_thresholds.png" 
                         alt="Conservation Across Thresholds" 
                         style="max-width: 100%; border: 1px solid var(--border-color); border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);"
                         onerror="this.style.display='none'">
                </div>
                <script>
                    // Show static image if Plotly fails or is missing
                    if (typeof Plotly === 'undefined' || !document.getElementById('conservation_plot').innerHTML) {{
                        document.getElementById('conservation_static').style.display = 'block';
                    }}
                </script>
                
                <!-- Conservation Pie Charts Section -->
                <div style="clear: both; margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border-color);">
                    <h3 style="margin-bottom: 20px; color: var(--text-color);">Conservation Distribution by Threshold</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 20px;">
""")
    
    # Color palette for conservation levels (from conserved to unique)
    # Using a gradient from green (conserved in all) to gray (unique)
    conservation_colors = [
        '#22c55e',  # All datasets (green)
        '#84cc16',  # N-1 (lime)
        '#eab308',  # N-2 (yellow)
        '#f59e0b',  # N-3 (orange)
        '#f97316',  # N-4 (orange-red)
        '#ef4444',  # Lower (red)
        '#94a3b8',  # Unique (gray)
    ]
    
    for threshold in thresholds:
        # Get aligned data to compute distribution
        aligned = analyzer.get_aligned_data(threshold)
        available = [d for d in dataset_names if d in aligned.columns]
        n = len(available)
        
        if aligned.empty or n == 0:
            continue
        
        # Compute edge distribution using vectorized operations
        edge_counts = {}  # {count: number_of_edges}
        # Count non-zero columns per row
        presence_matrix = (aligned[available] > 0).astype(int)
        counts_per_row = presence_matrix.sum(axis=1)
        edge_counts = counts_per_row[counts_per_row > 0].value_counts().to_dict()
        
        # Get path data from path_presence_matrix
        path_counts = {}  # {count: number_of_paths}
        try:
            # Use path_presence_matrix from comparison_report
            path_presence_matrix = None
            if hasattr(analyzer, 'comparison_report') and analyzer.comparison_report:
                path_presence_matrix = analyzer.comparison_report.get('path_presence_matrix', None)
            
            if path_presence_matrix is not None and not path_presence_matrix.empty:
                # Find columns for this threshold (sanitized names with _t{threshold} suffix)
                cols_for_threshold = []
                for d in dataset_names:
                    safe_name = analyzer.parameters._sanitize_name(d)
                    col_name = f'{safe_name}_t{threshold}'
                    if col_name in path_presence_matrix.columns:
                        cols_for_threshold.append(col_name)
                
                if cols_for_threshold:
                    # Vectorized: count how many datasets each path appears in at this threshold
                    sub_df = path_presence_matrix[cols_for_threshold]
                    # Convert to boolean - handle 'True', True, and numeric > 0
                    presence_bool = sub_df.apply(lambda col: col.map(lambda v: v == 'True' or v == True or (isinstance(v, (int, float)) and v > 0)))
                    counts_per_path = presence_bool.sum(axis=1)
                    path_counts = counts_per_path[counts_per_path > 0].value_counts().to_dict()
        except Exception as e:
            pass
        
        # Build data for pie charts
        edge_values = []
        edge_labels = []
        edge_colors = []
        
        for count in range(n, 0, -1):
            if count in edge_counts and edge_counts[count] > 0:
                edge_values.append(edge_counts[count])
                if count == n:
                    edge_labels.append(f'All {n} datasets')
                elif count == 1:
                    edge_labels.append('Unique (1)')
                else:
                    edge_labels.append(f'In {count} datasets')
                # Assign color based on position from top
                color_idx = min(n - count, len(conservation_colors) - 1)
                edge_colors.append(conservation_colors[color_idx])
        
        path_values = []
        path_labels = []
        path_colors = []
        
        for count in range(n, 0, -1):
            if count in path_counts and path_counts[count] > 0:
                path_values.append(path_counts[count])
                if count == n:
                    path_labels.append(f'All {n} datasets')
                elif count == 1:
                    path_labels.append('Unique (1)')
                else:
                    path_labels.append(f'In {count} datasets')
                color_idx = min(n - count, len(conservation_colors) - 1)
                path_colors.append(conservation_colors[color_idx])
        
        # Summary stats
        kf = key_findings.get(threshold, {})
        te, ce = kf.get('total_edges', 0), kf.get('common_edges', 0)
        tp, cp = kf.get('total_paths', 0), kf.get('common_paths', 0)
        er = (ce / te * 100) if te > 0 else 0
        pr = (cp / tp * 100) if tp > 0 else 0
        
        # Convert to JSON for JavaScript
        edge_values_json = json.dumps(edge_values)
        edge_labels_json = json.dumps(edge_labels)
        edge_colors_json = json.dumps(edge_colors)
        path_values_json = json.dumps(path_values)
        path_labels_json = json.dumps(path_labels)
        path_colors_json = json.dumps(path_colors)
        
        html_parts.append(f'''
            <div class="card" style="min-width: 350px;">
                <h3 style="font-size: 1rem; margin-bottom: 10px;">Conservation at Threshold = {threshold}</h3>
                <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;">
                    <div id="cons_edge_{threshold}" style="flex: 1; min-width: 160px; max-width: 200px; height: 220px;"></div>
                    <div id="cons_path_{threshold}" style="flex: 1; min-width: 160px; max-width: 200px; height: 220px;"></div>
                </div>
                <div style="text-align: center; color: var(--secondary-color); font-size: 0.8rem; margin-top: 8px;">
                    Edges: {ce}/{te} ({er:.1f}%) | Paths: {cp}/{tp} ({pr:.1f}%)
                </div>
            </div>
            <script>
                (function() {{
                    const edgeValues = {edge_values_json};
                    const edgeLabels = {edge_labels_json};
                    const edgeColors = {edge_colors_json};
                    const pathValues = {path_values_json};
                    const pathLabels = {path_labels_json};
                    const pathColors = {path_colors_json};
                    
                    if (edgeValues.length > 0) {{
                        Plotly.newPlot('cons_edge_{threshold}', [{{
                            values: edgeValues, 
                            labels: edgeLabels,
                            type: 'pie', 
                            hole: 0.4, 
                            marker: {{ colors: edgeColors }},
                            textinfo: 'percent',
                            textposition: 'inside',
                            textfont: {{ size: 10 }},
                            hoverinfo: 'label+value+percent'
                        }}], {{ 
                            title: {{ text: 'Edges', font: {{ size: 12 }} }}, 
                            showlegend: false,
                            margin: {{ t: 30, b: 10, l: 10, r: 10 }} 
                        }}, {{responsive: true}});
                    }} else {{
                        document.getElementById('cons_edge_{threshold}').innerHTML = '<p style="text-align:center;color:#999;">No edge data</p>';
                    }}
                    
                    if (pathValues.length > 0) {{
                        Plotly.newPlot('cons_path_{threshold}', [{{
                            values: pathValues, 
                            labels: pathLabels,
                            type: 'pie', 
                            hole: 0.4, 
                            marker: {{ colors: pathColors }},
                            textinfo: 'percent',
                            textposition: 'inside',
                            textfont: {{ size: 10 }},
                            hoverinfo: 'label+value+percent'
                        }}], {{ 
                            title: {{ text: 'Paths', font: {{ size: 12 }} }}, 
                            showlegend: false,
                            margin: {{ t: 30, b: 10, l: 10, r: 10 }} 
                        }}, {{responsive: true}});
                    }} else {{
                        document.getElementById('cons_path_{threshold}').innerHTML = '<p style="text-align:center;color:#999;">No path data</p>';
                    }}
                }})();
            </script>
''')
    
    # Links to conserved graph visualizations
    html_parts.append('''
                    </div>
                    <div class="card" style="margin-top: 30px;">
                        <h3>Conserved Graph Visualizations</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th>Threshold</th>
                                    <th>Conserved Paths</th>
                                    <th>Conserved Reciprocal Graph</th>
                                </tr>
                            </thead>
                            <tbody>
    ''')

    for threshold in thresholds:
        conserved_path_file = os.path.join(
            analyzer.parameters.full_output_path,
            'conserved_paths',
            f'conserved_network_t{threshold}_network.html'
        )
        conserved_recip_file = os.path.join(
            analyzer.parameters.full_output_path,
            'conserved_reciprocal_graph',
            f'conserved_reciprocal_t{threshold}_network.html'
        )
        html_parts.append(
            f'<tr><td><strong>t={threshold}</strong></td>'
            f'<td>{_make_link(conserved_path_file, analyzer.parameters.full_output_path)}</td>'
            f'<td>{_make_link(conserved_recip_file, analyzer.parameters.full_output_path)}</td></tr>'
        )

    html_parts.append('''
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    ''')
    return ''.join(html_parts)


def _generate_overlap_matrices_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                        nickname_map: Dict[str, str]) -> str:
    """Generate dataset overlap matrices section.
    
    Shows asymmetric N×N matrices where cell (i,j) = 
    number of edges/paths in dataset i that are also found in dataset j.
    Diagonal shows total count for each dataset.
    """
    html_parts = []
    nicknames = [nickname_map[d] for d in dataset_names]
    thresholds_json = json.dumps(thresholds)
    
    html_parts.append(f"""
        <div id="overlap-matrices" class="section">
            <div class="section-header">🔀 Dataset Overlap Matrices</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Asymmetric overlap matrices. Cell (row, col) shows how many edges/paths from the <strong>row</strong> dataset 
                    are also found in the <strong>column</strong> dataset. Diagonal = total count per dataset.
                </p>
                <div class="tabs">
                    <div class="tab-buttons">
""")
    
    for i, t in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(f'<button class="tab-btn {active}" onclick="showOverlapTab({t})">t = {t}</button>')
    html_parts.append('</div>')
    
    for i, threshold in enumerate(thresholds):
        aligned = analyzer.get_aligned_data(threshold)
        
        # Always include ALL datasets, even if they have no connections at this threshold
        # This ensures consistent matrix dimensions across thresholds
        available = dataset_names  # Use all datasets, not just those with data
        n = len(available)
        
        # Ensure aligned data has columns for all datasets (fill missing with 0)
        for d in dataset_names:
            if d not in aligned.columns:
                aligned[d] = 0
        
        if n == 0:
            active = 'active' if i == 0 else ''
            html_parts.append(f'<div id="overlap_tab_{threshold}" class="tab-content {active}"><p>No datasets configured.</p></div>')
            continue
        
        # Compute edge overlap matrix (asymmetric)
        edge_overlap = [[0 for _ in range(n)] for _ in range(n)]
        for i1, d1 in enumerate(available):
            edges_in_d1 = set(aligned.index[aligned[d1] > 0]) if d1 in aligned.columns else set()
            edge_overlap[i1][i1] = len(edges_in_d1)  # Diagonal = total
            for i2, d2 in enumerate(available):
                if i1 != i2:
                    edges_in_d2 = set(aligned.index[aligned[d2] > 0]) if d2 in aligned.columns else set()
                    # Edges in d1 that are also in d2
                    edge_overlap[i1][i2] = len(edges_in_d1 & edges_in_d2)
        
        # Compute path overlap matrix (asymmetric)
        path_overlap = [[0 for _ in range(n)] for _ in range(n)]
        try:
            path_data = analyzer._get_path_data_for_threshold(threshold)
            if path_data is not None and not path_data.empty:
                for i1, d1 in enumerate(available):
                    paths_in_d1 = set(path_data.index[path_data[d1] > 0]) if d1 in path_data.columns else set()
                    path_overlap[i1][i1] = len(paths_in_d1)
                    for i2, d2 in enumerate(available):
                        if i1 != i2:
                            paths_in_d2 = set(path_data.index[path_data[d2] > 0]) if d2 in path_data.columns else set()
                            path_overlap[i1][i2] = len(paths_in_d1 & paths_in_d2)
        except Exception as e:
            pass  # Path data may not be available
        
        # Calculate chart size - make matrices larger (~1/3 page width) and square
        # For a 1600px max-width container with padding, aim for ~450px per matrix
        chart_size = 450  # Fixed size for square matrices
        
        labels = [nickname_map[d] for d in available]
        edge_overlap_json = json.dumps(edge_overlap)
        path_overlap_json = json.dumps(path_overlap)
        labels_json = json.dumps(labels)
        
        active = 'active' if i == 0 else ''
        html_parts.append(f'''
            <div id="overlap_tab_{threshold}" class="tab-content {active}">
                <div class="card">
                    <h3>Dataset Overlap at Threshold = {threshold}</h3>
                    <div style="margin-bottom: 15px; text-align: center;">
                        <label style="margin-right: 20px; cursor: pointer;">
                            <input type="radio" name="overlap_mode_{threshold}" value="count" checked 
                                   onclick="updateOverlapMode_{threshold}('count')"> Show Count
                        </label>
                        <label style="cursor: pointer;">
                            <input type="radio" name="overlap_mode_{threshold}" value="proportion" 
                                   onclick="updateOverlapMode_{threshold}('proportion')"> Show Proportion
                        </label>
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 30px; justify-content: center;">
                        <div style="width: {chart_size}px;">
                            <h5 style="text-align: center; margin-bottom: 8px; color: var(--secondary-color);">Edge Overlap</h5>
                            <div id="edge_overlap_{threshold}" style="width: {chart_size}px; height: {chart_size}px;"></div>
                        </div>
                        <div style="width: {chart_size}px;">
                            <h5 style="text-align: center; margin-bottom: 8px; color: var(--secondary-color);">Path Overlap</h5>
                            <div id="path_overlap_{threshold}" style="width: {chart_size}px; height: {chart_size}px;"></div>
                        </div>
                    </div>
                    <p style="font-size: 0.8em; color: #64748b; margin-top: 10px; text-align: center;">
                        Read as: edges/paths from <strong>row</strong> dataset found in <strong>column</strong> dataset.
                        Diagonal = total in that dataset.
                    </p>
                </div>
            </div>
            <script>
                (function() {{
                    const labels_{threshold} = {labels_json};
                    const edgeOverlap_{threshold} = {edge_overlap_json};
                    const pathOverlap_{threshold} = {path_overlap_json};
                    
                    // Compute proportion matrices (row-normalized: what % of row's edges/paths are in col)
                    const edgeProportion_{threshold} = edgeOverlap_{threshold}.map((row, i) => 
                        row.map((val, j) => {{
                            const diag = edgeOverlap_{threshold}[i][i];
                            return diag > 0 ? val / diag : 0;
                        }})
                    );
                    const pathProportion_{threshold} = pathOverlap_{threshold}.map((row, i) => 
                        row.map((val, j) => {{
                            const diag = pathOverlap_{threshold}[i][i];
                            return diag > 0 ? val / diag : 0;
                        }})
                    );
                    
                    // Create text annotations for count mode
                    const edgeTextCount_{threshold} = edgeOverlap_{threshold}.map((row, i) => 
                        row.map((val, j) => {{
                            const diag = edgeOverlap_{threshold}[i][i];
                            const pct = diag > 0 ? (val / diag * 100).toFixed(0) : 0;
                            return i === j ? String(val) : val + ' (' + pct + '%)';
                        }})
                    );
                    const pathTextCount_{threshold} = pathOverlap_{threshold}.map((row, i) => 
                        row.map((val, j) => {{
                            const diag = pathOverlap_{threshold}[i][i];
                            const pct = diag > 0 ? (val / diag * 100).toFixed(0) : 0;
                            return i === j ? String(val) : val + ' (' + pct + '%)';
                        }})
                    );
                    
                    // Create text annotations for proportion mode
                    const edgeTextProp_{threshold} = edgeProportion_{threshold}.map((row, i) => 
                        row.map((val, j) => (val * 100).toFixed(1) + '%')
                    );
                    const pathTextProp_{threshold} = pathProportion_{threshold}.map((row, i) => 
                        row.map((val, j) => (val * 100).toFixed(1) + '%')
                    );
                    
                    // Square matrix layout with fixed aspect ratio and no grid
                    const baseLayout = {{
                        xaxis: {{ 
                            tickangle: -45, 
                            side: 'bottom', 
                            tickfont: {{size: 11}}, 
                            constrain: 'domain',
                            showgrid: false,
                            zeroline: false
                        }},
                        yaxis: {{ 
                            autorange: 'reversed', 
                            tickfont: {{size: 11}}, 
                            scaleanchor: 'x', 
                            scaleratio: 1,
                            showgrid: false,
                            zeroline: false
                        }},
                        margin: {{ l: 100, r: 30, t: 30, b: 100 }},
                        width: {chart_size},
                        height: {chart_size},
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)'
                    }};
                    
                    // Store current mode
                    window.overlapMode_{threshold} = 'count';
                    
                    // Update function for toggling modes
                    window.updateOverlapMode_{threshold} = function(mode) {{
                        window.overlapMode_{threshold} = mode;
                        
                        if (mode === 'count') {{
                            // Count mode
                            Plotly.react('edge_overlap_{threshold}', [{{
                                z: edgeOverlap_{threshold},
                                x: labels_{threshold},
                                y: labels_{threshold},
                                type: 'heatmap',
                                colorscale: [[0, '#f8fafc'], [0.5, '#93c5fd'], [1, '#2563eb']],
                                text: edgeTextCount_{threshold},
                                texttemplate: '%{{text}}',
                                textfont: {{ size: 10 }},
                                hovertemplate: '%{{y}} in %{{x}}: %{{z}} edges<extra></extra>',
                                showscale: true,
                                colorbar: {{ title: 'Count', len: 0.5 }}
                            }}], baseLayout);
                            
                            Plotly.react('path_overlap_{threshold}', [{{
                                z: pathOverlap_{threshold},
                                x: labels_{threshold},
                                y: labels_{threshold},
                                type: 'heatmap',
                                colorscale: [[0, '#f8fafc'], [0.5, '#c4b5fd'], [1, '#8b5cf6']],
                                text: pathTextCount_{threshold},
                                texttemplate: '%{{text}}',
                                textfont: {{ size: 10 }},
                                hovertemplate: '%{{y}} in %{{x}}: %{{z}} paths<extra></extra>',
                                showscale: true,
                                colorbar: {{ title: 'Count', len: 0.5 }}
                            }}], baseLayout);
                        }} else {{
                            // Proportion mode (row-normalized)
                            Plotly.react('edge_overlap_{threshold}', [{{
                                z: edgeProportion_{threshold},
                                x: labels_{threshold},
                                y: labels_{threshold},
                                type: 'heatmap',
                                colorscale: [[0, '#f8fafc'], [0.5, '#93c5fd'], [1, '#2563eb']],
                                text: edgeTextProp_{threshold},
                                texttemplate: '%{{text}}',
                                textfont: {{ size: 10 }},
                                hovertemplate: '%{{y}} in %{{x}}: %{{z:.1%}} of row edges<extra></extra>',
                                showscale: true,
                                colorbar: {{ title: 'Proportion', len: 0.5, tickformat: '.0%' }},
                                zmin: 0, zmax: 1
                            }}], baseLayout);
                            
                            Plotly.react('path_overlap_{threshold}', [{{
                                z: pathProportion_{threshold},
                                x: labels_{threshold},
                                y: labels_{threshold},
                                type: 'heatmap',
                                colorscale: [[0, '#f8fafc'], [0.5, '#c4b5fd'], [1, '#8b5cf6']],
                                text: pathTextProp_{threshold},
                                texttemplate: '%{{text}}',
                                textfont: {{ size: 10 }},
                                hovertemplate: '%{{y}} in %{{x}}: %{{z:.1%}} of row paths<extra></extra>',
                                showscale: true,
                                colorbar: {{ title: 'Proportion', len: 0.5, tickformat: '.0%' }},
                                zmin: 0, zmax: 1
                            }}], baseLayout);
                        }}
                    }};
                    
                    // Initial plot in count mode
                    Plotly.newPlot('edge_overlap_{threshold}', [{{
                        z: edgeOverlap_{threshold},
                        x: labels_{threshold},
                        y: labels_{threshold},
                        type: 'heatmap',
                        colorscale: [[0, '#f8fafc'], [0.5, '#93c5fd'], [1, '#2563eb']],
                        text: edgeTextCount_{threshold},
                        texttemplate: '%{{text}}',
                        textfont: {{ size: 10 }},
                        hovertemplate: '%{{y}} in %{{x}}: %{{z}} edges<extra></extra>',
                        showscale: true,
                        colorbar: {{ title: 'Count', len: 0.5 }}
                    }}], baseLayout, {{responsive: false}});
                    
                    Plotly.newPlot('path_overlap_{threshold}', [{{
                        z: pathOverlap_{threshold},
                        x: labels_{threshold},
                        y: labels_{threshold},
                        type: 'heatmap',
                        colorscale: [[0, '#f8fafc'], [0.5, '#c4b5fd'], [1, '#8b5cf6']],
                        text: pathTextCount_{threshold},
                        texttemplate: '%{{text}}',
                        textfont: {{ size: 10 }},
                        hovertemplate: '%{{y}} in %{{x}}: %{{z}} paths<extra></extra>',
                        showscale: true,
                        colorbar: {{ title: 'Count', len: 0.5 }}
                    }}], baseLayout, {{responsive: false}});
                }})();
            </script>
''')
    
    html_parts.append(f"""
                </div>
                <script>
                    function showOverlapTab(threshold) {{
                        document.querySelectorAll('#overlap-matrices .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#overlap-matrices .tab-btn').forEach(el => el.classList.remove('active'));
                        document.getElementById('overlap_tab_' + threshold).classList.add('active');
                        event.target.classList.add('active');
                    }}
                </script>
            </div>
        </div>
""")
    
    return ''.join(html_parts)


def _generate_statistics_section(analyzer, dataset_names: List[str], thresholds: List[int],
                                  nickname_map: Dict[str, str]) -> str:
    """Generate statistics section."""
    html_parts = []
    html_parts.append("""
        <div id="statistics" class="section">
            <div class="section-header">📉 Statistics</div>
            <div class="section-content">
                <p style="margin-bottom: 20px; color: var(--secondary-color);">
                    Detailed statistics per dataset and threshold.
                </p>
""")
    
    html_parts.append('<div class="tabs"><div class="tab-buttons">')
    for i, t in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(f'<button class="tab-btn {active}" onclick="showStatsTab({t})">t = {t}</button>')
    html_parts.append('</div>')
    
    for i, threshold in enumerate(thresholds):
        active = 'active' if i == 0 else ''
        html_parts.append(f'<div id="stats_tab_{threshold}" class="tab-content {active}">')
        html_parts.append(_generate_stats_table(analyzer, dataset_names, threshold, nickname_map))
        html_parts.append('</div>')
    
    html_parts.append("""
                </div>
                <script>
                    function showStatsTab(threshold) {
                        document.querySelectorAll('#statistics .tab-content').forEach(el => el.classList.remove('active'));
                        document.querySelectorAll('#statistics .tab-btn').forEach(el => el.classList.remove('active'));
                        document.getElementById('stats_tab_' + threshold).classList.add('active');
                        event.target.classList.add('active');
                    }
                </script>
""")
    
    # Generate unified 2x2 Similarity Trends plot across thresholds
    similarity_trends_html = _generate_similarity_trends_2x2_plot(analyzer, dataset_names, thresholds, nickname_map)
    html_parts.append(similarity_trends_html)
    
    html_parts.append("""
            </div>
        </div>
""")
    
    return ''.join(html_parts)


def _generate_similarity_trends_2x2_plot(analyzer, dataset_names: List[str], thresholds: List[int],
                                          nickname_map: Dict[str, str]) -> str:
    """
    Generate a 2x2 subplot showing all 4 similarity metrics across thresholds.
    
    Layout:
        Row 1: Jaccard (set overlap) | Edge Rank (all-edge ranking)
        Row 2: Cosine (all-edge directional) | Spearman (shared-edge ranking)
    """
    from .metrics import ComparisonMetrics
    import json
    
    metrics = ComparisonMetrics()
    
    # Collect data for all 4 metrics
    # Structure: {metric_name: {(d1, d2): {threshold: value}}}
    all_pair_data = {
        'jaccard': {},
        'edge_rank': {},
        'cosine': {},
        'spearman': {}
    }
    
    available_pairs = []
    for i, d1 in enumerate(dataset_names):
        for d2 in dataset_names[i+1:]:
            pair_key = (d1, d2)
            available_pairs.append(pair_key)
            for metric in all_pair_data:
                all_pair_data[metric][pair_key] = {}
    
    for threshold in thresholds:
        aligned = analyzer.get_aligned_data(threshold)
        if aligned.empty:
            continue
        
        available = [d for d in dataset_names if d in aligned.columns]
        
        for i, d1 in enumerate(available):
            for d2 in available[i+1:]:
                pair_key = (d1, d2)
                if pair_key not in all_pair_data['jaccard']:
                    pair_key = (d2, d1)
                
                # Jaccard (set overlap)
                c1, c2 = aligned[d1], aligned[d2]
                s1, s2 = set(aligned.index[c1 > 0]), set(aligned.index[c2 > 0])
                inter, union = len(s1 & s2), len(s1 | s2)
                jac = inter / union if union > 0 else 0
                all_pair_data['jaccard'][pair_key][threshold] = jac
                
                # Get edge weights as Series
                weights_a = aligned[d1].dropna()
                weights_b = aligned[d2].dropna()
                
                # Edge Rank (all edges, 0 for missing)
                edge_rank = metrics.calculate_edge_list_rank_correlation(weights_a, weights_b)
                all_pair_data['edge_rank'][pair_key][threshold] = edge_rank
                
                # Cosine (all edges, 0 for missing)
                cosine = metrics.calculate_cosine_similarity(weights_a, weights_b)
                all_pair_data['cosine'][pair_key][threshold] = cosine
                
                # Spearman (shared edges only)
                spearman = metrics.calculate_spearman_rank_correlation(weights_a, weights_b)
                all_pair_data['spearman'][pair_key][threshold] = spearman
    
    # Build Plotly subplot data
    colors = ['#3b82f6', '#f97316', '#22c55e', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#eab308']
    
    # Helper function to create traces for a metric
    # Plotly subplot indices: row1col1=1, row1col2=2, row2col1=3, row2col2=4
    def make_traces(metric_data, subplot_idx, show_legend=False):
        traces = []
        axis_suffix = '' if subplot_idx == 1 else str(subplot_idx)
        
        for idx, pair_key in enumerate(available_pairs):
            d1, d2 = pair_key
            n1, n2 = nickname_map.get(d1, d1), nickname_map.get(d2, d2)
            
            x_vals = []
            y_vals = []
            for t in thresholds:
                if t in metric_data[pair_key]:
                    val = metric_data[pair_key][t]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        x_vals.append(t)
                        y_vals.append(val)
            
            if x_vals:
                color = colors[idx % len(colors)]
                traces.append({
                    'x': x_vals,
                    'y': y_vals,
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': f'{n1} vs {n2}',
                    'line': {'color': color, 'width': 2},
                    'marker': {'size': 6},
                    'legendgroup': f'{n1} vs {n2}',
                    'showlegend': show_legend,
                    'xaxis': f'x{axis_suffix}',
                    'yaxis': f'y{axis_suffix}'
                })
        
        # Add average trace
        avg_x = []
        avg_y = []
        for t in thresholds:
            vals = [metric_data[pk].get(t) for pk in available_pairs if t in metric_data[pk]]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if vals:
                avg_x.append(t)
                avg_y.append(sum(vals) / len(vals))
        
        if avg_x:
            traces.append({
                'x': avg_x,
                'y': avg_y,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Average',
                'line': {'color': '#64748b', 'width': 3, 'dash': 'dash'},
                'marker': {'size': 8, 'symbol': 'star'},
                'legendgroup': 'Average',
                'showlegend': show_legend,
                'xaxis': f'x{axis_suffix}',
                'yaxis': f'y{axis_suffix}'
            })
        
        return traces
    
    # Build all traces (subplot indices: 1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right)
    all_traces = []
    all_traces.extend(make_traces(all_pair_data['jaccard'], 1, show_legend=True))   # Top-left: Jaccard
    all_traces.extend(make_traces(all_pair_data['edge_rank'], 2, show_legend=False)) # Top-right: Edge Rank
    all_traces.extend(make_traces(all_pair_data['cosine'], 3, show_legend=False))    # Bottom-left: Cosine
    all_traces.extend(make_traces(all_pair_data['spearman'], 4, show_legend=False))  # Bottom-right: Spearman
    
    # Layout with 2x2 subplots
    layout = {
        'grid': {'rows': 2, 'columns': 2, 'pattern': 'independent'},
        'annotations': [
            {'text': '<b>Jaccard</b> (set overlap) [0,1]', 'x': 0.22, 'y': 1.08, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 13}},
            {'text': '<b>Edge Rank</b> (all edges) [-1,1]', 'x': 0.78, 'y': 1.08, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 13}},
            {'text': '<b>Cosine</b> (all edges) [0,1]', 'x': 0.22, 'y': 0.45, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 13}},
            {'text': '<b>Spearman</b> (shared only) [-1,1]', 'x': 0.78, 'y': 0.45, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 13}}
        ],
        # Top-left: Jaccard [0, 1]
        'xaxis': {'title': '', 'type': 'category', 'domain': [0, 0.45]},
        'yaxis': {'title': 'Similarity', 'range': [0, 1], 'domain': [0.55, 1]},
        # Top-right: Edge Rank [-1, 1]
        'xaxis2': {'title': '', 'type': 'category', 'domain': [0.55, 1]},
        'yaxis2': {'title': '', 'range': [-1, 1], 'domain': [0.55, 1]},
        # Bottom-left: Cosine [0, 1]
        'xaxis3': {'title': 'Threshold', 'type': 'category', 'domain': [0, 0.45]},
        'yaxis3': {'title': 'Similarity', 'range': [0, 1], 'domain': [0, 0.42]},
        # Bottom-right: Spearman [-1, 1]
        'xaxis4': {'title': 'Threshold', 'type': 'category', 'domain': [0.55, 1]},
        'yaxis4': {'title': '', 'range': [-1, 1], 'domain': [0, 0.42]},
        # Legend and margins
        'legend': {'orientation': 'h', 'y': -0.15, 'x': 0.5, 'xanchor': 'center'},
        'margin': {'t': 60, 'b': 100, 'l': 60, 'r': 30},
        'hovermode': 'x unified',
        # Zero reference lines for [-1, 1] plots
        'shapes': [
            {'type': 'line', 'x0': 0, 'x1': 1, 'y0': 0, 'y1': 0, 'xref': 'x2 domain', 'yref': 'y2', 'line': {'color': 'gray', 'width': 1, 'dash': 'dot'}},
            {'type': 'line', 'x0': 0, 'x1': 1, 'y0': 0, 'y1': 0, 'xref': 'x4 domain', 'yref': 'y4', 'line': {'color': 'gray', 'width': 1, 'dash': 'dot'}}
        ]
    }
    
    plotly_data = json.dumps({'data': all_traces, 'layout': layout})
    
    return f'''
        <div class="card" style="margin-top: 30px;">
            <h3>Similarity Trends Across Thresholds</h3>
            <p style="color: var(--secondary-color); font-size: 0.85rem; margin-bottom: 15px;">
                How similarity metrics change with increasing threshold. <strong>Edge Rank</strong> and <strong>Cosine</strong> compare 
                all edges (assigning 0 to missing edges), while <strong>Spearman (shared)</strong> only compares edges present in both datasets.
                The dashed line shows the average across all dataset pairs.
            </p>
            <div id="similarity_trends_2x2" style="width: 100%; height: 700px;"></div>
            <script>
                (function() {{
                    try {{
                        var plotData = {plotly_data};
                        Plotly.newPlot('similarity_trends_2x2', plotData.data, plotData.layout, {{responsive: true}});
                    }} catch(e) {{
                        console.error("Failed to render similarity trends plot:", e);
                        document.getElementById('similarity_trends_2x2').innerHTML = '<p style="text-align:center;color:#999;">Chart failed to load</p>';
                    }}
                }})();
            </script>
        </div>
    '''


def _generate_jaccard_similarity_plot(analyzer, dataset_names: List[str], thresholds: List[int],
                                       nickname_map: Dict[str, str]) -> str:
    """Generate a line plot showing Jaccard similarity across thresholds for all dataset pairs."""
    
    # Collect Jaccard similarities for each pair at each threshold
    pair_data = {}  # {(d1, d2): {threshold: jaccard}}
    available_pairs = []
    
    for i, d1 in enumerate(dataset_names):
        for d2 in dataset_names[i+1:]:
            pair_key = (d1, d2)
            pair_data[pair_key] = {}
            available_pairs.append(pair_key)
    
    for threshold in thresholds:
        aligned = analyzer.get_aligned_data(threshold)
        if aligned.empty:
            continue
        
        available = [d for d in dataset_names if d in aligned.columns]
        
        for i, d1 in enumerate(available):
            for d2 in available[i+1:]:
                pair_key = (d1, d2)
                if pair_key not in pair_data:
                    pair_key = (d2, d1)  # Try reverse
                
                c1, c2 = aligned[d1], aligned[d2]
                s1, s2 = set(aligned.index[c1 > 0]), set(aligned.index[c2 > 0])
                inter, union = len(s1 & s2), len(s1 | s2)
                jac = inter / union if union > 0 else 0
                pair_data[pair_key][threshold] = jac
    
    # Build traces for Plotly
    traces = []
    colors = ['#3b82f6', '#f97316', '#22c55e', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#eab308']
    
    for idx, pair_key in enumerate(available_pairs):
        d1, d2 = pair_key
        n1, n2 = nickname_map.get(d1, d1), nickname_map.get(d2, d2)
        
        x_vals = []
        y_vals = []
        for t in thresholds:
            if t in pair_data[pair_key]:
                x_vals.append(t)
                y_vals.append(pair_data[pair_key][t])
        
        if x_vals:
            color = colors[idx % len(colors)]
            traces.append({
                'x': x_vals,
                'y': y_vals,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': f'{n1} vs {n2}',
                'line': {'color': color, 'width': 2},
                'marker': {'size': 8}
            })
    
    # Calculate average Jaccard across all pairs
    avg_x = []
    avg_y = []
    for t in thresholds:
        vals = [pair_data[pk].get(t) for pk in available_pairs if t in pair_data[pk]]
        vals = [v for v in vals if v is not None]
        if vals:
            avg_x.append(t)
            avg_y.append(sum(vals) / len(vals))
    
    if avg_x:
        traces.append({
            'x': avg_x,
            'y': avg_y,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Average',
            'line': {'color': '#64748b', 'width': 3, 'dash': 'dash'},
            'marker': {'size': 10, 'symbol': 'star'}
        })
    
    layout = {
        'title': {'text': 'Jaccard Similarity Across Thresholds', 'font': {'size': 16}},
        'xaxis': {'title': 'Threshold', 'type': 'category'},
        'yaxis': {'title': 'Jaccard Index', 'range': [0, 1]},
        'legend': {'orientation': 'h', 'y': -0.2, 'x': 0.5, 'xanchor': 'center'},
        'margin': {'t': 50, 'b': 80, 'l': 60, 'r': 30},
        'hovermode': 'x unified'
    }
    
    plotly_data = json.dumps({'data': traces, 'layout': layout})
    
    return f'''
        <div class="card" style="margin-top: 30px;">
            <h3>Jaccard Similarity Trend Across Thresholds</h3>
            <p style="color: var(--secondary-color); font-size: 0.85rem; margin-bottom: 15px;">
                How edge overlap between dataset pairs changes with increasing threshold. The dashed line shows the average across all pairs.
            </p>
            <div id="jaccard_trend_plot" style="width: 100%; height: 1050px;"></div>
            <script>
                (function() {{
                    try {{
                        var plotData = {plotly_data};
                        Plotly.newPlot('jaccard_trend_plot', plotData.data, plotData.layout, {{responsive: true}});
                    }} catch(e) {{
                        console.error("Failed to render Jaccard trend plot:", e);
                        document.getElementById('jaccard_trend_plot').innerHTML = '<p style="text-align:center;color:#999;">Chart failed to load</p>';
                    }}
                }})();
            </script>
        </div>
    '''


def _generate_edge_rank_correlation_plot(analyzer, dataset_names: List[str], thresholds: List[int],
                                          nickname_map: Dict[str, str]) -> str:
    """Generate a line plot showing Edge Rank Correlation across thresholds for all dataset pairs."""
    from .metrics import ComparisonMetrics
    
    metrics = ComparisonMetrics()
    
    # Collect edge rank correlations for each pair at each threshold
    pair_data = {}  # {(d1, d2): {threshold: correlation}}
    available_pairs = []
    
    for i, d1 in enumerate(dataset_names):
        for d2 in dataset_names[i+1:]:
            pair_key = (d1, d2)
            pair_data[pair_key] = {}
            available_pairs.append(pair_key)
    
    for threshold in thresholds:
        aligned = analyzer.get_aligned_data(threshold)
        if aligned.empty:
            continue
        
        available = [d for d in dataset_names if d in aligned.columns]
        
        for i, d1 in enumerate(available):
            for d2 in available[i+1:]:
                pair_key = (d1, d2)
                if pair_key not in pair_data:
                    pair_key = (d2, d1)  # Try reverse
                
                # Get edge weights as Series
                weights_a = aligned[d1].dropna()
                weights_b = aligned[d2].dropna()
                
                # Calculate edge rank correlation
                corr = metrics.calculate_edge_list_rank_correlation(weights_a, weights_b)
                pair_data[pair_key][threshold] = corr
    
    # Build traces for Plotly
    traces = []
    colors = ['#3b82f6', '#f97316', '#22c55e', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#eab308']
    
    for idx, pair_key in enumerate(available_pairs):
        d1, d2 = pair_key
        n1, n2 = nickname_map.get(d1, d1), nickname_map.get(d2, d2)
        
        x_vals = []
        y_vals = []
        for t in thresholds:
            if t in pair_data[pair_key]:
                val = pair_data[pair_key][t]
                # Skip NaN values
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    x_vals.append(t)
                    y_vals.append(val)
        
        if x_vals:
            color = colors[idx % len(colors)]
            traces.append({
                'x': x_vals,
                'y': y_vals,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': f'{n1} vs {n2}',
                'line': {'color': color, 'width': 2},
                'marker': {'size': 8}
            })
    
    # Calculate average across all pairs, ignoring NaN values
    avg_x = []
    avg_y = []
    for t in thresholds:
        vals = [pair_data[pk].get(t) for pk in available_pairs if t in pair_data[pk]]
        # Filter out None and NaN values
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if vals:
            avg_x.append(t)
            avg_y.append(sum(vals) / len(vals))
    
    if avg_x:
        traces.append({
            'x': avg_x,
            'y': avg_y,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Average',
            'line': {'color': '#64748b', 'width': 3, 'dash': 'dash'},
            'marker': {'size': 10, 'symbol': 'star'}
        })
    
    layout = {
        'title': {'text': 'Edge Rank Correlation Across Thresholds', 'font': {'size': 16}},
        'xaxis': {'title': 'Threshold', 'type': 'category'},
        'yaxis': {'title': 'Edge Rank Correlation', 'range': [-1, 1]},
        'legend': {'orientation': 'h', 'y': -0.2, 'x': 0.5, 'xanchor': 'center'},
        'margin': {'t': 50, 'b': 80, 'l': 60, 'r': 30},
        'hovermode': 'x unified'
    }
    
    plotly_data = json.dumps({'data': traces, 'layout': layout})
    
    return f'''
        <div class="card" style="margin-top: 30px;">
            <h3>Edge Rank Correlation Trend Across Thresholds</h3>
            <p style="color: var(--secondary-color); font-size: 0.85rem; margin-bottom: 15px;">
                Compares the ranking of edges by weight (using union of edges). Values range from -1 (inverse ranking) to +1 (identical ranking). The dashed line shows the average across all pairs.
            </p>
            <div id="edge_rank_trend_plot" style="width: 100%; height: 1050px;"></div>
            <script>
                (function() {{
                    try {{
                        var plotData = {plotly_data};
                        Plotly.newPlot('edge_rank_trend_plot', plotData.data, plotData.layout, {{responsive: true}});
                    }} catch(e) {{
                        console.error("Failed to render Edge Rank Correlation trend plot:", e);
                        document.getElementById('edge_rank_trend_plot').innerHTML = '<p style="text-align:center;color:#999;">Chart failed to load</p>';
                    }}
                }})();
            </script>
        </div>
    '''


def _generate_cosine_similarity_trend_plot(analyzer, dataset_names: List[str], thresholds: List[int],
                                          nickname_map: Dict[str, str]) -> str:
    """Generate a line plot showing Cosine Similarity across thresholds for all dataset pairs."""
    from .metrics import ComparisonMetrics
    
    metrics = ComparisonMetrics()
    
    # Collect cosine similarities for each pair at each threshold
    pair_data = {}  # {(d1, d2): {threshold: cosine_sim}}
    available_pairs = []
    
    for i, d1 in enumerate(dataset_names):
        for d2 in dataset_names[i+1:]:
            pair_key = (d1, d2)
            pair_data[pair_key] = {}
            available_pairs.append(pair_key)
    
    for threshold in thresholds:
        try:
            aligned = analyzer.get_aligned_data(threshold)
        except:
            continue
        
        if aligned is None or aligned.empty:
            continue
        
        available = [d for d in dataset_names if d in aligned.columns]
        
        for i, d1 in enumerate(available):
            for d2 in available[i+1:]:
                pair_key = (d1, d2)
                if pair_key not in pair_data:
                    pair_key = (d2, d1)  # Try reverse
                
                # Get edge weights as Series
                weights_a = aligned[d1].dropna()
                weights_b = aligned[d2].dropna()
                
                # Calculate cosine similarity
                cosine = metrics.calculate_cosine_similarity(weights_a, weights_b)
                pair_data[pair_key][threshold] = cosine
    
    # Build traces for Plotly
    traces = []
    colors = ['#3b82f6', '#f97316', '#22c55e', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#eab308']
    
    for idx, pair_key in enumerate(available_pairs):
        d1, d2 = pair_key
        n1, n2 = nickname_map.get(d1, d1), nickname_map.get(d2, d2)
        
        x_vals = []
        y_vals = []
        for t in thresholds:
            if t in pair_data[pair_key]:
                val = pair_data[pair_key][t]
                # Skip NaN values
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    x_vals.append(t)
                    y_vals.append(val)
        
        if x_vals:
            color = colors[idx % len(colors)]
            traces.append({
                'x': x_vals,
                'y': y_vals,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': f'{n1} vs {n2}',
                'line': {'color': color, 'width': 2},
                'marker': {'size': 8}
            })
    
    # Calculate average across all pairs, ignoring NaN values
    avg_x = []
    avg_y = []
    for t in thresholds:
        vals = [pair_data[pk].get(t) for pk in available_pairs if t in pair_data[pk]]
        # Filter out None and NaN values
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if vals:
            avg_x.append(t)
            avg_y.append(sum(vals) / len(vals))
    
    if avg_x:
        traces.append({
            'x': avg_x,
            'y': avg_y,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Average',
            'line': {'color': '#64748b', 'width': 3, 'dash': 'dash'},
            'marker': {'size': 10, 'symbol': 'star'}
        })
    
    layout = {
        'title': {'text': 'Cosine Similarity Across Thresholds', 'font': {'size': 16}},
        'xaxis': {'title': 'Threshold', 'type': 'category'},
        'yaxis': {'title': 'Cosine Similarity', 'range': [0, 1]},
        'legend': {'orientation': 'h', 'y': -0.2, 'x': 0.5, 'xanchor': 'center'},
        'margin': {'t': 50, 'b': 80, 'l': 60, 'r': 30},
        'hovermode': 'x unified'
    }
    
    plotly_data = json.dumps({'data': traces, 'layout': layout})
    
    return f'''
        <div class="card" style="margin-top: 30px;">
            <h3>Cosine Similarity Trend Across Thresholds</h3>
            <p style="color: var(--secondary-color); font-size: 0.85rem; margin-bottom: 15px;">
                Compares edge weight distributions using cosine similarity (scale-invariant). Higher values indicate more similar connectivity patterns. The dashed line shows the average across all pairs (N/A values excluded).
            </p>
            <div id="cosine_trend_plot" style="width: 100%; height: 1050px;"></div>
            <script>
                (function() {{
                    try {{
                        var plotData = {plotly_data};
                        Plotly.newPlot('cosine_trend_plot', plotData.data, plotData.layout, {{responsive: true}});
                    }} catch(e) {{
                        console.error("Failed to render Cosine Similarity trend plot:", e);
                        document.getElementById('cosine_trend_plot').innerHTML = '<p style="text-align:center;color:#999;">Chart failed to load</p>';
                    }}
                }})();
            </script>
        </div>
    '''


def _generate_path_rank_correlation_plot(analyzer, dataset_names: List[str], thresholds: List[int],
                                          nickname_map: Dict[str, str]) -> str:
    """Generate a line plot showing Path Rank Correlation across thresholds for all dataset pairs."""
    from .metrics import ComparisonMetrics
    
    metrics = ComparisonMetrics()
    
    # Collect path rank correlations for each pair at each threshold
    pair_data = {}  # {(d1, d2): {threshold: correlation}}
    available_pairs = []
    
    for i, d1 in enumerate(dataset_names):
        for d2 in dataset_names[i+1:]:
            pair_key = (d1, d2)
            pair_data[pair_key] = {}
            available_pairs.append(pair_key)
    
    for threshold in thresholds:
        try:
            path_df = analyzer._get_path_data_for_threshold(threshold)
        except:
            continue
        
        if path_df is None or path_df.empty:
            continue
        
        available = [d for d in dataset_names if d in path_df.columns]
        
        for i, d1 in enumerate(available):
            for d2 in available[i+1:]:
                pair_key = (d1, d2)
                if pair_key not in pair_data:
                    pair_key = (d2, d1)  # Try reverse
                
                # Get path weights as Series
                paths_a = path_df[d1].dropna()
                paths_b = path_df[d2].dropna()
                
                # Calculate path rank correlation
                corr = metrics.calculate_path_list_rank_correlation(paths_a, paths_b)
                pair_data[pair_key][threshold] = corr
    
    # Build traces for Plotly
    traces = []
    colors = ['#3b82f6', '#f97316', '#22c55e', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#eab308']
    
    for idx, pair_key in enumerate(available_pairs):
        d1, d2 = pair_key
        n1, n2 = nickname_map.get(d1, d1), nickname_map.get(d2, d2)
        
        x_vals = []
        y_vals = []
        for t in thresholds:
            if t in pair_data[pair_key]:
                val = pair_data[pair_key][t]
                # Skip NaN values in plotting
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    x_vals.append(t)
                    y_vals.append(val)
        
        if x_vals:
            color = colors[idx % len(colors)]
            traces.append({
                'x': x_vals,
                'y': y_vals,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': f'{n1} vs {n2}',
                'line': {'color': color, 'width': 2},
                'marker': {'size': 8}
            })
    
    # Calculate average across all pairs (excluding NaN values)
    avg_x = []
    avg_y = []
    for t in thresholds:
        vals = [pair_data[pk].get(t) for pk in available_pairs if t in pair_data[pk]]
        # Filter out None and NaN values
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if vals:
            avg_x.append(t)
            avg_y.append(sum(vals) / len(vals))
    
    if avg_x:
        traces.append({
            'x': avg_x,
            'y': avg_y,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Average',
            'line': {'color': '#64748b', 'width': 3, 'dash': 'dash'},
            'marker': {'size': 10, 'symbol': 'star'}
        })
    
    layout = {
        'title': {'text': 'Path Rank Correlation Across Thresholds', 'font': {'size': 16}},
        'xaxis': {'title': 'Threshold', 'type': 'category'},
        'yaxis': {'title': 'Path Rank Correlation', 'range': [-1, 1]},
        'legend': {'orientation': 'h', 'y': -0.2, 'x': 0.5, 'xanchor': 'center'},
        'margin': {'t': 50, 'b': 80, 'l': 60, 'r': 30},
        'hovermode': 'x unified',
        'shapes': [{
            'type': 'line',
            'x0': 0, 'x1': 1, 'xref': 'paper',
            'y0': 0, 'y1': 0,
            'line': {'color': 'gray', 'width': 1, 'dash': 'dot'}
        }]
    }
    
    plotly_data = json.dumps({'data': traces, 'layout': layout})
    
    return f'''
        <div class="card" style="margin-top: 30px;">
            <h3>Path Rank Correlation Trend Across Thresholds</h3>
            <p style="color: var(--secondary-color); font-size: 0.85rem; margin-bottom: 15px;">
                Compares the ranking of multi-hop paths by min_weight (using union of paths). Higher values indicate more similar path importance rankings. The dashed line shows the average across all pairs.
            </p>
            <div id="path_rank_trend_plot" style="width: 100%; height: 1050px;"></div>
            <script>
                (function() {{
                    try {{
                        var plotData = {plotly_data};
                        Plotly.newPlot('path_rank_trend_plot', plotData.data, plotData.layout, {{responsive: true}});
                    }} catch(e) {{
                        console.error("Failed to render Path Rank Correlation trend plot:", e);
                        document.getElementById('path_rank_trend_plot').innerHTML = '<p style="text-align:center;color:#999;">Chart failed to load</p>';
                    }}
                }})();
            </script>
        </div>
    '''


def _generate_stats_table(analyzer, dataset_names: List[str], threshold: int,
                           nickname_map: Dict[str, str]) -> str:
    """Generate statistics tables with threshold indication."""
    aligned = analyzer.get_aligned_data(threshold)
    
    if aligned.empty:
        return '<p>No data available.</p>'
    
    available = [d for d in dataset_names if d in aligned.columns]
    if not available:
        return '<p>No datasets available.</p>'
    
    html = [f'''<div class="card"><h3>Per-Dataset Statistics <span style="color: var(--primary-color);">(Threshold = {threshold})</span></h3>
        <table><thead><tr><th>Dataset</th><th>Edge Count</th><th>Total Weight</th><th>Mean Weight</th><th>Max Weight</th></tr></thead><tbody>''']
    
    for d in available:
        col = aligned[d]
        ec = int((col > 0).sum())
        tw = int(col.sum())
        mw = float(col[col > 0].mean()) if (col > 0).any() else 0
        mx = int(col.max()) if len(col) > 0 else 0
        html.append(f'<tr><td>{nickname_map[d]}</td><td>{ec}</td><td>{tw}</td><td>{mw:.2f}</td><td>{mx}</td></tr>')
    
    html.append('</tbody></table></div>')
    
    # Pairwise similarities
    html.append(f'''<div class="card"><h3>Pairwise Similarities <span style="color: var(--primary-color);">(Threshold = {threshold})</span></h3>
        <table><thead><tr><th>Dataset 1</th><th>Dataset 2</th><th>Jaccard</th><th>Rank Corr</th><th>Common</th></tr></thead><tbody>''')
    
    from scipy.stats import spearmanr
    for i, d1 in enumerate(available):
        for d2 in available[i+1:]:
            c1, c2 = aligned[d1], aligned[d2]
            s1, s2 = set(aligned.index[c1 > 0]), set(aligned.index[c2 > 0])
            inter, union = len(s1 & s2), len(s1 | s2)
            jac = inter / union if union > 0 else 0
            # Use rank correlation instead of cosine
            w1, w2 = c1.values, c2.values
            # Only compute rank correlation where both have values
            mask = (w1 > 0) & (w2 > 0)
            if mask.sum() >= 2:
                rank_corr, _ = spearmanr(w1[mask], w2[mask])
                rank_corr = rank_corr if not np.isnan(rank_corr) else 0
            else:
                rank_corr = 0
            html.append(f'<tr><td>{nickname_map[d1]}</td><td>{nickname_map[d2]}</td><td>{jac:.3f}</td><td>{rank_corr:.3f}</td><td>{inter}</td></tr>')
    
    html.append('</tbody></table></div>')
    return ''.join(html)


def _generate_footer() -> str:
    """Generate footer."""
    return """
        <button class="print-btn" onclick="window.print()">🖨️ Print Report</button>
        <script>
            // Final initialization: ensure visible network is properly rendered
            // This runs after all network scripts have executed
            // IMPORTANT: Only redraw networks in visible containers
            // vis.js cannot calculate dimensions for hidden containers (display:none)
            window.addEventListener('load', function() {
                setTimeout(function() {
                    // Only redraw the active/visible network on load
                    // The first threshold network is visible by default
                    if (window.allThresholds && window.allThresholds.length > 0) {
                        const firstThreshold = window.allThresholds[0];
                        if (window.allNetworks && window.allNetworks[firstThreshold]) {
                            const netData = window.allNetworks[firstThreshold];
                            if (netData && netData.network) {
                                netData.network.redraw();
                                netData.network.fit({ animation: false });
                            }
                        }
                    }
                }, 500);
            });
        </script>
    </div>
</body>
</html>
"""