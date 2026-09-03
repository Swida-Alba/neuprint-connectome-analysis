"""
ComparisonAnalyzer - Main orchestrator for cross-dataset comparison.

This module provides the primary interface for running path analyses across
multiple datasets and comparing results.

Optimized Workflow:
    1. Create ComparisonParameters with all settings (datasets, neurons, thresholds)
    2. Create ComparisonAnalyzer with parameters
    3. Run comparison and generate reports

Example:
    >>> params = ComparisonParameters(
    ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    ...     source_neurons=['MBON14.*_R'],
    ...     target_neurons=['KCg-d.*_R', 'PPL101.*_R'],
    ...     max_interlayer=2,
    ...     thresholds=[1, 3, 5, 10, 20],
    ... )
    >>> analyzer = ComparisonAnalyzer(params)
    >>> results = analyzer.run_comparison()
"""

import os
import json
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Optional, Any, Union, Tuple, Set
import pandas as pd
import numpy as np
from tqdm import tqdm

from .dataset_config import DatasetConfig
from .comparison_parameters import ComparisonParameters
from .label_mapper import LabelMapper
from .data_loader import DataLoader
from .metrics import ComparisonMetrics
from .interactive_heatmap import generate_interactive_heatmap


def _escape_cypher_string_fallback(value):
    """Inline escape fallback (only used when src.utils.api_utils is unavailable)."""
    if not isinstance(value, str):
        return str(value)
    return value.replace('\\', '\\\\').replace("'", "\\'")


def _wildcard_pattern_to_regex(pattern: str) -> str:
    """Convert a user wildcard pattern to a regex pattern.

    Every ``*`` means "any string".  Pre-existing ``.*`` sequences are already
    that wildcard and must not be mangled into ``..`` (the naive chained
    ``replace('*', '.*')`` turned ``X.*`` into ``X..*``, which no longer
    matched the bare ``X``).
    """
    placeholder = '\x00'
    return pattern.replace('.*', placeholder).replace('*', '.*').replace(placeholder, '.*')


class ComparisonAnalyzer:
    """
    Main orchestrator for cross-dataset comparison analysis.
    
    Handles:
    - Running path analysis on multiple datasets
    - Coordinating label mapping
    - Computing comparison metrics
    - Generating reports
    
    Example:
        >>> # Simple workflow - ComparisonParameters first
        >>> params = ComparisonParameters(
        ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
        ...     source_neurons=['MBON14.*_R'],
        ...     target_neurons=['KCg-d.*_R'],
        ...     max_interlayer=2,
        ...     thresholds=[1, 3, 5, 10, 20],
        ...     output_folder='./comparison_output'
        ... )
        >>> 
        >>> analyzer = ComparisonAnalyzer(params)
        >>> results = analyzer.run_comparison()
        >>> report = analyzer.generate_report()
    """
    
    # Normalization map for neurotransmitter names
    # Key: uppercase variation, Value: canonical name (lowercase)
    _NT_NORMALIZATION_MAP = {
        'ACH': 'acetylcholine',
        'ACETYLCHOLINE': 'acetylcholine',
        'GABA': 'gaba',
        'GLUT': 'glutamate',
        'GLUTAMATE': 'glutamate',
        'DA': 'dopamine',
        'DOPAMINE': 'dopamine', 
        'SER': 'serotonin',
        'SEROTONIN': 'serotonin',
        'OCT': 'octopamine',
        'OCTOPAMINE': 'octopamine',
        'HIS': 'histamine',
        'HISTAMINE': 'histamine',
        'UNKNOWN': 'unknown',
        'NONE': 'unknown',
        'NO_CONS': 'unknown'
    }

    def __init__(
        self,
        parameters: ComparisonParameters,
        label_mapper: Optional[LabelMapper] = None,
        verbose: bool = True
    ):
        """
        Initialize ComparisonAnalyzer.
        
        Args:
            parameters: ComparisonParameters with all analysis settings
            label_mapper: Optional LabelMapper for cross-dataset standardization
            verbose: Print progress messages
        """
        self.parameters = parameters
        self.label_mapper = label_mapper
        self.verbose = verbose
        self.separate_hemispheres = parameters.separate_hemispheres
        
        # Track if a mapper was explicitly provided
        has_user_mapper = self.label_mapper is not None or self.parameters.overall_label_mapper is not None
        
        # Extract and merge LabelMappers from parameters
        if self.label_mapper is None:
            # Initialize unified mapper
            self.label_mapper = LabelMapper()
            
            # Merge overall_label_mapper from parameters
            # Note: ComparisonParameters.__post_init__ ensures that source/target mappers
            # are already merged into overall_label_mapper if they existed.
            if self.parameters.overall_label_mapper:
                 self.label_mapper.merge(self.parameters.overall_label_mapper)

        # Initialize components
        self.metrics = ComparisonMetrics()
        
        # Storage for results
        self.raw_results: Dict[str, Dict[int, pd.DataFrame]] = {}
        self.aligned_results: Dict[int, pd.DataFrame] = {}
        self.comparison_report: Optional[Dict] = None
        
        # Cache for expensive calculations (reused across export/visualizations)
        self._similarity_cache: Dict[int, pd.DataFrame] = {}  # threshold -> similarities
        self._hemisphere_symmetry_cache: Dict[int, Dict[str, Dict]] = {}
        self._network_aligned_cache: Dict[int, pd.DataFrame] = {}
        self._output_base_printed: bool = False  # Track if base dir was printed
        
        # Resolve dataset configurations from strings
        self._dataset_configs: Dict[str, DatasetConfig] = {}
        self._resolve_dataset_configs()
        
        # Validate datasets if LabelMapper is provided
        if has_user_mapper and self.label_mapper:
            # Determine role to validate based on usage in parameters
            role_to_validate = 'both'
            
            # Check if mapper is used for source/target
            # Note: ComparisonParameters moves mapper to _source_mapper/_target_mapper in __post_init__
            is_source_mapper = getattr(self.parameters, '_source_mapper', None) is not None
            is_target_mapper = getattr(self.parameters, '_target_mapper', None) is not None
            
            # Fallback: check if source_neurons/target_neurons ARE mappers
            if not is_source_mapper:
                is_source_mapper = isinstance(self.parameters.source_neurons, LabelMapper)
            if not is_target_mapper:
                is_target_mapper = isinstance(self.parameters.target_neurons, LabelMapper)
            
            # If explicitly passed in init but not in params, assume 'both' (or check params types)
            # If in params, restrict validation to relevant role
            if is_source_mapper and not is_target_mapper:
                role_to_validate = 'source'
            elif is_target_mapper and not is_source_mapper:
                role_to_validate = 'target'
            
            # Get dataset names from parameters
            dataset_names = [
                ds.dataset if isinstance(ds, DatasetConfig) else ds 
                for ds in self.parameters.datasets
            ]
            self.label_mapper.validate_datasets(dataset_names, role=role_to_validate)
        
        # Setup output directory and data loader
        if parameters.output_folder:
            self.data_loader = DataLoader(parameters.full_output_path)
            self.data_loader.ensure_directories()
        else:
            self.data_loader = None
    
    def _resolve_dataset_configs(self):
        """
        Resolve dataset strings to DatasetConfig objects.
        
        Handles both string identifiers and existing DatasetConfig objects.
        For NeuPrint datasets, shares the client to avoid repeated login.
        """
        for ds in self.parameters.datasets:
            if isinstance(ds, str):
                # Create DatasetConfig from string
                # For NeuPrint datasets, we'll set the client when needed
                config = DatasetConfig.from_string(ds)
                self._dataset_configs[ds] = config
            elif isinstance(ds, DatasetConfig):
                # Use existing config
                self._dataset_configs[ds.dataset] = ds
            else:
                raise ValueError(f"Dataset must be string or DatasetConfig, got {type(ds)}")
    
    def _get_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """Get DatasetConfig for a dataset name."""
        if dataset_name in self._dataset_configs:
            return self._dataset_configs[dataset_name]
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _log(self, message: str, level: str = 'info'):
        """Print message if verbose mode enabled.
        
        Args:
            message: Message to print
            level: Log level ('info', 'warn', 'debug'). Debug messages only shown with extra verbosity.
        """
        if not self.verbose:
            return
        # Skip repetitive debug messages
        if level == 'debug':
            return
        prefix = "⚠️ " if level == 'warn' else ""
        # Use tqdm.write to avoid interfering with progress bars
        tqdm.write(f"[Comparison] {prefix}{message}")

    def _progress(self, step: int, total: int, label: str = ""):
        """Emit a structured step-progress event consumed by the web UI.

        The line ``[DROCAT][progress] <step>/<total> <label>`` drives the
        determinate progress bar + step label in the results panel; it is a
        control event, never shown in the execution log.  Uses ``tqdm.write``
        like :meth:`_log` so it never interleaves with an active bar.
        """
        if self.verbose:
            tqdm.write(
                f"[DROCAT][progress] {int(step)}/{int(total)} {label}".rstrip()
            )
    
    def _log_file(self, filepath: str, description: str = "Saved"):
        """Log file save with relative path (prints base dir only once).
        
        Args:
            filepath: Full file path
            description: Action description (default: "Saved")
        """
        if not self.verbose:
            return
        
        base_dir = self.parameters.full_output_path if self.parameters else None
        
        # Print base directory once at the start
        if base_dir and not self._output_base_printed:
            tqdm.write(f"[Comparison] Output directory: {base_dir}")
            self._output_base_printed = True
        
        # Show only relative path from base dir
        if base_dir and filepath.startswith(base_dir):
            rel_path = os.path.relpath(filepath, base_dir)
            tqdm.write(f"[Comparison] {description}: {rel_path}")
        else:
            tqdm.write(f"[Comparison] {description}: {filepath}")
    
    def _save_csv(self, df: pd.DataFrame, filepath: str, index: bool = False):
        """Save DataFrame to CSV with UTF-8 encoding for cross-platform compatibility.
        
        Uses polars for faster writes when available, falls back to pandas.
        Ensures Windows/macOS/Linux compatibility with explicit UTF-8 encoding.
        
        Args:
            df: DataFrame to save
            filepath: Output file path
            index: Whether to include row index (default: False)
        """
        if df is None or (hasattr(df, 'empty') and df.empty):
            return
        
        try:
            import polars as pl
            # Reset index if needed to avoid conversion issues
            if index:
                df_to_save = df.reset_index()
            else:
                df_to_save = df.reset_index(drop=True) if df.index.name or not df.index.equals(pd.RangeIndex(len(df))) else df
            
            # Convert pandas to polars and write - faster for large files
            pl_df = pl.from_pandas(df_to_save)
            pl_df.write_csv(filepath)
        except ImportError:
            # Fallback to pandas with explicit UTF-8 encoding
            df.to_csv(filepath, index=index, encoding='utf-8')
        except Exception:
            # Fallback for any polars conversion issues
            df.to_csv(filepath, index=index, encoding='utf-8')
    
    def _read_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Read CSV with polars (faster) and convert to pandas.
        
        Uses polars for faster reads when available, falls back to pandas.
        Ensures cross-platform compatibility.
        
        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments passed to pandas read_csv
            
        Returns:
            pandas DataFrame
        """
        try:
            import polars as pl
            # Use polars for faster reading ONLY when no pandas-specific kwargs
            # were passed. Silently dropping dtype=/index_col=/header= here
            # used to turn str bodyId columns into int64 and break joins.
            if kwargs:
                return pd.read_csv(filepath, encoding='utf-8', **kwargs)
            return pl.read_csv(filepath, infer_schema_length=10000).to_pandas()
        except ImportError:
            return pd.read_csv(filepath, encoding='utf-8', **kwargs)
        except Exception:
            # Fallback for polars issues (schema inference, etc.)
            return pd.read_csv(filepath, encoding='utf-8', **kwargs)
    
    def _collect_result_types(self) -> Set[str]:
        """Collect all neuron types present in comparison results.
        
        This scans raw_results for type names in result DataFrames.
        raw_results structure: {dataset: {threshold: DataFrame}}
        
        Returns:
            Set of all unique type names (strings only).
        """
        all_types = set()
        
        for dataset, thresh_results in self.raw_results.items():
            for threshold, result in thresh_results.items():
                # Handle DataFrame directly (path/edge analysis results)
                if isinstance(result, pd.DataFrame) and not result.empty:
                    # Check common type columns
                    for col in ['type_pre', 'type_post', 'from_type', 'to_type', 
                                'std_label_pre', 'std_label_post']:
                        if col in result.columns:
                            all_types.update(result[col].dropna().unique())
                
                # Handle dict structure if present (legacy format)
                elif isinstance(result, dict):
                    for key in ['type_level', 'edge_level']:
                        df = result.get(key)
                        if df is not None and hasattr(df, 'empty') and not df.empty:
                            for col in ['from', 'to', 'from_type', 'to_type',
                                        'type_pre', 'type_post']:
                                if col in df.columns:
                                    all_types.update(df[col].dropna().unique())
        
        # Filter to only string types
        return {t for t in all_types if isinstance(t, str)}

    def get_hemisphere_symmetry_summaries(self) -> Dict[int, Dict[str, Dict]]:
        """Load hemisphere symmetry summaries for all datasets and thresholds.

        Returns:
            Dict mapping threshold -> {dataset_name: summary_dict}
        """
        if not self.parameters.symmetry_analysis:
            return {}

        dataset_names = self.parameters.get_dataset_names()
        thresholds = self.parameters.thresholds
        summaries: Dict[int, Dict[str, Dict]] = {}

        for threshold in thresholds:
            if threshold in self._hemisphere_symmetry_cache:
                summaries[threshold] = self._hemisphere_symmetry_cache[threshold]
                continue

            threshold_summaries: Dict[str, Dict] = {}
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                summary_path = os.path.join(
                    self.parameters.full_output_path,
                    'dataset_data',
                    safe_name,
                    f'minsyn_{threshold}',
                    'hemisphere_symmetry',
                    'symmetry_summary.json'
                )
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, 'r', encoding='utf-8') as f:
                            threshold_summaries[dataset] = json.load(f)
                    except Exception as e:
                        self._log(f"Warning: Failed to load symmetry summary for {dataset} t={threshold}: {e}", level='warn')

            self._hemisphere_symmetry_cache[threshold] = threshold_summaries
            summaries[threshold] = threshold_summaries

        return summaries
    
    def get_mapped_results(self) -> Dict[str, Dict[int, pd.DataFrame]]:
        """Get raw_results with type mapping applied.
        
        Creates a copy of raw_results where type_pre and type_post columns
        are replaced with canonical (mapped) type names. This is used for
        visualizations and conservation analysis to properly compare types
        across datasets that may use different naming conventions.
        
        Returns:
            Dict mapping dataset -> threshold -> DataFrame with mapped types
        """
        if not self.parameters.auto_type_mapping or not self.parameters._auto_type_mapper:
            return self.raw_results
        
        mapped_results = {}
        
        for dataset, thresh_results in self.raw_results.items():
            mapped_results[dataset] = {}
            
            for threshold, df in thresh_results.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Create a copy to avoid modifying original
                    mapped_df = df.copy()
                    
                    # Map type columns to canonical names
                    if 'type_pre' in mapped_df.columns:
                        mapped_df['type_pre'] = mapped_df['type_pre'].apply(
                            lambda t: self._get_canonical_type(t, dataset) if pd.notna(t) else t
                        )
                    if 'type_post' in mapped_df.columns:
                        mapped_df['type_post'] = mapped_df['type_post'].apply(
                            lambda t: self._get_canonical_type(t, dataset) if pd.notna(t) else t
                        )
                    
                    mapped_results[dataset][threshold] = mapped_df
                else:
                    mapped_results[dataset][threshold] = df
        
        return mapped_results
    
    def _get_canonical_type(self, type_name: str, dataset: str) -> str:
        """Get canonical (male-cns) type name for a given type.
        
        If auto_type_mapping is enabled and a mapping exists, returns the 
        canonical (male-cns) name. Otherwise returns the original type name.
        
        Args:
            type_name: Original type name
            dataset: Dataset the type comes from
            
        Returns:
            Canonical type name (male-cns name if mapped, else original)
        """
        if not self.parameters.auto_type_mapping or not self.parameters._auto_type_mapper:
            return type_name
        return self.parameters._auto_type_mapper.get_canonical_type(type_name, dataset)
    
    def _get_display_type(self, canonical_name: str) -> str:
        """Get display name for a canonical type showing all dataset variants.
        
        If auto_type_mapping is enabled, returns format like 'MeVPaMe1(MTe46)'.
        Otherwise returns the canonical name unchanged.
        
        Args:
            canonical_name: Canonical (male-cns) type name
            
        Returns:
            Display name with variants in parentheses
        """
        if not self.parameters.auto_type_mapping or not self.parameters._auto_type_mapper:
            return canonical_name
        # Pass all datasets being compared to get full mapping info
        datasets = self.parameters.get_dataset_names()
        return self.parameters._auto_type_mapper.get_display_name(canonical_name, datasets)
    
    def _build_path_key_with_mapping(self, path_nodes: list, dataset: str) -> Tuple[str, str]:
        """Build canonical and display path keys from path nodes.
        
        Args:
            path_nodes: List of type names in the path
            dataset: Dataset the path comes from
            
        Returns:
            Tuple of (canonical_key, display_key) where:
            - canonical_key: Path with canonical names for merging
            - display_key: Path with display names (variants in parentheses)
        """
        # Get canonical names for each node
        canonical_nodes = [self._get_canonical_type(node, dataset) for node in path_nodes]
        canonical_key = ' → '.join(canonical_nodes)
        
        # Get display names for each canonical node
        display_nodes = [self._get_display_type(node) for node in canonical_nodes]
        display_key = ' → '.join(display_nodes)
        
        return canonical_key, display_key

    def _print_intermediate_mapping_summary(self):
        """Print summary of intermediate neuron type mappings.
        
        Shows:
        - Number of types with cross-dataset name differences
        - Count of N-to-1 and 1-to-N mappings
        - Reference to export file for details
        """
        if not self.parameters.auto_type_mapping or not self.parameters._auto_type_mapper:
            return
        
        # Use the helper method to collect all types from results
        str_types = self._collect_result_types()
        
        if not str_types:
            return
        
        # Get mapping summary
        mapper = self.parameters._auto_type_mapper
        dataset_names = self.parameters.get_dataset_names()
        summary = mapper.get_intermediate_mapping_summary(str_types, dataset_names)
        
        # Print summary
        output_path = self.parameters.full_output_path
        self._log("Auto type mapping summary for intermediate neurons:")
        self._log(f"  • {summary['total_types']} neuron types in comparison results")
        if summary['mapped_count'] > 0:
            self._log(f"  • {summary['mapped_count']} types have cross-dataset name differences")
        if summary['n_to_1_count'] > 0:
            self._log(f"  ⚠️ {summary['n_to_1_count']} N-to-1 type mappings (check aggregation)")
        if summary['one_to_n_count'] > 0:
            self._log(f"  ⚠️ {summary['one_to_n_count']} 1-to-N type mappings (check aggregation)")
        
        if summary['mapped_count'] > 0 or summary['n_to_1_count'] > 0 or summary['one_to_n_count'] > 0:
            self._log(f"  → Check auto_type_mapping.csv and auto_type_mapping_conflicts.csv in output folder")
            self._write_user_warning_notes_for_mapping(mapper, str_types, dataset_names)

    def _write_user_warning_notes_for_mapping(self, mapper, type_names, dataset_names):
        """Write auto-type-mapping caveats to user_warning_notes.txt.

        The run guide renders this file, so expanded (renamed) type
        mappings and N-to-1 / 1-to-N conflicts surface next to the results
        with a recommendation to double check them.
        """
        try:
            notes = mapper.build_user_warning_notes(type_names, dataset_names)
        except Exception as e:
            self._log(f"  Warning: could not build auto type mapping notes: {e}")
            return None
        if not notes:
            return None
        try:
            os.makedirs(self.parameters.full_output_path, exist_ok=True)
            notes_path = os.path.join(self.parameters.full_output_path, 'user_warning_notes.txt')
            with open(notes_path, 'w', encoding='utf-8') as f:
                f.write('User warning notes\n')
                f.write('==================\n\n')
                f.write('Auto type mapping changed how some neuron type names '
                        'were matched across datasets:\n\n')
                for note in notes:
                    f.write(f'- {note}\n')
            self._log(f"  ⚠️ Wrote {notes_path} - please double check the automatic mappings")
            return notes_path
        except Exception as e:
            self._log(f"  Warning: could not write user_warning_notes.txt: {e}")
            return None

    def _generate_mode_specific_note(self) -> str:
        """Generate HTML note specific to the comparison mode used."""
        mode = getattr(self.parameters, 'comparison_mode', 'path')
        
        if mode == 'edge':
            return '''
            <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px 16px; margin-top: 12px; border-radius: 0 6px 6px 0;">
                <strong>⚠️ Edge-Based Comparison Mode:</strong> This analysis uses edge-based filtering where each synapse connection 
                is evaluated independently by its weight. Edges are aggregated at the neuron type level. This mode avoids the 
                path-filtering artifacts where strong edges might appear absent due to weak intermediate edges on paths. 
                <br><br>
                <strong>Caveat 1 - Dead-ends:</strong> Edge-based comparison may include <strong>dead-end connections</strong> in the network—edges 
                that are strongly connected but don't contribute to any complete source→target path. These appear in individual 
                datasets but may not be functionally relevant to the circuit. Check the path presence matrix to identify 
                edges that form complete paths versus isolated strong connections.
                <br><br>
                <strong>Caveat 2 - Weight Mismatch:</strong> Edge weights in the <em>Edge Presence Matrix</em> represent the <strong>total 
                synapse count between all neuron pairs</strong> of the source and target types that meet the threshold. However, edge weights 
                shown in the <em>Path Presence Matrix</em> (hop weights) represent only synapses from <strong>neurons actually participating 
                in paths</strong>. The same A→B edge may show different weights: e.g., edge matrix shows 500 synapses (all type-A to type-B 
                connections), while path matrix shows 120 (only neurons on paths from source to target). This is expected behavior—path 
                weights are subsets of edge weights.
            </div>'''
        else:
            return '''
            <div style="background: #fee2e2; border-left: 4px solid #ef4444; padding: 12px 16px; margin-top: 12px; border-radius: 0 6px 6px 0;">
                <strong>⚠️ Path-Based Filtering Caveat:</strong> This analysis uses path-based filtering where edges are discovered 
                through paths from source to target neurons. <strong>Strong edges may appear absent</strong> (marked ❌) if they only 
                exist on paths with weaker intermediate edges that fall below the threshold. An edge marked as "non-existent" in one 
                dataset may actually exist but was filtered due to path context, not edge absence. Compare results across threshold 
                levels to identify such cases—if an edge appears at lower thresholds but disappears at higher ones, it may indicate 
                path-filtering artifacts rather than true biological differences.
            </div>'''
    
    # =========================================================================
    # Path Analysis
    # =========================================================================
    
    def run_path_analysis(
        self,
        dataset_name: str,
        threshold: int,
        verbose_mode: str = 'simple'
    ) -> pd.DataFrame:
        """
        Run path analysis for a single dataset at a specific threshold.
        
        Args:
            dataset_name: Dataset identifier string
            threshold: Weight threshold for path finding
            verbose_mode: Verbosity level for the path run ('simple', 'full', 'silent')
            
        Returns:
            DataFrame with path analysis results
        """
        # Import here to avoid circular imports
        # Use absolute import since src is on sys.path
        from coana import FindNeuronConnection
        
        # Only log when verbose (not 'silent')
        if verbose_mode != 'silent':
            self._log(f"Running analysis: \033[94m{dataset_name} @ threshold={threshold}\033[0m")
        
        # Get dataset config
        config = self._get_dataset_config(dataset_name)
        
        # Get source/target neurons from ComparisonParameters
        source_neurons = self.parameters.get_source_neurons_for_dataset(dataset_name)
        target_neurons = self.parameters.get_target_neurons_for_dataset(dataset_name)
        
        # Get max_interlayer from ComparisonParameters (shared across all datasets)
        max_interlayer = self.parameters.max_interlayer
        
        # Set up output path to redirect FNC output into comparison folder structure
        # Output goes to: {comparison_output}/dataset_data/{dataset}/minsyn_{threshold}/
        safe_dataset_name = self.parameters._sanitize_name(dataset_name)
        fnc_output_path = self.parameters.get_dataset_output_path(dataset_name, threshold)
        
        # Determine custom names based on labels
        # If single label provided, use it as custom name.
        # If multiple labels provided, leave empty to allow auto-naming (or group naming).
        custom_source_name = ''
        if self.parameters.source_labels and len(self.parameters.source_labels) == 1:
            custom_source_name = self.parameters.source_labels[0]
            
        custom_target_name = ''
        if self.parameters.target_labels and len(self.parameters.target_labels) == 1:
            custom_target_name = self.parameters.target_labels[0]

        # Let FindNeuronConnection handle client creation
        # It will auto-detect client_type from dataset name and create/reuse clients as needed
        # - If dataset contains 'flywire' or 'fafb' -> uses local data
        # - Otherwise -> uses NeuPrint (creates client using dataset name and token from env var)
        # Determine if force_API_fetching should be applied (only for FAFB/FlyWire datasets)
        is_fafb = 'flywire' in dataset_name.lower() or 'fafb' in dataset_name.lower()
        use_force_api = self.parameters.force_API_fetching if is_fafb else False
        
        fnc = FindNeuronConnection(
            sourceNeurons=source_neurons,
            targetNeurons=target_neurons,
            custom_source_name=custom_source_name,
            custom_target_name=custom_target_name,
            max_interlayer=max_interlayer,
            min_synapse_num=threshold,
            min_traversal_probability=0,  # Use 0 to match FindPath.py - default 0.001 can miss weak but important edges
            min_ratio=0,
            dataset=dataset_name,
            # Redirect output to comparison folder structure
            saveas=fnc_output_path,  # Absolute path - overrides data_folder
            verbose_mode=verbose_mode,  # Verbosity level for FindAllPath
            skip_bodyId=self.parameters.skip_bodyId,  # Skip bodyId-level processing if requested
            label_mapper=self.label_mapper,  # Pass label mapper for standardization
            pathfinding=self.parameters.pathfinding,  # Pass pathfinding algorithm
            graph_edge_limit_bodyid=self.parameters.graph_edge_limit_bodyid,  # bodyId edge limit (deep searches)
            edgeN_limit=self.parameters.edgeN_limit,  # Visualization Edge Limit
            search_columns=self.parameters.search_columns,  # Column scope for neuron name resolution
            force_API_fetching=use_force_api,  # Use CAVE API for FAFB if enabled
            cache_only=self.parameters.cache_only,  # Use cache-only mode if enabled
            separate_hemispheres=self.parameters.separate_hemispheres,
            symmetry_analysis=self.parameters.symmetry_analysis,
            keep_only_hemisphere_conserved_connections=self.parameters.keep_only_hemisphere_conserved_connections,
        )
        
        # Initialize and run analysis
        # Use FindAllPath()/FindShortestPath() as specified in TODO_comparison.md:
        # "DO NOT use the FindDirectConnection() function, because the FindAllPath() 
        # function can already include direct connections as 1-hop paths"
        # (FindShortestPath likewise includes direct connections as 1-hop
        # shortest paths).
        fnc.InitializeNeuronInfo()
        source_df = getattr(fnc, "source_df", None)
        target_df = getattr(fnc, "target_df", None)
        print(
            f"[DROCAT][neuron-match] source={len(source_df) if source_df is not None else 0} "
            f"target={len(target_df) if target_df is not None else 0}",
            flush=True,
        )
        if self.parameters.path_mode == 'shortest':
            fnc.FindShortestPath(find_reciprocal=self.parameters.find_reciprocal)
        else:
            fnc.FindAllPath(find_reciprocal=self.parameters.find_reciprocal)
        
        # Get results - FindAllPath saves both path data and connection data
        # For comparison metrics, we need the connection data (edge-level) format:
        # - data_details/connection_info_bodyId.csv has bodyId_pre, bodyId_post, weight, etc.
        # - This is the correct format for comparison metrics
        conn_df = pd.DataFrame()
        
        if hasattr(fnc, 'allpath_folder') and fnc.allpath_folder:
            # Try to load connection data (edge-level format for metrics)
            conn_file = os.path.join(
                fnc.allpath_folder, 'data_details', 'connection_info_bodyId.csv'
            )
            
            if os.path.exists(conn_file):
                try:
                    # Use Polars for faster CSV reading
                    import polars as pl
                    conn_df = pl.read_csv(conn_file, infer_schema_length=10000).to_pandas()
                    self._log(f"Loaded {len(conn_df)} connections from connection_info_bodyId.csv")
                except Exception as e:
                    self._log(f"Warning: Could not read connection file: {e}")
            else:
                # Fallback: try connection_type.csv (type-level)
                conn_type_file = os.path.join(
                    fnc.allpath_folder, 'data_details', 'connection_type.csv'
                )
                if os.path.exists(conn_type_file):
                    try:
                        # Use Polars for faster CSV reading
                        import polars as pl
                        conn_df = pl.read_csv(conn_type_file, infer_schema_length=10000).to_pandas()
                        self._log(f"Loaded {len(conn_df)} connections from connection_type.csv")
                    except Exception as e:
                        self._log(f"Warning: Could not read connection type file: {e}")
        
        # Add dataset identifier
        if not conn_df.empty:
            conn_df = conn_df.copy()
            conn_df['dataset'] = dataset_name
            conn_df['threshold'] = threshold
            
            # Apply label mapping if available
            if self.label_mapper and not self.label_mapper.is_empty:
                self._log(f"Applying label mapping to {dataset_name} results")
                conn_df = self.label_mapper.apply_to_dataframe(conn_df, dataset_name)
                
                # Overwrite original types with standardized labels
                # This ensures merging in downstream analysis and visualizations
                if 'std_label_pre' in conn_df.columns:
                    # Only overwrite if label is not empty
                    mask = conn_df['std_label_pre'] != ''
                    conn_df.loc[mask, 'type_pre'] = conn_df.loc[mask, 'std_label_pre']
                    
                if 'std_label_post' in conn_df.columns:
                    mask = conn_df['std_label_post'] != ''
                    conn_df.loc[mask, 'type_post'] = conn_df.loc[mask, 'std_label_post']
        
        return conn_df
    
    def run_edge_analysis(
        self,
        dataset_name: str,
        threshold: int,
        base_edges: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Run edge-based analysis for a single dataset at a specific threshold.
        
        Unlike path-based analysis, this queries edges directly between neuron types
        without path context. Edges are filtered by weight threshold independently.
        
        Args:
            dataset_name: Dataset identifier string
            threshold: Weight threshold for edge filtering
            base_edges: Optional pre-fetched edges at lowest threshold (for efficiency)
            
        Returns:
            DataFrame with edge analysis results
        """
        self._log(f"Running edge analysis: \033[94m{dataset_name} @ threshold={threshold}\033[0m")
        
        # Get source/target neurons
        source_neurons = self.parameters.get_source_neurons_for_dataset(dataset_name)
        target_neurons = self.parameters.get_target_neurons_for_dataset(dataset_name)
        
        # If base_edges provided, filter from it
        if base_edges is not None and not base_edges.empty:
            filtered = base_edges[base_edges['weight'] >= threshold].copy()
            filtered['threshold'] = threshold
            self._log(f"Filtered {len(filtered)} edges from base at threshold={threshold}", 'debug')
            return filtered
        
        # Otherwise, query edges directly
        conn_df = self._query_edges_for_dataset(
            dataset_name, source_neurons, target_neurons, threshold
        )
        
        if not conn_df.empty:
            conn_df = conn_df.copy()
            conn_df['dataset'] = dataset_name
            conn_df['threshold'] = threshold
            
            # Apply label mapping if available
            if self.label_mapper and not self.label_mapper.is_empty:
                self._log(f"Applying label mapping to {dataset_name} edge results")
                conn_df = self.label_mapper.apply_to_dataframe(conn_df, dataset_name)
                
                # Overwrite original types with standardized labels
                if 'std_label_pre' in conn_df.columns:
                    mask = conn_df['std_label_pre'] != ''
                    conn_df.loc[mask, 'type_pre'] = conn_df.loc[mask, 'std_label_pre']
                    
                if 'std_label_post' in conn_df.columns:
                    mask = conn_df['std_label_post'] != ''
                    conn_df.loc[mask, 'type_post'] = conn_df.loc[mask, 'std_label_post']
        
        return conn_df
    
    def _query_edges_for_dataset(
        self,
        dataset_name: str,
        source_neurons: List,
        target_neurons: List,
        min_weight: int = 1
    ) -> pd.DataFrame:
        """
        Query all edges between source and target neuron types.
        
        This queries edges directly without path context, capturing all connections
        between relevant neuron types regardless of path existence.
        
        Args:
            dataset_name: Dataset identifier
            source_neurons: List of source neuron types/patterns
            target_neurons: List of target neuron types/patterns
            min_weight: Minimum edge weight
            
        Returns:
            DataFrame with columns: bodyId_pre, bodyId_post, type_pre, type_post, weight
        """
        # Check if dataset is local (FlyWire/FAFB) or NeuPrint
        is_local = 'flywire' in dataset_name.lower() or 'fafb' in dataset_name.lower() or 'banc' in dataset_name.lower()
        
        if is_local:
            return self._query_edges_local(dataset_name, source_neurons, target_neurons, min_weight)
        else:
            return self._query_edges_neuprint(dataset_name, source_neurons, target_neurons, min_weight)
    
    def _query_edges_neuprint(
        self,
        dataset_name: str,
        source_neurons: List,
        target_neurons: List,
        min_weight: int = 1
    ) -> pd.DataFrame:
        """Query edges from NeuPrint database."""
        try:
            from neuprint import Client
            
            token = self.parameters.resolve_token()
            client = Client('neuprint.janelia.org', dataset=dataset_name, token=token)
            
            # Import API utilities for Cypher escaping
            try:
                from src.utils.api_utils import escape_cypher_string
            except ImportError:
                escape_cypher_string = _escape_cypher_string_fallback
            
            # Build type patterns for Cypher query
            # Handle regex patterns (convert .* to Cypher regex)
            def format_types_for_cypher(types_list):
                formatted = []
                for t in types_list:
                    if isinstance(t, str):
                        escaped_t = escape_cypher_string(t)
                        if '.*' in t or '*' in t:
                            # Convert to Cypher regex pattern
                            pattern = _wildcard_pattern_to_regex(escaped_t)
                            formatted.append(f"a.type =~ '{pattern}'")
                        else:
                            formatted.append(f"a.type = '{escaped_t}'")
                return formatted
            
            # Build source and target type conditions
            source_conditions = format_types_for_cypher(source_neurons)
            target_conditions = format_types_for_cypher(target_neurons)
            
            # Also include intermediate types (neurons that connect source to target)
            # We want edges where:
            # 1. pre is source type and post is any type (outgoing from sources)
            # 2. pre is any type and post is target type (incoming to targets)
            # 3. edges between any neurons that could be intermediates
            
            # For simplicity, query edges involving source or target types
            source_cond_str = ' OR '.join([c.replace('a.type', 'pre.type') for c in source_conditions]) if source_conditions else 'false'
            target_cond_str = ' OR '.join([c.replace('a.type', 'post.type') for c in target_conditions]) if target_conditions else 'false'
            
            # Query edges where pre is source-like OR post is target-like
            # This captures the relevant subgraph
            query = f"""
            MATCH (pre:Neuron)-[c:ConnectsTo]->(post:Neuron)
            WHERE c.weight >= {min_weight}
            AND (({source_cond_str}) OR ({target_cond_str}))
            AND pre.type IS NOT NULL AND post.type IS NOT NULL
            RETURN pre.bodyId AS bodyId_pre, pre.type AS type_pre,
                   post.bodyId AS bodyId_post, post.type AS type_post,
                   c.weight AS weight
            """
            
            with tqdm(total=1, desc=f"  ⏳ Querying NeuPrint for {dataset_name}",
                      bar_format='{desc}...', leave=False) as pbar:
                result = client.fetch_custom(query)
                pbar.update(1)
            
            if not result.empty:
                self._log(f"Queried {len(result)} edges from NeuPrint for {dataset_name}")
            
            return result
            
        except Exception as e:
            self._log(f"Warning: Failed to query NeuPrint edges for {dataset_name}: {e}")
            return pd.DataFrame()
    
    def _query_edges_local(
        self,
        dataset_name: str,
        source_neurons: List,
        target_neurons: List,
        min_weight: int = 1
    ) -> pd.DataFrame:
        """Query edges from local dataset files (FlyWire/FAFB/BANC)."""
        import re
        
        # Load local connection data
        safe_name = self.parameters._sanitize_name(dataset_name)
        datasets_folder = self._get_datasets_folder()
        
        # Try different file patterns for connections
        conn_files = [
            os.path.join(datasets_folder, safe_name, f'{safe_name}_merged_connections.parquet'),
            os.path.join(datasets_folder, safe_name, f'{safe_name}_merged_connections.csv'),
            os.path.join(datasets_folder, safe_name, f'{safe_name}_connections.parquet'),
            os.path.join(datasets_folder, safe_name, f'{safe_name}_connections.csv'),
            os.path.join(datasets_folder, safe_name, 'connections.parquet'),
            os.path.join(datasets_folder, safe_name, 'connections.csv'),
        ]
        
        conn_df = None
        for conn_file in conn_files:
            if os.path.exists(conn_file):
                try:
                    file_size_mb = os.path.getsize(conn_file) / (1024 * 1024)
                    with tqdm(total=1, desc=f"  ⏳ Loading connections ({file_size_mb:.1f} MB)", 
                              bar_format='{desc}', leave=False) as pbar:
                        if conn_file.endswith('.parquet'):
                            conn_df = pd.read_parquet(conn_file)
                        else:
                            conn_df = self._read_csv(conn_file)
                        pbar.update(1)
                    self._log(f"Loaded connections from {conn_file}")
                    break
                except Exception as e:
                    self._log(f"Warning: Could not load {conn_file}: {e}")
        
        if conn_df is None or conn_df.empty:
            self._log(f"Warning: No connection data found for {dataset_name}")
            return pd.DataFrame()
        
        # Standardize column names
        col_mapping = {
            'pre_pt_root_id': 'bodyId_pre',
            'post_pt_root_id': 'bodyId_post',
            'pre_type': 'type_pre',
            'post_type': 'type_post',
            'syn_count': 'weight',
            'neuropil': 'roi'
        }
        conn_df = conn_df.rename(columns={k: v for k, v in col_mapping.items() if k in conn_df.columns})
        
        # Filter by weight
        if 'weight' in conn_df.columns:
            conn_df = conn_df[conn_df['weight'] >= min_weight]
        
        # If type columns don't exist, join with neuron info to get them
        if 'type_pre' not in conn_df.columns or 'type_post' not in conn_df.columns:
            # Load neuron info file
            neuron_files = [
                os.path.join(datasets_folder, safe_name, f'{safe_name}_allneurons_neuron_df.parquet'),
                os.path.join(datasets_folder, safe_name, f'{safe_name}_allneurons_neuron_df.csv'),
                os.path.join(datasets_folder, safe_name, f'{safe_name}_neurons.parquet'),
                os.path.join(datasets_folder, safe_name, f'{safe_name}_neurons.csv'),
            ]
            
            neuron_df = None
            for neuron_file in neuron_files:
                if os.path.exists(neuron_file):
                    try:
                        if neuron_file.endswith('.parquet'):
                            neuron_df = pd.read_parquet(neuron_file)
                        else:
                            neuron_df = self._read_csv(neuron_file)
                        self._log(f"Loaded neuron info from {neuron_file}")
                        break
                    except Exception as e:
                        self._log(f"Warning: Could not load {neuron_file}: {e}")
            
            if neuron_df is not None and not neuron_df.empty:
                # Identify bodyId and type columns in neuron_df
                bodyid_col = None
                type_col = None
                for col in ['bodyId', 'root_id', 'pt_root_id', 'segment_id']:
                    if col in neuron_df.columns:
                        bodyid_col = col
                        break
                for col in ['type', 'cell_type', 'hemibrain_type']:
                    if col in neuron_df.columns:
                        type_col = col
                        break
                
                if bodyid_col and type_col:
                    # Create mapping dict for efficiency
                    type_map = dict(zip(neuron_df[bodyid_col], neuron_df[type_col]))
                    
                    # Map types to connections with progress bar
                    with tqdm(total=2, desc="  ⏳ Mapping bodyId → type", 
                              bar_format='{desc}: {n}/{total} columns', leave=False) as pbar:
                        if 'type_pre' not in conn_df.columns:
                            conn_df['type_pre'] = conn_df['bodyId_pre'].map(type_map)
                        pbar.update(1)
                        if 'type_post' not in conn_df.columns:
                            conn_df['type_post'] = conn_df['bodyId_post'].map(type_map)
                        pbar.update(1)
                    
                    self._log(f"Mapped types for {len(conn_df)} connections")
        
        # Filter by source/target types if type columns exist
        if 'type_pre' in conn_df.columns and 'type_post' in conn_df.columns:
            def matches_patterns(type_val, patterns):
                if pd.isna(type_val):
                    return False
                for p in patterns:
                    if isinstance(p, str):
                        if '.*' in p or '*' in p:
                            pattern = _wildcard_pattern_to_regex(p)
                            if re.match(pattern, str(type_val)):
                                return True
                        elif str(type_val) == p:
                            return True
                return False
            
            # Keep edges where pre matches source OR post matches target
            with tqdm(total=2, desc="  ⏳ Filtering by source/target types",
                      bar_format='{desc}: {n}/{total} masks', leave=False) as pbar:
                mask_pre = conn_df['type_pre'].apply(lambda x: matches_patterns(x, source_neurons))
                pbar.update(1)
                mask_post = conn_df['type_post'].apply(lambda x: matches_patterns(x, target_neurons))
                pbar.update(1)
            mask = mask_pre | mask_post
            conn_df = conn_df[mask]
        
        self._log(f"Filtered to {len(conn_df)} edges for {dataset_name}")
        return conn_df
    
    def run_all_analyses(self, skip_existing: bool = True) -> Dict[str, Dict[int, pd.DataFrame]]:
        """
        Run analysis for all datasets and thresholds.
        
        Uses comparison_mode from parameters:
        - 'path': Path-based analysis using FindAllPath()
        - 'edge': Edge-based analysis querying edges directly
        
        Args:
            skip_existing: Skip if results already cached
            
        Returns:
            Nested dict {dataset_name: {threshold: DataFrame}}
        """
        mode = self.parameters.comparison_mode
        self._log(f"Starting analysis across all datasets and thresholds (mode={mode})")
        
        dataset_names = self.parameters.get_dataset_names()
        
        if mode == 'edge':
            return self._run_all_edge_analyses(skip_existing)
        else:
            return self._run_all_path_analyses(skip_existing)
    
    def _run_all_path_analyses(self, skip_existing: bool = True) -> Dict[str, Dict[int, pd.DataFrame]]:
        """Run path-based analyses for all datasets and thresholds.
        
        Thresholds are processed in ascending order to enable graph caching:
        the lowest threshold is processed first, its graph is cached, and 
        higher thresholds reuse the cached graph with edge filtering.
        """
        dataset_names = self.parameters.get_dataset_names()
        # Sort thresholds ascending so lowest is processed first (enables graph cache)
        sorted_thresholds = sorted(self.parameters.thresholds)
        lowest_threshold = sorted_thresholds[0] if sorted_thresholds else None
        
        for dataset_name in dataset_names:
            if dataset_name not in self.raw_results:
                self.raw_results[dataset_name] = {}
            
            self._log(f"Processing \033[94m{dataset_name}\033[0m ({len(sorted_thresholds)} thresholds)")
            
            # Use progress bar for threshold iteration
            threshold_iter = tqdm(
                sorted_thresholds, 
                desc=f"  {dataset_name} thresholds",
                leave=False,
                unit="thr"
            )
            
            for threshold in threshold_iter:
                threshold_iter.set_postfix(threshold=threshold)
                
                # Check if already computed
                if skip_existing and threshold in self.raw_results[dataset_name]:
                    continue
                
                # Check if cached on disk
                if skip_existing and self.parameters.output_folder:
                    cached = self._try_load_cached(dataset_name, threshold)
                    if cached is not None:
                        self.raw_results[dataset_name][threshold] = cached
                        continue
                
                # Run path analysis
                # Use 'simple' verbose for lowest threshold (builds cache), 'silent' for others (uses cache)
                verbose = 'simple' if threshold == lowest_threshold else 'silent'
                result_df = self.run_path_analysis(dataset_name, threshold, verbose_mode=verbose)
                self.raw_results[dataset_name][threshold] = result_df
                
                # Save to disk
                if self.parameters.output_folder:
                    self._save_result(dataset_name, threshold, result_df)
        
        self._log(f"Completed path analysis for {len(dataset_names)} datasets")
        return self.raw_results
    
    def _run_all_edge_analyses(self, skip_existing: bool = True) -> Dict[str, Dict[int, pd.DataFrame]]:
        """
        Run edge-based analyses for all datasets and thresholds.
        
        Optimized Edge Mode Workflow:
        1. Run the path tool (FindAllPath, or FindShortestPath when
           path_mode='shortest') at LOWEST threshold to get bodyId connections
        2. Filter bodyId data by ALL thresholds and aggregate ALL to type-level immediately
        3. Run the path tool for remaining thresholds (only for output consistency, no aggregation)
        
        This approach:
        - Fetches connections only once (at lowest threshold)
        - Computes all edge aggregations upfront using FastGraph
        - Runs the path tool for other thresholds only to generate path output files
        """
        from core.fast_graph import FastGraph
        
        dataset_names = self.parameters.get_dataset_names()
        lowest_threshold = min(self.parameters.thresholds)
        sorted_thresholds = sorted(self.parameters.thresholds)
        path_tool = 'FindShortestPath' if self.parameters.path_mode == 'shortest' else 'FindAllPath'
        
        for dataset_name in dataset_names:
            if dataset_name not in self.raw_results:
                self.raw_results[dataset_name] = {}
            
            self._log(f"Edge mode analysis for \033[94m{dataset_name}\033[0m ({len(sorted_thresholds)} thresholds)")
            
            # ===== Step 1: Run the path tool for LOWEST threshold =====
            self._log(f"Running {path_tool} for {dataset_name} @ threshold={lowest_threshold}")
            self.run_path_analysis(dataset_name, lowest_threshold, verbose_mode='simple')
            
            # ===== Step 2: Get bodyId-level connections =====
            bodyid_df, label_map = self._get_bodyid_connections_for_dataset(
                dataset_name, lowest_threshold, skip_existing=True
            )
            
            if bodyid_df is None or bodyid_df.empty:
                self._log(f"Warning: No bodyId connections found for {dataset_name}")
                for threshold in self.parameters.thresholds:
                    self.raw_results[dataset_name][threshold] = pd.DataFrame()
                # Still run the path tool for other thresholds for output consistency
                remaining_thresholds = [t for t in sorted_thresholds if t != lowest_threshold]
                for threshold in tqdm(remaining_thresholds, desc=f"  {dataset_name} thresholds", leave=False, unit="thr"):
                    self.run_path_analysis(dataset_name, threshold, verbose_mode='silent')
                continue
            
            self._log(f"Loaded {len(bodyid_df)} bodyId-level connections from threshold={lowest_threshold}")
            
            # Get source/target types for path finding
            source_types = set(self.parameters.get_source_neurons_for_dataset(dataset_name))
            target_types = set(self.parameters.get_target_neurons_for_dataset(dataset_name))
            max_layers = self.parameters.max_interlayer + 1
            if self.parameters.path_mode == 'shortest' and self.parameters.max_interlayer <= 0:
                max_layers = None  # unlimited depth in shortest mode
            
            # ===== Step 3: Filter and aggregate for ALL thresholds at once =====
            self._log(f"Aggregating edges for all {len(sorted_thresholds)} thresholds...")
            for threshold in tqdm(sorted_thresholds, desc=f"  Aggregating", leave=False, unit="thr"):
                self._process_threshold_aggregation(
                    dataset_name, threshold, bodyid_df, label_map,
                    source_types, target_types, max_layers, skip_existing,
                    path_mode=self.parameters.path_mode
                )
            
            # ===== Step 4: Run the path tool for remaining thresholds (output consistency only) =====
            remaining_thresholds = [t for t in sorted_thresholds if t != lowest_threshold]
            if remaining_thresholds:
                for threshold in tqdm(remaining_thresholds, desc=f"  {dataset_name} thresholds", leave=False, unit="thr"):
                    self.run_path_analysis(dataset_name, threshold, verbose_mode='silent')
        
        self._log(f"Completed edge analysis for {len(dataset_names)} datasets")
        return self.raw_results
    
    def _process_threshold_aggregation(
        self,
        dataset_name: str,
        threshold: int,
        bodyid_df: pd.DataFrame,
        label_map: Dict,
        source_types: set,
        target_types: set,
        max_layers: Optional[int],
        skip_existing: bool,
        path_mode: str = 'all'
    ):
        """Process edge aggregation for a single threshold."""
        # Check if already computed in memory
        if skip_existing and threshold in self.raw_results[dataset_name]:
            existing = self.raw_results[dataset_name][threshold]
            if not existing.empty:
                self._log(f"  Skipping threshold={threshold} (already computed)")
                return
        
        # Filter bodyId edges by current threshold
        filtered_df = bodyid_df[bodyid_df['weight'] >= threshold].copy()
        
        if filtered_df.empty:
            self._log(f"  threshold={threshold}: No edges meet threshold")
            result_df = pd.DataFrame()
        else:
            # Aggregate to type-level using FastGraph
            result_df = self._aggregate_and_find_paths(
                filtered_df, label_map, source_types, target_types,
                max_layers, dataset_name, threshold, path_mode=path_mode
            )
            valid_count = result_df['has_valid_path'].sum() if not result_df.empty and 'has_valid_path' in result_df.columns else 0
            self._log(f"  threshold={threshold}: {len(result_df)} type edges ({valid_count} with valid paths)")
        
        self.raw_results[dataset_name][threshold] = result_df
        
        # Save aggregated edge data to edge_mode_data folder
        if self.parameters.output_folder and not result_df.empty:
            self._save_edge_mode_result(dataset_name, threshold, result_df)
    
    def _get_bodyid_connections_for_dataset(
        self, 
        dataset_name: str, 
        threshold: int,
        skip_existing: bool = True,
        run_findallpath: bool = False
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Get bodyId-level connections for a dataset at the lowest threshold.
        
        Args:
            dataset_name: Dataset identifier
            threshold: Threshold level
            skip_existing: Skip if cached data exists
            run_findallpath: If True, run FindAllPath to generate data (default False)
        
        Returns:
            Tuple of (bodyId DataFrame, label_map dict mapping bodyId -> type)
        """
        # Check for cached bodyId data
        if self.parameters.output_folder:
            safe_name = self.parameters._sanitize_name(dataset_name)
            cache_dir = os.path.join(
                self.parameters.full_output_path,
                'dataset_data',
                safe_name,
                f'minsyn_{threshold}'
            )
            bodyid_file = os.path.join(cache_dir, 'data_details', 'connection_info_bodyId.csv')
            
            if skip_existing and os.path.exists(bodyid_file):
                try:
                    df = self._read_csv(bodyid_file)
                    if not df.empty:
                        # Build label map from the data
                        label_map = self._build_label_map_from_df(df)
                        self._log(f"Loaded cached bodyId data from {bodyid_file}")
                        return df, label_map
                except Exception as e:
                    self._log(f"Warning: Could not load cached bodyId data: {e}")
        
        # Only run the path tool if explicitly requested (avoids duplicate runs)
        if run_findallpath:
            path_tool = 'FindShortestPath' if self.parameters.path_mode == 'shortest' else 'FindAllPath'
            self._log(f"Running {path_tool} for {dataset_name} @ threshold={threshold} to get bodyId connections")
            self.run_path_analysis(dataset_name, threshold)
            
            # Try to load the bodyId file that was generated
            if self.parameters.output_folder:
                safe_name = self.parameters._sanitize_name(dataset_name)
                cache_dir = os.path.join(
                    self.parameters.full_output_path,
                    'dataset_data',
                    safe_name,
                    f'minsyn_{threshold}'
                )
                bodyid_file = os.path.join(cache_dir, 'data_details', 'connection_info_bodyId.csv')
                
                if os.path.exists(bodyid_file):
                    try:
                        df = self._read_csv(bodyid_file)
                        if not df.empty:
                            label_map = self._build_label_map_from_df(df)
                            return df, label_map
                    except Exception:
                        pass
        
        # Fallback: query edges directly (this is the fast path when skip_bodyId=True)
        self._log(f"Querying bodyId edges directly for {dataset_name}")
        source_neurons = self.parameters.get_source_neurons_for_dataset(dataset_name)
        target_neurons = self.parameters.get_target_neurons_for_dataset(dataset_name)
        
        df = self._query_edges_for_dataset(dataset_name, source_neurons, target_neurons, threshold)
        if df is not None and not df.empty:
            label_map = self._build_label_map_from_df(df)
            return df, label_map
        
        return None, {}
    
    def _build_label_map_from_df(self, df: pd.DataFrame) -> Dict:
        """Build bodyId -> type label map from connection DataFrame."""
        label_map = {}
        
        # Check column names
        pre_id_col = 'bodyId_pre' if 'bodyId_pre' in df.columns else None
        post_id_col = 'bodyId_post' if 'bodyId_post' in df.columns else None
        pre_type_col = 'type_pre' if 'type_pre' in df.columns else None
        post_type_col = 'type_post' if 'type_post' in df.columns else None
        
        if pre_id_col and pre_type_col:
            for _, row in df[[pre_id_col, pre_type_col]].drop_duplicates().iterrows():
                if pd.notna(row[pre_id_col]) and pd.notna(row[pre_type_col]):
                    label_map[row[pre_id_col]] = row[pre_type_col]
        
        if post_id_col and post_type_col:
            for _, row in df[[post_id_col, post_type_col]].drop_duplicates().iterrows():
                if pd.notna(row[post_id_col]) and pd.notna(row[post_type_col]):
                    label_map[row[post_id_col]] = row[post_type_col]
        
        return label_map
    
    def _aggregate_and_find_paths(
        self,
        bodyid_df: pd.DataFrame,
        label_map: Dict,
        source_types: set,
        target_types: set,
        max_layers: Optional[int],
        dataset_name: str,
        threshold: int,
        path_mode: str = 'all'
    ) -> pd.DataFrame:
        """
        Aggregate bodyId edges to type-level and find valid paths.
        
        Args:
            bodyid_df: DataFrame with bodyId-level connections (already filtered by threshold)
            label_map: Dict mapping bodyId -> type
            source_types: Set of source neuron types
            target_types: Set of target neuron types
            max_layers: Maximum path length (None = unlimited, shortest mode only)
            dataset_name: Dataset name for metadata
            threshold: Current threshold for metadata
            path_mode: 'all' (every path within max_layers) or 'shortest'
                (edges valid only when they lie on a per-pair minimum-hop path)
            
        Returns:
            DataFrame with type-level edges and has_valid_path flag
        """
        from core.fast_graph import FastGraph
        
        # Determine column names
        pre_id_col = 'bodyId_pre' if 'bodyId_pre' in bodyid_df.columns else 'pre_pt_root_id'
        post_id_col = 'bodyId_post' if 'bodyId_post' in bodyid_df.columns else 'post_pt_root_id'
        weight_col = 'weight' if 'weight' in bodyid_df.columns else 'syn_count'
        
        # Build bodyId-level graph
        G_bodyid = FastGraph()
        G_bodyid.build_from_dataframe(bodyid_df, pre_id_col, post_id_col, weight_col)
        
        # Aggregate to type-level graph
        G_type, edge_df = G_bodyid.aggregate_by_label(label_map, return_edge_df=True)
        
        if edge_df.empty:
            return pd.DataFrame()
        
        # Find valid paths at type level
        # Get all type nodes that are sources or targets
        graph_source_types = [t for t in source_types if G_type.has_node(t)]
        graph_target_types = [t for t in target_types if G_type.has_node(t)]
        
        # Find paths (all paths, or only per-pair shortest paths)
        valid_edges = set()
        if graph_source_types and graph_target_types:
            try:
                if path_mode == 'shortest':
                    # Edges are valid only when they lie on a per-pair
                    # minimum-hop path; cutoff None = unlimited depth.
                    paths = list(G_type.find_paths_shortest(
                        graph_source_types, graph_target_types, cutoff=max_layers
                    ))
                else:
                    # Use memoized DFS for efficiency
                    paths = list(G_type.find_paths_memoized_dfs(
                        graph_source_types, graph_target_types, max_layers, 
                        direction='backward', verbose=False
                    ))
                
                # Extract edges from paths
                for path in paths:
                    for i in range(len(path) - 1):
                        valid_edges.add((path[i], path[i+1]))
                        
                self._log(f"Found {len(paths)} paths, {len(valid_edges)} unique edges in paths", 'debug')
            except Exception as e:
                self._log(f"Warning: Path finding failed: {e}")
        
        # Add metadata to edge DataFrame
        edge_df['has_valid_path'] = edge_df.apply(
            lambda r: (r['type_pre'], r['type_post']) in valid_edges, axis=1
        )
        edge_df['dataset'] = dataset_name
        edge_df['threshold'] = threshold
        
        # Compute additional metrics
        # Get total_post from bodyid_df if available
        if 'total_post' in bodyid_df.columns:
            # Aggregate total_post by type_post (take first value since it should be same for all bodyIds of same type)
            total_post_map = {}
            if 'type_post' in bodyid_df.columns:
                for post_type in edge_df['type_post'].unique():
                    mask = bodyid_df['type_post'] == post_type
                    if mask.any():
                        total_post_map[post_type] = bodyid_df.loc[mask, 'total_post'].iloc[0]
            
            edge_df['total_post'] = edge_df['type_post'].map(total_post_map)
            edge_df['connection_ratio'] = edge_df['weight'] / edge_df['total_post'].fillna(1)
        
        # Compute traversal probability if possible
        # traversal_prob = weight / sum(all outgoing weights from pre)
        outgoing_weights = edge_df.groupby('type_pre')['weight'].sum().to_dict()
        edge_df['traversal_probability'] = edge_df.apply(
            lambda r: r['weight'] / outgoing_weights.get(r['type_pre'], 1), axis=1
        )
        
        # Add conn_layer info based on path structure
        # This requires knowing which types are in which layer
        # For now, infer from source/target membership
        def get_conn_layer(row):
            pre, post = row['type_pre'], row['type_post']
            if pre in source_types:
                return '0->1'
            elif post in target_types:
                return '1->2'  # Assuming 2-hop max
            else:
                return '1->2'  # Default intermediate
        
        edge_df['conn_layer'] = edge_df.apply(get_conn_layer, axis=1)
        
        return edge_df
    
    # =========================================================================
    # Dataset Metadata Collection
    # =========================================================================
    
    def _get_datasets_folder(self) -> str:
        """Get the path to the datasets folder."""
        # Assume datasets folder is at project root level
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(src_dir)
        return os.path.join(project_root, 'datasets')
    
    def _get_metadata_path(self, dataset_name: str) -> str:
        """Get path to the metadata file for a dataset."""
        safe_name = self.parameters._sanitize_name(dataset_name)
        datasets_folder = self._get_datasets_folder()
        return os.path.join(datasets_folder, safe_name, f'{safe_name}_metadata.json')
    
    def _load_cached_metadata(self, dataset_name: str) -> Optional[Dict]:
        """Try to load cached metadata from local file."""
        metadata_path = self._get_metadata_path(dataset_name)
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self._log(f"Warning: Failed to load cached metadata for {dataset_name}: {e}")
        return None
    
    def _save_metadata(self, dataset_name: str, metadata: Dict) -> None:
        """Save metadata to local cache file."""
        metadata_path = self._get_metadata_path(dataset_name)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            self._log(f"Saved metadata to: {metadata_path}")
        except Exception as e:
            self._log(f"Warning: Failed to save metadata for {dataset_name}: {e}")
    
    def _fetch_neuprint_metadata(self, dataset_name: str) -> Dict:
        """Fetch metadata from NeuPrint server."""
        try:
            from neuprint import Client
            
            # Parse dataset name for server connection
            token = self.parameters.resolve_token()
            client = Client('neuprint.janelia.org', dataset=dataset_name, token=token)
            
            # Get neuron counts
            q = "MATCH (n:Neuron) RETURN count(n) AS total_neurons"
            result = client.fetch_custom(q)
            total_neurons = int(result.iloc[0]['total_neurons']) if not result.empty else 0
            
            # Get typed neuron count
            q = "MATCH (n:Neuron) WHERE n.type IS NOT NULL AND n.type <> '' RETURN count(n) AS typed_neurons"
            result = client.fetch_custom(q)
            typed_neurons = int(result.iloc[0]['typed_neurons']) if not result.empty else 0
            
            # Get synapse counts
            q = "MATCH (n:Neuron) RETURN sum(n.pre) AS total_pre, sum(n.post) AS total_post"
            result = client.fetch_custom(q)
            total_pre = int(result.iloc[0]['total_pre']) if not result.empty and result.iloc[0]['total_pre'] else 0
            total_post = int(result.iloc[0]['total_post']) if not result.empty and result.iloc[0]['total_post'] else 0
            
            # Get ROI coverage
            q = "MATCH (m:Meta) RETURN m.primaryRois AS rois"
            result = client.fetch_custom(q)
            rois = result.iloc[0]['rois'] if not result.empty else []
            
            # Get neuron counts per ROI
            roi_counts = {}
            if rois:
                for roi in rois[:20]:  # Limit to first 20 ROIs for performance
                    try:
                        q = f"MATCH (n:Neuron) WHERE n.`{roi}` = true RETURN count(n) AS count"
                        result = client.fetch_custom(q)
                        roi_counts[roi] = int(result.iloc[0]['count']) if not result.empty else 0
                    except:
                        pass
            
            metadata = {
                'dataset': dataset_name,
                'source': 'neuprint',
                'fetched_at': datetime.now().isoformat(),
                'neuron_counts': {
                    'total': total_neurons,
                    'typed': typed_neurons,
                    'untyped': total_neurons - typed_neurons,
                    'type_coverage': typed_neurons / total_neurons if total_neurons > 0 else 0
                },
                'synapse_counts': {
                    'total_presynaptic': total_pre,
                    'total_postsynaptic': total_post,
                    'total': total_pre + total_post
                },
                'roi_coverage': {
                    'roi_list': rois if rois else [],
                    'roi_count': len(rois) if rois else 0,
                    'neuron_counts_per_roi': roi_counts
                },
                'coverage_notes': self._get_coverage_notes(dataset_name)
            }
            
            return metadata
            
        except Exception as e:
            self._log(f"Warning: Failed to fetch NeuPrint metadata for {dataset_name}: {e}")
            return self._create_empty_metadata(dataset_name, str(e))
    
    def _fetch_local_metadata(self, dataset_name: str) -> Dict:
        """Fetch metadata from local dataset files."""
        safe_name = self.parameters._sanitize_name(dataset_name)
        datasets_folder = self._get_datasets_folder()
        dataset_path = os.path.join(datasets_folder, safe_name)
        
        neuron_file = os.path.join(dataset_path, f'{safe_name}_allneurons_neuron_df.csv')
        neuron_parquet = os.path.join(dataset_path, f'{safe_name}_allneurons_neuron_df.parquet')
        
        # Try to load neuron data
        neuron_df = None
        if os.path.exists(neuron_parquet):
            try:
                neuron_df = pd.read_parquet(neuron_parquet)
            except:
                pass
        if neuron_df is None and os.path.exists(neuron_file):
            try:
                neuron_df = self._read_csv(neuron_file)
            except:
                pass
        
        if neuron_df is None:
            return self._create_empty_metadata(dataset_name, "No local data found")
        
        total_neurons = len(neuron_df)
        
        # Count typed neurons
        type_col = None
        for col in ['type', 'cell_type', 'hemibrain_type', 'class']:
            if col in neuron_df.columns:
                type_col = col
                break
        
        if type_col:
            typed_neurons = neuron_df[type_col].notna().sum()
            typed_neurons = int(typed_neurons - (neuron_df[type_col] == '').sum())
        else:
            typed_neurons = 0
        
        # Synapse counts
        pre_col = 'pre' if 'pre' in neuron_df.columns else None
        post_col = 'post' if 'post' in neuron_df.columns else None
        
        total_pre = int(neuron_df[pre_col].sum()) if pre_col else 0
        total_post = int(neuron_df[post_col].sum()) if post_col else 0
        
        # Check for ROI data (parquet from current pulls, CSV from older ones)
        roi_file = os.path.join(dataset_path, f'{safe_name}_allneurons_roi_count_df.csv')
        roi_parquet = os.path.join(dataset_path, f'{safe_name}_allneurons_roi_count_df.parquet')
        roi_counts = {}
        rois = []

        roi_df = None
        if os.path.exists(roi_parquet):
            try:
                roi_df = pd.read_parquet(roi_parquet)
            except:
                pass
        if roi_df is None and os.path.exists(roi_file):
            try:
                roi_df = self._read_csv(roi_file)
            except:
                pass

        if roi_df is not None:
            try:
                # Get ROI columns (usually all except bodyId)
                roi_cols = [c for c in roi_df.columns if c not in ['bodyId', 'Unnamed: 0']]
                rois = roi_cols
                for col in roi_cols[:20]:  # Limit
                    roi_counts[col] = int((roi_df[col] > 0).sum())
            except:
                pass
        
        metadata = {
            'dataset': dataset_name,
            'source': 'local',
            'fetched_at': datetime.now().isoformat(),
            'neuron_counts': {
                'total': total_neurons,
                'typed': typed_neurons,
                'untyped': total_neurons - typed_neurons,
                'type_coverage': typed_neurons / total_neurons if total_neurons > 0 else 0
            },
            'synapse_counts': {
                'total_presynaptic': total_pre,
                'total_postsynaptic': total_post,
                'total': total_pre + total_post
            },
            'roi_coverage': {
                'roi_list': rois,
                'roi_count': len(rois),
                'neuron_counts_per_roi': roi_counts
            },
            'coverage_notes': self._get_coverage_notes(dataset_name)
        }
        
        return metadata
    
    def _create_empty_metadata(self, dataset_name: str, error_msg: str) -> Dict:
        """Create empty metadata structure with error message."""
        return {
            'dataset': dataset_name,
            'source': 'error',
            'fetched_at': datetime.now().isoformat(),
            'error': error_msg,
            'neuron_counts': {'total': 0, 'typed': 0, 'untyped': 0, 'type_coverage': 0},
            'synapse_counts': {'total_presynaptic': 0, 'total_postsynaptic': 0, 'total': 0},
            'roi_coverage': {'roi_list': [], 'roi_count': 0, 'neuron_counts_per_roi': {}},
            'coverage_notes': self._get_coverage_notes(dataset_name)
        }
    
    def _get_coverage_notes(self, dataset_name: str) -> str:
        """Get known coverage notes for a dataset."""
        notes = {
            'hemibrain': "Central brain only. Missing: optic lobe, ventral nerve cord, subesophageal zone.",
            'male-cns': "Full male CNS including central brain, optic lobes, VNC. Mostly bilateral symmetric.",
            'manc': "Male adult nerve cord (VNC) connectome.",
            'flywire': "Full adult female brain (FAFB). Complete brain coverage with optic lobes.",
            'fafb': "Full adult female brain. Complete brain coverage with optic lobes.",
            'optic-lobe': "Optic lobe only. Missing: central brain, VNC.",
            'banc': "Full brain and VNC connectome.",
        }
        
        dataset_lower = dataset_name.lower()
        for key, note in notes.items():
            if key in dataset_lower:
                return note
        return "Coverage information not available."
    
    def collect_dataset_metadata(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """
        Collect metadata for all datasets.
        
        Metadata is cached locally in datasets/{dataset}/{dataset}_metadata.json.
        If cached file exists and force_refresh=False, uses cached data.
        
        Args:
            force_refresh: If True, fetch fresh metadata even if cached exists
            
        Returns:
            Dict mapping dataset name to metadata dict
        """
        self._log("Collecting dataset metadata...")
        
        all_metadata = {}
        
        for dataset_name in self.parameters.get_dataset_names():
            # Try cached first
            if not force_refresh:
                cached = self._load_cached_metadata(dataset_name)
                if cached:
                    self._log(f"Loaded cached metadata for {dataset_name}")
                    all_metadata[dataset_name] = cached
                    continue
            
            # Fetch fresh metadata
            self._log(f"Fetching metadata for {dataset_name}...")
            
            # Determine if local or NeuPrint dataset
            dataset_lower = dataset_name.lower()
            if 'flywire' in dataset_lower or 'fafb' in dataset_lower or 'banc' in dataset_lower:
                metadata = self._fetch_local_metadata(dataset_name)
            else:
                metadata = self._fetch_neuprint_metadata(dataset_name)
            
            # Save to cache
            self._save_metadata(dataset_name, metadata)
            all_metadata[dataset_name] = metadata
        
        # Store for later use
        self._dataset_metadata = all_metadata
        
        return all_metadata
    
    def generate_metadata_comparison_table(self) -> pd.DataFrame:
        """
        Generate a comparison table from collected metadata.
        
        Returns:
            DataFrame comparing key metrics across datasets
        """
        if not hasattr(self, '_dataset_metadata') or not self._dataset_metadata:
            self.collect_dataset_metadata()
        
        rows = []
        for dataset_name, metadata in self._dataset_metadata.items():
            nc = metadata.get('neuron_counts', {})
            sc = metadata.get('synapse_counts', {})
            rc = metadata.get('roi_coverage', {})
            
            rows.append({
                'dataset': dataset_name,
                'total_neurons': nc.get('total', 0),
                'typed_neurons': nc.get('typed', 0),
                'untyped_neurons': nc.get('untyped', 0),
                'type_coverage_pct': round(nc.get('type_coverage', 0) * 100, 2),
                'total_presynaptic': sc.get('total_presynaptic', 0),
                'total_postsynaptic': sc.get('total_postsynaptic', 0),
                'total_synapses': sc.get('total', 0),
                'roi_count': rc.get('roi_count', 0),
                'coverage_notes': metadata.get('coverage_notes', '')
            })
        
        return pd.DataFrame(rows)
    
    def _try_load_cached(self, dataset_name: str, threshold: int) -> Optional[pd.DataFrame]:
        """Try to load cached result from disk."""
        if not self.parameters.output_folder:
            return None
        
        output_dir = self.parameters.get_dataset_output_path(dataset_name, threshold)
        
        # First try our own cached connections_edge.csv (edge mode output)
        filepath = os.path.join(output_dir, "connections_edge.csv")
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            try:
                df = self._read_csv(filepath)
                if not df.empty:
                    self._log(f"Loading cached: {dataset_name} @ {threshold}", 'debug')
                    return df
            except (pd.errors.EmptyDataError, Exception):
                pass  # File is empty or corrupted, try other sources
        
        # Also try legacy paths.csv (for backward compatibility)
        filepath = os.path.join(output_dir, "paths.csv")
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            try:
                df = self._read_csv(filepath)
                if not df.empty:
                    self._log(f"Loading cached: {dataset_name} @ {threshold}", 'debug')
                    return df
            except (pd.errors.EmptyDataError, Exception):
                pass
        
        # Also try to find the FindNeuronConnection output file (connection_info_bodyId.csv)
        conn_file = os.path.join(output_dir, 'data_details', 'connection_info_bodyId.csv')
        if os.path.exists(conn_file) and os.path.getsize(conn_file) > 0:
            try:
                df = self._read_csv(conn_file)
                if not df.empty:
                    self._log(f"Loading cached: {dataset_name} @ {threshold}", 'debug')
                    # Add dataset info if missing
                    if 'dataset' not in df.columns:
                        df['dataset'] = dataset_name
                        df['threshold'] = threshold
                    return df
            except (pd.errors.EmptyDataError, Exception):
                pass
        
        # Also try connection_type.csv (type-level path mode output)
        conn_type_file = os.path.join(output_dir, 'data_details', 'connection_type.csv')
        if os.path.exists(conn_type_file) and os.path.getsize(conn_type_file) > 0:
            try:
                df = self._read_csv(conn_type_file)
                if not df.empty:
                    self._log(f"Loading cached: {dataset_name} @ {threshold}", 'debug')
                    # Add dataset info if missing
                    if 'dataset' not in df.columns:
                        df['dataset'] = dataset_name
                        df['threshold'] = threshold
                    return df
            except (pd.errors.EmptyDataError, Exception):
                pass
        
        return None
    
    def _save_result(self, dataset_name: str, threshold: int, df: pd.DataFrame):
        """Save result to disk (edge mode output)."""
        if not self.parameters.output_folder:
            return
        
        # Don't save empty DataFrames - they cause read errors later
        if df is None or df.empty:
            self._log(f"Skipping save for {dataset_name} @ {threshold} (empty result)", 'debug')
            return
        
        dirpath = self.parameters.get_dataset_output_path(dataset_name, threshold)
        os.makedirs(dirpath, exist_ok=True)
        
        # Save to connections_edge.csv (edge mode cached version)
        filepath = os.path.join(dirpath, "connections_edge.csv")
        self._save_csv(df, filepath)
        self._log_file(filepath)
    
    def _save_edge_mode_result(self, dataset_name: str, threshold: int, df: pd.DataFrame):
        """Save aggregated edge mode result to edge_mode_data folder."""
        if not self.parameters.output_folder:
            return
        
        # Don't save empty DataFrames
        if df is None or df.empty:
            self._log(f"Skipping edge_mode save for {dataset_name} @ {threshold} (empty result)", 'debug')
            return
        
        # Save to: comparison_results_{}/edge_mode_data/{dataset}/connections_edge_{threshold}.csv
        safe_name = self.parameters._sanitize_name(dataset_name)
        edge_mode_dir = os.path.join(
            self.parameters.full_output_path,
            'edge_mode_data',
            safe_name
        )
        os.makedirs(edge_mode_dir, exist_ok=True)
        
        # Save aggregated connections_edge_{threshold}.csv
        filepath = os.path.join(edge_mode_dir, f"connections_edge_{threshold}.csv")
        self._save_csv(df, filepath)
        self._log_file(filepath)
    
    # =========================================================================
    # Comparison Analysis
    # =========================================================================
    
    def run_comparison(self, skip_existing: bool = True) -> Dict[str, Any]:
        """
        Run full comparison analysis: path analysis + comparison metrics.
        
        This is the main entry point for running a complete comparison.
        
        Args:
            skip_existing: Skip if results already cached
            
        Returns:
            Dictionary with all comparison metrics and findings
        """
        if self.parameters and self.parameters.full_output_path:
            self._log(f"📁 Output folder: {self.parameters.full_output_path}")
        self._progress(1, 5, "Resolving datasets and thresholds")
        # Run path analyses
        mode_label = ("Running edge analyses"
                      if getattr(self.parameters, "comparison_mode", "path") == 'edge'
                      else "Running path analyses")
        self._progress(2, 5, mode_label)
        self.run_all_analyses(skip_existing=skip_existing)
        
        # Compute comparison metrics
        self._progress(3, 5, "Computing cross-dataset metrics")
        return self.run_comparison_analysis()

    def run_comparison_analysis(self) -> Dict[str, Any]:
        """
        Run full comparison analysis on results.

        Returns:
            Dictionary with all comparison metrics and findings
        """
        # Ensure analyses have been run
        if not self.raw_results:
            self.run_all_analyses()
        
        self._log("Computing comparison metrics")
        self._log("  Step 1/2: Generating comparison summary...")

        dataset_names = self.parameters.get_dataset_names()
        
        # Get type mapper for auto type mapping (if enabled)
        type_mapper = self.parameters._auto_type_mapper if self.parameters.auto_type_mapping else None

        # Generate comprehensive summary
        # Pass label_mapper=None because raw_results are already mapped
        summary = self.metrics.generate_comparison_summary(
            results=self.raw_results,
            datasets=dataset_names,
            thresholds=self.parameters.thresholds,
            label_mapper=None,
            type_mapper=type_mapper,
            max_edges_for_metrics=self.parameters.max_edges_for_metrics
        )

        self._log("  Step 2/2: Calculating cross-threshold similarities...")

        # Calculate cross-threshold similarities and cache them
        # Pass label_mapper=None because raw_results are already mapped
        # Pass path_data_func to enable path rank correlation computation
        similarities = self.metrics.calculate_similarity_across_thresholds(
            results=self.raw_results,
            datasets=dataset_names,
            thresholds=self.parameters.thresholds,
            label_mapper=None,
            path_data_func=self._get_path_data_for_threshold,
            type_mapper=type_mapper,
            max_edges_for_metrics=self.parameters.max_edges_for_metrics
        )
        summary['threshold_similarities'] = similarities

        # Cache per-threshold similarities for reuse in visualizations
        if not similarities.empty and 'threshold' in similarities.columns:
            for threshold in self.parameters.thresholds:
                thresh_sims = similarities[similarities['threshold'] == threshold]
                if not thresh_sims.empty:
                    self._similarity_cache[threshold] = thresh_sims.copy()

        # Store for later use
        self.comparison_report = summary
        
        # Print intermediate type mapping summary if auto_type_mapping is enabled
        if self.parameters.auto_type_mapping and self.parameters._auto_type_mapper:
            self._print_intermediate_mapping_summary()
        
        return summary
    
    def get_cached_similarities(self, threshold: int) -> pd.DataFrame:
        """
        Get cached pairwise similarities at a threshold.
        
        Uses cached values if available, otherwise computes and caches.
        
        Args:
            threshold: Weight threshold
            
        Returns:
            DataFrame with pairwise similarities
        """
        if threshold in self._similarity_cache:
            return self._similarity_cache[threshold].copy()
        
        # Compute if not cached
        aligned = self.get_aligned_data(threshold)
        if aligned.empty:
            return pd.DataFrame()
        
        dataset_names = self.parameters.get_dataset_names()
        similarities = self.metrics.calculate_all_pairwise_similarities(
            aligned, dataset_names, threshold=1, include_advanced_metrics=True
        )
        
        # Cache for future use
        if not similarities.empty:
            self._similarity_cache[threshold] = similarities.copy()
        
        return similarities
    
    def get_aligned_data(self, threshold: int) -> pd.DataFrame:
        """
        Get aligned edge data at a specific threshold.
        
        Args:
            threshold: Weight threshold
            
        Returns:
            DataFrame with edges aligned across datasets
        """
        if threshold in self.aligned_results:
            return self.aligned_results[threshold]
        
        dataset_names = self.parameters.get_dataset_names()
        
        # Get type mapper for auto type mapping (if enabled)
        type_mapper = self.parameters._auto_type_mapper if self.parameters.auto_type_mapping else None
        
        # Pass label_mapper=None because raw_results are already mapped in run_path_analysis/run_edge_analysis
        aligned = self.metrics._align_results_at_threshold(
            self.raw_results,
            dataset_names,
            threshold,
            label_mapper=None,
            type_mapper=type_mapper
        )

        # Optionally filter edges to only hemisphere-conserved pairs
        # Works for edges with _L/_R/_U suffixes; edges without hemisphere info are kept as-is
        if self.parameters.keep_only_hemisphere_conserved_connections:
            aligned = self._filter_hemisphere_unconserved(aligned, dataset_names, threshold)
        
        self.aligned_results[threshold] = aligned
        return aligned

    def get_aligned_data_for_network(self, threshold: int) -> pd.DataFrame:
        """Get aligned edge data for network visualizations.

        When find_reciprocal=True, this uses reciprocal_connection_type.csv
        outputs (if available) to build the network graph.
        """
        if not self.parameters.find_reciprocal:
            return self.get_aligned_data(threshold)

        if threshold in self._network_aligned_cache:
            return self._network_aligned_cache[threshold]

        dataset_names = self.parameters.get_dataset_names()
        type_mapper = self.parameters._auto_type_mapper if self.parameters.auto_type_mapping else None

        # Build temporary raw_results from reciprocal files when available
        reciprocal_results: Dict[str, Dict[int, pd.DataFrame]] = {}
        for dataset in dataset_names:
            reciprocal_results[dataset] = {}
            safe_name = self.parameters._sanitize_name(dataset)
            reciprocal_path = os.path.join(
                self.parameters.full_output_path,
                'dataset_data',
                safe_name,
                f'minsyn_{threshold}',
                'find_reciprocal',
                'reciprocal_connection_type.csv'
            )
            if os.path.exists(reciprocal_path):
                try:
                    df = self._read_csv(reciprocal_path)
                    reciprocal_results[dataset][threshold] = df
                except Exception as e:
                    self._log(f"Warning: Failed to read reciprocal network data for {dataset} t={threshold}: {e}", level='warn')

        # If no reciprocal data found, fall back to standard aligned data
        has_any = any(threshold in reciprocal_results.get(ds, {}) for ds in dataset_names)
        if not has_any:
            return self.get_aligned_data(threshold)

        aligned = self.metrics._align_results_at_threshold(
            reciprocal_results,
            dataset_names,
            threshold,
            label_mapper=None,
            type_mapper=type_mapper
        )

        if self.parameters.keep_only_hemisphere_conserved_connections:
            aligned = self._filter_hemisphere_unconserved(aligned, dataset_names, threshold)

        self._network_aligned_cache[threshold] = aligned
        return aligned

    def _filter_hemisphere_unconserved(self, aligned: pd.DataFrame, dataset_names: List[str], 
                                        threshold: int = None) -> pd.DataFrame:
        """
        Filter out hemisphere-unconserved edges and save them to a separate file.
        
        An edge is considered "conserved" if both it and its mirror counterpart
        (L->L paired with R->R, or L->R paired with R->L) are present.
        
        Edges without hemisphere suffixes (_L/_R/_U) in their labels are kept as-is
        since they cannot be evaluated for hemisphere conservation.
        
        Args:
            aligned: Aligned edge data
            dataset_names: List of dataset names
            threshold: Optional threshold value for file naming
            
        Returns:
            Filtered DataFrame with only conserved edges
        """
        if aligned is None or aligned.empty:
            return aligned

        def extract_hemi(label: str):
            base = label.split('(')[0].strip() if '(' in label else label
            hemi = None
            if base.endswith(('_L', '_R', '_U')):
                hemi = base[-1]
                base = base[:-2]
            return base, hemi

        def opposite(hemi: str) -> str:
            return 'R' if hemi == 'L' else 'L'

        aligned = aligned.copy()
        aligned.index = aligned.index.astype(str)
        index_set = set(aligned.index)
        
        # Track unconserved edges for saving
        unconserved_edges = []
        unconserved_reasons = []

        for edge_key in aligned.index:
            if ' -> ' not in edge_key:
                continue
            pre, post = edge_key.split(' -> ', 1)
            base_pre, hemi_pre = extract_hemi(pre)
            base_post, hemi_post = extract_hemi(post)

            if hemi_pre not in ('L', 'R') or hemi_post not in ('L', 'R'):
                # Edge doesn't have proper L/R hemisphere info - keep it as-is
                # (Cannot evaluate hemisphere conservation without hemisphere suffixes)
                continue

            # Mirror counterpart: flip the hemisphere of both endpoints.
            # (opposite() handles same-side and cross-hemisphere edges alike.)
            counterpart = f"{base_pre}_{opposite(hemi_pre)} -> {base_post}_{opposite(hemi_post)}"

            if counterpart not in index_set:
                original_weights = {ds: aligned.at[edge_key, ds] for ds in dataset_names if ds in aligned.columns and aligned.at[edge_key, ds] > 0}
                if original_weights:
                    unconserved_edges.append(edge_key)
                    unconserved_reasons.append(f"Missing counterpart: {counterpart}")
                aligned.loc[edge_key, dataset_names] = 0
                continue

            for ds in dataset_names:
                if ds not in aligned.columns:
                    continue
                w = aligned.at[edge_key, ds]
                w2 = aligned.at[counterpart, ds]
                if not (w > 0 and w2 > 0):
                    if w > 0:
                        unconserved_edges.append(edge_key)
                        unconserved_reasons.append(f"Counterpart {counterpart} has weight=0 in {ds}")
                    aligned.at[edge_key, ds] = 0
        
        # Save unconserved edges to file
        if unconserved_edges and hasattr(self, 'parameters') and self.parameters.full_output_path:
            try:
                results_dir = os.path.join(self.parameters.full_output_path, 'comparison_results')
                os.makedirs(results_dir, exist_ok=True)
                
                threshold_suffix = f"_t{threshold}" if threshold else ""
                unconserved_file = os.path.join(results_dir, f"hemisphere_unconserved_edges{threshold_suffix}.csv")
                
                unconserved_df = pd.DataFrame({
                    'edge': unconserved_edges,
                    'reason': unconserved_reasons
                })
                unconserved_df.to_csv(unconserved_file, index=False)
                self._log(f"Saved {len(unconserved_edges)} unconserved edges to hemisphere_unconserved_edges{threshold_suffix}.csv")
            except Exception as e:
                self._log(f"Warning: Could not save unconserved edges: {e}")

        return aligned
    
    def get_common_connections(self, threshold: int) -> pd.DataFrame:
        """
        Get connections present in all datasets at threshold.
        
        Args:
            threshold: Weight threshold
            
        Returns:
            DataFrame with common connections
        """
        aligned = self.get_aligned_data(threshold)
        dataset_names = self.parameters.get_dataset_names()
        return self.metrics.find_common_connections(aligned, dataset_names, threshold)
    
    def get_unique_connections(self, threshold: int) -> Dict[str, pd.DataFrame]:
        """
        Get connections unique to each dataset.
        
        Args:
            threshold: Weight threshold
            
        Returns:
            Dict mapping dataset name to unique connections DataFrame
        """
        aligned = self.get_aligned_data(threshold)
        dataset_names = self.parameters.get_dataset_names()
        return self.metrics.find_unique_connections(aligned, dataset_names, threshold)
    
    def get_differential_connections(self, threshold: int, fold_threshold: float = 2.0) -> pd.DataFrame:
        """
        Get connections with large weight differences.
        
        Args:
            threshold: Weight threshold for presence
            fold_threshold: Minimum fold change
            
        Returns:
            DataFrame with differential connections
        """
        aligned = self.get_aligned_data(threshold)
        dataset_names = self.parameters.get_dataset_names()
        return self.metrics.find_differential_connections(aligned, dataset_names, fold_threshold)
    
    # =========================================================================
    # Report Generation
    # =========================================================================
    
    def generate_report(self, output_path: Optional[str] = None, _skip_log: bool = False) -> str:
        """
        Generate human-readable comparison report.
        
        Args:
            output_path: Optional path to save report
            _skip_log: Internal flag to skip logging (used when called from export_results)
            
        Returns:
            Report text
        """
        if not _skip_log:
            self._log("Generating comparison report text...")
        
        # Ensure comparison has been run
        if self.comparison_report is None:
            self.run_comparison_analysis()
        
        thresholds_str = ', '.join(str(t) for t in self.parameters.thresholds)
        mid_threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]
        dataset_names = self.parameters.get_dataset_names()
        
        lines = []
        lines.append("=" * 70)
        lines.append("CROSS-DATASET COMPARISON REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")
        
        # Datasets
        lines.append("DATASETS:")
        lines.append("-" * 40)
        for dataset_name in dataset_names:
            lines.append(f"  • {dataset_name}")
        lines.append("")
        
        # Analysis parameters with explicit threshold info
        lines.append("ANALYSIS PARAMETERS:")
        lines.append("-" * 40)
        lines.append(f"  Source neurons: {self.parameters.source_neurons}")
        lines.append(f"  Target neurons: {self.parameters.target_neurons}")
        lines.append(f"  Max interlayer: {self.parameters.max_interlayer}")
        lines.append(f"  Comparison mode: {self.parameters.comparison_mode}")
        lines.append("")
        
        # Explicit threshold section
        lines.append("SYNAPSE CUTOFF THRESHOLDS (min_synapse_num):")
        lines.append("-" * 40)
        lines.append(f"  Thresholds analyzed: {thresholds_str}")
        lines.append("  NOTE: Results are highly sensitive to threshold choice.")
        lines.append("")
        
        # Key metrics by threshold (matches HTML summary section)
        lines.append("KEY METRICS BY THRESHOLD:")
        lines.append("-" * 70)
        lines.append(f"{'Threshold':>10} | {'Total Edges':>12} | {'Conserved':>10} | {'Edge Rate':>10} | {'Total Paths':>12} | {'Path Rate':>10}")
        lines.append("-" * 70)
        
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            available_ds = [d for d in dataset_names if d in aligned.columns]
            total_edges = len(aligned)
            
            # Conserved edges (present in ALL datasets)
            if available_ds:
                mask_all = (aligned[available_ds] > 0).all(axis=1)
                common_edges = int(mask_all.sum())
            else:
                common_edges = 0
            
            edge_rate = (common_edges / total_edges * 100) if total_edges > 0 else 0
            
            # Path data
            try:
                path_data = self._get_path_data_for_threshold(threshold)
                if not path_data.empty:
                    total_paths = len(path_data)
                    path_mask = (path_data[available_ds] > 0).all(axis=1) if available_ds else pd.Series([False])
                    common_paths = int(path_mask.sum())
                    path_rate = (common_paths / total_paths * 100) if total_paths > 0 else 0
                else:
                    total_paths, common_paths, path_rate = 0, 0, 0
            except:
                total_paths, common_paths, path_rate = 0, 0, 0
            
            lines.append(f"{threshold:>10} | {total_edges:>12} | {common_edges:>10} | {edge_rate:>9.1f}% | {total_paths:>12} | {path_rate:>9.1f}%")
        
        lines.append("")

        # Hemisphere symmetry summary (per dataset & threshold)
        symmetry_summaries = self.get_hemisphere_symmetry_summaries()
        if symmetry_summaries:
            lines.append("HEMISPHERE SYMMETRY SUMMARY:")
            lines.append("-" * 70)
            for threshold in self.parameters.thresholds:
                summaries = symmetry_summaries.get(threshold, {})
                if not summaries:
                    continue
                lines.append(f"Threshold t={threshold}:")
                lines.append("  Dataset | Ipsi Jaccard | Contra Jaccard | Ipsi Conserved/Union | Contra Conserved/Union | Types Conserved/Union | Counts L/R")
                lines.append("  " + "-" * 64)
                for dataset in dataset_names:
                    summary = summaries.get(dataset)
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
                    lines.append(
                        f"  {dataset:<7} | {ipsi_j:>11.3f} | {contra_j:>13.3f} | {ipsi_cons:>19} | {contra_cons:>20} | {types_cons:>20} | {lr_counts:>9}"
                    )
                lines.append("")
        else:
            lines.append("HEMISPHERE SYMMETRY SUMMARY:")
            lines.append("-" * 70)
            lines.append("  (No hemisphere symmetry summaries found)")
            lines.append("")
        
        # Edge counts per dataset per threshold
        lines.append("EDGE COUNTS PER DATASET:")
        lines.append("-" * 70)
        header = f"{'Threshold':>10}"
        for d in dataset_names:
            short_name = d.split(':')[0][:10] if ':' in d else d[:10]
            header += f" | {short_name:>12}"
        lines.append(header)
        lines.append("-" * 70)
        
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            row = f"{threshold:>10}"
            for d in dataset_names:
                count = int((aligned[d] > 0).sum()) if d in aligned.columns else 0
                row += f" | {count:>12}"
            lines.append(row)
        
        lines.append("")
        
        # Pairwise similarities at ALL thresholds
        lines.append("PAIRWISE SIMILARITIES AT ALL THRESHOLD LEVELS:")
        lines.append("-" * 70)
        
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            sim_t = self.metrics.calculate_all_pairwise_similarities(aligned, dataset_names, threshold=1)
            if sim_t.empty:
                continue
            
            lines.append(f"\n  Threshold = {threshold}:")
            for _, row in sim_t.iterrows():
                jaccard = row.get('jaccard_similarity', 0)
                svd = row.get('svd_similarity', 0)
                if not self.raw_results: 
                    svd = 0
                pearson = row.get('pearson_correlation', 0)
                if pd.isna(pearson):
                    pearson = 0
                common = row.get('common_edges', 0)
                unique_d1 = row.get('unique_to_d1', 0)
                unique_d2 = row.get('unique_to_d2', 0)
                lines.append(f"    {row['dataset_1']} vs {row['dataset_2']}:")
                lines.append(f"      Jaccard: {jaccard:.3f} | SVD: {svd:.3f} | Weight Corr: {pearson:.3f}")
                lines.append(f"      Common: {common} | Unique to D1: {unique_d1} | Unique to D2: {unique_d2}")
        
        lines.append("")
        
        # Dataset overlap summary
        lines.append("DATASET OVERLAP SUMMARY:")
        lines.append("-" * 70)
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            available_ds = [d for d in dataset_names if d in aligned.columns]
            n = len(available_ds)
            
            lines.append(f"\n  Threshold = {threshold}:")
            # Edge overlap
            for i, d1 in enumerate(available_ds):
                edges_in_d1 = set(aligned.index[aligned[d1] > 0])
                for j, d2 in enumerate(available_ds):
                    if i < j:
                        edges_in_d2 = set(aligned.index[aligned[d2] > 0])
                        overlap = len(edges_in_d1 & edges_in_d2)
                        pct1 = (overlap / len(edges_in_d1) * 100) if edges_in_d1 else 0
                        pct2 = (overlap / len(edges_in_d2) * 100) if edges_in_d2 else 0
                        short1 = d1.split(':')[0][:8] if ':' in d1 else d1[:8]
                        short2 = d2.split(':')[0][:8] if ':' in d2 else d2[:8]
                        lines.append(f"    {short1} ∩ {short2}: {overlap} edges ({pct1:.0f}% of D1, {pct2:.0f}% of D2)")
        
        lines.append("")
        lines.append("=" * 70)
        lines.append("For interactive visualizations, see: comparison_report.html")
        lines.append("For connectivity profile verification, see: connectivity_profile_comparison.html")
        lines.append("=" * 70)
        
        report_text = "\n".join(lines)
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            self._log(f"Report saved to: {output_path}")
        
        return report_text
    
    def export_results(self, output_dir: Optional[str] = None):
        """
        Export all results to files following TODO_comparison.md structure.
        
        Output structure:
            {output_folder}/{saveas}/
            ├── parameters.json
            ├── comparison_report.txt
            ├── comparison_report.html          # Interactive HTML report
            ├── comparison_visualizations/       # Visualizations (renamed from visualizations/)
            │   ├── path_counts.png
            │   ├── edge_heatmap.png
            │   ├── similarity_matrix.png
            │   ├── similarity_per_threshold.png
            │   ├── edge_overlap.png
            │   ├── threshold_comparison.png
            │   ├── heatmaps/                   # Interactive HTML heatmaps
            │   │   ├── edge_heatmap.html
            │   │   └── edge_heatmap_all_thresholds.html
            │   └── visualization_data/         # Data files for visualizations
            │       ├── path_counts.csv
            │       ├── edge_heatmap.csv
            │       ├── similarity_matrix.csv
            │       └── ...
            ├── dataset_data/               # Raw FindNeuronConnection outputs (created by FNC)
            │   ├── {dataset_1}/minsyn_{threshold}/
            │   └── {dataset_2}/minsyn_{threshold}/
            └── comparison_results/         # Cross-dataset comparison outputs
                ├── path_count_comparison.csv
                ├── edge_weight_comparison.csv
                ├── edge_presence_matrix_minsyn_{threshold}.csv
                ├── path_presence_matrix_minsyn_{threshold}.csv
                └── ...
        
        Args:
            output_dir: Directory to save results (defaults to parameters.full_output_path)
        """
        out_dir = output_dir or self.parameters.full_output_path
        if not out_dir:
            raise ValueError("No output directory specified")
        
        self._log("Exporting results...")
        self._progress(4, 5, "Exporting reports and result tables")
        self._log("  Step 1: Saving parameters and report...")
        
        # Ensure data loader is initialized
        if self.data_loader is None:
            self.data_loader = DataLoader(out_dir)
        self.data_loader.ensure_directories()
        
        # Create comparison_results subfolder
        comparison_results_dir = os.path.join(out_dir, "comparison_results")
        os.makedirs(comparison_results_dir, exist_ok=True)
        
        # Save parameters
        params_path = os.path.join(out_dir, "parameters.json")
        with open(params_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(self.parameters.to_dict(), f, indent=2, default=str)
            
        # Save label mapping (always generate a compatible JSON)
        label_map_path = os.path.join(out_dir, "label_map.json")
        self._export_label_map(label_map_path)
        
        # Export auto type mapping if enabled (filtered to result types and used datasets only)
        if self.parameters.auto_type_mapping and self.parameters._auto_type_mapper:
            try:
                # Collect types from results for filtered export
                result_types = self._collect_result_types()
                dataset_names = self.parameters.get_dataset_names()
                
                auto_map_path = os.path.join(out_dir, "auto_type_mapping.csv")
                self.parameters._auto_type_mapper.export_mapping(
                    auto_map_path, 
                    filter_types=result_types if result_types else None,
                    datasets=dataset_names,
                    only_different=True  # Only export mappings where types differ across datasets
                )
                self._log_file(auto_map_path, "Auto type mapping")
                
                # Also export conflicts if any (filtered to result types)
                if self.parameters._auto_type_mapper._conflicts:
                    conflicts_path = os.path.join(out_dir, "auto_type_mapping_conflicts.csv")
                    self.parameters._auto_type_mapper.export_conflicts(
                        conflicts_path,
                        filter_types=result_types if result_types else None
                    )
                    self._log_file(conflicts_path, "Type mapping conflicts")

                # Bridge visualizations for the run's mapped pairs
                try:
                    mapper = self.parameters._auto_type_mapper
                    flows = []
                    seen_pairs = set()
                    for result_type in (result_types or [])[:300]:
                        source_ds = (
                            mapper._detect_type_source(result_type)
                            or (dataset_names[0] if dataset_names else "")
                        )
                        if not source_ds:
                            continue
                        for dataset_name in dataset_names:
                            pair = (result_type, dataset_name)
                            if pair in seen_pairs:
                                continue
                            bridges = mapper.get_type_bridges(
                                result_type, source_ds, dataset_name
                            )
                            if not bridges:
                                continue
                            seen_pairs.add(pair)
                            final = bridges[0][-1]["value"]
                            target_key = mapper._get_type_mapping_key(dataset_name)
                            count = len(
                                mapper._dataset_types.get(target_key, {}).get(
                                    bridges[0][-1]["value"], set())
                            ) or 1
                            flows.append({
                                "source_dataset": source_ds,
                                "target_dataset": dataset_name,
                                "source_type": result_type,
                                "source_count": count,
                                "foreign_type": final,
                                "foreign_count": count,
                                "matched_origin": "auto type mapping",
                                "bridges": bridges,
                            })
                    if flows:
                        from comparison.mapping_visualization import (
                            build_mapping_sankey_figure,
                            write_mapping_network_html,
                        )
                        sankey_path = os.path.join(out_dir, "mapping_sankey.html")
                        build_mapping_sankey_figure(flows).write_html(
                            sankey_path, include_plotlyjs="cdn"
                        )
                        self._log_file(sankey_path, "Mapping bridge Sankey")
                        network_path = os.path.join(out_dir, "mapping_network.html")
                        write_mapping_network_html(flows, network_path, open_browser=False)
                        self._log_file(network_path, "Mapping bridge network")
                except Exception as map_exc:
                    self._log(f"Mapping visualization skipped: {map_exc}")
            except Exception as e:
                self._log(f"Warning: Failed to export auto type mapping: {e}", level='warn')
        
        # Save report (skip logging since we already logged "Saving parameters and report")
        report_path = os.path.join(out_dir, "comparison_report.txt")
        self.generate_report(report_path, _skip_log=True)
        
        self._log("  Step 2: Exporting cross-dataset comparisons...")
        
        # === Cross-dataset comparison results ===
        self._export_cross_dataset_comparisons(comparison_results_dir)
        
        # === Intra-dataset threshold sensitivity ===
        self._export_intra_dataset_comparisons(comparison_results_dir)

        self._log("  Step 3: Generating visualizations...")
        self._progress(5, 5, "Generating comparison visualizations and HTML report")
        
        # Generate matplotlib visualizations to comparison_visualizations/ at base level
        try:
            self._generate_visualizations(out_dir)
        except Exception as e:
            self._log(f"Warning: Failed to generate visualizations: {e}")
        
        # NOTE: Connectivity profile verification is NOT run here automatically.
        # Call run_connectivity_profile_verification() separately after export_results()
        # if needed. Parameters can be set in ComparisonParameters:
        #   - verification_direction, verification_mode, verification_top_k, etc.
        # Example:
        #   analyzer.run_connectivity_profile_verification()  # Uses params from ComparisonParameters
        
        self._log("  Step 4: Generating HTML report...")
        
        # Generate interactive HTML report at base level
        try:
            html_report_path = os.path.join(out_dir, "comparison_report.html")
            self.generate_html_report(html_report_path)
        except Exception as e:
            self._log(f"Warning: Failed to generate HTML report: {e}")
        
        self._log(f"All results exported to: {out_dir}")
        self._log(f"Note: Run connectivity_profile_comparison() separately for profile verification.")
    
    def _export_label_map(self, filepath: str):
        """
        Export label mapping to JSON, creating one from parameters if needed.
        
        Ensures that a valid LabelMapper JSON is always saved, merging data from
        an existing LabelMapper and any list-based parameters.
        
        When auto_type_mapping=True:
        - Saves the auto-mapped types per dataset (e.g., MeVPLo2 → MTe07 for FAFB)
        - Removes types that don't exist in specific datasets from those dataset entries
        """
        output = {}
        dataset_names = self.parameters.get_dataset_names()
        auto_mapper = self.parameters._auto_type_mapper if self.parameters.auto_type_mapping else None
        
        # Helper for smart labels
        def generate_smart_labels(groups, user_labels, suffix=""):
            if user_labels:
                if isinstance(user_labels, list) and len(user_labels) == len(groups):
                    return user_labels
                if isinstance(user_labels, str) and len(groups) == 1:
                    return [user_labels]
            
            # Generate defaults
            labels = []
            for i, g in enumerate(groups):
                if len(g) == 1:
                    labels.append(str(g[0]))
                else:
                    labels.append(f"Group_{i+1}{suffix}")
            return labels
        
        def resolve_types_for_dataset(neurons: list, dataset: str) -> list:
            """Resolve neuron types for a specific dataset using auto mapping."""
            if not auto_mapper:
                return neurons
            
            resolved = []
            for neuron in neurons:
                if isinstance(neuron, list):
                    # Handle grouped neurons
                    resolved_group = resolve_types_for_dataset(neuron, dataset)
                    if resolved_group:  # Only add if not empty
                        resolved.append(resolved_group)
                elif isinstance(neuron, str):
                    # Skip regex patterns - pass through as-is
                    if '*' in neuron or ('.' in neuron and '.*' in neuron):
                        resolved.append(neuron)
                        continue
                    
                    # Try to resolve type using auto mapper
                    source_ds = auto_mapper._detect_type_source(neuron)
                    if source_ds:
                        mapped = auto_mapper.get_mapped_type(neuron, source_ds, dataset)
                        if mapped:
                            resolved.append(mapped)
                        # If no mapping found, the type doesn't exist in this dataset - skip it
                    else:
                        # Type not found in any dataset, pass through (might be regex or new type)
                        resolved.append(neuron)
                else:
                    # Non-string (bodyId), pass through
                    resolved.append(neuron)
            return resolved

        # --- Source Mapping ---
        # Priority 1: Analyzer's label_mapper (contains merged overall/source/target mappers)
        if self.label_mapper:
            d = self.label_mapper.to_dict()
            if 'source_mapping' in d and d['source_mapping']:
                output['source_mapping'] = d['source_mapping']
        
        # Priority 2: List in parameters (fallback if no explicit mapping)
        if 'source_mapping' not in output and self.parameters.source_neurons:
            groups = self.parameters.get_source_groups()
            labels = generate_smart_labels(groups, self.parameters.source_labels, suffix="_source")
            
            source_data = {'custom_label': labels}
            
            # If auto_type_mapping is enabled, resolve types per dataset
            if auto_mapper:
                for ds in dataset_names:
                    resolved_groups = []
                    for group in groups:
                        if isinstance(group, list):
                            resolved_group = resolve_types_for_dataset(group, ds)
                        else:
                            resolved_group = resolve_types_for_dataset([group], ds)
                        if resolved_group:  # Only add non-empty groups
                            resolved_groups.append(resolved_group)
                    source_data[ds] = resolved_groups if resolved_groups else groups
            else:
                for ds in dataset_names:
                    source_data[ds] = groups
                    
            output['source_mapping'] = source_data

        # --- Target Mapping ---
        # Priority 1: Analyzer's label_mapper
        if self.label_mapper:
            d = self.label_mapper.to_dict()
            if 'target_mapping' in d and d['target_mapping']:
                output['target_mapping'] = d['target_mapping']
                
        # Priority 2: List in parameters (fallback)
        if 'target_mapping' not in output and self.parameters.target_neurons:
            groups = self.parameters.get_target_groups()
            labels = generate_smart_labels(groups, self.parameters.target_labels, suffix="_target")
            
            target_data = {'custom_label': labels}
            
            # If auto_type_mapping is enabled, resolve types per dataset
            if auto_mapper:
                for ds in dataset_names:
                    resolved_groups = []
                    for group in groups:
                        if isinstance(group, list):
                            resolved_group = resolve_types_for_dataset(group, ds)
                        else:
                            resolved_group = resolve_types_for_dataset([group], ds)
                        if resolved_group:  # Only add non-empty groups
                            resolved_groups.append(resolved_group)
                    target_data[ds] = resolved_groups if resolved_groups else groups
            else:
                for ds in dataset_names:
                    target_data[ds] = groups
                    
            output['target_mapping'] = target_data

        # --- Intermediate Mapping ---
        if self.label_mapper:
            d = self.label_mapper.to_dict()
            if 'intermediate_mapping' in d:
                output['intermediate_mapping'] = d['intermediate_mapping']
        
        # Add auto_type_mapping metadata
        output['metadata'] = {
            'auto_type_mapping': self.parameters.auto_type_mapping,
            'description': 'Type mappings per dataset. When auto_type_mapping=True, types are resolved to their dataset-specific equivalents.'
        }

        if output:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2)

    def _export_cross_dataset_comparisons(self, comparison_results_dir: str):
        """
        Export cross-dataset comparison results at each threshold level.
        
        Compares connections across datasets at the same synapse count cutoff.
        """
        dataset_names = self.parameters.get_dataset_names()
        
        # 1. Path count comparison across datasets
        path_counts = []
        for dataset in dataset_names:
            for threshold in self.parameters.thresholds:
                df = self.raw_results.get(dataset, {}).get(threshold, pd.DataFrame())
                count = len(df) if not df.empty else 0
                total_weight = df['weight'].sum() if 'weight' in df.columns else 0
                path_counts.append({
                    'dataset': dataset,
                    'threshold': threshold,
                    'connection_count': count,
                    'total_weight': total_weight,
                    'avg_weight': total_weight / count if count > 0 else 0
                })
        
        if path_counts:
            path_count_df = pd.DataFrame(path_counts)
            self._save_csv(path_count_df, os.path.join(comparison_results_dir, "path_count_comparison.csv"))
            self._log("Saved: path_count_comparison.csv")
        
        # 2. Common and unique connections at each threshold
        # To-Do List 5 Item 2: Remove redundant files
        # - unique_to_*.csv per threshold → merged into unique_to_{dataset}.csv 
        # - common_connections_*.csv → redundant with edge_presence_matrix
        # - conserved_strong_connections_*.csv → redundant with edge_presence_matrix
        # - aligned_data/ folder → removed
        
        # Collect all motif data for unified export (To-Do List 5 Item 6)
        all_motif_data = []
        
        # Use progress bar for threshold exports
        threshold_iter = tqdm(
            self.parameters.thresholds,
            desc="  Exporting matrices",
            unit="thr",
            leave=False
        ) if self.verbose else self.parameters.thresholds
        
        for threshold in threshold_iter:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            # Export presence matrices directly to comparison_results/ (no output_cutoffs subfolder)
            self._export_presence_matrix(comparison_results_dir, threshold, silent=True)
            
            # Export path presence matrix (multi-hop paths)
            self._export_path_presence_matrix(comparison_results_dir, threshold, silent=True)
            
            # Collect motif analysis data (for unified export)
            motif_data = self._export_motif_analysis(comparison_results_dir, threshold)
            if motif_data:
                all_motif_data.extend(motif_data)
        
        # Log summary after loop
        self._log(f"Exported edge/path presence matrices for {len(self.parameters.thresholds)} thresholds")
        
        # To-Do List 5 Item 6: Save unified motif_analysis.csv with all thresholds
        if all_motif_data:
            motif_df = pd.DataFrame(all_motif_data)
            # Reorder columns: dataset, threshold first
            cols = ['dataset', 'threshold'] + [c for c in motif_df.columns if c not in ['dataset', 'threshold']]
            motif_df = motif_df[cols]
            self._save_csv(motif_df, os.path.join(comparison_results_dir, "motif_analysis.csv"))
            self._log(f"Saved: motif_analysis.csv (unified, {len(motif_df)} rows)")
        
        # 3. Edge weight comparison matrix - includes all datasets, thresholds, with presence/difference cols
        # Use vectorized operations for performance
        all_threshold_dfs = []
        
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            available = [d for d in dataset_names if d in aligned.columns]
            safe_names = {d: self.parameters._sanitize_name(d) for d in available}
            
            # Build threshold dataframe using vectorized operations
            # Ensure index is a flat string index (not MultiIndex)
            if isinstance(aligned.index, pd.MultiIndex):
                edge_keys = pd.Series([f"{idx[0]} -> {idx[1]}" for idx in aligned.index], index=aligned.index)
            else:
                edge_keys = aligned.index.astype(str)
            
            # Parse edge keys vectorized with defensive handling
            try:
                split_keys = edge_keys.str.split(' -> ', n=1, expand=True)
                if isinstance(split_keys, pd.DataFrame):
                    source_col = split_keys[0].fillna(edge_keys)
                    target_col = split_keys[1].fillna('') if 1 in split_keys.columns else pd.Series('', index=aligned.index)
                else:
                    source_col = edge_keys
                    target_col = pd.Series('', index=aligned.index)
            except Exception:
                source_col = edge_keys
                target_col = pd.Series('', index=aligned.index)
            
            threshold_df = pd.DataFrame({
                'edge_key': edge_keys,
                'source': source_col.values,
                'target': target_col.values,
                'threshold': threshold,
            }, index=aligned.index)
            
            # Add weight columns for each dataset
            for dataset in available:
                safe_name = safe_names[dataset]
                threshold_df[f'weight_{safe_name}'] = aligned[dataset]
            
            # Calculate presence count vectorized
            threshold_df['presence_count'] = (aligned[available] > 0).sum(axis=1)
            threshold_df['total_datasets'] = len(available)
            
            # Calculate statistics vectorized
            weight_values = aligned[available].copy()
            weight_values = weight_values.replace(0, np.nan)  # Exclude zeros from stats
            
            threshold_df['max_weight'] = weight_values.max(axis=1).fillna(0)
            threshold_df['avg_weight'] = weight_values.mean(axis=1).round(2).fillna(0)
            
            # Weight diff and ratio (only meaningful when >1 dataset has the edge)
            has_multiple = weight_values.notna().sum(axis=1) > 1
            max_vals = weight_values.max(axis=1)
            min_vals = weight_values.min(axis=1)
            
            threshold_df['weight_diff'] = (max_vals - min_vals).where(has_multiple, 0).fillna(0)
            
            # Weight ratio - avoid division by zero
            ratio = (max_vals / min_vals).round(2)
            threshold_df['weight_ratio'] = ratio.where(has_multiple & (min_vals > 0), '')
            # Handle cases where min is 0 but has_multiple is True
            threshold_df.loc[has_multiple & (min_vals == 0), 'weight_ratio'] = ''
            threshold_df.loc[~has_multiple, 'weight_ratio'] = 1.0
            
            all_threshold_dfs.append(threshold_df.reset_index(drop=True))
        
        if all_threshold_dfs:
            edge_weight_df = pd.concat(all_threshold_dfs, ignore_index=True)
            # Order columns logically
            col_order = ['edge_key', 'source', 'target', 'threshold', 'presence_count', 'total_datasets']
            # Add weight columns
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                weight_col = f'weight_{safe_name}'
                if weight_col in edge_weight_df.columns:
                    col_order.append(weight_col)
            col_order.extend(['max_weight', 'avg_weight', 'weight_diff', 'weight_ratio'])
            col_order = [c for c in col_order if c in edge_weight_df.columns]
            edge_weight_df = edge_weight_df[col_order]
            
            self._save_csv(edge_weight_df, os.path.join(comparison_results_dir, "edge_weight_comparison.csv"))
            self._log(f"Saved: edge_weight_comparison.csv ({len(edge_weight_df)} edges)")
        
        # 4. Top edges comparison
        self._export_top_edges_comparison(comparison_results_dir)
        
        # 5. Degree distribution analysis
        self._export_degree_distribution(comparison_results_dir)
        
        # 6. Dataset metadata comparison
        self._export_metadata_comparison(comparison_results_dir)
        
        # 7. Unified summary CSVs (merged across thresholds)
        self._export_unified_summary(comparison_results_dir)
        
        # 8. Source/target neuron counts comparison
        self._export_neuron_counts_comparison(comparison_results_dir)
    
    def _export_intra_dataset_comparisons(self, comparison_results_dir: str):
        """
        Export intra-dataset threshold sensitivity analysis.
        
        Compares how connections change across different threshold levels within each dataset.
        """
        dataset_names = self.parameters.get_dataset_names()
        
        sensitivity_data = []
        
        for dataset in dataset_names:
            prev_edges = None
            prev_threshold = None
            
            for threshold in self.parameters.thresholds:
                df = self.raw_results.get(dataset, {}).get(threshold, pd.DataFrame())
                
                if df.empty:
                    sensitivity_data.append({
                        'dataset': dataset,
                        'threshold': threshold,
                        'edge_count': 0,
                        'edges_retained_from_prev': None,
                        'retention_rate': None,
                        'edges_lost': None,
                    })
                    prev_edges = set()
                    prev_threshold = threshold
                    continue
                
                # Create edge identifiers
                if 'type_pre' in df.columns and 'type_post' in df.columns:
                    current_edges = set(zip(df['type_pre'], df['type_post']))
                elif 'bodyId_pre' in df.columns and 'bodyId_post' in df.columns:
                    current_edges = set(zip(df['bodyId_pre'], df['bodyId_post']))
                else:
                    current_edges = set(range(len(df)))
                
                edge_count = len(current_edges)
                
                if prev_edges is not None:
                    retained = len(current_edges & prev_edges)
                    lost = len(prev_edges - current_edges)
                    retention_rate = retained / len(prev_edges) if prev_edges else None
                else:
                    retained = None
                    lost = None
                    retention_rate = None
                
                sensitivity_data.append({
                    'dataset': dataset,
                    'threshold': threshold,
                    'edge_count': edge_count,
                    'edges_retained_from_prev': retained,
                    'retention_rate': retention_rate,
                    'edges_lost': lost,
                })
                
                prev_edges = current_edges
                prev_threshold = threshold
        
        if sensitivity_data:
            sensitivity_df = pd.DataFrame(sensitivity_data)
            self._save_csv(sensitivity_df, os.path.join(comparison_results_dir, "threshold_sensitivity.csv"))
            self._log("Saved: threshold_sensitivity.csv")
    
    def _export_top_edges_comparison(self, comparison_results_dir: str):
        """Export report tables capped by the top_edges parameter.

        This report-row cap is independent of graph discovery and the
        per-visualization drawn-edge limit.
        """
        dataset_names = self.parameters.get_dataset_names()
        top_n = self.parameters.top_edges
        
        # Use the middle threshold for top edges analysis
        mid_threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]
        aligned = self.get_aligned_data(mid_threshold)
        
        if aligned.empty:
            return
        
        # Get top edges per dataset
        top_edges = self.metrics.get_top_edges_per_dataset(aligned, dataset_names, top_n)
        if not top_edges.empty:
            self._save_csv(top_edges, os.path.join(comparison_results_dir, "top_edges_comparison.csv"))
            self._log("Saved: top_edges_comparison.csv")
        
        # Get overlap statistics
        overlap = self.metrics.compare_top_edges_overlap(aligned, dataset_names, top_n)
        if not overlap.empty:
            self._save_csv(overlap, os.path.join(comparison_results_dir, "top_edges_overlap.csv"))
            self._log("Saved: top_edges_overlap.csv")
    
    def _export_degree_distribution(self, comparison_results_dir: str):
        """
        Export degree distribution analysis to CSV.
        
        To-Do List 5 Item 5: Renamed files to degree_out and degree_in, 
        added threshold indication and merged all thresholds into unified files.
        """
        dataset_names = self.parameters.get_dataset_names()
        
        # Collect degree data for ALL thresholds (To-Do List 5 Item 5)
        all_out_degree = []
        all_in_degree = []
        
        for threshold in self.parameters.thresholds:
            degree_data = self.metrics.calculate_degree_distribution(
                self.raw_results, dataset_names, threshold
            )
            
            # Add threshold column to each row
            out_degree = degree_data.get('out_degree', pd.DataFrame())
            if not out_degree.empty:
                out_degree = out_degree.copy()
                out_degree['threshold'] = threshold
                all_out_degree.append(out_degree)
            
            in_degree = degree_data.get('in_degree', pd.DataFrame())
            if not in_degree.empty:
                in_degree = in_degree.copy()
                in_degree['threshold'] = threshold
                all_in_degree.append(in_degree)
        
        # Save unified out-degree data (renamed from out_degree_distribution.csv)
        if all_out_degree:
            unified_out = pd.concat(all_out_degree, ignore_index=True)
            # Reorder columns: threshold first
            cols = ['threshold'] + [c for c in unified_out.columns if c != 'threshold']
            unified_out = unified_out[cols]
            self._save_csv(unified_out, os.path.join(comparison_results_dir, "degree_out.csv"))
            self._log("Saved: degree_out.csv (unified across thresholds)")
        
        # Save unified in-degree data (renamed from in_degree_distribution.csv)
        if all_in_degree:
            unified_in = pd.concat(all_in_degree, ignore_index=True)
            # Reorder columns: threshold first
            cols = ['threshold'] + [c for c in unified_in.columns if c != 'threshold']
            unified_in = unified_in[cols]
            self._save_csv(unified_in, os.path.join(comparison_results_dir, "degree_in.csv"))
            self._log("Saved: degree_in.csv (unified across thresholds)")
        
        # Save degree statistics summary for ALL thresholds
        all_degree_stats = []
        
        for threshold in self.parameters.thresholds:
            degree_data = self.metrics.calculate_degree_distribution(
                self.raw_results, dataset_names, threshold
            )
            
            if degree_data.get('out_degree', pd.DataFrame()).empty and degree_data.get('in_degree', pd.DataFrame()).empty:
                continue
            
            degree_stats = self.metrics.calculate_degree_statistics(degree_data)
            if not degree_stats.empty:
                degree_stats['threshold'] = threshold
                all_degree_stats.append(degree_stats)
        
        if all_degree_stats:
            unified_stats = pd.concat(all_degree_stats, ignore_index=True)
            # Reorder columns: threshold first
            cols = ['threshold'] + [c for c in unified_stats.columns if c != 'threshold']
            unified_stats = unified_stats[cols]
            self._save_csv(unified_stats, os.path.join(comparison_results_dir, "degree_statistics.csv"))
            self._log("Saved: degree_statistics.csv (all thresholds)")
    
    def _export_metadata_comparison(self, comparison_results_dir: str):
        """Export dataset metadata comparison to CSV."""
        try:
            # Collect metadata (uses cache if available)
            self.collect_dataset_metadata(force_refresh=False)
            
            # Generate comparison table
            metadata_df = self.generate_metadata_comparison_table()
            if not metadata_df.empty:
                # Save to comparison_results folder
                self._save_csv(metadata_df, os.path.join(comparison_results_dir, "dataset_metadata_comparison.csv"))
                self._log("Saved: dataset_metadata_comparison.csv")
                
                # Also save to the main output folder
                out_dir = os.path.dirname(comparison_results_dir)
                self._save_csv(metadata_df, os.path.join(out_dir, "dataset_metadata_comparison.csv"))
        except Exception as e:
            self._log(f"Warning: Failed to export metadata comparison: {e}")
    
    def _export_unified_summary(self, comparison_results_dir: str):
        """
        Export unified summary CSVs that merge data across all thresholds.
        
        Creates:
        1. unified_edge_comparison.csv - All edges across all thresholds with presence/weights
        2. unified_summary.csv - High-level summary per dataset per threshold
        
        This reduces the number of output files and provides a comprehensive view.
        """
        dataset_names = self.parameters.get_dataset_names()
        
        # 1. Unified edge comparison across all thresholds
        unified_edges = []
        
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            available = [d for d in dataset_names if d in aligned.columns]
            
            for edge_key, row in aligned.iterrows():
                if ' -> ' in str(edge_key):
                    parts = str(edge_key).split(' -> ')
                    source = parts[0]
                    target = parts[1] if len(parts) > 1 else ''
                else:
                    source = str(edge_key)
                    target = ''
                
                edge_data = {
                    'edge_key': edge_key,
                    'source': source,
                    'target': target,
                    'threshold': threshold,
                }
                
                # Add weights per dataset
                conservation_count = 0
                for dataset in available:
                    safe_name = self.parameters._sanitize_name(dataset)
                    weight = row[dataset] if dataset in row else 0
                    edge_data[f'{safe_name}_weight'] = weight
                    edge_data[f'{safe_name}_present'] = True if weight > 0 else 0
                    if weight > 0:
                        conservation_count += 1
                
                edge_data['conservation'] = f"{conservation_count}/{len(available)}"
                unified_edges.append(edge_data)
        
        if unified_edges:
            unified_df = pd.DataFrame(unified_edges)
            self._save_csv(unified_df, os.path.join(comparison_results_dir, "unified_edge_comparison.csv"))
            self._log(f"Saved: unified_edge_comparison.csv ({len(unified_df)} entries)")
        
        # 2. Unified summary per dataset per threshold
        summary_data = []
        
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            
            for threshold in self.parameters.thresholds:
                df = self.raw_results.get(dataset, {}).get(threshold, pd.DataFrame())
                
                if df.empty:
                    summary_data.append({
                        'dataset': safe_name,
                        'threshold': threshold,
                        'total_edges': 0,
                        'total_weight': 0,
                        'mean_weight': 0,
                        'unique_sources': 0,
                        'unique_targets': 0,
                    })
                    continue
                
                # Extract source/target columns
                if 'type_pre' in df.columns and 'type_post' in df.columns:
                    sources = df['type_pre'].nunique()
                    targets = df['type_post'].nunique()
                elif 'source' in df.columns and 'target' in df.columns:
                    sources = df['source'].nunique()
                    targets = df['target'].nunique()
                else:
                    sources = 0
                    targets = 0
                
                weight_col = 'weight' if 'weight' in df.columns else None
                total_weight = df[weight_col].sum() if weight_col else len(df)
                mean_weight = df[weight_col].mean() if weight_col else 1
                
                summary_data.append({
                    'dataset': safe_name,
                    'threshold': threshold,
                    'total_edges': len(df),
                    'total_weight': round(total_weight, 2),
                    'mean_weight': round(mean_weight, 2),
                    'unique_sources': sources,
                    'unique_targets': targets,
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            self._save_csv(summary_df, os.path.join(comparison_results_dir, "unified_summary.csv"))
            self._log(f"Saved: unified_summary.csv ({len(summary_df)} entries)")
        
        # 3. Export unified presence matrix with all thresholds as columns
        self._export_unified_presence_matrix(comparison_results_dir)
        
        # 4. Export merged unique connections per dataset
        self._export_merged_unique_connections(comparison_results_dir)
        
        # 5. To-Do List 5 Item 1: Export unified path presence matrix with all thresholds
        self._export_unified_path_presence_matrix(comparison_results_dir)

    def _export_unified_presence_matrix(self, comparison_results_dir: str):
        """
        Export a unified edge presence matrix with all thresholds expanded horizontally.
        
        Creates edge_presence_matrix.csv with columns:
        - edge_key, source, target
        - For each dataset and threshold: presence marker (✔️/❌)
        - For each dataset and threshold: weight
        - Conservation summary
        
        This provides a single-file view of how edges are affected by threshold changes.
        """
        dataset_names = self.parameters.get_dataset_names()
        thresholds = self.parameters.thresholds
        
        # Collect all unique edges across all thresholds
        all_edges = {}
        
        for threshold in thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            available = [d for d in dataset_names if d in aligned.columns]
            
            for edge_key, row in aligned.iterrows():
                if edge_key not in all_edges:
                    # Parse source/target from edge key
                    if ' -> ' in str(edge_key):
                        parts = str(edge_key).split(' -> ')
                        source = parts[0]
                        target = parts[1] if len(parts) > 1 else ''
                    else:
                        source = str(edge_key)
                        target = ''
                    
                    all_edges[edge_key] = {
                        'edge_key': edge_key,
                        'source': source,
                        'target': target,
                    }
                
                # Add threshold-specific presence and weight for each dataset
                for dataset in available:
                    safe_name = self.parameters._sanitize_name(dataset)
                    weight = row[dataset] if dataset in row else 0
                    
                    # Presence marker: dataset_threshold (True/0 for CSV readability)
                    pres_col = f'{safe_name}_t{threshold}'
                    all_edges[edge_key][pres_col] = True if weight > 0 else 0
                    
                    # Weight column: weight_dataset_threshold
                    weight_col = f'w_{safe_name}_t{threshold}'
                    all_edges[edge_key][weight_col] = weight if weight > 0 else ''
        
        if not all_edges:
            return
        
        # Build DataFrame
        rows = list(all_edges.values())
        presence_df = pd.DataFrame(rows)
        
        # Add summary columns - count thresholds where edge is present per dataset
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            # Count how many thresholds this edge appears in for this dataset
            presence_cols = [f'{safe_name}_t{t}' for t in thresholds if f'{safe_name}_t{t}' in presence_df.columns]
            if presence_cols:
                presence_df[f'{safe_name}_count'] = presence_df[presence_cols].apply(
                    lambda x: sum(1 for v in x if v == True), axis=1
                )
        
        # Add total conservation count (edge present in any dataset at lowest threshold)
        lowest_threshold = min(thresholds)
        available = [d for d in dataset_names if f'{self.parameters._sanitize_name(d)}_t{lowest_threshold}' in presence_df.columns]
        presence_cols = [f'{self.parameters._sanitize_name(d)}_t{lowest_threshold}' for d in available]
        if presence_cols:
            presence_df['conserved_at_lowest'] = presence_df[presence_cols].apply(
                lambda x: sum(1 for v in x if v == True), axis=1
            )
        
        # Reorder columns: edge info first, then by threshold
        col_order = ['edge_key', 'source', 'target']
        
        # Add presence columns grouped by threshold
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                pres_col = f'{safe_name}_t{threshold}'
                if pres_col in presence_df.columns:
                    col_order.append(pres_col)
        
        # Add weight columns grouped by threshold
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                weight_col = f'w_{safe_name}_t{threshold}'
                if weight_col in presence_df.columns:
                    col_order.append(weight_col)
        
        # Add summary columns
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            count_col = f'{safe_name}_count'
            if count_col in presence_df.columns:
                col_order.append(count_col)
        
        if 'conserved_at_lowest' in presence_df.columns:
            col_order.append('conserved_at_lowest')
        
        # Filter and reorder
        col_order = [c for c in col_order if c in presence_df.columns]
        presence_df = presence_df[col_order]
        
        # Sort by conservation at lowest threshold, then alphabetically
        if 'conserved_at_lowest' in presence_df.columns:
            presence_df = presence_df.sort_values(['conserved_at_lowest', 'edge_key'], ascending=[False, True])
        
        # Save
        self._save_csv(presence_df, os.path.join(comparison_results_dir, "edge_presence_matrix.csv"))
        self._log(f"Saved: edge_presence_matrix.csv (unified, {len(presence_df)} edges, {len(thresholds)} thresholds)")

    def _export_merged_unique_connections(self, comparison_results_dir: str):
        """
        Export merged unique connections - one file per dataset with all thresholds.
        
        Instead of separate unique_to_{dataset}_minsyn_{threshold}.csv files,
        creates a single unique_to_{dataset}.csv with a threshold column.
        
        Columns include:
        - edge_key, source, target
        - threshold (synapse cutoff level)
        - weight
        - Additional context columns
        """
        dataset_names = self.parameters.get_dataset_names()
        
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            merged_unique = []
            
            for threshold in self.parameters.thresholds:
                unique = self.get_unique_connections(threshold)
                if safe_name not in unique and dataset not in unique:
                    continue
                
                # Get unique_df - check safe_name first, then dataset name
                # Cannot use `or` with DataFrames due to ambiguity
                if safe_name in unique:
                    unique_df = unique[safe_name]
                elif dataset in unique:
                    unique_df = unique[dataset]
                else:
                    unique_df = pd.DataFrame()
                
                if unique_df.empty:
                    continue
                
                for _, row in unique_df.iterrows():
                    # Get edge info from row or index
                    edge_key = row.get('edge_key', row.name if hasattr(row, 'name') else '')
                    if not edge_key:
                        # Try to construct from source/target
                        source = row.get('source_type', row.get('source', row.get('type_pre', '')))
                        target = row.get('target_type', row.get('target', row.get('type_post', '')))
                        edge_key = f"{source} -> {target}"
                    else:
                        # Parse from edge_key
                        if ' -> ' in str(edge_key):
                            parts = str(edge_key).split(' -> ')
                            source = parts[0]
                            target = parts[1] if len(parts) > 1 else ''
                        else:
                            source = str(edge_key)
                            target = ''
                    
                    weight = row.get('weight', row.get(safe_name, row.get(dataset, 0)))
                    
                    merged_unique.append({
                        'edge_key': edge_key,
                        'source': source,
                        'target': target,
                        'threshold': threshold,
                        'weight': weight,
                        'unique_to': safe_name,
                    })
            
            if merged_unique:
                merged_df = pd.DataFrame(merged_unique)
                merged_df = merged_df.sort_values(['threshold', 'weight'], ascending=[True, False])
                self._save_csv(merged_df, os.path.join(comparison_results_dir, f"unique_to_{safe_name}.csv"))
                self._log(f"Saved: unique_to_{safe_name}.csv ({len(merged_df)} unique edges)")

    def _export_neuron_counts_comparison(self, comparison_results_dir: str):
        """
        Export source/target neuron counts comparison across datasets.
        
        Creates:
        1. neuron_counts_summary.csv - Total counts per dataset (source/target)
        2. neuron_counts_by_type.csv - Count per neuron type per dataset
        3. neuron_counts_by_group.csv - Count per custom group per dataset (if custom groups exist)
        
        Data is loaded from source_neurons.csv and target_neurons.csv saved by FindAllPath.
        """
        dataset_names = self.parameters.get_dataset_names()
        lowest_threshold = min(self.parameters.thresholds)
        
        # Collect neuron data from each dataset
        all_source_data = []
        all_target_data = []
        summary_data = []
        type_counts = {}  # type -> {dataset: count}
        group_counts = {}  # group -> {dataset: count}

        def _count_hemisphere(df: pd.DataFrame) -> Dict[str, int]:
            if df is None or df.empty:
                return {'L': 0, 'R': 0, 'U': 0}
            hemi_col = None
            for col in ['hemisphere', 'hemisphere_code', 'hemisphere_label', 'Soma side', 'soma_side']:
                if col in df.columns:
                    hemi_col = col
                    break
            counts = {'L': 0, 'R': 0, 'U': 0}
            if hemi_col:
                vals = df[hemi_col].astype(str).str.strip().str.upper()
                counts['L'] = int((vals == 'L').sum())
                counts['R'] = int((vals == 'R').sum())
                counts['U'] = int((~vals.isin(['L', 'R'])).sum())
                return counts
            if 'type' in df.columns:
                types = df['type'].astype(str)
                counts['L'] = int(types.str.endswith('_L').sum())
                counts['R'] = int(types.str.endswith('_R').sum())
                counts['U'] = int(types.str.endswith('_U').sum())
            return counts
        
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            
            # Load from the lowest threshold output (source/target neurons are the same across thresholds)
            dataset_output_path = self.parameters.get_dataset_output_path(dataset, lowest_threshold)
            source_file = os.path.join(dataset_output_path, 'data_details', 'source_neurons.csv')
            target_file = os.path.join(dataset_output_path, 'data_details', 'target_neurons.csv')
            
            source_count = 0
            target_count = 0
            source_df = None
            target_df = None
            source_hemi = {'L': 0, 'R': 0, 'U': 0}
            target_hemi = {'L': 0, 'R': 0, 'U': 0}
            
            # Load source neurons
            if os.path.exists(source_file):
                try:
                    source_df = self._read_csv(source_file, dtype={'bodyId': str})
                    source_count = len(source_df)
                    source_hemi = _count_hemisphere(source_df)
                    
                    # Count by type
                    if 'type' in source_df.columns:
                        for type_val in source_df['type'].dropna().unique():
                            type_cnt = len(source_df[source_df['type'] == type_val])
                            if type_val not in type_counts:
                                type_counts[type_val] = {'role': 'source'}
                            type_counts[type_val][f'{safe_name}_source'] = type_cnt
                    
                    # Count by custom group
                    if 'custom_group' in source_df.columns:
                        for group_val in source_df['custom_group'].dropna().unique():
                            group_cnt = len(source_df[source_df['custom_group'] == group_val])
                            if group_val not in group_counts:
                                group_counts[group_val] = {'role': 'source'}
                            group_counts[group_val][f'{safe_name}_source'] = group_cnt
                            
                except Exception as e:
                    self._log(f"Warning: Could not load source neurons for {dataset}: {e}")
            
            # Load target neurons
            if os.path.exists(target_file):
                try:
                    target_df = self._read_csv(target_file, dtype={'bodyId': str})
                    target_count = len(target_df)
                    target_hemi = _count_hemisphere(target_df)
                    
                    # Count by type
                    if 'type' in target_df.columns:
                        for type_val in target_df['type'].dropna().unique():
                            type_cnt = len(target_df[target_df['type'] == type_val])
                            if type_val not in type_counts:
                                type_counts[type_val] = {'role': 'target'}
                            type_counts[type_val][f'{safe_name}_target'] = type_cnt
                    
                    # Count by custom group  
                    if 'custom_group' in target_df.columns:
                        for group_val in target_df['custom_group'].dropna().unique():
                            group_cnt = len(target_df[target_df['custom_group'] == group_val])
                            if group_val not in group_counts:
                                group_counts[group_val] = {'role': 'target'}
                            group_counts[group_val][f'{safe_name}_target'] = group_cnt
                            
                except Exception as e:
                    self._log(f"Warning: Could not load target neurons for {dataset}: {e}")
            
            # Summary row
            summary_data.append({
                'dataset': safe_name,
                'source_count': source_count,
                'target_count': target_count,
                'total_neurons': source_count + target_count,
                'source_L': source_hemi['L'],
                'source_R': source_hemi['R'],
                'source_U': source_hemi['U'],
                'target_L': target_hemi['L'],
                'target_R': target_hemi['R'],
                'target_U': target_hemi['U'],
                'total_L': source_hemi['L'] + target_hemi['L'],
                'total_R': source_hemi['R'] + target_hemi['R'],
                'total_U': source_hemi['U'] + target_hemi['U'],
                'source_types': source_df['type'].nunique() if source_df is not None and 'type' in source_df.columns else 0,
                'target_types': target_df['type'].nunique() if target_df is not None and 'type' in target_df.columns else 0,
            })
        
        # Save summary CSV
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            self._save_csv(summary_df, os.path.join(comparison_results_dir, "neuron_counts_summary.csv"))
            self._log(f"Saved: neuron_counts_summary.csv ({len(summary_df)} datasets)")
        
        # Save type counts CSV (presence matrix style)
        if type_counts:
            type_rows = []
            for type_val, counts in type_counts.items():
                row = {'type': type_val}
                row.update(counts)
                type_rows.append(row)
            
            type_df = pd.DataFrame(type_rows)
            # Sort by type name
            type_df = type_df.sort_values('type')
            self._save_csv(type_df, os.path.join(comparison_results_dir, "neuron_counts_by_type.csv"))
            self._log(f"Saved: neuron_counts_by_type.csv ({len(type_df)} types)")
            
            # Store for HTML report
            self._neuron_type_counts = type_df
        
        # Save group counts CSV (if any groups exist)
        if group_counts:
            group_rows = []
            for group_val, counts in group_counts.items():
                row = {'custom_group': group_val}
                row.update(counts)
                group_rows.append(row)
            
            group_df = pd.DataFrame(group_rows)
            group_df = group_df.sort_values('custom_group')
            self._save_csv(group_df, os.path.join(comparison_results_dir, "neuron_counts_by_group.csv"))
            self._log(f"Saved: neuron_counts_by_group.csv ({len(group_df)} groups)")
            
            # Store for HTML report
            self._neuron_group_counts = group_df
        
        # Store summary for HTML report
        self._neuron_counts_summary = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()

    def _export_unified_path_presence_matrix(self, comparison_results_dir: str):
        """
        To-Do List 5 Item 1: Export unified path presence matrix with all thresholds horizontally.
        
        Similar to edge_presence_matrix.csv but for multi-hop paths, with columns:
        - path_key, source, target, hops, intermediates
        - For each dataset and threshold: presence marker (✔️/❌)
        - For each dataset and threshold: weight
        - For each dataset: hop weights as [w1, w2, ...]
        - Conservation summary
        """
        import ast
        
        dataset_names = self.parameters.get_dataset_names()
        thresholds = self.parameters.thresholds
        
        # Collect all paths across all thresholds
        all_paths = {}  # path_key -> {base_info, threshold_data}
        
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                
                # Find path data file
                dataset_output_path = self.parameters.get_dataset_output_path(dataset, threshold)
                
                path_files_to_try = [
                    os.path.join(dataset_output_path, f"minsyn_{threshold}_data_original_paths.csv"),
                ]
                
                if os.path.exists(dataset_output_path):
                    for f in os.listdir(dataset_output_path):
                        if f.endswith('_allpaths_type.csv'):
                            path_files_to_try.append(os.path.join(dataset_output_path, f))
                
                path_df = None
                for path_file in path_files_to_try:
                    if os.path.exists(path_file):
                        try:
                            path_df = self._read_csv(path_file)
                            break
                        except:
                            continue
                
                if path_df is None or path_df.empty:
                    continue
                
                # Determine format: path column vs source/target columns
                path_col = 'path' if 'path' in path_df.columns else 'path_str' if 'path_str' in path_df.columns else None
                has_source_target_format = 'source' in path_df.columns and 'target' in path_df.columns
                
                if path_col is None and not has_source_target_format:
                    continue
                
                # Extract path information
                for _, row in path_df.iterrows():
                    # Handle source/target format (e.g., from data_original_paths.csv)
                    if path_col is None and has_source_target_format:
                        source_node = str(row.get('source', ''))
                        target_node = str(row.get('target', ''))
                        if not source_node or source_node == 'nan' or not target_node or target_node == 'nan':
                            continue
                        path_nodes = [source_node, target_node]
                    else:
                        # Handle path/path_str format
                        path_str = str(row.get('path', row.get('path_str', '')))
                        if not path_str or path_str == 'nan':
                            continue
                        
                        # Parse path string
                        if '->' in path_str:
                            path_nodes = [n.strip() for n in path_str.split('->')]
                        elif path_str.startswith('['):
                            try:
                                path_nodes = ast.literal_eval(path_str)
                            except:
                                continue
                        else:
                            continue
                    
                    if len(path_nodes) < 2:
                        continue
                    
                    # Build path key using canonical names for cross-dataset merging
                    canonical_key, display_key = self._build_path_key_with_mapping(path_nodes, dataset)
                    
                    # Get canonical node names
                    canonical_nodes = [self._get_canonical_type(node, dataset) for node in path_nodes]
                    source = canonical_nodes[0]
                    target = canonical_nodes[-1]
                    intermediates = canonical_nodes[1:-1] if len(canonical_nodes) > 2 else []
                    
                    # Use canonical_key for merging
                    path_key = canonical_key
                    
                    # Initialize path data (use display names for output)
                    if path_key not in all_paths:
                        all_paths[path_key] = {
                            'path_key': display_key,
                            'source': self._get_display_type(source) if self.parameters.auto_type_mapping else source,
                            'target': self._get_display_type(target) if self.parameters.auto_type_mapping else target,
                            'hops': len(path_nodes) - 1,
                            'intermediates': ' → '.join([self._get_display_type(i) for i in intermediates]) if intermediates else '',
                        }
                    
                    # Add threshold-specific data
                    weight = row.get('min_weight', row.get('weight', 1))
                    if pd.isna(weight):
                        weight = 1
                    
                    # Presence and weight columns (True/0 for CSV readability)
                    pres_col = f'{safe_name}_t{threshold}'
                    weight_col = f'w_{safe_name}_t{threshold}'
                    all_paths[path_key][pres_col] = True
                    all_paths[path_key][weight_col] = float(weight)
                    
                    # Parse hop weights
                    weights_str = str(row.get('weights', row.get('hop_weights', '')))
                    if weights_str and weights_str != 'nan':
                        hop_weights_col = f'hop_{safe_name}_t{threshold}'
                        if weights_str.startswith('['):
                            all_paths[path_key][hop_weights_col] = weights_str
                        elif ',' in weights_str:
                            try:
                                hw = [float(w.strip()) for w in weights_str.split(',')]
                                all_paths[path_key][hop_weights_col] = f"[{', '.join(str(int(w)) for w in hw)}]"
                            except:
                                pass
        
        if not all_paths:
            self._log("No path data available for unified path presence matrix")
            return
        
        # Build DataFrame
        rows = list(all_paths.values())
        path_df = pd.DataFrame(rows)
        
        # Fill missing presence markers with 0 (absent)
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                pres_col = f'{safe_name}_t{threshold}'
                if pres_col in path_df.columns:
                    path_df[pres_col] = path_df[pres_col].fillna(0)
                else:
                    path_df[pres_col] = 0
        
        # Add conservation count at lowest threshold
        lowest_threshold = min(thresholds)
        presence_cols = [f'{self.parameters._sanitize_name(d)}_t{lowest_threshold}' for d in dataset_names]
        presence_cols = [c for c in presence_cols if c in path_df.columns]
        if presence_cols:
            path_df['conserved_at_lowest'] = path_df[presence_cols].apply(
                lambda x: sum(1 for v in x if v == True), axis=1
            )
        
        # Reorder columns
        col_order = ['path_key', 'source', 'target', 'hops', 'intermediates']
        
        # Add presence columns by threshold
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                pres_col = f'{safe_name}_t{threshold}'
                if pres_col in path_df.columns:
                    col_order.append(pres_col)
        
        # Add weight columns by threshold
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                weight_col = f'w_{safe_name}_t{threshold}'
                if weight_col in path_df.columns:
                    col_order.append(weight_col)
        
        # Add hop weight columns by threshold  
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                hop_col = f'hop_{safe_name}_t{threshold}'
                if hop_col in path_df.columns:
                    col_order.append(hop_col)
        
        if 'conserved_at_lowest' in path_df.columns:
            col_order.append('conserved_at_lowest')
        
        col_order = [c for c in col_order if c in path_df.columns]
        path_df = path_df[col_order]
        
        # Sort by conservation
        if 'conserved_at_lowest' in path_df.columns:
            path_df = path_df.sort_values(['conserved_at_lowest', 'path_key'], ascending=[False, True])
        
        # Save
        self._save_csv(path_df, os.path.join(comparison_results_dir, "path_presence_matrix.csv"))
        self._log(f"Saved: path_presence_matrix.csv (unified, {len(path_df)} paths, {len(thresholds)} thresholds)")
        
        # Update comparison report with path presence matrix for visualizations
        if self.comparison_report is not None:
            self.comparison_report['path_presence_matrix'] = path_df

    def _export_presence_matrix(self, comparison_results_dir: str, threshold: int, silent: bool = False):
        """
        Export edge and path presence matrices showing conservation across datasets.
        
        Creates unified tables with:
        - ✔️/❌ presence markers per dataset
        - Weights per dataset
        - Conservation count (number of datasets with edge)
        - Conservation metrics (CV, max_weight)
        
        Args:
            comparison_results_dir: Directory to save output files
            threshold: Weight threshold for analysis
            silent: If True, suppress per-file logging
        """
        dataset_names = self.parameters.get_dataset_names()
        aligned = self.get_aligned_data(threshold)
        
        if aligned.empty:
            self._log(f"No aligned data for presence matrix at threshold {threshold}")
            return
        
        # Get top edges union from all datasets
        top_n = self.parameters.top_edges
        
        if top_n > 0:
            top_edges_union = set()
            
            for dataset in dataset_names:
                if dataset in aligned.columns:
                    dataset_top = set(aligned.nlargest(top_n, dataset).index)
                    top_edges_union.update(dataset_top)
            
            # Limit total rows to reasonable number (2x top_edges)
            max_rows = top_n * 2
            if len(top_edges_union) > max_rows:
                # Sort by max weight across all datasets
                available = [d for d in dataset_names if d in aligned.columns]
                aligned['_max_weight'] = aligned[available].max(axis=1)
                top_edges_union = set(aligned.nlargest(max_rows, '_max_weight').index)
                aligned = aligned.drop(columns=['_max_weight'])
            
            # Filter to top edges
            matrix_df = aligned.loc[aligned.index.isin(top_edges_union)].copy()
        else:
            # Include all edges if top_edges <= 0
            matrix_df = aligned.copy()
        
        if matrix_df.empty:
            return
        
        # Build presence matrix using vectorized operations (much faster than iterrows)
        available = [d for d in dataset_names if d in matrix_df.columns]
        
        # Ensure index is a flat string index (not MultiIndex)
        if isinstance(matrix_df.index, pd.MultiIndex):
            # Convert MultiIndex to string format "source -> target"
            edge_keys = pd.Series([f"{idx[0]} -> {idx[1]}" for idx in matrix_df.index], index=matrix_df.index)
        else:
            edge_keys = matrix_df.index.astype(str)
        
        # Parse edge keys to source/target using vectorized string operations
        # Use try/except to handle edge cases where split returns unexpected types
        try:
            split_keys = edge_keys.str.split(' -> ', n=1, expand=True)
            # Ensure split_keys is a DataFrame with at least 2 columns
            if isinstance(split_keys, pd.DataFrame):
                source_types = split_keys[0].fillna(edge_keys)
                target_types = split_keys[1].fillna('') if 1 in split_keys.columns else pd.Series('', index=matrix_df.index)
            else:
                # Fallback: split didn't return DataFrame (edge case)
                source_types = edge_keys
                target_types = pd.Series('', index=matrix_df.index)
        except Exception:
            # Ultimate fallback
            source_types = edge_keys
            target_types = pd.Series('', index=matrix_df.index)
        
        # Start building the presence DataFrame
        presence_df = pd.DataFrame({
            'edge_key': edge_keys,
            'source_type': source_types.values,
            'target_type': target_types.values,
        }, index=matrix_df.index)
        
        # Build safe name mapping
        safe_names = {d: self.parameters._sanitize_name(d) for d in available}
        
        # Add presence markers and weight columns for each dataset (vectorized)
        for dataset in available:
            safe_name = safe_names[dataset]
            weights = matrix_df[dataset]
            is_present = weights > 0
            
            # Presence marker (True/0 for CSV readability)
            presence_df[safe_name] = is_present.map({True: True, False: 0})
            
            # Weight column (show weight if present, else empty string)
            presence_df[f'weight_{safe_name}'] = weights.where(is_present, '')
        
        # Calculate conservation count (number of datasets with edge > 0)
        presence_cols = [safe_names[d] for d in available]
        # Convert True/0 to 1/0 for counting
        presence_df['conservation_count'] = (matrix_df[available] > 0).sum(axis=1)
        
        # Calculate statistics (vectorized)
        weight_values = matrix_df[available].copy()
        # Replace 0 with NaN for statistics (so we only consider present edges)
        weight_values = weight_values.replace(0, np.nan)
        
        presence_df['max_weight'] = weight_values.max(axis=1)
        presence_df['avg_weight'] = weight_values.mean(axis=1)
        
        # Calculate CV only where we have more than 1 present dataset
        has_multiple = (weight_values.notna().sum(axis=1) > 1)
        cv_values = weight_values.std(axis=1) / weight_values.mean(axis=1)
        presence_df['weight_cv'] = cv_values.round(3).where(has_multiple, '')
        
        # Replace NaN with empty string for display
        presence_df['max_weight'] = presence_df['max_weight'].fillna('')
        presence_df['avg_weight'] = presence_df['avg_weight'].fillna('')
        
        if presence_df.empty:
            return
        
        # Order columns: edge_key, source, target, conservation_count, presence markers, weights, stats
        col_order = ['edge_key', 'source_type', 'target_type', 'conservation_count']
        
        # Add presence marker columns
        for dataset in available:
            safe_name = self.parameters._sanitize_name(dataset)
            if safe_name in presence_df.columns:
                col_order.append(safe_name)
        
        # Add weight columns
        for dataset in available:
            safe_name = self.parameters._sanitize_name(dataset)
            weight_col = f'weight_{safe_name}'
            if weight_col in presence_df.columns:
                col_order.append(weight_col)
        
        # Add statistics
        col_order.extend(['max_weight', 'avg_weight', 'weight_cv'])
        
        # Ensure all columns exist
        col_order = [c for c in col_order if c in presence_df.columns]
        presence_df = presence_df[col_order]
        
        # Sort by conservation count (desc) then max weight (desc)
        presence_df = presence_df.sort_values(
            ['conservation_count', 'max_weight'],
            ascending=[False, False]
        )
        
        # Save edge presence matrix
        self._save_csv(presence_df, os.path.join(comparison_results_dir, f"edge_presence_matrix_minsyn_{threshold}.csv"))
        if not silent:
            self._log(f"Saved: edge_presence_matrix_minsyn_{threshold}.csv ({len(presence_df)} edges)")
        
        # Also save a threshold-independent version at the middle threshold
        mid_threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]
        if threshold == mid_threshold:
            self._save_csv(presence_df, os.path.join(comparison_results_dir, "edge_presence_matrix.csv"))
            if not silent:
                self._log("Saved: edge_presence_matrix.csv (default)")
    
    def _export_path_presence_matrix(self, comparison_results_dir: str, threshold: int, silent: bool = False):
        """
        Export path presence matrix showing multi-hop path conservation across datasets.
        
        Similar to edge presence matrix but for source → intermediate → target paths.
        Creates unified tables with:
        - ✔️/❌ presence markers per dataset
        - Path structure (source → inter1 → inter2 → target)
        - Hop weights as [w1, w2, ...] for each dataset
        - Conservation metrics
        
        Args:
            comparison_results_dir: Directory to save output files
            threshold: Weight threshold for analysis
            silent: If True, suppress per-file logging
        """
        import ast
        
        dataset_names = self.parameters.get_dataset_names()
        
        # Limit paths to prevent hanging on large datasets
        max_paths_per_dataset = 5000  # Safety limit (reduced from 10000)
        
        # Collect paths from path CSV files (not connection data)
        path_data = {}  # path_key -> {dataset: True/False}
        path_details = {}  # path_key -> {source, intermediates, target, weights, hop_weights}
        
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            
            # Find path data file: minsyn_X_data_original_paths.csv or {source}_to_{target}_allpaths_type.csv
            dataset_output_path = self.parameters.get_dataset_output_path(dataset, threshold)
            
            # Try multiple path file patterns
            path_files_to_try = [
                os.path.join(dataset_output_path, f"minsyn_{threshold}_data_original_paths.csv"),
            ]
            
            # Also check for source_to_target_allpaths_type.csv pattern
            if os.path.exists(dataset_output_path):
                for f in os.listdir(dataset_output_path):
                    if f.endswith('_allpaths_type.csv'):
                        path_files_to_try.append(os.path.join(dataset_output_path, f))
            
            path_df = None
            for path_file in path_files_to_try:
                if os.path.exists(path_file):
                    try:
                        path_df = self._read_csv(path_file)
                        if not silent:
                            self._log(f"Loaded path data from {os.path.basename(path_file)} for {dataset}")
                        break
                    except Exception as e:
                        if not silent:
                            self._log(f"Warning: Could not read {path_file}: {e}")
            
            if path_df is None or path_df.empty:
                if not silent:
                    self._log(f"No path data found for {dataset} at threshold {threshold}")
                continue
            
            # Extract path information from DataFrame using vectorized operations where possible
            # Expected columns: 
            # Format 1 (allpaths_type.csv): path/path_str, weights, min_weight, length
            # Format 2 (data_original_paths.csv): source, target, weight, weights, layer
            
            # Determine format: check for 'path' or 'path_str' column vs 'source'+'target' columns
            path_col = 'path' if 'path' in path_df.columns else 'path_str' if 'path_str' in path_df.columns else None
            has_source_target_format = 'source' in path_df.columns and 'target' in path_df.columns
            
            if path_col is None and not has_source_target_format:
                if not silent:
                    self._log(f"Path file for {dataset} has no 'path' or 'source/target' columns")
                continue
            
            # Handle source/target format (e.g., from data_original_paths.csv)
            if path_col is None and has_source_target_format:
                # Create path strings from source/target columns
                valid_mask = path_df['source'].notna() & path_df['target'].notna()
                valid_paths = path_df[valid_mask].copy()
                
                if valid_paths.empty:
                    continue
                    
                # Limit paths to prevent hanging on large datasets
                if len(valid_paths) > max_paths_per_dataset:
                    if 'weight' in valid_paths.columns:
                        valid_paths = valid_paths.nlargest(max_paths_per_dataset, 'weight')
                    else:
                        valid_paths = valid_paths.head(max_paths_per_dataset)
                    if not silent:
                        self._log(f"  Limiting to top {max_paths_per_dataset} paths for {dataset}")
                
                # Process source-target format directly
                for idx, row in valid_paths.iterrows():
                    source_node = str(row['source'])
                    target_node = str(row['target'])
                    path_nodes = [source_node, target_node]
                    
                    # Build path key using canonical names for cross-dataset merging
                    canonical_key, display_key = self._build_path_key_with_mapping(path_nodes, dataset)
                    canonical_nodes = [self._get_canonical_type(node, dataset) for node in path_nodes]
                    source = canonical_nodes[0]
                    target = canonical_nodes[-1]
                    path_key = canonical_key
                    
                    # Initialize path data if not exists
                    if path_key not in path_data:
                        path_data[path_key] = {'_display_key': display_key}
                        path_details[path_key] = {
                            'source': source,
                            'target': target,
                            'intermediates': [],
                            'weights': {},
                            'hop_weights': {}
                        }
                    
                    # Mark as present
                    path_data[path_key][safe_name] = True
                    
                    # Parse hop weights from weights column
                    hop_weights_list = []
                    weights_str = str(row.get('weights', ''))
                    if weights_str and weights_str != 'nan':
                        if weights_str.startswith('['):
                            try:
                                hop_weights_list = ast.literal_eval(weights_str)
                            except:
                                pass
                        elif ',' in weights_str:
                            try:
                                hop_weights_list = [float(w.strip()) for w in weights_str.split(',')]
                            except:
                                pass
                    
                    if hop_weights_list:
                        path_details[path_key]['hop_weights'][safe_name] = hop_weights_list
                    
                    # Get weight value
                    weight = row.get('weight', 1)
                    if pd.isna(weight):
                        weight = hop_weights_list[0] if hop_weights_list else 1
                    
                    if safe_name not in path_details[path_key]['weights']:
                        path_details[path_key]['weights'][safe_name] = []
                    path_details[path_key]['weights'][safe_name].append(float(weight))
                
                continue  # Done with this dataset, skip the path_col logic below
            
            # Original path_col format handling (allpaths_type.csv with 'path' or 'path_str' column)
            # Vectorized: filter out null/empty paths
            valid_mask = path_df[path_col].notna() & (path_df[path_col].astype(str) != 'nan') & (path_df[path_col].astype(str) != '')
            valid_paths = path_df[valid_mask].copy()
            
            if valid_paths.empty:
                continue
            
            # Limit paths to prevent hanging on large datasets
            if len(valid_paths) > max_paths_per_dataset:
                # Sort by min_weight if available and take top paths
                if 'min_weight' in valid_paths.columns:
                    valid_paths = valid_paths.nlargest(max_paths_per_dataset, 'min_weight')
                else:
                    valid_paths = valid_paths.head(max_paths_per_dataset)
                if not silent:
                    self._log(f"  Limiting to top {max_paths_per_dataset} paths for {dataset}")
            
            # Process paths - we need to iterate here due to complex parsing, but limit to valid rows only
            for idx, row in valid_paths.iterrows():
                path_str = str(row[path_col])
                
                # Parse path string to extract nodes
                # Handle both "A->B->C" and "['A', 'B', 'C']" formats
                if '->' in path_str:
                    path_nodes = [n.strip() for n in path_str.split('->')]
                elif path_str.startswith('['):
                    # Parse list format like "['aMe12', 'KCg-d', 'PPL101']"
                    try:
                        path_nodes = ast.literal_eval(path_str)
                    except:
                        continue
                else:
                    continue
                
                if len(path_nodes) < 2:
                    continue
                
                # Build path key using canonical names for cross-dataset merging
                # and display names for the output
                canonical_key, display_key = self._build_path_key_with_mapping(path_nodes, dataset)
                
                # Get canonical node names for source/intermediates/target
                canonical_nodes = [self._get_canonical_type(node, dataset) for node in path_nodes]
                source = canonical_nodes[0]
                target = canonical_nodes[-1]
                intermediates = canonical_nodes[1:-1] if len(canonical_nodes) > 2 else []
                
                # Use canonical_key for data merging (consistent across datasets)
                path_key = canonical_key
                
                # Initialize path data if not exists
                if path_key not in path_data:
                    path_data[path_key] = {'_display_key': display_key}
                    path_details[path_key] = {
                        'source': source,
                        'target': target,
                        'intermediates': intermediates,
                        'weights': {},
                        'hop_weights': {}  # Store individual hop weights
                    }
                
                # Mark as present
                path_data[path_key][safe_name] = True
                
                # Parse hop weights from weights column (e.g., "[10, 5]" or "10,5")
                hop_weights_list = []
                weights_str = str(row.get('weights', row.get('hop_weights', '')))
                if weights_str and weights_str != 'nan':
                    if weights_str.startswith('['):
                        try:
                            hop_weights_list = ast.literal_eval(weights_str)
                        except:
                            pass
                    elif ',' in weights_str:
                        try:
                            hop_weights_list = [float(w.strip()) for w in weights_str.split(',')]
                        except:
                            pass
                
                # Store hop weights
                if hop_weights_list:
                    path_details[path_key]['hop_weights'][safe_name] = hop_weights_list
                
                # Record min weight as the path weight
                weight = row.get('min_weight', row.get('weight', 1))
                if pd.isna(weight):
                    if hop_weights_list:
                        weight = min(hop_weights_list)
                    else:
                        weight = 1
                
                if safe_name not in path_details[path_key]['weights']:
                    path_details[path_key]['weights'][safe_name] = []
                path_details[path_key]['weights'][safe_name].append(float(weight))
        
        if not path_data:
            if not silent:
                self._log(f"No path data for path presence matrix at threshold {threshold}")
            return
        
        # Build presence matrix rows
        rows = []
        available_datasets = [self.parameters._sanitize_name(d) for d in dataset_names]
        
        for path_key, presence in path_data.items():
            details = path_details[path_key]
            
            # Use display_key for human-readable output (shows type variants)
            display_key = presence.get('_display_key', path_key)
            
            row_data = {
                'path_key': display_key,
                'source': self._get_display_type(details['source']) if self.parameters.auto_type_mapping else details['source'],
                'target': self._get_display_type(details['target']) if self.parameters.auto_type_mapping else details['target'],
                'hops': len(details['intermediates']) + 1,
                'intermediates': ' → '.join([self._get_display_type(i) for i in details['intermediates']]) if details['intermediates'] else '',
            }
            
            # Add presence markers, weights, and hop weights
            conservation_count = 0
            all_weights = []
            
            for safe_name in available_datasets:
                is_present = presence.get(safe_name, False)
                row_data[safe_name] = True if is_present else 0  # True/0 for CSV readability
                
                if is_present:
                    conservation_count += 1
                    weights = details['weights'].get(safe_name, [])
                    hop_weights = details['hop_weights'].get(safe_name, [])
                    
                    if weights:
                        avg_weight = np.mean(weights)
                        row_data[f'weight_{safe_name}'] = round(avg_weight, 2)
                        all_weights.extend(weights)
                    else:
                        row_data[f'weight_{safe_name}'] = ''
                    
                    # Add hop weights as formatted string -w1-w2- with dashes
                    if hop_weights:
                        row_data[f'hop_weights_{safe_name}'] = '-' + '-'.join(str(int(w)) for w in hop_weights) + '-'
                    else:
                        row_data[f'hop_weights_{safe_name}'] = ''
                else:
                    row_data[f'weight_{safe_name}'] = ''
                    row_data[f'hop_weights_{safe_name}'] = ''
            
            row_data['conservation_count'] = conservation_count
            
            # Calculate statistics
            if all_weights:
                row_data['max_weight'] = round(max(all_weights), 2)
                row_data['avg_weight'] = round(np.mean(all_weights), 2)
                if len(all_weights) > 1:
                    row_data['weight_cv'] = round(np.std(all_weights) / np.mean(all_weights), 3)
                else:
                    row_data['weight_cv'] = ''
            else:
                row_data['max_weight'] = ''
                row_data['avg_weight'] = ''
                row_data['weight_cv'] = ''
            
            rows.append(row_data)
        
        if not rows:
            return
        
        # Create DataFrame and sort
        path_presence_df = pd.DataFrame(rows)
        
        # Reorder columns for better readability
        col_order = ['path_key', 'source', 'target', 'hops', 'intermediates', 'conservation_count']
        
        # Add presence columns
        for safe_name in available_datasets:
            if safe_name in path_presence_df.columns:
                col_order.append(safe_name)
        
        # Add weight columns  
        for safe_name in available_datasets:
            weight_col = f'weight_{safe_name}'
            if weight_col in path_presence_df.columns:
                col_order.append(weight_col)
        
        # Add hop weights columns
        for safe_name in available_datasets:
            hop_col = f'hop_weights_{safe_name}'
            if hop_col in path_presence_df.columns:
                col_order.append(hop_col)
        
        # Add statistics
        col_order.extend(['max_weight', 'avg_weight', 'weight_cv'])
        col_order = [c for c in col_order if c in path_presence_df.columns]
        path_presence_df = path_presence_df[col_order]
        
        path_presence_df = path_presence_df.sort_values(
            ['conservation_count', 'max_weight'],
            ascending=[False, False]
        )
        
        # Limit to top paths only when a positive report-row cap is configured.
        # Zero/negative values mean include all rows, matching the edge
        # presence matrix and the top-edge metric helpers.
        if self.parameters.top_edges > 0:
            max_paths = self.parameters.top_edges * 2
            if len(path_presence_df) > max_paths:
                path_presence_df = path_presence_df.head(max_paths)
        
        # Save
        self._save_csv(path_presence_df, os.path.join(comparison_results_dir, f"path_presence_matrix_minsyn_{threshold}.csv"))
        if not silent:
            self._log(f"Saved: path_presence_matrix_minsyn_{threshold}.csv ({len(path_presence_df)} paths)")
        
        # Save default version at middle threshold
        mid_threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]
        if threshold == mid_threshold:
            self._save_csv(path_presence_df, os.path.join(comparison_results_dir, "path_presence_matrix.csv"))
            if not silent:
                self._log("Saved: path_presence_matrix.csv (default)")
    
    def _export_motif_analysis(self, comparison_results_dir: str, threshold: int):
        """
        Export network motif analysis for each dataset.
        
        Detects and compares common network motifs:
        - Feedforward loops (A→B→C, A→C)
        - Feedback loops (A→B→A)
        - Fan-in patterns (multiple inputs to one node)
        - Fan-out patterns (one node with multiple outputs)
        - Reciprocal connections
        
        Args:
            comparison_results_dir: Directory to save output files
            threshold: Weight threshold for analysis
        """
        dataset_names = self.parameters.get_dataset_names()
        motif_data = []
        
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            
            # Build graph from connection data
            df = self.raw_results.get(dataset, {}).get(threshold, pd.DataFrame())
            if df.empty:
                continue
            
            # Extract edges using vectorized operations
            source_col = 'type_pre' if 'type_pre' in df.columns else 'source'
            target_col = 'type_post' if 'type_post' in df.columns else 'target'
            
            # Filter valid rows
            valid_mask = (
                df[source_col].notna() & 
                df[target_col].notna() & 
                (df[source_col].astype(str) != 'nan') & 
                (df[target_col].astype(str) != 'nan')
            )
            valid_df = df[valid_mask]
            
            # Build edge set and adjacency structures
            sources = valid_df[source_col].astype(str).values
            targets = valid_df[target_col].astype(str).values
            
            edge_set = set(zip(sources, targets))
            nodes = set(sources) | set(targets)
            
            # Build adjacency dict for fast neighbor lookup
            out_neighbors = {}  # node -> set of outgoing neighbors
            for s, t in edge_set:
                if s not in out_neighbors:
                    out_neighbors[s] = set()
                out_neighbors[s].add(t)
            
            # Calculate motif counts
            feedforward_loops = 0
            feedback_loops = 0
            reciprocal_connections = 0
            
            # Fan-in/fan-out analysis using vectorized counting
            from collections import Counter
            out_degree = dict(Counter(sources))
            in_degree = dict(Counter(targets))
            
            # Find reciprocal connections - only count each pair once
            for (s, t) in edge_set:
                if s < t and (t, s) in edge_set:  # Only count when s < t to avoid double counting
                    reciprocal_connections += 1
                elif s == t:  # Self-loop (edge to itself)
                    pass  # Don't count self-loops as reciprocal
                elif s > t and (t, s) in edge_set:
                    pass  # Skip - already counted
            
            # Find feedforward loops (A→B→C where A→C also exists)
            # Use adjacency dict for O(E) instead of O(E*N)
            for a in out_neighbors:
                a_neighbors = out_neighbors.get(a, set())
                for b in a_neighbors:
                    b_neighbors = out_neighbors.get(b, set())
                    # Check which C nodes (neighbors of B) are also neighbors of A
                    common = a_neighbors & b_neighbors
                    feedforward_loops += len(common - {a})  # Exclude A itself
            
            # Find feedback loops (A→B→A cycles) - same as reciprocal
            feedback_loops = reciprocal_connections
            
            # Calculate hub metrics
            max_out_degree = max(out_degree.values()) if out_degree else 0
            max_in_degree = max(in_degree.values()) if in_degree else 0
            avg_out_degree = np.mean(list(out_degree.values())) if out_degree else 0
            avg_in_degree = np.mean(list(in_degree.values())) if in_degree else 0
            
            # Find fan-in/fan-out hubs
            fan_out_hubs = [n for n, d in out_degree.items() if d >= 3]  # Nodes with 3+ outputs
            fan_in_hubs = [n for n, d in in_degree.items() if d >= 3]  # Nodes with 3+ inputs
            
            motif_data.append({
                'dataset': safe_name,
                'threshold': threshold,  # To-Do List 5 Item 6: Add threshold column
                'total_nodes': len(nodes),
                'total_edges': len(edge_set),
                'feedforward_loops': feedforward_loops,
                'feedback_loops': feedback_loops,
                'reciprocal_connections': reciprocal_connections,
                'fan_out_hubs': len(fan_out_hubs),
                'fan_in_hubs': len(fan_in_hubs),
                'max_out_degree': max_out_degree,
                'max_in_degree': max_in_degree,
                'avg_out_degree': round(avg_out_degree, 2),
                'avg_in_degree': round(avg_in_degree, 2),
                'density': round(len(edge_set) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0, 4),
            })
        
        if not motif_data:
            self._log(f"No motif data available at threshold {threshold}")
            return motif_data  # Return for later merging
        
        return motif_data  # Return for merging in export_cross_dataset_comparisons

    def _generate_visualizations(self, output_dir: str):
        """
        Generate and save all matplotlib visualizations.
        
        Creates visualization PNG files in the comparison_visualizations/ subfolder at base level.
        Also generates interactive HTML heatmaps using VisualizePath.
        Uses cached similarities to avoid redundant calculations.
        
        Args:
            output_dir: Base output directory
        """
        try:
            from .visualizations import ComparisonVisualizer
        except ImportError:
            self._log("Warning: ComparisonVisualizer not available, skipping visualizations")
            return
        
        # Save visualizations to comparison_visualizations/ at base level (not inside comparison_results/)
        vis_dir = os.path.join(output_dir, "comparison_visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        mid_threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]
        aligned = self.get_aligned_data(mid_threshold)
        dataset_names = self.parameters.get_dataset_names()
        
        # Get pairwise similarities if available
        pairwise_sim = pd.DataFrame()
        if self.comparison_report and 'pairwise_similarities' in self.comparison_report:
            pairwise_sim = self.comparison_report['pairwise_similarities']
            if pairwise_sim is None:
                pairwise_sim = pd.DataFrame()
        
        # Get path presence matrix if available
        path_presence = pd.DataFrame()
        if self.comparison_report and 'path_presence_matrix' in self.comparison_report:
            path_presence = self.comparison_report['path_presence_matrix']
        
        try:
            visualizer = ComparisonVisualizer(verbose=self.verbose)
            
            # Build nickname map from parameters
            dataset_names = self.parameters.get_dataset_names()
            nicknames = self.parameters.get_dataset_nicknames()
            nickname_map = dict(zip(dataset_names, nicknames))
            
            # Get type-mapped results for proper cross-dataset comparison
            # This ensures types like MeVPaMe1 (male-cns) and MTe46 (FAFB) are recognized as the same
            mapped_results = self.get_mapped_results()
            
            # Generate all standard plots, passing cached similarity function
            visualizer.save_all_plots(
                results=mapped_results,
                aligned_data=aligned,
                similarities=pairwise_sim,
                output_dir=vis_dir,
                thresholds=self.parameters.thresholds,
                align_func=self.get_aligned_data,  # Pass function to get aligned data at any threshold
                similarity_func=self.get_cached_similarities,  # Pass cached similarity function
                current_threshold=mid_threshold,
                path_data_func=self._get_path_data_for_threshold,
                ratio_data_func=self._get_ratio_data_for_threshold,
                prob_data_func=self._get_prob_data_for_threshold,
                output_base_path=self.parameters.full_output_path,
                nickname_map=nickname_map,
                path_presence_matrix=path_presence,  # Pass path presence matrix for accurate path counts
                silent=True  # Suppress per-file messages, show summary instead
            )
            
            self._log_file(vis_dir, "Saved visualizations")
        except Exception as e:
            self._log(f"Warning: Failed to generate some visualizations: {e}")
        
        # Generate VisualizePath interactive heatmaps (no separate network files)
        self._generate_vispath_visualizations(vis_dir)

        # Generate conserved reciprocal graphs when enabled
        if getattr(self.parameters, 'find_reciprocal', False):
            try:
                self.visualize_conserved_reciprocal_graph_all_thresholds()
            except Exception as e:
                self._log(f"Warning: Failed to generate conserved reciprocal graphs: {e}")
    
    def _get_path_data_for_threshold(self, threshold: int) -> pd.DataFrame:
        """
        Get path min_weight data aligned across all datasets for a given threshold.
        
        Reads from dataset_data/{dataset}/minsyn_{threshold}/*_original_paths.csv
        or *_allpaths_type.csv and extracts min_weight values, aligning paths across datasets.
        
        Args:
            threshold: The threshold level
            
        Returns:
            DataFrame with path index and dataset columns containing min_weight
        """
        dataset_names = self.parameters.get_dataset_names()
        all_path_data = {}
        
        for dataset_name in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset_name)
            dataset_output_path = os.path.join(
                self.parameters.full_output_path,
                'dataset_data',
                safe_name,
                f'minsyn_{threshold}'
            )
            
            # Try multiple path file patterns
            path_files_to_try = [
                os.path.join(dataset_output_path, f'minsyn_{threshold}_data_original_paths.csv'),
            ]
            
            # Also check for source_to_target_allpaths_type.csv pattern
            if os.path.exists(dataset_output_path):
                for f in os.listdir(dataset_output_path):
                    if f.endswith('_allpaths_type.csv'):
                        path_files_to_try.append(os.path.join(dataset_output_path, f))
            
            # Try to read from available files
            df = None
            for path_file in path_files_to_try:
                if os.path.exists(path_file):
                    try:
                        df = self._read_csv(path_file)
                        break
                    except Exception as e:
                        self._log(f"Warning: Could not read {path_file}: {e}")
            
            if df is not None:
                # Handle two formats:
                # Format 1 (allpaths_type.csv): 'path' and 'min_weight' columns
                # Format 2 (data_original_paths.csv): 'source', 'target', and 'weight' columns
                has_path_format = 'path' in df.columns and 'min_weight' in df.columns
                has_source_target_format = 'source' in df.columns and 'target' in df.columns and 'weight' in df.columns
                
                if has_path_format:
                    for _, row in df.iterrows():
                        original_path_key = row['path']
                        min_weight = row['min_weight']
                        
                        # Apply type mapping to path key if auto_type_mapping is enabled
                        if self.parameters.auto_type_mapping and self.parameters._auto_type_mapper:
                            # Parse path nodes
                            if '->' in str(original_path_key):
                                path_nodes = [n.strip() for n in str(original_path_key).split('->')]
                            elif ' → ' in str(original_path_key):
                                path_nodes = [n.strip() for n in str(original_path_key).split(' → ')]
                            else:
                                path_nodes = [str(original_path_key)]
                            
                            # Build canonical and display keys
                            canonical_key, display_key = self._build_path_key_with_mapping(path_nodes, dataset_name)
                            path_key = canonical_key  # Use canonical key for merging
                        else:
                            path_key = original_path_key
                            display_key = original_path_key
                        
                        if path_key not in all_path_data:
                            all_path_data[path_key] = {'_display_key': display_key}
                        all_path_data[path_key][dataset_name] = min_weight
                
                elif has_source_target_format:
                    # Handle source/target format from data_original_paths.csv
                    for _, row in df.iterrows():
                        source = str(row['source'])
                        target = str(row['target'])
                        weight = row['weight']
                        
                        if pd.isna(source) or pd.isna(target) or source == 'nan' or target == 'nan':
                            continue
                        
                        path_nodes = [source, target]
                        
                        # Apply type mapping if enabled
                        if self.parameters.auto_type_mapping and self.parameters._auto_type_mapper:
                            canonical_key, display_key = self._build_path_key_with_mapping(path_nodes, dataset_name)
                            path_key = canonical_key
                        else:
                            path_key = f"{source} → {target}"
                            display_key = path_key
                        
                        if path_key not in all_path_data:
                            all_path_data[path_key] = {'_display_key': display_key}
                        all_path_data[path_key][dataset_name] = weight
        
        if not all_path_data:
            return pd.DataFrame()
        
        # Build result DataFrame using display keys for index
        result_rows = []
        for canonical_key, data in all_path_data.items():
            display_key = data.pop('_display_key', canonical_key)
            row_data = {d: data.get(d, 0) for d in dataset_names}
            result_rows.append((display_key, row_data))
        
        if not result_rows:
            return pd.DataFrame()
        
        result_df = pd.DataFrame([r[1] for r in result_rows], index=[r[0] for r in result_rows])
        return result_df.fillna(0)
    
    def _get_path_hop_weights_for_threshold(self, threshold: int) -> Dict[str, Dict[str, List[float]]]:
        """
        Get hop weights for each path across all datasets for a given threshold.
        
        Returns:
            Dict mapping path_key -> {safe_dataset_name: [hop_weight1, hop_weight2, ...]}
        """
        import ast
        dataset_names = self.parameters.get_dataset_names()
        all_hop_weights = {}  # path_key -> {safe_name: [weights]}
        
        for dataset_name in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset_name)
            dataset_output_path = os.path.join(
                self.parameters.full_output_path,
                'dataset_data',
                safe_name,
                f'minsyn_{threshold}'
            )
            
            # Try multiple path file patterns
            path_files_to_try = [
                os.path.join(dataset_output_path, f'minsyn_{threshold}_data_original_paths.csv'),
            ]
            
            # Also check for source_to_target_allpaths_type.csv pattern
            if os.path.exists(dataset_output_path):
                for f in os.listdir(dataset_output_path):
                    if f.endswith('_allpaths_type.csv'):
                        path_files_to_try.append(os.path.join(dataset_output_path, f))
            
            # Try to read from available files
            df = None
            for path_file in path_files_to_try:
                if os.path.exists(path_file):
                    try:
                        df = self._read_csv(path_file)
                        break
                    except Exception as e:
                        self._log(f"Warning: Could not read {path_file}: {e}")
            
            if df is not None and 'path' in df.columns and 'weights' in df.columns:
                for _, row in df.iterrows():
                    original_path_key = row['path']
                    weights_str = str(row.get('weights', ''))
                    
                    # Parse weights column
                    hop_weights = []
                    if weights_str and weights_str != 'nan':
                        if weights_str.startswith('['):
                            try:
                                hop_weights = ast.literal_eval(weights_str)
                            except:
                                pass
                        elif ',' in weights_str:
                            try:
                                hop_weights = [float(w.strip()) for w in weights_str.split(',')]
                            except:
                                pass
                    
                    if hop_weights:
                        # Apply type mapping to path key if auto_type_mapping is enabled
                        if self.parameters.auto_type_mapping and self.parameters._auto_type_mapper:
                            # Parse path nodes
                            if '->' in str(original_path_key):
                                path_nodes = [n.strip() for n in str(original_path_key).split('->')]
                            elif ' → ' in str(original_path_key):
                                path_nodes = [n.strip() for n in str(original_path_key).split(' → ')]
                            else:
                                path_nodes = [str(original_path_key)]
                            
                            # Build canonical key for merging
                            canonical_key, _ = self._build_path_key_with_mapping(path_nodes, dataset_name)
                            path_key = canonical_key
                        else:
                            path_key = original_path_key
                        
                        if path_key not in all_hop_weights:
                            all_hop_weights[path_key] = {}
                        all_hop_weights[path_key][safe_name] = hop_weights
        
        return all_hop_weights
    
    def _get_ratio_data_for_threshold(self, threshold: int) -> pd.DataFrame:
        """
        Get min_ratio data aligned across all datasets for a given threshold.
        
        Reads from dataset_data/{dataset}/minsyn_{threshold}/*_original_paths.csv
        or *_allpaths_type.csv and extracts min_ratio values.
        
        Args:
            threshold: The threshold level
            
        Returns:
            DataFrame with path index and dataset columns containing min_ratio
        """
        dataset_names = self.parameters.get_dataset_names()
        all_ratio_data = {}
        
        for dataset_name in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset_name)
            dataset_output_path = os.path.join(
                self.parameters.full_output_path,
                'dataset_data',
                safe_name,
                f'minsyn_{threshold}'
            )
            
            # Try multiple path file patterns
            path_files_to_try = [
                os.path.join(dataset_output_path, f'minsyn_{threshold}_data_original_paths.csv'),
            ]
            
            # Also check for source_to_target_allpaths_type.csv pattern
            if os.path.exists(dataset_output_path):
                for f in os.listdir(dataset_output_path):
                    if f.endswith('_allpaths_type.csv'):
                        path_files_to_try.append(os.path.join(dataset_output_path, f))
            
            # Try to read from available files
            df = None
            for path_file in path_files_to_try:
                if os.path.exists(path_file):
                    try:
                        df = self._read_csv(path_file)
                        break
                    except Exception as e:
                        self._log(f"Warning: Could not read {path_file}: {e}")
            
            if df is not None and 'path' in df.columns and 'min_ratio' in df.columns:
                for _, row in df.iterrows():
                    original_path_key = row['path']
                    min_ratio = row['min_ratio']
                    
                    # Apply type mapping to path key if auto_type_mapping is enabled
                    if self.parameters.auto_type_mapping and self.parameters._auto_type_mapper:
                        # Parse path nodes
                        if '->' in str(original_path_key):
                            path_nodes = [n.strip() for n in str(original_path_key).split('->')]
                        elif ' → ' in str(original_path_key):
                            path_nodes = [n.strip() for n in str(original_path_key).split(' → ')]
                        else:
                            path_nodes = [str(original_path_key)]
                        
                        # Build canonical and display keys
                        canonical_key, display_key = self._build_path_key_with_mapping(path_nodes, dataset_name)
                        path_key = canonical_key  # Use canonical key for merging
                    else:
                        path_key = original_path_key
                        display_key = original_path_key
                    
                    if path_key not in all_ratio_data:
                        all_ratio_data[path_key] = {'_display_key': display_key}
                    all_ratio_data[path_key][dataset_name] = min_ratio
        
        if not all_ratio_data:
            return pd.DataFrame()
        
        # Build result DataFrame using display keys for index
        result_rows = []
        for canonical_key, data in all_ratio_data.items():
            display_key = data.pop('_display_key', canonical_key)
            row_data = {d: data.get(d, 0) for d in dataset_names}
            result_rows.append((display_key, row_data))
        
        if not result_rows:
            return pd.DataFrame()
        
        result_df = pd.DataFrame([r[1] for r in result_rows], index=[r[0] for r in result_rows])
        return result_df.fillna(0)
    
    def _get_prob_data_for_threshold(self, threshold: int) -> pd.DataFrame:
        """
        Get traversal probability (path_prob) data aligned across datasets for a threshold.
        
        Reads from dataset_data/{dataset}/minsyn_{threshold}/*_original_paths.csv
        or *_allpaths_type.csv and extracts path_prob values.
        
        Args:
            threshold: The threshold level
            
        Returns:
            DataFrame with path index and dataset columns containing path_prob
        """
        dataset_names = self.parameters.get_dataset_names()
        all_prob_data = {}
        
        for dataset_name in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset_name)
            dataset_output_path = os.path.join(
                self.parameters.full_output_path,
                'dataset_data',
                safe_name,
                f'minsyn_{threshold}'
            )
            
            # Try multiple path file patterns
            path_files_to_try = [
                os.path.join(dataset_output_path, f'minsyn_{threshold}_data_original_paths.csv'),
            ]
            
            # Also check for source_to_target_allpaths_type.csv pattern
            if os.path.exists(dataset_output_path):
                for f in os.listdir(dataset_output_path):
                    if f.endswith('_allpaths_type.csv'):
                        path_files_to_try.append(os.path.join(dataset_output_path, f))
            
            # Try to read from available files
            df = None
            for path_file in path_files_to_try:
                if os.path.exists(path_file):
                    try:
                        df = self._read_csv(path_file)
                        break
                    except Exception as e:
                        self._log(f"Warning: Could not read {path_file}: {e}")
            
            if df is not None and 'path' in df.columns and 'path_prob' in df.columns:
                for _, row in df.iterrows():
                    original_path_key = row['path']
                    path_prob = row['path_prob']
                    
                    # Apply type mapping to path key if auto_type_mapping is enabled
                    if self.parameters.auto_type_mapping and self.parameters._auto_type_mapper:
                        # Parse path nodes
                        if '->' in str(original_path_key):
                            path_nodes = [n.strip() for n in str(original_path_key).split('->')]
                        elif ' → ' in str(original_path_key):
                            path_nodes = [n.strip() for n in str(original_path_key).split(' → ')]
                        else:
                            path_nodes = [str(original_path_key)]
                        
                        # Build canonical and display keys
                        canonical_key, display_key = self._build_path_key_with_mapping(path_nodes, dataset_name)
                        path_key = canonical_key  # Use canonical key for merging
                    else:
                        path_key = original_path_key
                        display_key = original_path_key
                    
                    if path_key not in all_prob_data:
                        all_prob_data[path_key] = {'_display_key': display_key}
                    all_prob_data[path_key][dataset_name] = path_prob
        
        if not all_prob_data:
            return pd.DataFrame()
        
        # Build result DataFrame using display keys for index
        result_rows = []
        for canonical_key, data in all_prob_data.items():
            display_key = data.pop('_display_key', canonical_key)
            row_data = {d: data.get(d, 0) for d in dataset_names}
            result_rows.append((display_key, row_data))
        
        if not result_rows:
            return pd.DataFrame()
        
        result_df = pd.DataFrame([r[1] for r in result_rows], index=[r[0] for r in result_rows])
        return result_df.fillna(0)

    def _get_edge_ratio_data_for_threshold(self, threshold: int) -> pd.DataFrame:
        """
        Get edge-level connection_ratio data aligned across datasets for a threshold.
        
        Reads from:
        1. dataset_data/{dataset}/minsyn_{threshold}/connections_edge.csv (edge mode)
        2. dataset_data/{dataset}/minsyn_{threshold}/data_details/connection_type.csv (path mode fallback)
        
        connection_ratio = w_ij / W_j (edge weight / total post-synaptic sites)
        
        Args:
            threshold: The threshold level
            
        Returns:
            DataFrame with edge index and dataset columns containing connection_ratio
        """
        dataset_names = self.parameters.get_dataset_names()
        all_ratio_data = {}
        
        for dataset_name in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset_name)
            dataset_output_path = os.path.join(
                self.parameters.full_output_path,
                'dataset_data',
                safe_name,
                f'minsyn_{threshold}'
            )
            
            df = None
            
            # Try to read connections_edge.csv first (edge mode output)
            conn_file = os.path.join(dataset_output_path, 'connections_edge.csv')
            file_exists = os.path.exists(conn_file)
            if file_exists:
                try:
                    # Check file is not empty before reading
                    if os.path.getsize(conn_file) > 0:
                        df = self._read_csv(conn_file)
                except pd.errors.EmptyDataError:
                    pass  # File is empty or has no columns, try fallback
                except Exception:
                    pass  # Other errors, try fallback
            
            # Fallback to data_details/connection_type.csv (path mode output)
            if df is None or df.empty or 'connection_ratio' not in df.columns:
                conn_type_file = os.path.join(dataset_output_path, 'data_details', 'connection_type.csv')
                if os.path.exists(conn_type_file):
                    try:
                        if os.path.getsize(conn_type_file) > 0:
                            df = self._read_csv(conn_type_file)
                    except pd.errors.EmptyDataError:
                        pass  # File is empty
                    except Exception:
                        pass  # Other errors
            
            if df is None or df.empty or 'connection_ratio' not in df.columns:
                continue
            
            # Determine edge columns
            if 'std_label_pre' in df.columns and 'std_label_post' in df.columns:
                pre_col, post_col = 'std_label_pre', 'std_label_post'
            elif 'type_pre' in df.columns and 'type_post' in df.columns:
                pre_col, post_col = 'type_pre', 'type_post'
            else:
                pre_col, post_col = 'bodyId_pre', 'bodyId_post'
            
            # Aggregate by edge - use mean for connection_ratio
            for _, row in df.iterrows():
                pre_type = str(row[pre_col])
                post_type = str(row[post_col])
                
                # Apply type mapping if auto_type_mapping is enabled
                if self.parameters.auto_type_mapping and self.parameters._auto_type_mapper:
                    canonical_pre = self._get_canonical_type(pre_type, dataset_name)
                    canonical_post = self._get_canonical_type(post_type, dataset_name)
                    # _get_display_type takes canonical name, not (type, dataset)
                    display_pre = self._get_display_type(canonical_pre)
                    display_post = self._get_display_type(canonical_post)
                    edge_key = f"{canonical_pre} -> {canonical_post}"
                    display_edge_key = f"{display_pre} -> {display_post}"
                else:
                    edge_key = f"{pre_type} -> {post_type}"
                    display_edge_key = edge_key
                
                ratio_val = row['connection_ratio']
                if pd.notna(ratio_val):
                    if edge_key not in all_ratio_data:
                        all_ratio_data[edge_key] = {'_display_key': display_edge_key}
                    # Store ratio, will be averaged later if multiple edges
                    if dataset_name not in all_ratio_data[edge_key]:
                        all_ratio_data[edge_key][dataset_name] = []
                    all_ratio_data[edge_key][dataset_name].append(ratio_val)
        
        if not all_ratio_data:
            return pd.DataFrame()
        
        # Convert lists to averages and build result DataFrame
        dataset_names = self.parameters.get_dataset_names()
        result_rows = []
        for edge_key, data in all_ratio_data.items():
            display_key = data.pop('_display_key', edge_key)
            row_data = {}
            for ds in dataset_names:
                vals = data.get(ds, [])
                if isinstance(vals, list) and vals:
                    row_data[ds] = sum(vals) / len(vals)
                else:
                    row_data[ds] = 0.0
            result_rows.append((display_key, row_data))
        
        if not result_rows:
            return pd.DataFrame()
        
        result_df = pd.DataFrame([r[1] for r in result_rows], index=[r[0] for r in result_rows])
        return result_df.fillna(0)

    def _get_edge_nt_details(self, threshold: int) -> Dict[str, Dict[str, str]]:
        """
        Get NT type per edge per dataset at a threshold.
        
        Returns:
            Dict[canonical_edge_key, Dict[dataset_name, nt_type]]
        """
        dataset_names = self.parameters.get_dataset_names()
        nt_by_edge: Dict[str, Dict[str, str]] = {} # canonical_key -> {dataset: nt}

        for dataset_name in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset_name)
            
            # Try reciprocal first, then standard
            # We want the most specific NT info available
            dataset_output_path = os.path.join(
                self.parameters.full_output_path,
                'dataset_data',
                safe_name,
                f'minsyn_{threshold}',
                'find_reciprocal'
            )
            conn_file = os.path.join(dataset_output_path, 'reciprocal_connection_type.csv')
            
            df = None
            if os.path.exists(conn_file) and os.path.getsize(conn_file) > 0:
                try:
                    df = self._read_csv(conn_file)
                except Exception:
                    df = None

            # Fallback to standard connections
            if df is None or df.empty:
                dataset_output_path = os.path.join(
                    self.parameters.full_output_path,
                    'dataset_data',
                    safe_name,
                    f'minsyn_{threshold}'
                )
                conn_file = os.path.join(dataset_output_path, 'connections_edge.csv')
                if os.path.exists(conn_file) and os.path.getsize(conn_file) > 0:
                    try:
                        df = self._read_csv(conn_file)
                    except Exception:
                        df = None

            if (df is None or df.empty):
                # Fallback to data_details
                dataset_output_path = os.path.join(
                    self.parameters.full_output_path,
                    'dataset_data',
                    safe_name,
                    f'minsyn_{threshold}'
                )
                fallback = os.path.join(dataset_output_path, 'data_details', 'connection_type.csv')
                if os.path.exists(fallback) and os.path.getsize(fallback) > 0:
                    try:
                        df = self._read_csv(fallback)
                    except Exception:
                        df = None

            if df is None or df.empty:
                continue

            if 'std_label_pre' in df.columns and 'std_label_post' in df.columns:
                pre_col, post_col = 'std_label_pre', 'std_label_post'
            elif 'type_pre' in df.columns and 'type_post' in df.columns:
                pre_col, post_col = 'type_pre', 'type_post'
            elif 'bodyId_pre' in df.columns and 'bodyId_post' in df.columns:
                pre_col, post_col = 'bodyId_pre', 'bodyId_post'
            else:
                continue

            if 'nt_type_pre' in df.columns:
                nt_col = 'nt_type_pre'
            elif 'nt_type' in df.columns:
                nt_col = 'nt_type'
            else:
                continue

            for _, row in df.iterrows():
                pre_type = str(row[pre_col])
                post_type = str(row[post_col])
                nt_val = row.get(nt_col, None)
                if pd.isna(nt_val):
                    continue
                nt_str = str(nt_val).strip()
                if not nt_str:
                    continue
                
                # Normalize NT string
                nt_upper = nt_str.upper()
                nt_str = self._NT_NORMALIZATION_MAP.get(nt_upper, nt_str.lower())

                if self.parameters.auto_type_mapping and self.parameters._auto_type_mapper:
                    canonical_pre = self._get_canonical_type(pre_type, dataset_name)
                    canonical_post = self._get_canonical_type(post_type, dataset_name)
                else:
                    canonical_pre = pre_type
                    canonical_post = post_type

                edge_key = f"{canonical_pre} -> {canonical_post}"
                if edge_key not in nt_by_edge:
                    nt_by_edge[edge_key] = {}
                
                # Check if we already have a value for this dataset (avoid overwrite if duplicates, or just take first)
                if dataset_name not in nt_by_edge[edge_key]:
                    nt_by_edge[edge_key][dataset_name] = nt_str
        
        return nt_by_edge

    def _get_reciprocal_edge_ratio_data_for_threshold(self, threshold: int) -> pd.DataFrame:
        """
        Get edge-level connection_ratio data from reciprocal outputs for a threshold.

        Reads from:
        dataset_data/{dataset}/minsyn_{threshold}/find_reciprocal/reciprocal_connection_type.csv
        """
        dataset_names = self.parameters.get_dataset_names()
        all_ratio_data = {}

        for dataset_name in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset_name)
            dataset_output_path = os.path.join(
                self.parameters.full_output_path,
                'dataset_data',
                safe_name,
                f'minsyn_{threshold}',
                'find_reciprocal'
            )

            df = None
            conn_file = os.path.join(dataset_output_path, 'reciprocal_connection_type.csv')
            if os.path.exists(conn_file) and os.path.getsize(conn_file) > 0:
                try:
                    df = self._read_csv(conn_file)
                except Exception:
                    df = None

            if df is None or df.empty or 'connection_ratio' not in df.columns:
                continue

            if 'type_pre' in df.columns and 'type_post' in df.columns:
                df['edge_key'] = df['type_pre'].astype(str) + ' -> ' + df['type_post'].astype(str)
            else:
                continue

            ratio_series = df.groupby('edge_key')['connection_ratio'].mean()
            all_ratio_data[dataset_name] = ratio_series

        if not all_ratio_data:
            return pd.DataFrame()

        result_df = pd.DataFrame(all_ratio_data).fillna(0)
        return result_df

    def _generate_vispath_visualizations(self, vis_dir: str):
        """
        Generate interactive HTML visualizations using VisualizePath.
        
        Note: Interactive heatmaps have been removed (redundant with PNG visualizations
        and comparison_report.html). Network visualizations are embedded directly
        in comparison_report.html using Cytoscape.js.
        
        This method is kept for future extension but currently is a no-op.
        
        Args:
            vis_dir: Directory to save visualization files
        """
        # Note: _generate_vispath_heatmaps and heatmaps/ folder removed as redundant
        # All heatmap needs are served by PNG visualizations in visualizations/ folder
        pass

    def _create_combined_network_html(self, conn_rows: List[Dict], datasets: List[str],
                                       threshold: int, output_path: str, title: str):
        """
        Create a combined network HTML with all datasets side by side.
        
        Uses vis.js with dataset-based coloring to distinguish connections.
        """
        import json
        
        # Build nodes and edges
        nodes = []
        edges = []
        node_ids = {}
        node_counter = 0
        
        # Color palette for datasets
        colors = ['#3b82f6', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4']
        dataset_colors = {ds: colors[i % len(colors)] for i, ds in enumerate(datasets)}
        
        for row in conn_rows:
            source = row['source']
            target = row['target']
            weight = row['weight']
            dataset = row.get('dataset', 'unknown')
            
            # Add source node
            if source not in node_ids:
                node_ids[source] = node_counter
                nodes.append({
                    'id': node_counter, 
                    'label': source, 
                    'group': 'source',
                    'title': f'{source}'
                })
                node_counter += 1
            
            # Add target node  
            if target not in node_ids:
                node_ids[target] = node_counter
                nodes.append({
                    'id': node_counter, 
                    'label': target, 
                    'group': 'target',
                    'title': f'{target}'
                })
                node_counter += 1
            
            # Add edge with dataset color
            edge_color = dataset_colors.get(self.parameters._sanitize_name(dataset), '#64748b')
            edges.append({
                'from': node_ids[source],
                'to': node_ids[target],
                'value': int(weight),
                'title': f"{source} → {target}: {int(weight)} ({dataset})",
                'arrows': 'to',
                'color': {'color': edge_color, 'highlight': edge_color}
            })
        
        # Create legend HTML
        legend_items = ''.join([
            f'<span style="color:{dataset_colors.get(self.parameters._sanitize_name(ds), "#64748b")}; margin-right:15px;">● {self.parameters._sanitize_name(ds)}</span>'
            for ds in datasets
        ])
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        h1 {{ color: #2563eb; }}
        .legend {{ margin: 10px 0; font-size: 14px; }}
        #network {{ width: 100%; height: 700px; border: 1px solid #e2e8f0; border-radius: 8px; }}
        .stats {{ background: #f8fafc; padding: 10px 15px; border-radius: 8px; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="stats">
        <strong>{len(nodes)}</strong> neurons | <strong>{len(edges)}</strong> connections | 
        <strong>{len(datasets)}</strong> datasets
    </div>
    <div class="legend">{legend_items}</div>
    <div id="network"></div>
    <script>
        const nodes = new vis.DataSet({json.dumps(nodes)});
        const edges = new vis.DataSet({json.dumps(edges)});
        const container = document.getElementById('network');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
            nodes: {{
                shape: 'dot',
                size: 20,
                font: {{ size: 12 }},
                borderWidth: 2
            }},
            edges: {{
                width: 2,
                arrows: {{ to: {{ enabled: true, scaleFactor: 0.8 }} }},
                smooth: {{ type: 'curvedCW', roundness: 0.15 }}
            }},
            groups: {{
                source: {{ color: {{ background: '#3b82f6', border: '#1d4ed8' }} }},
                target: {{ color: {{ background: '#8b5cf6', border: '#6d28d9' }} }},
                intermediate: {{ color: {{ background: '#22c55e', border: '#15803d' }} }}
            }},
            layout: {{
                hierarchical: {{
                    enabled: true,
                    direction: 'LR',
                    sortMethod: 'directed',
                    levelSeparation: 200,
                    nodeSpacing: 80
                }}
            }},
            physics: {{ enabled: false }},
            interaction: {{ hover: true, tooltipDelay: 100 }}
        }};
        new vis.Network(container, data, options);
    </script>
</body>
</html>'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        self._log(f"Saved: {os.path.basename(output_path)}")

    def _create_threshold_comparison_network_html(self, conn_rows: List[Dict], 
                                                   dataset: str, thresholds: List[int],
                                                   output_path: str, title: str):
        """
        Create a network HTML comparing all thresholds for a single dataset.
        
        Uses edge width/opacity to show threshold sensitivity.
        """
        import json
        
        # Aggregate connections across thresholds
        # For each edge, track which thresholds it appears at and max weight
        edge_data = {}
        for row in conn_rows:
            source = row['source']
            target = row['target']
            weight = row['weight']
            threshold = row['threshold']
            
            key = (source, target)
            if key not in edge_data:
                edge_data[key] = {'thresholds': [], 'weights': [], 'max_weight': 0}
            edge_data[key]['thresholds'].append(threshold)
            edge_data[key]['weights'].append(weight)
            edge_data[key]['max_weight'] = max(edge_data[key]['max_weight'], weight)
        
        # Build nodes and edges
        nodes = []
        edges = []
        node_ids = {}
        node_counter = 0
        
        # Color by threshold count (more thresholds = more "conserved")
        num_thresholds = len(thresholds)
        
        for (source, target), data in edge_data.items():
            # Add source node
            if source not in node_ids:
                node_ids[source] = node_counter
                nodes.append({
                    'id': node_counter, 
                    'label': source, 
                    'group': 'source',
                    'title': f'{source}'
                })
                node_counter += 1
            
            # Add target node
            if target not in node_ids:
                node_ids[target] = node_counter
                nodes.append({
                    'id': node_counter, 
                    'label': target, 
                    'group': 'target',
                    'title': f'{target}'
                })
                node_counter += 1
            
            # Edge color based on threshold conservation
            t_count = len(data['thresholds'])
            if t_count == num_thresholds:
                color = '#22c55e'  # Green: in all thresholds
            elif t_count >= num_thresholds * 0.5:
                color = '#f59e0b'  # Orange: in half
            else:
                color = '#ef4444'  # Red: in few
            
            t_list = ', '.join(str(t) for t in sorted(data['thresholds']))
            edges.append({
                'from': node_ids[source],
                'to': node_ids[target],
                'value': int(data['max_weight']),
                'title': f"{source} → {target}\\nMax weight: {int(data['max_weight'])}\\nThresholds: {t_list}",
                'arrows': 'to',
                'color': {'color': color, 'highlight': color}
            })
        
        # Legend
        legend_html = '''
        <span style="color:#22c55e; margin-right:15px;">● All thresholds</span>
        <span style="color:#f59e0b; margin-right:15px;">● Half thresholds</span>
        <span style="color:#ef4444; margin-right:15px;">● Few thresholds</span>
        '''
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        h1 {{ color: #2563eb; }}
        .legend {{ margin: 10px 0; font-size: 14px; }}
        #network {{ width: 100%; height: 700px; border: 1px solid #e2e8f0; border-radius: 8px; }}
        .stats {{ background: #f8fafc; padding: 10px 15px; border-radius: 8px; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="stats">
        <strong>{len(nodes)}</strong> neurons | <strong>{len(edges)}</strong> unique connections | 
        Thresholds: <strong>{', '.join(str(t) for t in thresholds)}</strong>
    </div>
    <div class="legend">{legend_html}</div>
    <div id="network"></div>
    <script>
        const nodes = new vis.DataSet({json.dumps(nodes)});
        const edges = new vis.DataSet({json.dumps(edges)});
        const container = document.getElementById('network');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
            nodes: {{
                shape: 'dot',
                size: 20,
                font: {{ size: 12 }},
                borderWidth: 2
            }},
            edges: {{
                width: 2,
                arrows: {{ to: {{ enabled: true, scaleFactor: 0.8 }} }},
                smooth: {{ type: 'curvedCW', roundness: 0.15 }}
            }},
            groups: {{
                source: {{ color: {{ background: '#3b82f6', border: '#1d4ed8' }} }},
                target: {{ color: {{ background: '#8b5cf6', border: '#6d28d9' }} }},
                intermediate: {{ color: {{ background: '#22c55e', border: '#15803d' }} }}
            }},
            layout: {{
                hierarchical: {{
                    enabled: true,
                    direction: 'LR',
                    sortMethod: 'directed',
                    levelSeparation: 200,
                    nodeSpacing: 80
                }}
            }},
            physics: {{ enabled: false }},
            interaction: {{ hover: true, tooltipDelay: 100 }}
        }};
        new vis.Network(container, data, options);
    </script>
</body>
</html>'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        self._log(f"Saved: {os.path.basename(output_path)}")

    # =========================================================================
    # Connectivity Profile Comparison
    # =========================================================================
    
    def direct_comparison(
        self,
        neurons_a: Optional[Union[str, int, List[Union[str, int]]]] = None,
        neurons_b: Optional[Union[str, int, List[Union[str, int]]]] = None,
        dataset_a: Optional[str] = None,
        dataset_b: Optional[str] = None,
        direction: Optional[str] = None,
        comparison_mode: str = 'type',
        output_dir: Optional[str] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Direct comparison of specific neurons between datasets.
        
        This is a convenience method for directly comparing neurons by type name
        or bodyId. Uses ProfileComparator.direct_comparison() under the hood.
        
        If neurons_a/neurons_b are not provided, uses source_neurons/target_neurons
        from ComparisonParameters.
        
        Args:
            neurons_a: Neurons to compare from dataset_a (default: params.source_neurons)
            neurons_b: Neurons to compare from dataset_b (default: params.target_neurons)
            dataset_a: First dataset (default: first in params.datasets)
            dataset_b: Second dataset (default: second in params.datasets)
            direction: 'upstream', 'downstream', or 'both' (default: params.verification_direction)
            comparison_mode: 'type' (aggregate) or 'bodyid' (individual)
            output_dir: Where to save results (default: params output folder)
            save_results: If True, save results to CSV
        
        Returns:
            Dict with 'results' DataFrame and 'summary' statistics
        
        Example:
            >>> analyzer = ComparisonAnalyzer(params)
            >>> # Compare specific types
            >>> results = analyzer.direct_comparison('aMe12', 'aMe12')
            >>> # Or use defaults from params
            >>> results = analyzer.direct_comparison()
        """
        from .profile_comparator import ProfileComparator
        from .connectivity_profiler import ConnectivityProfiler, ProfilerConfig
        
        p = self.parameters
        
        # Resolve defaults
        datasets = p.get_dataset_names()
        ds_a = dataset_a or (datasets[0] if len(datasets) > 0 else None)
        ds_b = dataset_b or (datasets[1] if len(datasets) > 1 else ds_a)
        direction = direction or p.verification_direction
        
        # Use source/target neurons if not specified
        if neurons_a is None:
            neurons_a = p.source_neurons
        if neurons_b is None:
            neurons_b = p.target_neurons
        
        # Resolve output directory
        if output_dir is None:
            output_dir = os.path.join(p.full_output_path, "direct_comparison")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create profiler
        config = ProfilerConfig(
            top_k_bodyid=p.verification_top_k,
            top_m_type=p.verification_top_m,
            min_synapse_threshold=p.verification_min_synapse_threshold,
            include_untyped_partners=p.verification_include_untyped,
            use_cache=True
        )
        
        profiler = ConnectivityProfiler(
            datasets=[ds_a, ds_b],
            config=config,
            token=p.resolve_token(),
            verbose=self.verbose
        )
        
        # Run direct comparison
        self._log(f"Running direct comparison: {neurons_a} ({ds_a}) vs {neurons_b} ({ds_b})")
        
        results = ProfileComparator.direct_comparison(
            neurons_a=neurons_a,
            neurons_b=neurons_b,
            dataset_a=ds_a,
            dataset_b=ds_b,
            profiler=profiler,
            direction=direction,
            comparison_mode=comparison_mode,
            label_mapper=self.label_mapper,
            score_weights=p.verification_score_weights,
            verbose=self.verbose
        )
        
        # Save results
        if save_results and not results['results'].empty:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(output_dir, f'direct_comparison_{timestamp}.csv')
            self._save_csv(results['results'], filepath)
            self._log(f"Saved: {filepath}")
            results['output_file'] = filepath
        
        return results
    
    def connectivity_profile_comparison(
        self,
        output_dir: Optional[str] = None,
        neuron_types: Optional[List[str]] = None,
        direction: Optional[str] = None,
        comparison_mode: Optional[str] = None,
        include_visualizations: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_m: Optional[int] = None,
        min_synapse_threshold: Optional[int] = None,
        include_untyped_partners: Optional[bool] = None,
        score_weights: Optional[Dict[str, float]] = None,
        _skip_html_regeneration: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Reconstructed connectivity profile comparison.
        
        Workflow:
        1) For each neuron type, run pairwise direct comparisons across all dataset pairs
           using ProfileComparator.direct_comparison (same-label only).
        2) Merge pairwise results into a comparison matrix (type x dataset-pair).
        3) Compute per-type average similarity across all datasets.
        4) Save CSVs and an optional HTML + heatmap visualization.
        
        Only the essential arguments are kept; unused legacy flags were removed.
        """
        # Defaults from parameters
        p = self.parameters
        direction = direction if direction is not None else p.verification_direction
        comparison_mode = comparison_mode if comparison_mode is not None else p.verification_mode
        include_visualizations = include_visualizations if include_visualizations is not None else p.verification_include_visualizations
        top_k = top_k if top_k is not None else p.verification_top_k
        top_m = top_m if top_m is not None else p.verification_top_m
        min_synapse_threshold = min_synapse_threshold if min_synapse_threshold is not None else p.verification_min_synapse_threshold
        include_untyped_partners = include_untyped_partners if include_untyped_partners is not None else p.verification_include_untyped
        score_weights = score_weights if score_weights is not None else p.verification_score_weights

        # Visuals can be suppressed either by caller or via the internal skip flag
        visuals_enabled = bool(include_visualizations and not _skip_html_regeneration)

        # Validate datasets
        dataset_names = self.parameters.get_dataset_names()
        if len(dataset_names) < 2:
            self._log("Need at least 2 datasets for profile comparison")
            return {}

        # Resolve neuron types
        if neuron_types is None:
            src_types, tgt_types, inter_types = self._extract_types_from_results()
            ordered = list(src_types) + [t for t in tgt_types if t not in src_types]
            seen = set(ordered)
            ordered += [t for t in inter_types if t not in seen]
            neuron_types = ordered
        neuron_types = list(dict.fromkeys(neuron_types))  # dedupe, keep order

        if not neuron_types:
            self._log("No neuron types available for profile comparison")
            return {}

        if comparison_mode not in ['loose', 'strict']:
            self._log(f"Warning: Invalid comparison_mode '{comparison_mode}', using 'loose'")
            comparison_mode = 'loose'

        # Output dir
        if output_dir is None:
            output_dir = os.path.join(self.parameters.full_output_path, "connectivity_profile_comparison")
        os.makedirs(output_dir, exist_ok=True)

        # Profiler
        try:
            from .connectivity_profiler import ConnectivityProfiler, ProfilerConfig
            from .profile_comparator import ProfileComparator
        except ImportError as e:
            self._log(f"Warning: Connectivity profile modules not available: {e}")
            return {}

        profiler = ConnectivityProfiler(
            datasets=dataset_names,
            config=ProfilerConfig(
                top_k_bodyid=top_k,
                top_m_type=top_m,
                min_synapse_threshold=min_synapse_threshold,
                include_untyped_partners=include_untyped_partners,
                use_cache=True,
            ),
            token=self.parameters.resolve_token(),
            verbose=self.verbose,
        )

        # Pairwise comparisons
        pairwise_records: List[Dict[str, Any]] = []
        matrix_store: Dict[str, Dict[str, float]] = {t: {} for t in neuron_types}

        for ds_a, ds_b in combinations(dataset_names, 2):
            pair_label = f"{ds_a} vs {ds_b}"
            self._log(f"Comparing {len(neuron_types)} types: {pair_label}")
            try:
                res = ProfileComparator.direct_comparison(
                    neurons_a=neuron_types,
                    neurons_b=neuron_types,
                    dataset_a=ds_a,
                    dataset_b=ds_b,
                    profiler=profiler,
                    direction=direction,
                    comparison_mode=comparison_mode,
                    label_mapper=self.label_mapper,
                    score_weights=score_weights,
                    top_k=top_k,
                    top_m=top_m,
                    min_synapse_threshold=min_synapse_threshold,
                    include_untyped_partners=include_untyped_partners,
                    same_label_only=True,
                    verbose=self.verbose,
                )
            except Exception as e:
                self._log(f"Direct comparison failed for {pair_label}: {e}")
                continue

            # Prefer type_summary; fallback to results
            type_df = res.get('type_summary')
            if type_df is None or (hasattr(type_df, 'empty') and type_df.empty):
                type_df = res.get('results', pd.DataFrame())

            if type_df is None or type_df.empty:
                self._log(f"No results for {pair_label}")
                continue

            # Collect all available metrics
            metric_map = {
                'avg_rank_corr': 'rank_corr', 'rank_corr': 'rank_corr',
                'avg_rank_union': 'rank_union', 'rank_union': 'rank_union',
                'avg_cosine': 'cosine', 'cosine': 'cosine',
                'avg_jaccard': 'jaccard', 'jaccard': 'jaccard'
            }
            
            # Initialize stores for each metric if not exists
            if not hasattr(self, '_matrix_stores'):
                self._matrix_stores = {m: {t: {} for t in neuron_types} for m in set(metric_map.values())}

            found_any = False
            for col, canonical in metric_map.items():
                if col in type_df.columns:
                    for _, row in type_df.iterrows():
                        ntype = row.get('neuron_type') or row.get('type') or row.get('pair') or row.get('type_a')
                        if pd.isna(ntype):
                            continue
                        val = row.get(col)
                        if pd.isna(val):
                            continue
                        self._matrix_stores[canonical].setdefault(str(ntype), {})[pair_label] = float(val)
                        
                        # Add to pairwise records (only once per canonical metric per pair)
                        # We might overwrite if multiple cols map to same canonical, but that's fine (prefer last/best?)
                        # Actually, let's just append all and filter later or just keep it simple
                        pairwise_records.append({
                            'neuron_type': str(ntype),
                            'dataset_a': ds_a,
                            'dataset_b': ds_b,
                            'metric': canonical,
                            'value': float(val),
                        })
                    found_any = True
            
            if not found_any:
                self._log(f"No similarity columns found for {pair_label}")

        if not pairwise_records:
            self._log("No pairwise comparison results produced")
            return {}

        # Build matrix DataFrames for each metric
        matrices = {}
        pair_cols = [f"{a} vs {b}" for a, b in combinations(dataset_names, 2)]
        
        for metric, store in self._matrix_stores.items():
            # Check if store has any data
            has_data = any(store.values())
            if not has_data:
                continue
                
            df = pd.DataFrame(store).T.reset_index().rename(columns={'index': 'neuron_type'})
            existing_cols = [c for c in pair_cols if c in df.columns]
            if not existing_cols:
                continue
                
            df = df[['neuron_type'] + existing_cols]
            matrices[metric] = df

        if not matrices:
            self._log("No valid matrices could be built")
            return {}

        # Use 'rank_corr' as primary for sorting/display when present
        primary_metric = 'rank_corr' if 'rank_corr' in matrices else list(matrices.keys())[0]
        
        # Create summary DataFrame with averages for ALL metrics
        # Start with all unique neuron types across all matrices
        all_types = set()
        for df in matrices.values():
            all_types.update(df['neuron_type'].tolist())
        
        summary_df = pd.DataFrame({'neuron_type': sorted(list(all_types))})
        
        # Calculate and merge averages for each metric
        for metric, df in matrices.items():
            value_cols = [c for c in df.columns if c != 'neuron_type']
            # Calculate mean for this metric
            avg_series = df.set_index('neuron_type')[value_cols].mean(axis=1, skipna=True)
            avg_df = avg_series.reset_index().rename(columns={0: f'avg_{metric}'})
            summary_df = pd.merge(summary_df, avg_df, on='neuron_type', how='left')
            
        # Add n_pairs count (using primary metric)
        primary_df = matrices[primary_metric]
        primary_cols = [c for c in primary_df.columns if c != 'neuron_type']
        count_series = primary_df.set_index('neuron_type')[primary_cols].count(axis=1)
        count_df = count_series.reset_index().rename(columns={0: 'n_pairs'})
        summary_df = pd.merge(summary_df, count_df, on='neuron_type', how='left')
        
        # Sort by primary metric average
        if f'avg_{primary_metric}' in summary_df.columns:
            summary_df = summary_df.sort_values(f'avg_{primary_metric}', ascending=False)

        pairwise_df = pd.DataFrame(pairwise_records)

        # Save outputs
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # Save all matrices
        for metric, df in matrices.items():
            path = os.path.join(output_dir, f'comparison_matrix_{metric}_{timestamp}.csv')
            self._save_csv(df, path)
            
        summary_path = os.path.join(output_dir, f'comparison_summary_{timestamp}.csv')
        pairwise_path = os.path.join(output_dir, f'pairwise_results_{timestamp}.csv')
        self._save_csv(summary_df, summary_path)
        self._save_csv(pairwise_df, pairwise_path)
        self._log(f"Saved matrices, summary, and pairwise results to {output_dir}")

        report_path = None
        heatmap_paths = {}

        # Optional visualization
        if visuals_enabled:
            try:
                import matplotlib.pyplot as plt
                
                for metric, df in matrices.items():
                    cols = [c for c in df.columns if c != 'neuron_type']
                    if not cols:
                        continue
                        
                    fig, ax = plt.subplots(figsize=(max(6, len(cols)*0.8), max(6, len(df)*0.25)))
                    heatmap_data = df.set_index('neuron_type')[cols]
                    
                    # Determine range and colormap based on data
                    data_min = heatmap_data.min().min()
                    if data_min < 0:
                        # Use diverging colormap for data with negative values (e.g. correlations)
                        cmap = 'RdBu_r'
                        vmin = -1
                        vmax = 1
                    else:
                        # Use sequential colormap for positive-only data (e.g. Jaccard)
                        cmap = 'viridis'
                        vmin = 0
                        vmax = 1
                        
                    im = ax.imshow(heatmap_data.fillna(np.nan), aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_xticks(range(len(cols)))
                    ax.set_xticklabels(cols, rotation=45, ha='right')
                    ax.set_yticks(range(len(heatmap_data.index)))
                    ax.set_yticklabels(heatmap_data.index)
                    ax.set_title(f'Connectivity Profile Similarity ({metric})')
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    
                    path = os.path.join(output_dir, f'comparison_heatmap_{metric}_{timestamp}.png')
                    fig.tight_layout()
                    fig.savefig(path, dpi=200)
                    plt.close(fig)
                    heatmap_paths[metric] = path
                    self._log(f"Saved heatmap: {path}")
            except Exception as e:
                self._log(f"Heatmap generation failed: {e}")

            # Interactive Heatmap
            interactive_path = None
            try:
                interactive_path = os.path.join(output_dir, f'comparison_interactive_{timestamp}.html')
                interactive_matrices = {}
                for metric, df in matrices.items():
                    if 'neuron_type' in df.columns:
                        interactive_matrices[metric] = df.set_index('neuron_type')
                    else:
                        interactive_matrices[metric] = df.copy()
                
                generate_interactive_heatmap(
                    interactive_matrices, 
                    interactive_path, 
                    title=f"Connectivity Profile Comparison ({timestamp})",
                    showfig=False,
                    verbose=self.verbose
                )
                self._log(f"Saved interactive heatmap: {interactive_path}")
            except Exception as e:
                self._log(f"Interactive heatmap generation failed: {e}")

            # HTML report
            try:
                report_path = os.path.join(output_dir, f'comparison_report_{timestamp}.html')
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("<html><head><title>Connectivity Profile Comparison</title>")
                    f.write("<style>body{font-family:sans-serif; margin:20px;} table{border-collapse:collapse; width:100%;} th,td{border:1px solid #ddd; padding:8px; text-align:left;} th{background-color:#f2f2f2;} img{max-width:100%; height:auto; margin-bottom:20px;}</style>")
                    f.write("</head><body>")
                    f.write("<h2>Connectivity Profile Comparison</h2>")
                    
                    if interactive_path:
                        f.write(f"<p><a href='{os.path.basename(interactive_path)}' target='_blank' style='font-size:16px; font-weight:bold; color:#4CAF50;'>👉 Open Interactive Heatmap</a></p>")
                    
                    f.write("<h3>Summary (Primary Metric: " + primary_metric + ")</h3>")
                    f.write(summary_df.to_html(index=False))
                    
                    # Heatmaps section
                    f.write("<h3>Similarity Heatmaps</h3>")
                    for metric, path in heatmap_paths.items():
                        f.write(f"<h4>{metric}</h4>")
                        f.write(f"<p><img src='{os.path.basename(path)}'/></p>")
                    
                    # Matrices section
                    f.write("<h3>Similarity Matrices</h3>")
                    for metric, df in matrices.items():
                        f.write(f"<h4>{metric}</h4>")
                        f.write(df.to_html(index=False))
                        
                    f.write("<h3>Pairwise Records</h3>")
                    f.write(pairwise_df.head(2000).to_html(index=False))
                    f.write("</body></html>")
                self._log(f"Saved HTML report: {report_path}")
            except Exception as e:
                self._log(f"HTML report generation failed: {e}")

        return {
            'matrix': matrices.get(primary_metric),
            'matrices': matrices,
            'summary': summary_df,
            'pairwise_results': pairwise_df,
            'comparison_mode': comparison_mode,
            'report_path': report_path,
            'heatmap_path': heatmap_paths.get(primary_metric),
            'heatmap_paths': heatmap_paths,
            'include_visualizations': visuals_enabled,
        }
    
    # Alias for backward compatibility
    def run_connectivity_profile_verification(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Alias for connectivity_profile_comparison() for backward compatibility.
        
        Deprecated: Use connectivity_profile_comparison() instead.
        """
        # Remove arguments that were removed from the main method
        kwargs.pop('parallel', None)
        kwargs.pop('max_workers', None)
        return self.connectivity_profile_comparison(**kwargs)
    

    def _extract_types_from_results(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Extract source, target, and intermediate neuron types from comparison results.
        
        Returns:
            Tuple of (source_types, target_types, intermediate_types)
        """
        source_types = set()
        target_types = set()
        intermediate_types = set()
        
        # Get from parameters
        source_neurons = self.parameters.source_neurons
        target_neurons = self.parameters.target_neurons
        
        # Process source neurons - extract concrete types if possible
        for sn in source_neurons:
            if isinstance(sn, str) and '.*' not in sn and '*' not in sn:
                source_types.add(sn)
        
        # Process target neurons
        for tn in target_neurons:
            if isinstance(tn, str) and '.*' not in tn and '*' not in tn:
                target_types.add(tn)
        
        # Extract from raw results if available
        for dataset_name, threshold_results in self.raw_results.items():
            for threshold, df in threshold_results.items():
                if df.empty:
                    continue
                
                # Source types from type_pre column
                if 'type_pre' in df.columns:
                    for t in df['type_pre'].dropna().unique():
                        t_str = str(t)
                        if t_str and t_str != 'nan':
                            # Check if it matches source pattern
                            for sn in source_neurons:
                                if self._type_matches_pattern(t_str, sn):
                                    source_types.add(t_str)
                                    break
                            else:
                                # Not a source, might be intermediate
                                for tn in target_neurons:
                                    if self._type_matches_pattern(t_str, tn):
                                        target_types.add(t_str)
                                        break
                                else:
                                    intermediate_types.add(t_str)
                
                # Target types from type_post column
                if 'type_post' in df.columns:
                    for t in df['type_post'].dropna().unique():
                        t_str = str(t)
                        if t_str and t_str != 'nan':
                            for tn in target_neurons:
                                if self._type_matches_pattern(t_str, tn):
                                    target_types.add(t_str)
                                    break
                            else:
                                # Check if source
                                for sn in source_neurons:
                                    if self._type_matches_pattern(t_str, sn):
                                        source_types.add(t_str)
                                        break
                                else:
                                    intermediate_types.add(t_str)
        
        return list(source_types), list(target_types), list(intermediate_types)
    
    def _type_matches_pattern(self, type_name: str, pattern: Union[str, int]) -> bool:
        """Check if type_name matches the pattern (supports regex patterns)."""
        import re
        
        if isinstance(pattern, int):
            return str(pattern) == type_name
        
        if '.*' in pattern or '*' in pattern:
            regex_pattern = _wildcard_pattern_to_regex(pattern)
            try:
                return bool(re.match(f'^{regex_pattern}$', type_name))
            except re.error:
                return pattern == type_name
        
        return pattern == type_name
    
    # =========================================================================
    # Conserved Path Visualization
    # =========================================================================
    
    def visualize_conserved_paths(
        self,
        threshold: Optional[int] = None,
        trim_dead_ends: bool = True,
        output_folder: Optional[str] = None,
        showfig: bool = False,
        network_layout: str = 'hierarchical',
        edge_width_scale: str = 'log',
        **vispath_kwargs
    ) -> Optional[str]:
        """
        Visualize conserved edges/paths across all datasets using VisualizePath.
        
        Creates standalone network visualizations showing only edges that are
        conserved (present in all datasets), with synapse strengths from each
        dataset shown in the hover labels.
        
        Args:
            threshold: Weight threshold to use. If None, uses middle threshold.
            trim_dead_ends: If True, removes nodes that don't connect source to target.
                          Edges leading to dead-ends are removed.
            output_folder: Output folder for visualizations. If None, uses
                          {comparison_output}/comparison_visualizations/conserved_paths/
            showfig: If True, opens the visualization in browser.
            network_layout: Layout algorithm ('hierarchical', 'spring', 'circular').
            edge_width_scale: Edge width scaling ('log', 'linear', 'sqrt', 'none').
            **vispath_kwargs: Additional keyword arguments for VisualizePath.
            
        Returns:
            Path to the generated HTML file, or None if no conserved edges found.
            
        Example:
            >>> analyzer = ComparisonAnalyzer(params)
            >>> analyzer.run_comparison()
            >>> analyzer.visualize_conserved_paths(threshold=5, trim_dead_ends=True)
        """
        # Import VisualizePath
        try:
            from vispath_pkg import VisualizePath
        except ImportError:
            self._log("Warning: vispath_pkg not available. Cannot generate conserved path visualization.")
            return None
        
        # Ensure comparison has been run
        if not self.raw_results:
            self._log("Warning: No comparison results. Run run_comparison() first.")
            return None
        
        dataset_names = self.parameters.get_dataset_names()
        dataset_nickname_map = self.parameters.get_nickname_map()
        
        # Determine threshold
        if threshold is None:
            threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]
        
        self._log(f"Generating conserved path visualization @ threshold={threshold}...")
        
        # Get aligned data
        aligned = self.get_aligned_data(threshold)
        if aligned.empty:
            self._log("Warning: No aligned data at this threshold.")
            return None
        
        # Find conserved edges (present in ALL datasets)
        available_ds = [d for d in dataset_names if d in aligned.columns]
        if not available_ds:
            self._log("Warning: No datasets with aligned data.")
            return None
        
        # Mask for edges present in all datasets
        mask_all = (aligned[available_ds] > 0).all(axis=1)
        conserved_edges = aligned[mask_all].copy()
        
        if conserved_edges.empty:
            self._log("Warning: No conserved edges found at this threshold.")
            return None
        
        self._log(f"  Found {len(conserved_edges)} conserved edges")
        
        # Get ratio and probability data for enhanced hover labels
        ratio_data = self._get_edge_ratio_data_for_threshold(threshold)
        # Note: prob_data is path-level, not edge-level. We'll include it if edge matches
        
        # Get NT details for all edges
        nt_details = self._get_edge_nt_details(threshold)
        
        # Build edge list with weights from each dataset
        edge_list = []
        edge_labels = {}  # {(source, target): {dataset: {'weight': w, 'ratio': r}, ...}}
        
        for edge_key, row in conserved_edges.iterrows():
            # Parse source/target from edge key
            if ' -> ' in str(edge_key):
                parts = str(edge_key).split(' -> ')
                source = parts[0].strip()
                target = parts[1].strip() if len(parts) > 1 else ''
            else:
                continue
            
            # Get weights and ratios from all datasets
            # Format for vispath edge_labels: {key: value} where key-value pairs are shown in tooltip
            edge_info = {}  # Will contain: {'MCNS weight': 5, 'MCNS ratio': 0.01, ...}
            avg_weight = 0
            
            # Get NT info for this edge if available
            edge_nts = nt_details.get(str(edge_key), {})
            unique_nts = set(v for v in edge_nts.values() if v)
            nt_consensus = next(iter(unique_nts)) if len(unique_nts) == 1 else None
            
            for dataset in available_ds:
                weight = row[dataset]
                if weight > 0:
                    # Use the collision-aware nickname map so two releases
                    # from the same dataset family remain distinguishable.
                    nickname = dataset_nickname_map.get(
                        dataset, self.parameters._sanitize_name(dataset)
                    )
                    
                    # Add weight for this dataset
                    edge_info[f'{nickname} wt'] = int(weight)
                    
                    # Try to get ratio data for this edge
                    if not ratio_data.empty and dataset in ratio_data.columns:
                        # Look for matching edge in ratio_data
                        edge_key_for_ratio = str(edge_key)
                        if edge_key_for_ratio in ratio_data.index:
                            ratio_val = ratio_data.loc[edge_key_for_ratio, dataset]
                            # Handle case where loc returns a Series (duplicate indices)
                            if isinstance(ratio_val, pd.Series):
                                ratio_val = ratio_val.iloc[0]
                            if ratio_val > 0:
                                edge_info[f'{nickname} ratio'] = round(ratio_val, 4)
                    
                    # Add NT info for this dataset if available
                    if dataset in edge_nts:
                        edge_info[f'{nickname} nt'] = edge_nts[dataset]
                    
                    avg_weight += weight
            
            avg_weight = avg_weight / len(available_ds) if available_ds else 0
            
            edge_data = {
                'source': source,
                'target': target,
                'weight': avg_weight,  # Use average weight for visualization
            }
            if nt_consensus:
                edge_data['nt_type'] = nt_consensus
                
            edge_list.append(edge_data)
            edge_labels[(source, target)] = edge_info
        
        if not edge_list:
            self._log("Warning: No valid edges to visualize.")
            return None
        
        # Convert to DataFrame
        edges_df = pd.DataFrame(edge_list)
        
        # Enable NT coloring if we have NT types
        if 'nt_type' in edges_df.columns and not edges_df['nt_type'].isna().all():
            if 'color_edges_by_nt' not in vispath_kwargs:
                vispath_kwargs['color_edges_by_nt'] = True

        # Helper to extract canonical name from display name
        def get_canonical_name(display_name: str) -> str:
            if '(' in display_name:
                return display_name.split('(')[0].strip()
            return display_name
        
        # Transform node labels to display format with dataset-specific names
        # Format: {canonical}({alt1}/{alt2}) for types that differ across datasets
        type_mapper = self.parameters._auto_type_mapper if self.parameters.auto_type_mapping else None
        display_name_map = {}  # {canonical_name: display_name}
        node_dataset_info = {}  # {display_name: {code: name_in_that_dataset}} for hover labels
        dataset_legend = {}  # {short_code: full_dataset_name} for legend
        
        if type_mapper:
            # Get all unique nodes and compute display names + dataset info for hover
            all_unique_nodes = set(edges_df['source'].unique()) | set(edges_df['target'].unique())
            for node in all_unique_nodes:
                display_name, ds_info = type_mapper.get_display_name_with_dataset_info(node, dataset_names)
                if display_name != node:
                    display_name_map[node] = display_name
                if ds_info:
                    node_dataset_info[display_name] = ds_info
            
            # Get dataset legend info
            dataset_legend = type_mapper.get_all_dataset_short_codes(dataset_names)
            
            # Apply display name transformation to edges_df
            if display_name_map:
                edges_df['source'] = edges_df['source'].apply(lambda x: display_name_map.get(x, x))
                edges_df['target'] = edges_df['target'].apply(lambda x: display_name_map.get(x, x))
                
                # Update edge_labels keys to use display names
                new_edge_labels = {}
                for (src, tgt), info in edge_labels.items():
                    new_src = display_name_map.get(src, src)
                    new_tgt = display_name_map.get(tgt, tgt)
                    new_edge_labels[(new_src, new_tgt)] = info
                edge_labels = new_edge_labels
                
                self._log(f"  Applied display names to {len(display_name_map)} nodes with cross-dataset name differences")

        # Identify source and target nodes from parameters
        # CRITICAL: Include ALL mapped type names across all datasets, not just original query types
        source_patterns = set(self.parameters._ensure_flat_list(self.parameters.source_neurons))
        target_patterns = set(self.parameters._ensure_flat_list(self.parameters.target_neurons))
        
        # If auto_type_mapping is enabled, also include resolved type names for each dataset
        if self.parameters.auto_type_mapping:
            for dataset in dataset_names:
                src_resolved = self.parameters.get_source_neurons_for_dataset(dataset)
                source_patterns.update(src_resolved)
                tgt_resolved = self.parameters.get_target_neurons_for_dataset(dataset)
                target_patterns.update(tgt_resolved)
        
        # Convert back to list for matching function
        source_patterns = list(source_patterns)
        target_patterns = list(target_patterns)
        
        # Get all unique nodes
        all_nodes = set(edges_df['source'].unique()) | set(edges_df['target'].unique())
        
        # Helper to extract base name without hemisphere suffix
        def get_base_name(label: str) -> str:
            """Extract base name from label, removing hemisphere suffix like _L, _R, _U."""
            base = get_canonical_name(label)
            if base.endswith(('_L', '_R', '_U')):
                return base[:-2]
            return base
        
        # Classify nodes as source, target, or intermediate
        import re
        separate_hemispheres = bool(getattr(self.parameters, 'separate_hemispheres', False))
        
        def matches_patterns(node: str, patterns: list) -> bool:
            """Check if node matches any pattern. Handles merged display names and hemisphere suffixes."""
            # Get canonical name for matching (handles merged display names)
            canonical = get_canonical_name(node)
            base = get_base_name(node)
            # Check the full label, canonical name, and base name
            names_to_check = list(set([node, canonical, base]))
            
            for name in names_to_check:
                for pattern in patterns:
                    if isinstance(pattern, str):
                        # Handle regex patterns
                        if '.*' in pattern or '*' in pattern:
                            regex = _wildcard_pattern_to_regex(pattern)
                            if re.match(f'^{regex}$', name, re.IGNORECASE):
                                return True
                        elif name.lower() == pattern.lower():
                            return True
                        # If separating hemispheres, allow exact base name to match suffixed labels
                        elif separate_hemispheres and '.*' not in pattern and '*' not in pattern:
                            if name.lower().startswith(pattern.lower() + '_'):
                                suffix = name[len(pattern) + 1:]
                                if suffix.upper() in ('L', 'R', 'U'):
                                    return True
            return False
        
        source_nodes = {n for n in all_nodes if matches_patterns(n, source_patterns)}
        target_nodes = {n for n in all_nodes if matches_patterns(n, target_patterns)}
        intermediate_nodes = all_nodes - source_nodes - target_nodes
        
        # Trim dead-ends if requested
        if trim_dead_ends and source_nodes and target_nodes:
            self._log("  Trimming dead-end nodes...")
            
            # Build adjacency for reachability analysis
            from collections import defaultdict
            forward_adj = defaultdict(set)  # node -> downstream nodes
            backward_adj = defaultdict(set)  # node -> upstream nodes
            
            for _, row in edges_df.iterrows():
                forward_adj[row['source']].add(row['target'])
                backward_adj[row['target']].add(row['source'])
            
            # Find nodes reachable from sources (forward)
            reachable_from_source = set()
            from collections import deque
            queue = deque(source_nodes)
            while queue:
                node = queue.popleft()
                if node in reachable_from_source:
                    continue
                reachable_from_source.add(node)
                for next_node in forward_adj[node]:
                    if next_node not in reachable_from_source:
                        queue.append(next_node)
            
            # Find nodes that can reach targets (backward)
            can_reach_target = set()
            queue = deque(target_nodes)
            while queue:
                node = queue.popleft()
                if node in can_reach_target:
                    continue
                can_reach_target.add(node)
                for prev_node in backward_adj[node]:
                    if prev_node not in can_reach_target:
                        queue.append(prev_node)
            
            # Keep only nodes that are on paths from source to target
            valid_nodes = reachable_from_source & can_reach_target
            
            # Filter edges to only include valid nodes
            edges_df = edges_df[
                edges_df['source'].isin(valid_nodes) & 
                edges_df['target'].isin(valid_nodes)
            ].copy()
            
            # Filter edge labels
            edge_labels = {
                k: v for k, v in edge_labels.items() 
                if k[0] in valid_nodes and k[1] in valid_nodes
            }
            
            removed_count = len(all_nodes) - len(valid_nodes)
            if removed_count > 0:
                self._log(f"  Removed {removed_count} dead-end nodes, {len(edges_df)} edges remaining")
        
        if edges_df.empty:
            self._log("Warning: No edges remaining after trimming dead-ends.")
            return None
        
        # Setup output path (at root level, parallel to comparison_report.html)
        if output_folder is None:
            output_folder = os.path.join(
                self.parameters.full_output_path,
                "conserved_paths"
            )
        os.makedirs(output_folder, exist_ok=True)
        
        base_filename = f"conserved_network_t{threshold}"
        
        self._log(f"  Creating VisualizePath visualization with {len(edges_df)} edges...")
        
        # Create VisualizePath with conserved edges
        vp = VisualizePath(
            path_file=edges_df,
            output_folder=output_folder,
            showfig=showfig,
            network_layout=network_layout,
            edge_width_scale=edge_width_scale,
            edge_labels=edge_labels,  # Multi-dataset synapse strengths
            dataset_legend=dataset_legend,  # Dataset short code legend for display names
            node_dataset_info=node_dataset_info,  # Node-level dataset info for hover labels
            verbose=self.verbose,
            separate_hemispheres=self.parameters.separate_hemispheres,
            **vispath_kwargs
        )
        
        # Override base filename
        vp.base_filename = base_filename
        
        # Build network and create visualization
        vp.build_network()
        
        # Set node types for coloring
        for node in vp.G_network.nodes():
            if node in source_nodes:
                vp.G_network.nodes[node]['node_type'] = 'source'
            elif node in target_nodes:
                vp.G_network.nodes[node]['node_type'] = 'target'
            else:
                vp.G_network.nodes[node]['node_type'] = 'intermediate'
        
        # Generate network visualization
        output_path = vp.create_network()
        output_path_heatmap = vp.create_heatmap()
        
        self._log_file(output_path, "Saved conserved path visualization")
        
        return output_path
    
    def visualize_conserved_paths_all_thresholds(
        self,
        trim_dead_ends: bool = True,
        output_folder: Optional[str] = None,
        showfig: bool = False,
        **vispath_kwargs
    ) -> List[str]:
        """
        Generate conserved path visualizations for all thresholds.
        
        Creates a subfolder 'conserved_paths/' containing one network 
        visualization per threshold, each showing only edges conserved
        across all datasets with multi-dataset synapse strengths.
        
        Args:
            trim_dead_ends: If True, removes dead-end nodes.
            output_folder: Output folder for visualizations. If None, uses
                          {comparison_output}/comparison_visualizations/conserved_paths/
            showfig: If True, opens visualizations in browser.
            **vispath_kwargs: Additional keyword arguments for VisualizePath.
            
        Returns:
            List of paths to generated HTML files.
        """
        # Set up output folder for all thresholds (at root level, parallel to comparison_report.html)
        if output_folder is None:
            output_folder = os.path.join(
                self.parameters.full_output_path,
                "conserved_paths"
            )
        os.makedirs(output_folder, exist_ok=True)
        
        self._log(f"Generating conserved path visualizations for {len(self.parameters.thresholds)} thresholds...")
        
        output_paths = []
        
        for threshold in self.parameters.thresholds:
            result = self.visualize_conserved_paths(
                threshold=threshold,
                trim_dead_ends=trim_dead_ends,
                output_folder=output_folder,  # Use shared folder
                showfig=showfig,
                **vispath_kwargs
            )
            if result:
                output_paths.append(result)
        
        if output_paths:
            self._log(f"  Generated {len(output_paths)} conserved path visualizations in: conserved_paths/")
        
        return output_paths

    # =========================================================================
    # Conserved Reciprocal Graph Visualization
    # =========================================================================

    def visualize_conserved_reciprocal_graph(
        self,
        threshold: Optional[int] = None,
        trim_dead_ends: bool = True,
        output_folder: Optional[str] = None,
        showfig: bool = False,
        network_layout: str = 'hierarchical',
        edge_width_scale: str = 'log',
        **vispath_kwargs
    ) -> Optional[str]:
        """Visualize conserved edges from reciprocal graphs across datasets."""
        try:
            from vispath_pkg import VisualizePath
        except ImportError:
            self._log("Warning: vispath_pkg not available. Cannot generate conserved reciprocal visualization.")
            return None

        if not self.raw_results:
            self._log("Warning: No comparison results. Run run_comparison() first.")
            return None

        dataset_names = self.parameters.get_dataset_names()
        dataset_nickname_map = self.parameters.get_nickname_map()

        if threshold is None:
            threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]

        self._log(f"Generating conserved reciprocal graph @ threshold={threshold}...")

        aligned = self.get_aligned_data_for_network(threshold)
        if aligned.empty:
            self._log("Warning: No aligned reciprocal data at this threshold.")
            return None

        available_ds = [d for d in dataset_names if d in aligned.columns]
        if not available_ds:
            self._log("Warning: No datasets with aligned reciprocal data.")
            return None

        mask_all = (aligned[available_ds] > 0).all(axis=1)
        conserved_edges = aligned[mask_all].copy()
        if conserved_edges.empty:
            self._log("Warning: No conserved reciprocal edges found at this threshold.")
            return None

        self._log(f"  Found {len(conserved_edges)} conserved reciprocal edges")

        ratio_data = self._get_reciprocal_edge_ratio_data_for_threshold(threshold)

        edge_list = []
        edge_labels = {}

        for edge_key, row in conserved_edges.iterrows():
            if ' -> ' in str(edge_key):
                parts = str(edge_key).split(' -> ')
                source = parts[0].strip()
                target = parts[1].strip() if len(parts) > 1 else ''
            else:
                continue

            edge_info = {}
            avg_weight = 0
            for dataset in available_ds:
                weight = row[dataset]
                if weight > 0:
                    # Keep release-qualified aliases unique in reciprocal
                    # graph hover labels as well.
                    nickname = dataset_nickname_map.get(
                        dataset, self.parameters._sanitize_name(dataset)
                    )
                    edge_info[f'{nickname} wt'] = int(weight)

                    if not ratio_data.empty and dataset in ratio_data.columns:
                        edge_key_for_ratio = str(edge_key)
                        if edge_key_for_ratio in ratio_data.index:
                            ratio_val = ratio_data.loc[edge_key_for_ratio, dataset]
                            if isinstance(ratio_val, pd.Series):
                                ratio_val = ratio_val.iloc[0]
                            if ratio_val > 0:
                                edge_info[f'{nickname} ratio'] = round(ratio_val, 4)

                    avg_weight += weight

            avg_weight = avg_weight / len(available_ds) if available_ds else 0

            edge_list.append({'source': source, 'target': target, 'weight': avg_weight})
            edge_labels[(source, target)] = edge_info

        if not edge_list:
            self._log("Warning: No valid edges to visualize.")
            return None

        edges_df = pd.DataFrame(edge_list)

        source_patterns = set(self.parameters._ensure_flat_list(self.parameters.source_neurons))
        target_patterns = set(self.parameters._ensure_flat_list(self.parameters.target_neurons))
        if self.parameters.auto_type_mapping:
            for dataset in dataset_names:
                source_patterns.update(self.parameters.get_source_neurons_for_dataset(dataset))
                target_patterns.update(self.parameters.get_target_neurons_for_dataset(dataset))
        source_patterns = list(source_patterns)
        target_patterns = list(target_patterns)

        all_nodes = set(edges_df['source'].unique()) | set(edges_df['target'].unique())

        import re
        separate_hemispheres = bool(getattr(self.parameters, 'separate_hemispheres', False))

        def get_canonical_name(display_name: str) -> str:
            """Extract canonical name from display name with potential annotations."""
            if '(' in display_name:
                return display_name.split('(')[0].strip()
            return display_name.strip()

        def get_base_name(label: str) -> str:
            base = get_canonical_name(label)
            if base.endswith(('_L', '_R', '_U')):
                return base[:-2]
            return base

        def matches_patterns(node: str, patterns: list) -> bool:
            canonical = get_canonical_name(node)
            base = get_base_name(node)
            names_to_check = list(set([node, canonical, base]))
            for name in names_to_check:
                for pattern in patterns:
                    if isinstance(pattern, str):
                        if '.*' in pattern or '*' in pattern:
                            regex = _wildcard_pattern_to_regex(pattern)
                            if re.match(f'^{regex}$', name, re.IGNORECASE):
                                return True
                        elif name.lower() == pattern.lower():
                            return True
                        elif separate_hemispheres and '.*' not in pattern and '*' not in pattern:
                            if name.lower().startswith(pattern.lower() + '_'):
                                suffix = name[len(pattern) + 1:]
                                if suffix.upper() in ('L', 'R', 'U'):
                                    return True
            return False

        source_nodes = {n for n in all_nodes if matches_patterns(n, source_patterns)}
        target_nodes = {n for n in all_nodes if matches_patterns(n, target_patterns)}
        intermediate_nodes = all_nodes - source_nodes - target_nodes

        if trim_dead_ends and source_nodes and target_nodes:
            from collections import defaultdict
            forward_adj = defaultdict(set)
            backward_adj = defaultdict(set)
            for _, row in edges_df.iterrows():
                forward_adj[row['source']].add(row['target'])
                backward_adj[row['target']].add(row['source'])

            reachable_from_source = set()
            from collections import deque
            queue = deque(source_nodes)
            while queue:
                node = queue.popleft()
                if node in reachable_from_source:
                    continue
                reachable_from_source.add(node)
                for next_node in forward_adj[node]:
                    if next_node not in reachable_from_source:
                        queue.append(next_node)

            can_reach_target = set()
            queue = deque(target_nodes)
            while queue:
                node = queue.popleft()
                if node in can_reach_target:
                    continue
                can_reach_target.add(node)
                for prev_node in backward_adj[node]:
                    if prev_node not in can_reach_target:
                        queue.append(prev_node)

            valid_nodes = reachable_from_source & can_reach_target
            edges_df = edges_df[
                edges_df['source'].isin(valid_nodes) &
                edges_df['target'].isin(valid_nodes)
            ].copy()
            edge_labels = {
                k: v for k, v in edge_labels.items()
                if k[0] in valid_nodes and k[1] in valid_nodes
            }

        if edges_df.empty:
            self._log("Warning: No edges remaining after trimming dead-ends.")
            return None

        if output_folder is None:
            output_folder = os.path.join(
                self.parameters.full_output_path,
                "conserved_reciprocal_graph"
            )
        os.makedirs(output_folder, exist_ok=True)

        base_filename = f"conserved_reciprocal_t{threshold}"

        vp = VisualizePath(
            path_file=edges_df,
            output_folder=output_folder,
            showfig=showfig,
            network_layout=network_layout,
            edge_width_scale=edge_width_scale,
            edge_labels=edge_labels,
            verbose=self.verbose,
            separate_hemispheres=self.parameters.separate_hemispheres,
            **vispath_kwargs
        )

        vp.base_filename = base_filename
        vp.build_network()

        for node in vp.G_network.nodes():
            if node in source_nodes:
                vp.G_network.nodes[node]['node_type'] = 'source'
            elif node in target_nodes:
                vp.G_network.nodes[node]['node_type'] = 'target'
            else:
                vp.G_network.nodes[node]['node_type'] = 'intermediate'

        output_path = vp.create_network()
        self._log_file(output_path, "Saved conserved reciprocal graph")

        return output_path

    def visualize_conserved_reciprocal_graph_all_thresholds(
        self,
        trim_dead_ends: bool = True,
        output_folder: Optional[str] = None,
        showfig: bool = False,
        **vispath_kwargs
    ) -> List[str]:
        """Generate conserved reciprocal graph visualizations for all thresholds."""
        if output_folder is None:
            output_folder = os.path.join(
                self.parameters.full_output_path,
                "conserved_reciprocal_graph"
            )
        os.makedirs(output_folder, exist_ok=True)

        self._log(f"Generating conserved reciprocal graphs for {len(self.parameters.thresholds)} thresholds...")

        output_paths = []
        for threshold in self.parameters.thresholds:
            result = self.visualize_conserved_reciprocal_graph(
                threshold=threshold,
                trim_dead_ends=trim_dead_ends,
                output_folder=output_folder,
                showfig=showfig,
                **vispath_kwargs
            )
            if result:
                output_paths.append(result)

        if output_paths:
            self._log(f"  Generated {len(output_paths)} conserved reciprocal graphs in: conserved_reciprocal_graph/")

        return output_paths
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_dataset_config(self, name: str) -> Optional[DatasetConfig]:
        """Get dataset configuration by name."""
        return self._dataset_configs.get(name)
    
    def clear_results(self):
        """Clear all cached results."""
        self.raw_results.clear()
        self.aligned_results.clear()
        self.comparison_report = None
        self._log("Results cleared")
    
    def set_label_mapper(self, mapper: LabelMapper):
        """
        Set or update label mapper.
        
        Args:
            mapper: LabelMapper instance
        """
        self.label_mapper = mapper
        # Clear aligned results since labels may change
        self.aligned_results.clear()
        self._log("Label mapper updated")
    
    def generate_html_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate an interactive HTML report with Plotly charts.
        
        Creates a self-contained HTML file with:
        - Executive summary with key findings
        - Interactive path count charts
        - Edge weight heatmap
        - Conservation analysis tables
        - Dataset comparison dashboard
        
        Args:
            output_path: Path to save the HTML report (default: comparison_results/comparison_report.html)
            
        Returns:
            Path to the generated HTML file
        """
        # Ensure comparison has been run
        if self.comparison_report is None:
            self.run_comparison_analysis()
        
        if output_path is None:
            output_path = os.path.join(
                self.parameters.full_output_path,
                "comparison_results",
                "comparison_report.html"
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Try to import plotly, fall back to basic HTML if not available
        try:
            import plotly  # noqa: F401
            has_plotly = True
        except ImportError:
            has_plotly = False
            self._log("Warning: Plotly not installed. Generating basic HTML report.")
        
        html_content = self._generate_html_content(has_plotly)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self._log(f"HTML report saved to: {output_path}")
        return output_path
    
    def _generate_html_content(self, has_plotly: bool = True) -> str:
        """Generate the HTML content for the report - static version showing all thresholds."""
        from .html_report_generator import generate_html_report
        
        dataset_names = self.parameters.get_dataset_names()
        thresholds = self.parameters.thresholds
        
        # Generate mode-specific note
        mode_specific_note = self._generate_mode_specific_note()
        
        # Collect path count data for charts
        path_count_data = []
        for dataset in dataset_names:
            for threshold in thresholds:
                df = self.raw_results.get(dataset, {}).get(threshold, pd.DataFrame())
                count = len(df) if not df.empty else 0
                path_count_data.append({
                    'dataset': self.parameters._sanitize_name(dataset),
                    'threshold': threshold,
                    'count': count
                })
        
        # Get key findings per threshold and add path stats
        key_findings_per_threshold = self.comparison_report.get('key_findings_per_threshold', {})
        
        # Get path presence matrix for accurate path counts
        path_presence_matrix = self.comparison_report.get('path_presence_matrix', pd.DataFrame())
        
        # Add path statistics to key findings using path presence matrix
        for threshold in thresholds:
            if not path_presence_matrix.empty:
                # Use path presence matrix (sanitized names with _t{threshold} suffix)
                total_paths = 0
                common_paths = 0
                
                # Find columns for this threshold
                cols_for_threshold = []
                for d in dataset_names:
                    safe_name = self.parameters._sanitize_name(d)
                    col_name = f'{safe_name}_t{threshold}'
                    if col_name in path_presence_matrix.columns:
                        cols_for_threshold.append(col_name)
                
                if cols_for_threshold:
                    # Count paths present in at least one dataset at this threshold
                    is_any = pd.Series(False, index=path_presence_matrix.index)
                    for col in cols_for_threshold:
                        vals = path_presence_matrix[col]
                        if vals.dtype == object:
                            is_any |= ((vals == 'True') | (vals == True))
                        elif vals.dtype == bool:
                            is_any |= vals
                        else:
                            is_any |= (vals > 0)
                    total_paths = int(is_any.sum())
                    
                    # Count common paths (present in ALL datasets at this threshold)
                    if len(cols_for_threshold) == len(dataset_names):
                        is_common = pd.Series(True, index=path_presence_matrix.index)
                        for col in cols_for_threshold:
                            vals = path_presence_matrix[col]
                            if vals.dtype == object:
                                is_common &= ((vals == 'True') | (vals == True))
                            elif vals.dtype == bool:
                                is_common &= vals
                            else:
                                is_common &= (vals > 0)
                        common_paths = int(is_common.sum())
                
                if threshold in key_findings_per_threshold:
                    key_findings_per_threshold[threshold]['total_paths'] = total_paths
                    key_findings_per_threshold[threshold]['common_paths'] = common_paths
                    if total_paths > 0:
                        key_findings_per_threshold[threshold]['path_conservation_rate'] = common_paths / total_paths
        
        # Generate HTML using the new generator
        return generate_html_report(
            analyzer=self,
            dataset_names=dataset_names,
            thresholds=thresholds,
            mode_specific_note=mode_specific_note,
            path_count_data=path_count_data,
            key_findings_per_threshold=key_findings_per_threshold
        )


def quick_compare(
    datasets: List[Union[str, DatasetConfig]],
    source_neurons: List[Union[str, int]],
    target_neurons: List[Union[str, int]],
    max_interlayer: int = 2,
    thresholds: Optional[List[int]] = None,
    label_mapper: Optional[LabelMapper] = None,
    output_folder: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for quick cross-dataset comparison.
    
    Args:
        datasets: List of dataset identifiers (strings) or DatasetConfig objects
        source_neurons: Source neuron types/patterns (shared across all datasets)
        target_neurons: Target neuron types/patterns (shared across all datasets)
        max_interlayer: Maximum interlayer hops (default: 2)
        thresholds: List of weight thresholds (default: [1, 3, 5, 10, 20])
        label_mapper: Optional LabelMapper for standardization
        output_folder: Optional output directory
        verbose: Print progress
        
    Returns:
        Comparison results dictionary
        
    Example:
        >>> results = quick_compare(
        ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
        ...     source_neurons=['MBON14.*_R'],
        ...     target_neurons=['KCg-d.*_R'],
        ...     max_interlayer=2,
        ...     thresholds=[1, 5, 10]
        ... )
    """
    if thresholds is None:
        thresholds = [1, 3, 5, 10, 20]
    
    params = ComparisonParameters(
        datasets=datasets,
        source_neurons=source_neurons,
        target_neurons=target_neurons,
        max_interlayer=max_interlayer,
        thresholds=thresholds,
        output_folder=output_folder or '.',
    )
    
    analyzer = ComparisonAnalyzer(params, label_mapper, verbose)
    results = analyzer.run_comparison()
    
    if output_folder:
        analyzer.export_results()
    
    return results
