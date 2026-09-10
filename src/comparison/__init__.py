"""
Cross-Dataset Comparison Module

This module provides tools for systematic comparison of connectome data across multiple datasets.
Supports hemibrain, male-cns, FAFB, BANC, and optic-lobe datasets.

OPTIMIZED WORKFLOW:
    1. Create ComparisonParameters (PRIMARY ENTRY POINT) with:
       - datasets: List of dataset strings ('hemibrain:v1.2.1', 'male-cns:v0.9', etc.)
       - source_neurons: List of neuron patterns (shared across all datasets)
       - target_neurons: List of neuron patterns (shared across all datasets)
       - max_interlayer: Maximum path hops (shared across all datasets)
       - thresholds: Synapse count cutoffs for analysis
    
    2. Create ComparisonAnalyzer with parameters
    3. Run comparison: analyzer.run_comparison()
    4. Generate report: analyzer.generate_report()

Main components:
- ComparisonParameters: PRIMARY ENTRY POINT - all settings defined here
- ComparisonAnalyzer: Main orchestrator class for running comparisons
- LabelMapper: Cross-dataset label standardization (optional)
- DatasetConfig: Per-dataset configuration (auto-created from strings)

Quick Start:
    >>> from src.comparison import ComparisonParameters, ComparisonAnalyzer
    >>> 
    >>> params = ComparisonParameters(
    ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    ...     source_neurons=['MBON14.*_R'],
    ...     target_neurons=['KCg-d.*_R'],
    ...     max_interlayer=2,
    ...     thresholds=[1, 3, 5, 10, 20],
    ... )
    >>> 
    >>> analyzer = ComparisonAnalyzer(params)
    >>> results = analyzer.run_comparison()
    >>> report = analyzer.generate_report()

Or use quick_compare for one-liner:
    >>> from src.comparison import quick_compare
    >>> 
    >>> results = quick_compare(
    ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    ...     source_neurons=['MBON14.*_R'],
    ...     target_neurons=['KCg-d.*_R'],
    ... )
"""

from .comparison_parameters import ComparisonParameters
from .comparison_analyzer import ComparisonAnalyzer, quick_compare
from .label_mapper import LabelMapper
from .dataset_config import DatasetConfig
from .data_loader import DataLoader
from .metrics import ComparisonMetrics
from .visualizations import ComparisonVisualizer

# Connectivity Profile modules (new)
from .connectivity_profiler import (
    ConnectivityProfile,
    ConnectivityProfiler,
    ConnectivityStatus,
    ProfilerConfig,
    FuzzyMatchConfig,
    normalize_partner_type,
    compute_ranks,
    clear_profiler_conn_cache,
    get_profiler_conn_cache_info,
    DataNotAvailableError,
)
from .profile_comparator import (
    ProfileComparator,
    ComparisonResult,
    DEFAULT_SCORE_WEIGHTS,
    HomologFinder,
    ConnectivityProfileComparer,
)
from .profile_visualizations import ProfileVisualizer
from .cross_dataset_type_mapper import (
    CrossDatasetTypeMapper,
    TypeMappingWarning,
    TypeMappingConflict,
    get_type_mapper,
    DATASET_PRIORITY,
)

__all__ = [
    # Primary entry point
    'ComparisonParameters',
    'ComparisonAnalyzer',
    'quick_compare',
    
    # Optional components
    'LabelMapper',
    'DatasetConfig',
    
    # Cross-dataset type mapping
    'CrossDatasetTypeMapper',
    'TypeMappingWarning',
    'TypeMappingConflict',
    'get_type_mapper',
    'DATASET_PRIORITY',
    
    # Advanced usage
    'DataLoader',
    'ComparisonMetrics',
    'ComparisonVisualizer',
    
    # Connectivity Profile module (new)
    'ConnectivityProfile',
    'ConnectivityProfiler',
    'ConnectivityStatus',
    'ProfilerConfig',
    'FuzzyMatchConfig',
    'normalize_partner_type',
    'compute_ranks',
    'clear_profiler_conn_cache',
    'get_profiler_conn_cache_info',
    'DataNotAvailableError',
    'ProfileComparator',
    'ComparisonResult',
    'DEFAULT_SCORE_WEIGHTS',
    'HomologFinder',
    'ConnectivityProfileComparer',
    'ProfileVisualizer',
]
