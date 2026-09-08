#!/usr/bin/env python3
"""
Build Connectivity Profile Cache

Simplified script that uses FindNeuronConnection.build_connectivity_profile_cache()
to pre-build connectivity profiles for all neuron types in a dataset.

Connectivity profiles are used for homolog finding and cross-dataset comparisons.
The profile cache stores for each neuron type:
- Upstream and downstream partner weights and ranks
- Unique types counts
- 2-hop expansion data for untyped partners

Usage:
    python build_connectivity_profile_cache.py <dataset> [options]
    
Examples:
    # Build cache for hemibrain
    python build_connectivity_profile_cache.py hemibrain:v1.2.1
    
    # Build cache with custom parameters
    python build_connectivity_profile_cache.py hemibrain:v1.2.1 --top-k 15 --top-m 8
    
    # Limit to first 100 neurons (for testing)
    python build_connectivity_profile_cache.py hemibrain:v1.2.1 --max-neurons 100
    
    # Show cache statistics
    python build_connectivity_profile_cache.py --stats hemibrain:v1.2.1

Author: Hemibrain Analysis Team
Date: December 2024
"""

import os
import sys
import time
import argparse
from pathlib import Path

try:
    from .utils.naming_utils import canonical_dataset_name
except ImportError:  # pragma: no cover - src laid bare on sys.path
    from utils.naming_utils import canonical_dataset_name

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# DEFAULT PARAMETERS - Edit these to run the script directly
# =============================================================================
DEFAULT_DATASET = 'hemibrain:v1.2.1'  # Dataset name (use : format for NeuPrint)
DEFAULT_TOKEN = None  # NeuPrint API token (or set NEUPRINT_TOKEN env var)
DEFAULT_SERVER = 'https://neuprint.janelia.org'  # NeuPrint server
DEFAULT_TOP_K = 20  # Store top N partners by weight
DEFAULT_TOP_M = 10  # Ensure at least M unique types via expansion
DEFAULT_EXPAND_2HOP = True  # Enable 2-hop expansion for untyped partners
DEFAULT_MAX_NEURONS = None  # Limit to first N neurons (None = all)
# =============================================================================


def build_cache(
    dataset: str,
    token: str = None,
    server: str = DEFAULT_SERVER,
    top_k: int = DEFAULT_TOP_K,
    top_m: int = DEFAULT_TOP_M,
    expand_2hop: bool = DEFAULT_EXPAND_2HOP,
    max_neurons: int = None,
    neuron_types: list = None,
    force: bool = False
) -> dict:
    """
    Build connectivity profile cache using FindNeuronConnection.
    
    Parameters:
        dataset: Dataset name (e.g., 'hemibrain:v1.2.1', 'flywire_FAFB_v783')
        token: NeuPrint API token (required for NeuPrint datasets)
        server: NeuPrint server URL
        top_k: Store top N partners by weight
        top_m: Ensure at least M unique types via expansion
        expand_2hop: Enable 2-hop expansion for untyped partners
        max_neurons: Limit to first N neurons (for testing)
        neuron_types: Specific neuron types to cache
        force: Force rebuild even if cache exists
        
    Returns:
        dict: Summary with total_profiles, profiles, failed_types, elapsed_time
    """
    from coana import FindNeuronConnection
    
    # Get token through the standard chain: config.json wins per key, then
    # the gitignored config_local.json, then the environment (canonical
    # NEUPRINT_APPLICATION_CREDENTIALS; NEUPRINT_TOKEN as a legacy alias).
    if token is None:
        try:
            from utils.token_manager import token_manager
            token = token_manager.get_neuprint_token()
        except ImportError:
            token = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS') \
                or os.environ.get('NEUPRINT_TOKEN')
        if token:
            print("[INFO] Using NeuPrint token from the config/env chain")
    
    print("=" * 60)
    print("Building Connectivity Profile Cache")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Parameters: top_k={top_k}, top_m={top_m}, expand_2hop={expand_2hop}")
    if max_neurons:
        print(f"Max neurons: {max_neurons}")
    print()
    
    # Progress bar callback
    def progress_callback(current: int, total: int, neuron_type: str):
        pct = 100 * current / total if total > 0 else 0
        bar_len = 40
        filled = int(bar_len * current / total) if total > 0 else 0
        bar = '█' * filled + '░' * (bar_len - filled)
        type_display = neuron_type[:30] if neuron_type else ''
        print(f"\r[{bar}] {pct:5.1f}% ({current}/{total}) - {type_display:30s}", 
              end='', flush=True)
    
    # Initialize FNC with cache enabled
    fnc = FindNeuronConnection(
        dataset=dataset,
        server=server,
        token=token,
        use_cache=True,
        verbose_mode='minimal'
    )
    
    # Build cache using FNC's built-in method
    result = fnc.build_connectivity_profile_cache(
        neuron_types=neuron_types,
        top_k=top_k,
        top_m=top_m,
        expand_2hop=expand_2hop,
        max_neurons=max_neurons,
        force_refresh=force,
        progress_callback=progress_callback
    )
    
    print()  # New line after progress bar
    
    # Show sample profile
    profiles = result.get('profiles', {})
    if profiles:
        sample_type = list(profiles.keys())[0]
        sample_profile = profiles[sample_type]
        
        print(f"\nSample Profile: {sample_type}")
        print("-" * 40)
        print(f"  Upstream partners: {len(sample_profile.upstream_partners or {})}")
        print(f"  Downstream partners: {len(sample_profile.downstream_partners or {})}")
        print(f"  Unique types upstream: {sample_profile.unique_types_upstream}")
        print(f"  Unique types downstream: {sample_profile.unique_types_downstream}")
    
    return result


def show_stats(dataset: str) -> None:
    """Show cache statistics for a dataset."""
    import numpy as np
    
    # Normalize dataset name
    safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
    
    # Get paths
    src_dir = Path(__file__).parent
    project_root = src_dir.parent
    cache_path = project_root / 'cache' / safe_name / 'connectivity_profiles.parquet'
    
    print()
    print("=" * 60)
    print(f"Connectivity Profile Cache Statistics: {dataset}")
    print("=" * 60)
    
    if not cache_path.exists():
        print(f"\n[ERROR] Cache not found: {cache_path}")
        print(f"        Run: python build_connectivity_profile_cache.py {dataset}")
        return
    
    # Basic file info
    print(f"\nCache file: {cache_path}")
    print(f"File size: {cache_path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"Last modified: {time.ctime(cache_path.stat().st_mtime)}")
    
    try:
        from comparison.connectivity_profiler import ConnectivityProfiler, ProfilerConfig
        
        profiler = ConnectivityProfiler(ProfilerConfig(verbose=False))
        profiles = profiler.read_connectivity_profile_cache(dataset)
        
        if profiles:
            print(f"\nTotal profiles: {len(profiles)}")
            
            # Compute statistics
            up_counts = []
            down_counts = []
            has_2hop = 0
            
            for p in profiles.values():
                up_counts.append(len(p.upstream_partners or {}))
                down_counts.append(len(p.downstream_partners or {}))
                if p.untyped_upstream_2hop or p.untyped_downstream_2hop:
                    has_2hop += 1
            
            print(f"\nPartner Statistics:")
            print(f"  Upstream partners: avg={np.mean(up_counts):.1f}, max={max(up_counts)}")
            print(f"  Downstream partners: avg={np.mean(down_counts):.1f}, max={max(down_counts)}")
            print(f"  Profiles with 2-hop data: {has_2hop} ({100*has_2hop/len(profiles):.1f}%)")
            
            # Sample types
            print(f"\nSample types (first 5):")
            for ntype in list(profiles.keys())[:5]:
                print(f"  {ntype}")
    except Exception as e:
        print(f"\n[WARNING] Could not read profiles: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build connectivity profile cache using FindNeuronConnection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default parameters (edit DEFAULT_* at top of file)
    python build_connectivity_profile_cache.py
    
    # Build cache for hemibrain
    python build_connectivity_profile_cache.py hemibrain:v1.2.1
    
    # Build cache with custom parameters
    python build_connectivity_profile_cache.py hemibrain:v1.2.1 --top-k 15 --top-m 8
    
    # Build cache for FAFB (BANC is a separate public-release source)
    python build_connectivity_profile_cache.py flywire_FAFB_v783
    
    # Limit to first 100 neurons (for testing)
    python build_connectivity_profile_cache.py hemibrain:v1.2.1 --max-neurons 100
    
    # Show cache statistics
    python build_connectivity_profile_cache.py --stats
"""
    )
    
    parser.add_argument('dataset', nargs='?', default=DEFAULT_DATASET,
                        help='Dataset name (e.g., hemibrain:v1.2.1)')
    parser.add_argument('--token', '-t', type=str, default=DEFAULT_TOKEN,
                        help='NeuPrint API token')
    parser.add_argument('--server', '-s', type=str, default=DEFAULT_SERVER,
                        help='NeuPrint server URL')
    parser.add_argument('--top-k', '-k', type=int, default=DEFAULT_TOP_K,
                        help='Store top N partners by weight (default: 10)')
    parser.add_argument('--top-m', '-m', type=int, default=DEFAULT_TOP_M,
                        help='Ensure at least M unique types (default: 5)')
    parser.add_argument('--expand-2hop', action='store_true', default=DEFAULT_EXPAND_2HOP,
                        help='Enable 2-hop expansion (default)')
    parser.add_argument('--no-expand-2hop', dest='expand_2hop', action='store_false',
                        help='Disable 2-hop expansion')
    parser.add_argument('--max-neurons', '-n', type=int, default=DEFAULT_MAX_NEURONS,
                        help='Limit to first N neurons (for testing)')
    parser.add_argument('--types', nargs='+', default=None,
                        help='Specific neuron types to cache')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force rebuild even if cache exists')
    parser.add_argument('--stats', action='store_true',
                        help='Show cache statistics')
    
    args = parser.parse_args()
    
    if args.stats:
        show_stats(args.dataset)
        return
    
    build_cache(
        dataset=args.dataset,
        token=args.token,
        server=args.server,
        top_k=args.top_k,
        top_m=args.top_m,
        expand_2hop=args.expand_2hop,
        max_neurons=args.max_neurons,
        neuron_types=args.types,
        force=args.force
    )


if __name__ == '__main__':
    main()
