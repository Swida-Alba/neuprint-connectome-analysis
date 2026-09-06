"""
CrossDatasetTypeMapper - Automatic type name mapping across datasets.

This module provides automatic type name mapping using the male-cns neuron_df file
which contains cross-dataset type columns (flywireType, hemibrainType, mancType).

The mapping is bodyId-based: each neuron in male-cns has its own type AND the
corresponding type name in other datasets. This allows for accurate cross-dataset
comparison even when type names differ.

FlyWire datasets also publish an extra additional-type column (FAFB:
``additional_type(s)``, BANC: ``Alternative Cell Type(s)``) that records type
renames: neurons whose type changed name keep the old name there while the
primary ``type`` column holds the current one. When a male-cns ``flywireType``
value is no longer a primary type in the target dataset but appears in that
column, the mapping resolves to the current primary name (e.g. male-cns
SLP249 -> FAFB APDN3). Crosswalk cells and additional-type cells may both list
several names separated by ','; each name is split out before mapping.

Key Features:
- Auto-loads type mappings from male-cns_v1_0_allneurons_neuron_df.csv
- Resolves renamed flywire types via the FlyWire additional Type(S) columns
- Handles 1-to-1, N-to-1, and 1-to-N type relationships
- Warns about N-to-1 aggregations that should be avoided
- Priority-based resolution: male-cns > flywire > manc > hemibrain > optic-lobe
- Graceful handling of missing mappings
- Integration with LabelMapper (LabelMapper has higher priority)
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from utils.naming_utils import dataset_abbrev
from collections import defaultdict
import pandas as pd

from comparison.label_mapper import LabelMapper

try:
    from ..utils.naming_utils import dataset_version, make_unique_dataset_labels
except ImportError:  # pragma: no cover - supports direct ``comparison`` imports
    from utils.naming_utils import dataset_version, make_unique_dataset_labels

# Dataset priority for type name resolution (lower index = higher priority)
DATASET_PRIORITY = [
    'male-cns:v1.0',
    'male-cns_v1_0',
    'flywire_FAFB_v783',
    'banc_v626',
    'flywire_FAFB',
    'banc',
    'manc:v1.0',
    'manc:v1.2.1',
    'manc_v1_0',
    'manc_v1_2_1',
    'hemibrain:v1.2.1',
    'hemibrain_v1_2_1',
    'optic-lobe:v1.1',
    'optic-lobe_v1_1',
]

# Mapping from dataset names to neuron_df column names
DATASET_TO_TYPE_COL = {
    'male-cns:v1.0': 'type',
    'male-cns_v1_0': 'type',
    'flywire_FAFB_v783': 'flywireType',
    'banc_v626': 'flywireType',  # BANC uses same flywireType col
    'flywire_FAFB': 'flywireType',
    'banc': 'flywireType',
    'manc:v1.0': 'mancType',
    'manc:v1.2.1': 'mancType',
    'manc_v1_0': 'mancType',
    'manc_v1_2_1': 'mancType',
    'hemibrain:v1.2.1': 'hemibrainType',
    'hemibrain_v1_2_1': 'hemibrainType',
    # optic-lobe not in male-cns mapping
}

# FlyWire schema namespaces kept in the type mappings.  FAFB and the BANC
# releases share the male-cns ``flywireType`` crosswalk column, but each
# release renames types independently through its own additional-type
# column, so their resolved names live under separate mapping keys —
# selecting BANC v888 must resolve against the v888 tables, never v626
# (§version control, user 2026-09-06).
FLYWIRE_MAPPING_KEYS = ('flywire_FAFB_v783', 'banc_v626', 'banc_v888')

# Per FlyWire namespace: which neuron table carries the primary ``type``
# column and which additional-type column records renamed types.
FLYWIRE_TYPE_SOURCES = {
    'flywire_FAFB_v783': {
        'dataset_dir': 'flywire_FAFB_v783',
        'neuron_df': 'flywire_FAFB_v783_allneurons_neuron_df.csv',
        'alt_column': 'additional_type(s)',
    },
    'banc_v626': {
        'dataset_dir': 'banc_v626',
        'neuron_df': 'banc_v626_allneurons_neuron_df.csv',
        'alt_column': 'Alternative Cell Type(s)',
    },
    'banc_v888': {
        'dataset_dir': 'banc_v888',
        'neuron_df': 'banc_v888_allneurons_neuron_df.csv',
        'alt_column': 'Alternative Cell Type(s)',
    },
}


# Standardized linker decomposition per ordered dataset pair: the ordered
# metadata columns that carry the bridge between the two `type` identities
# (§4 of _plan/plan-type-mapping-bodyid-bridge.md). Pairs absent from the
# registry derive their linkers from the chain hops.
BRIDGE_STANDARD = {
    ('male-cns:v1.0', 'flywire_FAFB_v783'): (
        ('flywireType', 'male-cns:v1.0'),
        ('additional_type(s)', 'flywire_FAFB_v783'),
    ),
    ('male-cns:v1.0', 'hemibrain:v1.2.1'): (
        ('hemibrainType', 'male-cns:v1.0'),
    ),
    ('male-cns:v1.0', 'manc:v1.0'): (
        ('mancType', 'male-cns:v1.0'),
    ),
    ('male-cns:v1.0', 'manc:v1.2.1'): (
        ('mancType', 'male-cns:v1.0'),
    ),
    ('male-cns:v1.0', 'manc:v1.2.3'): (
        ('mancType', 'male-cns:v1.0'),
    ),
    # BANC pairs map by type name (same-name or routed through FAFB), so
    # their standard carries no own linker columns.
}

# One color per matched linker column (the detailed linker graph paints
# each linker node by its column so same-named linkers disambiguate).
LINKER_COLORS = {
    'flywireType': '#f59e0b',
    'additional_type(s)': '#a855f7',
    'Alternative Cell Type(s)': '#c084fc',
    'hemibrainType': '#14b8a6',
    'mancType': '#ef4444',
}


CROSSWALK_COLUMNS = ("flywireType", "hemibrainType", "mancType")
ANNOTATION_COLUMNS = ("additional_type(s)", "Alternative Cell Type(s)")

# ---------------------------------------------------------------------------
# The declarative source map of the valid derivation bridges (§9I).
#
# Every non-``type`` edge the derivation walk may use MUST be licensed by an
# entry here: ``(home registry key, column) -> {registry keys the column's
# values land in}``.  Valid bridges are derived from this map — never by
# guessing from the data layout:
#
#   - ``type`` is the universal same-name identity between namespaces;
#   - the male-cns crosswalk columns route from the male-cns metadata into
#     their target families: flywireType reaches BOTH flywire datasets (FAFB
#     and BANC — 96%/97% of its values hit either namespace), hemibrainType
#     only hemibrain (100%), mancType only manc (the crosswalk was built
#     against MANC v1.0);
#   - the annotation columns are intra-namespace: ``additional_type(s)``
#     maps to flywire datasets only (it lives on the FAFB rows) and
#     ``Alternative Cell Type(s)`` only works for the BANC.
#
# ``bridge_is_valid`` (below) plus the walk enforce one more rule on top of
# the map: a crosswalk hop is only evidence when the bridge's ENDPOINT
# namespaces include a namespace the column routes to — a hemibrainType hop
# on a male-cns↔BANC bridge describes a third dataset's naming and is
# invalid, while the hemibrain↔flywire route through male-cns (hemibrain
# type → hemibrainType → male-cns type → flywireType → flywire type) is
# licensed on both legs.
# ---------------------------------------------------------------------------

BRIDGE_IDENTITY = "*"

# The crosswalk host: the namespace whose metadata table physically
# carries the crosswalk columns (the map's crosswalk home).
CROSSWALK_HOME = "male-cns:v1.0"

BRIDGE_SOURCE_MAP: Dict[tuple, Set[str]] = {
    (BRIDGE_IDENTITY, "type"): set(),  # same-name identity, any namespace
    (CROSSWALK_HOME, "flywireType"): {
        "flywire_FAFB_v783", "banc_v626", "banc_v888"},
    (CROSSWALK_HOME, "hemibrainType"): {"hemibrain:v1.2.1"},
    # the same male-cns column carries the names for both manc releases
    (CROSSWALK_HOME, "mancType"): {"manc:v1.0", "manc:v1.2.1"},
    ("flywire_FAFB_v783", "additional_type(s)"): {"flywire_FAFB_v783"},
    ("banc_v626", "Alternative Cell Type(s)"): {
        "banc_v626"},
    ("banc_v888", "Alternative Cell Type(s)"): {
        "banc_v888"},
}

# §bridge rules (2026-09-06): the licensed single-intermediate routes.
# MCNS connects the neuprint datasets (hemibrain, manc) to the flywire
# family and to each other; FAFB connects MCNS and BANC; BANC is never a
# connector (never between MCNS and FAFB).  Every other pair — including
# MCNS <-> FAFB and FAFB <-> BANC — is direct-only, plus the universal
# same-name identity.  All bridges are bidirectional, and a two-linker
# chain cannot flip its linker order (enforced structurally: a flipped
# chain would have to revisit a namespace, which the walk forbids).
ROUTE_MIDS: Dict[frozenset, Set[str]] = {
    frozenset({"male-cns:v1.0", "banc_v626"}): {"flywire_FAFB_v783"},
    frozenset({"male-cns:v1.0", "banc_v888"}): {"flywire_FAFB_v783"},
    frozenset({"hemibrain:v1.2.1", "flywire_FAFB_v783"}): {"male-cns:v1.0"},
    frozenset({"hemibrain:v1.2.1", "banc_v626"}): {"male-cns:v1.0"},
    frozenset({"hemibrain:v1.2.1", "banc_v888"}): {"male-cns:v1.0"},
    frozenset({"manc:v1.0", "flywire_FAFB_v783"}): {"male-cns:v1.0"},
    frozenset({"manc:v1.0", "banc_v626"}): {"male-cns:v1.0"},
    frozenset({"manc:v1.0", "banc_v888"}): {"male-cns:v1.0"},
    frozenset({"manc:v1.2.1", "flywire_FAFB_v783"}): {"male-cns:v1.0"},
    frozenset({"manc:v1.2.1", "banc_v626"}): {"male-cns:v1.0"},
    frozenset({"manc:v1.2.1", "banc_v888"}): {"male-cns:v1.0"},
    frozenset({"hemibrain:v1.2.1", "manc:v1.0"}): {"male-cns:v1.0"},
    frozenset({"hemibrain:v1.2.1", "manc:v1.2.1"}): {"male-cns:v1.0"},
}


def source_map_targets(home_key: str, column: str) -> Optional[Set[str]]:
    """Landing namespaces licensed for one ``(home, column)`` edge.

    ``None`` means the universal same-name ``type`` identity; an empty
    set means the column is not licensed at all.
    """
    if column == "type":
        return None
    return set(BRIDGE_SOURCE_MAP.get((home_key, column), ()))


def crosswalk_route_licensed(column: str, source_key: str,
                             target_key: str) -> bool:
    """Endpoint-family licensing for one crosswalk hop (§9I).

    ``column`` may appear in a bridge between ``source_key`` and
    ``target_key`` only when one of the endpoints is a namespace the
    column routes to.
    """
    for (home, col), targets in BRIDGE_SOURCE_MAP.items():
        if col != column or home == BRIDGE_IDENTITY:
            continue
        if targets & {source_key, target_key}:
            return True
    return False


def hop_home(hop: Dict[str, str], source_dataset: str) -> str:
    """The dataset whose metadata physically holds a chain hop's column.

    Crosswalk columns (flywireType/hemibrainType/mancType) physically
    live on the male-cns crosswalk rows (CROSSWALK_HOME) no matter which
    direction the chain travels — the forward leg reads a male-cns
    type's cell, the reverse leg matches a foreign name against the
    cells — while every other hop (``type`` identities in
    intermediate/final namespaces, annotation columns) belongs to the
    dataset recorded on the hop itself.  Using the hop's own dataset
    for ``type`` hops keeps transitive same-name chains honest (e.g.
    DN1pA[MCNS·type] → DN1pA[FAFB·type], not two MCNS hops) and never
    renames the final target identity.
    """
    if hop.get("column") in CROSSWALK_COLUMNS:
        return CROSSWALK_HOME
    return hop.get("dataset", source_dataset)


def preferred_bridge_chain(chains, source_dataset: str,
                           target_dataset: str):
    """The most representative chain of one mapped pair.

    Among the chains that end at the pair's foreign type, prefer the one
    standardized with the most DIRECT (registry) linkers — the transitive
    same-name routes (via other namespaces) and hub detours are kept as
    alternative bridges but must not drive bodyId pooling or the
    primary hover.  Ties: fewer linkers, then the shorter chain.
    """
    best = None
    best_key = None
    for chain in chains or []:
        if not chain:
            continue
        linkers = standardize_bridge(chain, source_dataset, target_dataset)
        direct = sum(1 for l in linkers
                     if l["kind"] == "linker" and not l["indirect"])
        total = len(linkers)
        # bare name-equality chains sink below ANY linker chain — a
        # metadata verification (even a non-registry crosswalk hop)
        # always outranks the unverified name echo
        key = (0 if total else 1, -direct, total, len(chain))
        if best_key is None or key < best_key:
            best, best_key = chain, key
    return best


def bridge_linker_text(chains: List[List[Dict[str, str]]],
                       source_dataset: str, target_dataset: str,
                       foreign_type: str) -> Dict[str, Any]:
    """Deduplicated linker entries + display text for one mapped pair.

    ``chains`` are the derivation chains for
    ``(source_type → foreign_type)`` — only chains whose final hop lands
    on ``foreign_type`` are used.  Returns ``{'entries': […],
    'text': …}`` where each entry is a standardized linker
    (``column``/``value``/``home``/``indirect``/``text``) deduplicated
    by ``(column, value)``; indirect (hub-route) linkers sort last and
    carry the hub note, e.g. ``additional_type(s) 'LTe71' (via FAFB)``.
    The display text always includes the linker VALUES.
    """
    entries: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    hubs: Set[str] = set()
    usable = [c for c in chains or []
              if c and c[-1].get("value") == foreign_type]
    same_name_direct = False
    for chain in usable:
        # hub datasets: intermediate namespaces the route passes through
        # (e.g. BANC routes through the FAFB annotation hub)
        hubs.update(hop["dataset"] for hop in chain[1:-1]
                    if hop["dataset"] not in (source_dataset, target_dataset))
        linkers = standardize_bridge(chain, source_dataset, target_dataset)
        if not linkers:
            # a derivation-free chain: the two `type` identities agree
            # directly — worth stating even when other chains add linkers
            same_name_direct = True
            continue
        for linker in linkers:
            if linker["kind"] != "linker":
                hubs.add(linker.get("home", ""))
                continue
            key = (linker["column"], linker["value"])
            if key in seen:
                continue
            seen.add(key)
            entries.append(dict(linker))
    entries.sort(key=lambda e: e["indirect"])
    if same_name_direct:
        entries.insert(0, {
            "column": "type", "value": foreign_type,
            "home": target_dataset, "kind": "same_name",
            "indirect": False, "text": "same name",
        })
    for entry in entries:
        base = f"{entry['column']} '{entry['value']}'"
        if entry["kind"] == "same_name":
            entry["text"] = "same name"
            continue
        if entry["indirect"] and hubs:
            base += f" (via {'/'.join(sorted(hubs))})"
        entry["text"] = base
    if same_name_direct and not any(
            e["kind"] == "linker" for e in entries):
        # the ONLY evidence is name equality — tell the user to verify
        display = "same name — no metadata verification " \
                  "(please double check)"
    else:
        display = " + ".join(entry["text"] for entry in entries)
    return {"entries": entries, "text": display}


def standardize_bridge(chain, source_dataset: str,
                       target_dataset: str) -> List[Dict[str, Any]]:
    """Standardized linker nodes of one derivation chain.

    The chain's two ends are always ``type`` identities; every middle hop
    plus the final annotation hop's ``via`` becomes one linker node
    ``{column, value, home, kind}``. Same-name middle hops are identity
    pass-throughs (``kind: 'same_name_pass'``) — they continue the type
    into a hub dataset and render as no extra node; annotation/crosswalk
    linkers are ``kind: 'linker'`` and at most two per known pair (one
    crosswalk + one annotation). Hops outside the pair registry are
    flagged ``indirect``.
    """
    registry = BRIDGE_STANDARD.get((source_dataset, target_dataset), ())
    registry_columns = {column for column, _home in registry}
    linkers: List[Dict[str, Any]] = []
    middle = chain[1:-1]
    for hop in middle:
        if hop['column'] == 'type':
            linkers.append({
                'column': 'type', 'value': hop['value'],
                'home': hop['dataset'], 'kind': 'same_name_pass',
                'indirect': True,
            })
            continue
        linkers.append({
            'column': hop['column'], 'value': hop['value'],
            'home': hop_home(hop, source_dataset), 'kind': 'linker',
            'indirect': hop['column'] not in registry_columns,
        })
    last = chain[-1] if chain else {}
    if len(chain) >= 2 and last.get("column") in CROSSWALK_COLUMNS:
        # crosswalk-arrival chain (e.g. [type, flywireType]): the terminal
        # metadata hop IS the verification linker — a same-name pair is
        # corroborated by the source's crosswalk cell naming the target.
        # §pooling fix (user 2026-09-07): the linker value is the CELL
        # content (``via``) — the forward direction has via == value, but
        # the REVERSE leg's hop value is the arrival (male-cns) type name
        # while the cell carries the foreign token (e.g. MCNS rows typed
        # 5thsLNv_LNd6 whose flywireType cell is 's-LNv_a,LNd_a'): using
        # the arrival name emptied the target-side bodyId pool.
        linkers.append({
            'column': last['column'],
            'value': last.get("via") or last["value"],
            'home': hop_home(last, source_dataset), 'kind': 'linker',
            'indirect': last['column'] not in registry_columns,
        })
    elif len(chain) >= 2 and last.get('via') and last.get('column') != 'type':
        final = {
            'column': last['column'], 'value': last['via'],
            'home': hop_home(last, source_dataset), 'kind': 'linker',
            'indirect': last['column'] not in registry_columns,
        }
        # A cross-namespace annotation landing standardizes to the same
        # (column, value) as the arrival hop's via (the shared token): one
        # linker represents both legs of the composed bridge.  Only a
        # literal repeat is collapsed — the raw self-loop class is still
        # rejected by bridge_is_valid's consecutive-hop check.
        if not linkers or (final['column'], final['value']) != (
                linkers[-1]['column'], linkers[-1]['value']):
            linkers.append(final)
    return linkers


def bridge_is_valid(chain, source_dataset: str, target_dataset: str,
                    key_of=None) -> bool:
    """Map-derived validity check for one derivation chain (§9I).

    Every non-``type`` hop must be licensed by ``BRIDGE_SOURCE_MAP``:
    its landing namespace is among the entry's targets AND the entry's
    home agrees with the hop's physical home (``hop_home``) — a
    crosswalk hop is the exception, because its columns only exist on
    the male-cns rows (the map's crosswalk home) while the hop records
    the namespace it REACHES — the reverse leg is the mirror case, it
    lands on the crosswalk home itself.  A crosswalk hop additionally
    requires endpoint-family licensing; and consecutive standardized
    linkers must never be identical — the ping-pong self-loop class
    (e.g. additional_type(s) 'X' → additional_type(s) 'X').
    ``key_of`` maps dataset names to registry keys.
    """
    if not chain:
        return False
    resolve = key_of or (lambda name: name)
    endpoints = {resolve(source_dataset), resolve(target_dataset)}
    for hop in chain[1:]:
        column = hop.get("column", "")
        if column == "type":
            continue
        landing = resolve(hop.get("dataset", ""))
        physical_home = None if column in CROSSWALK_COLUMNS \
            else resolve(hop_home(hop, source_dataset))
        licensed = [
            targets
            for (home, col), targets in BRIDGE_SOURCE_MAP.items()
            if col == column and home != BRIDGE_IDENTITY
            and (landing in targets
                 or (column in CROSSWALK_COLUMNS
                     and landing == resolve(CROSSWALK_HOME)))
            and (physical_home is None or home == physical_home)
        ]
        if not licensed:
            return False
        if (column in CROSSWALK_COLUMNS
                and not any(targets & endpoints for targets in licensed)):
            return False
    # Raw self-loop check: two consecutive hops with the same
    # (dataset, column, value) — the ping-pong class (e.g.
    # additional_type(s) 'X' followed by additional_type(s) 'X' on one
    # dataset).  The composed annotation bridge's landing+arrival pair
    # differs in value (token -> primary) and stays valid; its
    # standardized linkers are deduped instead.
    prev_hop = None
    for hop in chain[1:]:
        hop_key = (hop.get("dataset", ""), hop.get("column", ""),
                   hop.get("value", ""))
        if hop_key == prev_hop:
            return False
        prev_hop = hop_key
    return True


class TypeMappingWarning(UserWarning):
    """Warning for type mapping issues like N-to-1 relationships."""
    pass


class TypeMappingConflict:
    """Represents a type mapping conflict (N-to-1 or 1-to-N relationship)."""
    
    def __init__(
        self,
        source_dataset: str,
        target_dataset: str,
        source_type: str,
        target_types: Set[str],
        relationship: str,  # 'N-to-1' or '1-to-N'
        origin: str = '',
    ):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.source_type = source_type
        self.target_types = target_types
        self.relationship = relationship
        # Derivation provenance: '' for crosswalk-derived conflicts, a
        # short description (e.g. 'annotation bridge') for overlay ones.
        self.origin = origin
    
    def __repr__(self):
        return (f"TypeMappingConflict({self.source_type} in {self.source_dataset} "
                f"-> {self.target_types} in {self.target_dataset}, {self.relationship})")


class CrossDatasetTypeMapper:
    """
    Automatic cross-dataset type name mapping using male-cns neuron_df.
    
    This class provides:
    1. Loading of type mapping from male-cns neuron_df
    2. Resolution of type names across datasets
    3. Detection and warning for N-to-1/1-to-N relationships
    4. Priority-based type name selection
    
    Example:
        >>> mapper = CrossDatasetTypeMapper(workspace_path='/path/to/project')
        >>> 
        >>> # Get equivalent type in target dataset
        >>> flywire_type = mapper.get_mapped_type('aMe12', 'male-cns:v1.0', 'flywire_FAFB_v783')
        >>> 
        >>> # Resolve a type name to all equivalent types across datasets
        >>> type_map = mapper.resolve_type_across_datasets('MeVPLo2', ['male-cns:v1.0', 'flywire_FAFB_v783'])
        >>> 
        >>> # Get canonical display name
        >>> display_name = mapper.get_display_name('MeVPLo2', datasets=['male-cns:v1.0', 'flywire_FAFB_v783'])
    """
    
    def __init__(
        self,
        workspace_path: Optional[str] = None,
        neuron_df_path: Optional[str] = None,
        verbose: bool = True,
        flywire_neuron_df_paths: Optional[Dict[str, Optional[str]]] = None,
    ):
        """
        Initialize CrossDatasetTypeMapper.

        Args:
            workspace_path: Path to the project workspace (containing datasets/ folder).
                           If None, will try to auto-detect from file location.
            neuron_df_path: Explicit path to the male-cns neuron_df file.
                           If provided, overrides workspace_path detection.
            verbose: Print loading and warning messages.
            flywire_neuron_df_paths: Optional per-FlyWire-namespace override for the
                           neuron tables used to resolve renamed types (see
                           FLYWIRE_TYPE_SOURCES).  A ``None`` value disables the
                           additional-type resolution for that namespace.  When not
                           provided, the tables are looked up under
                           ``<workspace_path>/datasets/`` (only if a workspace path
                           is known; mappers built from an explicit ``neuron_df_path``
                           without a workspace stay hermetic).
        """
        self.verbose = verbose
        self._neuron_df: Optional[pd.DataFrame] = None
        self._loaded = False

        # Type mappings: {source_dataset: {source_type: {target_dataset: target_type}}}
        self._type_mappings: Dict[str, Dict[str, Dict[str, str]]] = {}

        # Reverse mappings for lookup
        self._reverse_mappings: Dict[str, Dict[str, str]] = {}  # {dataset: {type: canonical_type}}

        # Conflict tracking
        self._conflicts: List[TypeMappingConflict] = []
        self._n_to_1_types: Dict[str, Set[str]] = defaultdict(set)  # {target_type: {source_types}}

        # Annotation-bridge overlay provenance (built with the mappings):
        # {(src_key, src_type, dst_key): {'kind', 'target', 'via',
        #  'source_column', 'target_column'}} for the pairs the same-name
        # identity + additional Type(S) ⇄ Alternative Cell Type(s) bridge
        # resolved or refused.
        self._bridge_provenance: Dict[tuple, Dict[str, Any]] = {}

        # Dataset types index: {dataset: {type: set(bodyIds)}}
        self._dataset_types: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

        # Additional-type resolution tables, keyed by FlyWire mapping key:
        # {key: {additional_name: {candidate primary names}}}.  Only names
        # that are not themselves a primary type are indexed.
        self._flywire_alt_to_primary: Dict[str, Dict[str, Set[str]]] = {}
        self._flywire_primary_to_alts: Dict[str, Dict[str, Set[str]]] = {}

        # Per FlyWire mapping key, the dataset's own primary type names.
        self._flywire_primaries: Dict[str, Set[str]] = {}

        # Derived lookup for get_alias_candidates; rebuilt with the mappings.
        self._alias_n_to_1_cache: Optional[Dict[str, TypeMappingConflict]] = None

        # Reverse crosswalk index ({cell value: {male-cns primaries}});
        # lazily built by _crosswalk_reverse, reset with the mappings.
        self._crosswalk_reverse_cache: Optional[
            Dict[str, Dict[str, Set[str]]]] = None

        # Unsupported releases are reported once per mapper instance.  The
        # mapping file is release-specific, so an unknown release must not be
        # silently treated as the nearest supported release.
        self._unsupported_dataset_warnings: Set[str] = set()

        # Determine path to neuron_df
        if neuron_df_path:
            self._neuron_df_path = neuron_df_path
        else:
            if workspace_path is None:
                # Try to auto-detect from this file's location
                workspace_path = str(Path(__file__).parent.parent.parent)

            self._neuron_df_path = os.path.join(
                workspace_path,
                'datasets',
                'male-cns_v1_0',
                'male-cns_v1_0_allneurons_neuron_df.csv'
            )

        self._workspace_path = workspace_path

        overrides = flywire_neuron_df_paths or {}
        self._flywire_neuron_df_paths: Dict[str, Optional[str]] = {}
        for key in FLYWIRE_MAPPING_KEYS:
            if key in overrides:
                self._flywire_neuron_df_paths[key] = overrides[key]
            elif self._workspace_path:
                source = FLYWIRE_TYPE_SOURCES[key]
                self._flywire_neuron_df_paths[key] = os.path.join(
                    self._workspace_path,
                    'datasets',
                    source['dataset_dir'],
                    source['neuron_df'],
                )
            else:
                # Hermetic mapper (explicit neuron_df_path, no workspace):
                # no dataset tables to resolve renames against.
                self._flywire_neuron_df_paths[key] = None

    @staticmethod
    def _split_hemi_suffix(type_name: str) -> Tuple[str, str]:
        """Split hemisphere suffix (_L/_R/_U) from a type name.

        Returns (base, suffix) where suffix includes leading underscore.
        """
        if not isinstance(type_name, str):
            return type_name, ''
        for suffix in ('_L', '_R', '_U'):
            if type_name.endswith(suffix):
                return type_name[:-2], suffix
        return type_name, ''

    @staticmethod
    def _split_type_cell(value) -> List[str]:
        """Split a multi-name cell like ``'A, B,C'`` into clean type names.

        Crosswalk columns (flywireType/hemibrainType/mancType) and the
        FlyWire additional-type columns may list several names separated by
        commas; each name is used individually.
        """
        if not isinstance(value, str):
            return []
        return [name.strip() for name in value.split(',') if name.strip()]

    def _load_flywire_type_tables(self):
        """Index primary and additional types from the FAFB/BANC neuron tables.

        For each FlyWire namespace this builds ``additional name ->
        {primary names}`` from rows whose additional Type(S) column lists a
        name that is not itself a primary type anywhere in the dataset.
        Those entries are the type renames (e.g. FAFB SLP249 -> APDN3) that
        male-cns ``flywireType`` crosswalk values can resolve to.  Missing
        tables only disable the rename resolution for that namespace.
        """
        self._flywire_alt_to_primary = {}
        self._flywire_primary_to_alts = {}
        # UNFILTERED annotation view (annotation value -> primaries): unlike
        # ``_flywire_alt_to_primary`` it KEEPS values that are themselves
        # primaries — FAFB rows typed C2 annotated 'C3' genuinely link the
        # pair, and the annotation-reverse walk (MCNS C3 -> FAFB C2) needs
        # it.  The filtered table stays authoritative for rename semantics.
        self._flywire_annotation_primaries = {}
        self._flywire_primaries = {}
        for key in FLYWIRE_MAPPING_KEYS:
            path = self._flywire_neuron_df_paths.get(key)
            if not path:
                continue
            if not os.path.exists(path):
                self._log(
                    f"FlyWire neuron table not found for {key} "
                    f"({os.path.basename(path)}); renamed types (additional "
                    "Type(S)) will not be resolved for it.",
                    level='warn',
                )
                continue

            alt_column = FLYWIRE_TYPE_SOURCES[key]['alt_column']
            try:
                table = pd.read_csv(
                    path,
                    usecols=lambda c, col=alt_column: c in ('type', col),
                    low_memory=False,
                )
            except Exception as e:
                self._log(f"Could not read {path}: {e}", level='warn')
                continue

            if 'type' not in table.columns or alt_column not in table.columns:
                self._log(
                    f"{os.path.basename(path)} lacks a 'type' or "
                    f"'{alt_column}' column; skipping rename resolution.",
                    level='warn',
                )
                continue

            primaries = set(table['type'].dropna().astype(str).str.strip()) - {''}
            # Primary types double as the authoritative "does this name exist
            # in the dataset" check for alias candidates.
            self._flywire_primaries[key] = primaries
            # §T3 (user 2026-09-06): a FAFB additional_type(s) cell may list
            # SEVERAL names; only the in-use ones (a FAFB primary, a male-cns
            # type name — the crosswalk's left side, e.g. APDN3 rows
            # annotating 'LMTe01, CL125' — or a male-cns flywireType cell
            # value) may carry bridge evidence — the never-used leftovers
            # are historical noise and are dropped from the bridge tables
            # when the cell also names a used name (when NOTHING in the
            # cell is used, the names are indistinguishable and all stay).
            used_values = None
            if (key == 'flywire_FAFB_v783'
                    and getattr(self, '_neuron_df', None) is not None
                    and 'flywireType' in self._neuron_df.columns):
                used_values = set(primaries)
                used_values.update(
                    self._neuron_df['type'].dropna().astype(str).str.strip())
                for cw_cell in self._neuron_df['flywireType'].dropna().astype(str):
                    used_values.update(self._split_type_cell(cw_cell))
            alt_to_primary: Dict[str, Set[str]] = {}
            annotation_primaries: Dict[str, Set[str]] = defaultdict(set)
            for cell, primary in zip(table[alt_column], table['type']):
                names = self._split_type_cell(cell)
                if not names or not isinstance(primary, str):
                    continue
                if used_values is not None and len(names) >= 2:
                    filtered = [n for n in names if n in used_values]
                    if filtered:
                        names = filtered
                primary = primary.strip()
                if not primary:
                    continue
                for name in names:
                    # The unfiltered annotation view keeps every value that
                    # is a DIFFERENT type — a primary's own name in its own
                    # cell is a self-alias and adds only self-loop noise.
                    if name == primary:
                        continue
                    annotation_primaries.setdefault(name, set()).add(primary)
                    if name in primaries:
                        # A primary type keeps its own identity; the
                        # additional listing is just an alias.
                        continue
                    alt_to_primary.setdefault(name, set()).add(primary)

            self._flywire_alt_to_primary[key] = alt_to_primary
            self._flywire_annotation_primaries[key] = {
                value: set(primaries_of_value)
                for value, primaries_of_value in annotation_primaries.items()
            }
            # Inverse view: every primary type and the additional Type(S)
            # values listed on its rows. The linker bridge walks these
            # edges in BOTH directions — alt -> primary resolves a rename,
            # primary -> alt pools the bodyIds whose annotation column
            # carries the linked crosswalk value (e.g. BANC/FlyWire types
            # routed through FAFB annotations into male-cns).
            # The walker-facing reverse table uses the UNFILTERED
            # annotation view: primary↔primary annotation links (e.g.
            # FAFB pC2la rows annotated 'AVLP567') are real pair evidence
            # and must be walkable.  ``_flywire_alt_to_primary`` stays
            # filtered for rename-semantics consumers.
            primary_to_alts: Dict[str, Set[str]] = {}
            for alt, primaries_of_alt in self._flywire_annotation_primaries[
                    key].items():
                for primary in primaries_of_alt:
                    primary_to_alts.setdefault(primary, set()).add(alt)
            self._flywire_primary_to_alts[key] = primary_to_alts
            unambiguous = sum(1 for v in alt_to_primary.values() if len(v) == 1)
            self._log(
                f"Indexed {len(primaries):,} primary types and "
                f"{len(alt_to_primary):,} additional-only types "
                f"({unambiguous:,} unambiguous renames) from "
                f"{os.path.basename(path)}")

    def _resolve_flywire_names(self, names: List[str], mapping_key: str) -> Set[str]:
        """Resolve crosswalk type names against one FlyWire namespace.

        Names that are already a primary type pass through unchanged; names
        that only exist in the additional Type(S) column resolve to the
        current primary name(s) (one candidate -> rename, several -> the
        split candidates); unknown names pass through unchanged.
        """
        alt_to_primary = self._flywire_alt_to_primary.get(mapping_key)
        if not alt_to_primary:
            return set(names)
        resolved: Set[str] = set()
        for name in names:
            resolved.update(alt_to_primary.get(name) or (name,))
        return resolved
    
    def _crosswalk_reverse(self, crosswalk_col: str) -> Dict[str, Set[str]]:
        """Reverse crosswalk index: {cell value: {male-cns primaries}}.

        The lazy inverse of ``_crosswalk_parts`` — used by the
        reverse-crosswalk walk legs (§bridge rules bidirectionality).
        """
        cache = getattr(self, '_crosswalk_reverse_cache', None)
        if cache is None:
            cache = {}
            self._crosswalk_reverse_cache = cache
        if crosswalk_col not in cache:
            index: Dict[str, Set[str]] = defaultdict(set)
            neuron_df = getattr(self, '_neuron_df', None)
            if (
                neuron_df is not None
                and crosswalk_col in neuron_df.columns
                and 'type' in neuron_df.columns
            ):
                for local_type, cell in zip(neuron_df['type'],
                                            neuron_df[crosswalk_col]):
                    if not isinstance(local_type, str) or not local_type:
                        continue
                    for part in self._split_type_cell(cell):
                        index[part].add(local_type)
            cache[crosswalk_col] = dict(index)
        return cache[crosswalk_col]

    def _log(self, message: str, level: str = 'info'):
        """Print message if verbose mode enabled."""
        if self.verbose:
            prefix = '⚠️ ' if level == 'warn' else ''
            print(f"[TypeMapper] {prefix}{message}")
    
    def load(self, force_reload: bool = False) -> bool:
        """
        Load type mappings from neuron_df file.
        
        Args:
            force_reload: Reload even if already loaded.
            
        Returns:
            True if loading succeeded, False otherwise.
        """
        if self._loaded and not force_reload:
            return True
        
        if not os.path.exists(self._neuron_df_path):
            self._log(f"Neuron DF file not found: {self._neuron_df_path}", level='warn')
            self._log("Auto type mapping will be disabled. Initialize male-cns dataset first.", level='warn')
            return False
        
        try:
            self._log(f"Loading type mappings from {os.path.basename(self._neuron_df_path)}...")
            
            # Read only the columns we need for efficiency
            cols_needed = ['bodyId', 'type', 'flywireType', 'hemibrainType', 'mancType']
            self._neuron_df = pd.read_csv(
                self._neuron_df_path,
                usecols=lambda c: c in cols_needed,
                dtype={'bodyId': str},
                low_memory=False,
            )

            # Load the FlyWire primary/additional type tables used to
            # resolve renamed types (best effort: missing tables only
            # disable that resolution).
            self._load_flywire_type_tables()

            # Build mappings
            self._build_type_mappings()
            # Map grounding runs LAST: the overlap stats need every
            # namespace index built (§9I).
            self._verify_source_map()
            self._loaded = True
            
            self._log(f"Loaded {len(self._neuron_df):,} neurons with type mappings")
            return True
            
        except Exception as e:
            # Source-map grounding failures (BRIDGE_SOURCE_MAP: …) are
            # attributed as such — a drifted table must not read like a
            # generic read error.
            prefix = ("Type mapper not loaded"
                      if str(e).startswith("BRIDGE_SOURCE_MAP:")
                      else "Error loading neuron_df")
            self._log(f"{prefix}: {e}", level='warn')
            return False
    
    def _verify_source_map(self) -> None:
        """Verify BRIDGE_SOURCE_MAP against the loaded tables (§9I).

        Every declared (home, column) must exist in the home dataset's
        neuron table — a missing column on a LOADED table is data drift
        and raises (valid bridges derive from the map, so a drifted
        table must not silently change what is bridgeable).  Absent
        optional tables (e.g. no FlyWire side table) only disable their
        bridges.  Loaded crosswalk tables also log their overlap stats,
        grounding the map's licensing numbers.
        """

        def _namespace_names(key: str) -> Set[str]:
            names = set(self._dataset_types.get(key, {}))
            names.update(self._flywire_primaries.get(key, ()))
            return names

        for (home, column), targets in BRIDGE_SOURCE_MAP.items():
            if home == BRIDGE_IDENTITY or column == "type":
                continue
            if home == CROSSWALK_HOME:
                if (self._neuron_df is None
                        or column not in self._neuron_df.columns):
                    raise ValueError(
                        f"BRIDGE_SOURCE_MAP: the male-cns table lacks the "
                        f"declared '{column}' column (data drift)")
                values: set = set()
                for cell in self._neuron_df[column].dropna().astype(str):
                    values.update(self._split_type_cell(cell))
                for target in sorted(targets):
                    hit = len(values & _namespace_names(target))
                    self._log(
                        f"BRIDGE_SOURCE_MAP: {column} → {target}: "
                        f"{hit:,}/{len(values):,} cell values hit "
                        f"({100 * hit / max(1, len(values)):.0f}%)")
            elif home in FLYWIRE_TYPE_SOURCES:
                path = self._flywire_neuron_df_paths.get(home)
                if not path or not os.path.exists(path):
                    # optional namespace: an absent table only disables
                    # its rename resolution (existing behavior)
                    self._log(
                        f"BRIDGE_SOURCE_MAP: {home} table unavailable; "
                        f"'{column}' bridges are disabled for it.",
                        level='warn')
                    continue
                try:
                    present = column in pd.read_csv(path, nrows=0).columns
                except Exception:
                    present = False
                if not present:
                    raise ValueError(
                        f"BRIDGE_SOURCE_MAP: column '{column}' of {home} "
                        "is missing from its dataset table (data drift)")
                values = set(self._flywire_annotation_primaries.get(home, {}))
                hit = len(values & _namespace_names(home))
                self._log(
                    f"BRIDGE_SOURCE_MAP: {column} → {home}: "
                    f"{len(values):,} distinct cell values, "
                    f"{hit:,} of them also primary names")
            else:
                raise ValueError(
                    f"BRIDGE_SOURCE_MAP: unknown home namespace {home!r}")

    def _build_type_mappings(self):
        """Build internal type mapping dictionaries."""
        if self._neuron_df is None:
            return

        # Drop the alias-candidate lookup caches; they derive from the
        # conflicts rebuilt below.
        self._alias_n_to_1_cache = None
        self._crosswalk_parts_cache = None
        self._crosswalk_reverse_cache = None

        df = self._neuron_df.copy()
        
        # Clean up: fill NaN with empty string, strip whitespace
        for col in ['type', 'flywireType', 'hemibrainType', 'mancType']:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str).str.strip()
        
        # Build per-row mappings
        # Each row represents one bodyId with its type in each dataset
        male_cns_types = set()
        hemibrain_types = set()
        manc_types = set()

        # FlyWire mappings are built per namespace (FAFB and BANC resolve
        # renames independently through their additional Type(S) columns).
        mcns_to_flywire: Dict[str, Dict[str, Set[str]]] = {
            key: defaultdict(set) for key in FLYWIRE_MAPPING_KEYS
        }
        flywire_to_mcns: Dict[str, Dict[str, Set[str]]] = {
            key: defaultdict(set) for key in FLYWIRE_MAPPING_KEYS
        }

        # Track: mcns_type -> {flywire_types}, etc.
        mcns_to_hemibrain: Dict[str, Set[str]] = defaultdict(set)
        mcns_to_manc: Dict[str, Set[str]] = defaultdict(set)

        # Reverse mappings
        hemibrain_to_mcns: Dict[str, Set[str]] = defaultdict(set)
        manc_to_mcns: Dict[str, Set[str]] = defaultdict(set)

        for _, row in df.iterrows():
            mcns_type = row.get('type', '')
            body_id = row.get('bodyId', '')

            # Skip empty types
            if not mcns_type:
                continue

            male_cns_types.add(mcns_type)
            self._dataset_types['male-cns:v1.0'][mcns_type].add(body_id)

            # Crosswalk cells may carry several names separated by ',';
            # resolve each name against each FlyWire namespace.
            fw_names = self._split_type_cell(row.get('flywireType', ''))
            for fw_key in FLYWIRE_MAPPING_KEYS:
                for fw_type in self._resolve_flywire_names(fw_names, fw_key):
                    mcns_to_flywire[fw_key][mcns_type].add(fw_type)
                    flywire_to_mcns[fw_key][fw_type].add(mcns_type)
                    self._dataset_types[fw_key][fw_type].add(body_id)

            for hemibrain_type in self._split_type_cell(row.get('hemibrainType', '')):
                hemibrain_types.add(hemibrain_type)
                mcns_to_hemibrain[mcns_type].add(hemibrain_type)
                hemibrain_to_mcns[hemibrain_type].add(mcns_type)
                self._dataset_types['hemibrain:v1.2.1'][hemibrain_type].add(body_id)

            for manc_name in self._split_type_cell(row.get('mancType', '')):
                manc_types.add(manc_name)
                mcns_to_manc[mcns_type].add(manc_name)
                manc_to_mcns[manc_name].add(mcns_type)
                self._dataset_types['manc:v1.0'][manc_name].add(body_id)
                self._dataset_types['manc:v1.2.1'][manc_name].add(body_id)
        
        # Build final mappings (only 1-to-1 or 1-to-N that we can handle)
        self._type_mappings = {
            'male-cns:v1.0': {},
            'flywire_FAFB_v783': {},
            'banc_v626': {},
            'banc_v888': {},
            'hemibrain:v1.2.1': {},
            'manc:v1.0': {},
            'manc:v1.2.1': {},
        }
        
        # Process male-cns to other datasets
        for mcns_type in male_cns_types:
            self._type_mappings['male-cns:v1.0'][mcns_type] = {}

            # FlyWire mappings, resolved per namespace
            for fw_key in FLYWIRE_MAPPING_KEYS:
                fw_types = mcns_to_flywire[fw_key].get(mcns_type, set())
                if len(fw_types) == 1:
                    self._type_mappings['male-cns:v1.0'][mcns_type][fw_key] = next(iter(fw_types))
                elif len(fw_types) > 1:
                    # N-to-1 from male-cns perspective (one mcns type maps to
                    # multiple types in this FlyWire namespace): the mcns
                    # type is a superset - record a conflict instead of
                    # guessing one target name.
                    self._conflicts.append(TypeMappingConflict(
                        source_dataset='male-cns:v1.0',
                        target_dataset=fw_key,
                        source_type=mcns_type,
                        target_types=fw_types,
                        relationship='1-to-N',
                    ))
            
            # Hemibrain mapping
            hb_types = mcns_to_hemibrain.get(mcns_type, set())
            if len(hb_types) == 1:
                hb_type = next(iter(hb_types))
                self._type_mappings['male-cns:v1.0'][mcns_type]['hemibrain:v1.2.1'] = hb_type
            elif len(hb_types) > 1:
                self._conflicts.append(TypeMappingConflict(
                    source_dataset='male-cns:v1.0',
                    target_dataset='hemibrain:v1.2.1',
                    source_type=mcns_type,
                    target_types=hb_types,
                    relationship='1-to-N',
                ))
            
            # MANC mapping
            manc_types_mapped = mcns_to_manc.get(mcns_type, set())
            if len(manc_types_mapped) == 1:
                manc_type = next(iter(manc_types_mapped))
                self._type_mappings['male-cns:v1.0'][mcns_type]['manc:v1.0'] = manc_type
                self._type_mappings['male-cns:v1.0'][mcns_type]['manc:v1.2.1'] = manc_type
            elif len(manc_types_mapped) > 1:
                self._conflicts.append(TypeMappingConflict(
                    source_dataset='male-cns:v1.0',
                    target_dataset='manc:v1.0',
                    source_type=mcns_type,
                    target_types=manc_types_mapped,
                    relationship='1-to-N',
                ))
        
        # Process reverse mappings (flywire/hemibrain/manc to male-cns)
        for fw_key in FLYWIRE_MAPPING_KEYS:
            for fw_type in flywire_to_mcns[fw_key]:
                mcns_types_for_fw = flywire_to_mcns[fw_key][fw_type]
                if len(mcns_types_for_fw) == 1:
                    mcns_type = next(iter(mcns_types_for_fw))
                    self._type_mappings[fw_key][fw_type] = {'male-cns:v1.0': mcns_type}
                elif len(mcns_types_for_fw) > 1:
                    # N-to-1: multiple mcns types map to the same type in
                    # this FlyWire namespace. This should NOT be aggregated.
                    self._conflicts.append(TypeMappingConflict(
                        source_dataset=fw_key,
                        target_dataset='male-cns:v1.0',
                        source_type=fw_type,
                        target_types=mcns_types_for_fw,
                        relationship='N-to-1',
                    ))
                    self._n_to_1_types[fw_key].add(fw_type)
                    for mt in mcns_types_for_fw:
                        self._n_to_1_types['male-cns:v1.0'].add(mt)

        # Transitive mappings from each FlyWire namespace: its reverse
        # entries gain the male-cns type's other targets, including the
        # sibling FlyWire namespace (FAFB <-> BANC names can legitimately
        # differ after rename resolution).
        for fw_key in FLYWIRE_MAPPING_KEYS:
            for fw_type, target_maps in self._type_mappings[fw_key].items():
                mcns_type = target_maps.get('male-cns:v1.0')
                if not mcns_type or mcns_type not in self._type_mappings['male-cns:v1.0']:
                    continue
                for target_ds, target_type in self._type_mappings['male-cns:v1.0'][mcns_type].items():
                    if target_ds == fw_key:
                        continue
                    target_maps[target_ds] = target_type
        
        # Similarly for hemibrain
        for hb_type in hemibrain_types:
            mcns_types = hemibrain_to_mcns.get(hb_type, set())
            if len(mcns_types) == 1:
                mcns_type = next(iter(mcns_types))
                self._type_mappings['hemibrain:v1.2.1'][hb_type] = {'male-cns:v1.0': mcns_type}
                
                # Transitive mappings
                if mcns_type in self._type_mappings['male-cns:v1.0']:
                    for target_ds, target_type in self._type_mappings['male-cns:v1.0'][mcns_type].items():
                        if target_ds != 'hemibrain:v1.2.1':
                            if hb_type not in self._type_mappings['hemibrain:v1.2.1']:
                                self._type_mappings['hemibrain:v1.2.1'][hb_type] = {}
                            self._type_mappings['hemibrain:v1.2.1'][hb_type][target_ds] = target_type
            elif len(mcns_types) > 1:
                self._conflicts.append(TypeMappingConflict(
                    source_dataset='hemibrain:v1.2.1',
                    target_dataset='male-cns:v1.0',
                    source_type=hb_type,
                    target_types=mcns_types,
                    relationship='N-to-1',
                ))
                self._n_to_1_types['hemibrain:v1.2.1'].add(hb_type)
                for mt in mcns_types:
                    self._n_to_1_types['male-cns:v1.0'].add(mt)
        
        # Similarly for MANC (both versions share the same mancType column)
        for manc_type in manc_types:
            mcns_types = manc_to_mcns.get(manc_type, set())
            if len(mcns_types) == 1:
                mcns_type = next(iter(mcns_types))
                self._type_mappings['manc:v1.0'][manc_type] = {'male-cns:v1.0': mcns_type}
                self._type_mappings['manc:v1.2.1'][manc_type] = {'male-cns:v1.0': mcns_type}
                
                # Transitive mappings
                if mcns_type in self._type_mappings['male-cns:v1.0']:
                    for target_ds, target_type in self._type_mappings['male-cns:v1.0'][mcns_type].items():
                        if target_ds not in ['manc:v1.0', 'manc:v1.2.1']:
                            self._type_mappings['manc:v1.0'][manc_type][target_ds] = target_type
                            self._type_mappings['manc:v1.2.1'][manc_type][target_ds] = target_type
            elif len(mcns_types) > 1:
                self._conflicts.append(TypeMappingConflict(
                    source_dataset='manc:v1.0',
                    target_dataset='male-cns:v1.0',
                    source_type=manc_type,
                    target_types=mcns_types,
                    relationship='N-to-1',
                ))
                self._n_to_1_types['manc:v1.0'].add(manc_type)
                self._n_to_1_types['manc:v1.2.1'].add(manc_type)
                for mt in mcns_types:
                    self._n_to_1_types['male-cns:v1.0'].add(mt)
        
        # Annotation-bridge overlay (plan-type-mapper-additional-type-bridge):
        # same-name identity + additional Type(S) ⇄ Alternative Cell
        # Type(s) evidence for flywire pairs the crosswalk does not
        # resolve.  Runs BEFORE the reverse lookup so overlay entries flow
        # through it.
        self._apply_annotation_bridge_overlay()

        # Build reverse lookup for fast type resolution
        self._build_reverse_lookup()
        
        # Store conflict counts for later reference (don't print now)
        self._n_to_1_count = sum(1 for c in self._conflicts if c.relationship == 'N-to-1')
        self._one_to_n_count = sum(1 for c in self._conflicts if c.relationship == '1-to-N')
    
    @staticmethod
    def _is_untyped_value(value) -> bool:
        """True for labels that mean 'untyped': empty, an explicit
        Unknown/none sentinel, or a bare number.  Mirrors the analyzer's
        drop_untyped semantics — such labels must never become bridge
        targets or annotation-bridge candidates."""
        s = str(value).strip()
        if not s or s.lower() in {"unknown", "nan", "none"}:
            return True
        return s.isdigit()

    def _has_conflict(self, source_dataset: str, target_dataset: str,
                      source_type: str) -> bool:
        """True when a conflict for this ordered pair already exists (the
        annotation-bridge overlay must not duplicate crosswalk conflicts)."""
        return any(
            c.source_dataset == source_dataset
            and c.target_dataset == target_dataset
            and c.source_type == source_type
            for c in self._conflicts)

    def _annotation_bridge_candidates(self, src_key: str, src_type: str,
                                      dst_key: str) -> Dict[str, Set[str]]:
        """Annotation-bridge evidence from one src primary type.

        Collects the dst-namespace primaries reachable through the src
        type's own annotation values: a token V annotated on src_type's
        rows that also appears in dst annotation cells (or is itself a
        dst primary) evidences src_type -> those primaries.  Untyped
        sentinels are excluded on both sides.  Returns
        ``{candidate: {evidence tokens}}``."""
        evidence: Dict[str, Set[str]] = defaultdict(set)
        dst_primaries = self._flywire_primaries.get(dst_key, set())
        dst_ann = self._flywire_annotation_primaries.get(dst_key, {})
        dst_alts = self._flywire_alt_to_primary.get(dst_key, {})
        for token in sorted(
                self._flywire_primary_to_alts.get(src_key, {}).get(src_type, ())):
            if self._is_untyped_value(token):
                continue
            if token in dst_primaries:
                evidence[token].add(token)
            for primary in dst_ann.get(token, ()):
                if not self._is_untyped_value(primary):
                    evidence[primary].add(token)
            for primary in dst_alts.get(token, ()):
                if not self._is_untyped_value(primary):
                    evidence[primary].add(token)
        return dict(evidence)

    def _apply_annotation_bridge_overlay(self):
        """Same-name identity + annotation-bridge overlay for FlyWire pairs.

        For each ordered flywire pair (A, B) and each A primary T that the
        crosswalk does not already resolve for B:

        1. same-name identity: T is also a B primary — the universal
           BRIDGE_IDENTITY rule maps T -> T (e.g. FAFB APDN3 -> BANC
           APDN3); without it a same-name type present in both datasets
           resolves to nothing and vanishes from cross-dataset queries;
        2. annotation bridge: T's annotation values (FAFB
           ``additional_type(s)`` / BANC ``Alternative Cell Type(s)``,
           each licensed in its own namespace) that also appear in B's
           annotation cells evidence T -> B's annotated primaries.
           Exactly one candidate becomes a mapping entry; several become
           a 1-to-N conflict (never guessed), deduplicated against
           crosswalk conflicts.

        Provenance lands in ``_bridge_provenance`` keyed by
        ``(src_key, src_type, dst_key)`` for the mapping exports; the
        derivation chains themselves stay derivable via
        ``get_type_bridges`` (the cross-namespace landing hop).
        """
        self._bridge_provenance = {}
        # Same-name identity needs only the primary sets; annotation
        # candidates additionally need the annotation tables (absent
        # tables simply yield no candidates).
        keys = [k for k in FLYWIRE_MAPPING_KEYS
                if self._flywire_primaries.get(k)]
        added_maps = 0
        added_conflicts = 0
        for src_key in keys:
            src_column = self._flywire_alt_column(src_key)
            for dst_key in keys:
                if dst_key == src_key:
                    continue
                if (src_key.startswith('banc')
                        and dst_key.startswith('banc')):
                    # §version control: BANC <-> BANC cross-release mapping
                    # is not licensed — each release resolves against its
                    # own tables, with no crosswalk evidence between them.
                    continue
                dst_column = self._flywire_alt_column(dst_key)
                for src_type in sorted(self._flywire_primaries.get(src_key, ())):
                    if self._is_untyped_value(src_type):
                        continue
                    existing = self._type_mappings.get(src_key, {}).get(src_type)
                    if existing and existing.get(dst_key):
                        continue  # crosswalk route already resolves it
                    if src_type in self._flywire_primaries.get(dst_key, ()):
                        # 1. same-name identity (exact, 1-to-1 by
                        # definition; annotation candidates would only
                        # add noise next to an exact match)
                        candidates = {src_type: {src_type}}
                        kind = 'same name'
                    else:
                        # 2. annotation bridge via shared tokens
                        candidates = self._annotation_bridge_candidates(
                            src_key, src_type, dst_key)
                        kind = 'annotation bridge'
                    if not candidates:
                        continue
                    ordered = sorted(candidates)
                    prov = {
                        'kind': kind,
                        'source_column': src_column,
                        'target_column': dst_column,
                        'via': sorted(
                            {t for toks in candidates.values() for t in toks}),
                    }
                    if len(ordered) == 1:
                        target = ordered[0]
                        self._type_mappings.setdefault(src_key, {}).setdefault(
                            src_type, {})[dst_key] = target
                        prov['target'] = target
                        added_maps += 1
                    else:
                        prov['target'] = None
                        if not self._has_conflict(src_key, dst_key, src_type):
                            self._conflicts.append(TypeMappingConflict(
                                source_dataset=src_key,
                                target_dataset=dst_key,
                                source_type=src_type,
                                target_types=set(ordered),
                                relationship='1-to-N',
                                origin=kind,
                            ))
                            added_conflicts += 1
                    self._bridge_provenance[(src_key, src_type, dst_key)] = prov
        if added_maps or added_conflicts:
            self._log(
                f"Annotation bridge overlay: {added_maps} flywire pair "
                f"mappings (same-name identity + additional Type(S) / "
                f"Alternative Cell Type(s) evidence) and "
                f"{added_conflicts} 1-to-N conflicts added")

    def _build_reverse_lookup(self):
        """Build reverse lookup tables for fast type name resolution."""
        # For each dataset, map type names to their canonical form (male-cns name)
        for src_dataset, type_maps in self._type_mappings.items():
            if src_dataset not in self._reverse_mappings:
                self._reverse_mappings[src_dataset] = {}
            
            for src_type, target_maps in type_maps.items():
                # The src_type in src_dataset maps to these target types
                mcns_type = target_maps.get('male-cns:v1.0', src_type) if src_dataset != 'male-cns:v1.0' else src_type
                self._reverse_mappings[src_dataset][src_type] = mcns_type
    
    def get_mapped_type(
        self, 
        type_name: str, 
        source_dataset: str, 
        target_dataset: str
    ) -> Optional[str]:
        """
        Get the equivalent type name in target dataset.
        
        Args:
            type_name: Type name in source dataset.
            source_dataset: Source dataset name.
            target_dataset: Target dataset name.
            
        Returns:
            Mapped type name, or None if no mapping exists.
        """
        if not self._loaded:
            if not self.load():
                return None
        
        self._warn_if_unsupported_dataset(source_dataset)
        self._warn_if_unsupported_dataset(target_dataset)

        # Normalize dataset names
        src_ds = self._normalize_dataset_name(source_dataset)
        tgt_ds = self._normalize_dataset_name(target_dataset)
        src_mapping_key = self._get_type_mapping_key(source_dataset)
        tgt_mapping_key = self._get_type_mapping_key(target_dataset)

        # Different releases can share one type namespace.  In that case the
        # native name is already the correct name in the target release.
        if src_ds == tgt_ds or src_mapping_key == tgt_mapping_key:
            return type_name

        base_name, hemi_suffix = self._split_hemi_suffix(type_name)
        
        if src_mapping_key in self._type_mappings:
            if base_name in self._type_mappings[src_mapping_key]:
                mapped = self._type_mappings[src_mapping_key][base_name].get(tgt_mapping_key)
                if mapped and hemi_suffix:
                    return f"{mapped}{hemi_suffix}"
                return mapped
        
        return None
    
    def _normalize_dataset_name(self, dataset: str) -> str:
        """Normalize a dataset name without discarding its release.

        The neuron mapping file currently describes a specific set of
        releases (male-cns v1.0, FAFB v783, BANC v626, and so on).  Older code
        collapsed every family to those releases, which made a selected
        ``male-cns:v0.9`` indistinguishable from ``male-cns:v1.0`` and BANC
        v888 indistinguishable from v626.  Keep the release token in the
        normalized key so unsupported releases remain native and can be
        reported explicitly by the caller.

        Bare family names retain the historical default release because they
        are aliases for the supported mapping columns (for example ``banc``
        means BANC v626).  Explicit versions are always preserved.
        """
        if dataset is None:
            return dataset

        raw_dataset = str(dataset).strip()
        ds_lower = raw_dataset.lower()
        version = dataset_version(raw_dataset)

        if 'male-cns' in ds_lower or 'male_cns' in ds_lower:
            return f"male-cns:{version or 'v1.0'}"
        if 'banc' in ds_lower:
            return f"banc_{version or 'v626'}"
        if 'fafb' in ds_lower or ('flywire' in ds_lower and 'banc' not in ds_lower):
            return f"flywire_FAFB_{version or 'v783'}"
        if 'hemibrain' in ds_lower:
            return f"hemibrain:{version or 'v1.2.1'}"
        if 'manc' in ds_lower:
            return f"manc:{version or 'v1.0'}"
        if 'optic' in ds_lower:
            return f"optic-lobe:{version or 'v1.1'}"

        return raw_dataset

    def _get_type_mapping_key(self, dataset: str) -> str:
        """Return the crosswalk namespace used for *dataset*.

        Dataset identifiers remain release-specific for data access, cache
        paths, labels, and legends.  Type names use a broader schema
        namespace, however:

        * Male-CNS v0.9 and v1.0 use the Male-CNS ``type`` namespace.
        * FAFB releases share the FlyWire ``flywireType`` crosswalk that the
          v1.0 neuron table stores under the ``flywire_FAFB_v783`` mapping
          key.
        * BANC releases draw on the same crosswalk column, but BANC renames
          types independently through its ``Alternative Cell Type(s)``
          column, so its resolved names live under the ``banc_v626``
          mapping key (legacy ``flywire_BANC_*`` identifiers normalize
          into the same namespace).

        Keeping this translation separate prevents a release collision from
        either losing a valid mapping or renaming one release into another.
        """
        normalized = self._normalize_dataset_name(dataset)

        if normalized == 'male-cns:v0.9':
            # §version control: v0.9 is a distinct release whose type names
            # the v1.0 crosswalk cannot verify — it keeps its own (empty)
            # namespace instead of silently resolving through v1.0.
            return normalized
        if normalized.startswith('male-cns:'):
            return 'male-cns:v1.0'

        if normalized.startswith('flywire_FAFB_'):
            return 'flywire_FAFB_v783'

        # BANC releases are per-release namespaces (§version control):
        # each release resolves against its own neuron tables, so a
        # banc_v888 selection can never land v626 names ("via banc v626")
        # or pool the other release's bodyIds.
        if normalized.startswith(('banc_', 'flywire_BANC_')):
            return normalized if normalized.startswith('banc_') \
                else 'banc_v626'

        return normalized

    def _warn_if_unsupported_dataset(self, dataset: str) -> None:
        """Log once when a selected release has no validated type mapping."""
        if not self._loaded:
            return

        normalized = self._normalize_dataset_name(dataset)
        mapping_key = self._get_type_mapping_key(dataset)
        if mapping_key in self._type_mappings:
            return
        if normalized in self._unsupported_dataset_warnings:
            return

        self._unsupported_dataset_warnings.add(normalized)
        self._log(
            f"No release-specific cross-dataset type mapping is available for "
            f"'{dataset}' (normalized as '{normalized}'). Keeping its native "
            "type names; it will not be treated as another release.",
            level='warn',
        )
    
    def resolve_type_across_datasets(
        self,
        type_name: str,
        datasets: List[str],
        source_dataset: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        """
        Resolve a type name to equivalent types in multiple datasets.
        
        Args:
            type_name: Type name to resolve.
            datasets: List of target datasets.
            source_dataset: Optional source dataset hint. If None, will auto-detect.
            
        Returns:
            Dict mapping dataset -> equivalent type name (or None if not found).
        """
        if not self._loaded:
            if not self.load():
                return {ds: None for ds in datasets}

        for dataset in datasets:
            self._warn_if_unsupported_dataset(dataset)
        if source_dataset is not None:
            self._warn_if_unsupported_dataset(source_dataset)
        
        result = {}
        
        # Determine source dataset if not provided
        if source_dataset is None:
            source_dataset = self._detect_type_source(type_name)
        
        if source_dataset:
            src_ds = self._normalize_dataset_name(source_dataset)
            src_mapping_key = self._get_type_mapping_key(source_dataset)
            for ds in datasets:
                tgt_ds = self._normalize_dataset_name(ds)
                tgt_mapping_key = self._get_type_mapping_key(ds)
                if tgt_ds == src_ds or tgt_mapping_key == src_mapping_key:
                    result[ds] = type_name
                else:
                    result[ds] = self.get_mapped_type(type_name, source_dataset, ds)
        else:
            # Type not found in any known dataset
            for ds in datasets:
                result[ds] = None
        
        return result
    
    def _detect_type_source(self, type_name: str) -> Optional[str]:
        """
        Detect which dataset a type name belongs to based on priority.
        
        Args:
            type_name: Type name to look up.
            
        Returns:
            Dataset name where type was found, or None.
        """
        if not self._loaded:
            return None
        
        base_name, _ = self._split_hemi_suffix(type_name)

        # Check in priority order
        for dataset in DATASET_PRIORITY:
            norm_ds = self._normalize_dataset_name(dataset)
            if norm_ds in self._type_mappings:
                if type_name in self._type_mappings[norm_ds] or base_name in self._type_mappings[norm_ds]:
                    return norm_ds
            
            # Also check the dataset_types index
            if norm_ds in self._dataset_types:
                if type_name in self._dataset_types[norm_ds] or base_name in self._dataset_types[norm_ds]:
                    return norm_ds
        
        return None
    
    # Dataset short codes for display names
    DATASET_SHORT_CODES = {
        'male-cns:v1.0': 'M',
        'male-cns_v1_0': 'M',
        'flywire_FAFB_v783': 'F',
        'flywire_FAFB': 'F',
        'banc_v626': 'B',
        'banc': 'B',
        'hemibrain:v1.2.1': 'H',
        'hemibrain_v1_2_1': 'H',
        'manc:v1.0': 'N',  # N for MANC
        'manc:v1.2.1': 'N',
        'manc_v1_0': 'N',
        'manc_v1_2_1': 'N',
        'optic-lobe:v1.1': 'O',
        'optic-lobe_v1_1': 'O',
    }
    
    # Full dataset names for hover info
    DATASET_FULL_NAMES = {
        'male-cns:v1.0': 'male-cns v1.0',
        'male-cns_v1_0': 'male-cns v1.0',
        'flywire_FAFB_v783': 'FlyWire FAFB v783',
        'flywire_FAFB': 'FlyWire FAFB',
        'banc_v626': 'BANC v626',
        'banc': 'BANC',
        'hemibrain:v1.2.1': 'hemibrain v1.2.1',
        'hemibrain_v1_2_1': 'hemibrain v1.2.1',
        'manc:v1.0': 'MANC v1.0',
        'manc:v1.2.1': 'MANC v1.2.1',
        'manc_v1_0': 'MANC v1.0',
        'manc_v1_2_1': 'MANC v1.2.1',
        'optic-lobe:v1.1': 'optic-lobe v1.1',
        'optic-lobe_v1_1': 'optic-lobe v1.1',
    }
    
    def _get_base_dataset_short_code(self, dataset: str) -> str:
        """Return the family code before release disambiguation."""
        norm_ds = self._normalize_dataset_name(dataset)
        code = self.DATASET_SHORT_CODES.get(norm_ds)
        if code:
            return code

        # Unknown releases still get the same family code as their supported
        # siblings; make_unique_dataset_labels() adds the release only when
        # that family occurs more than once in the selected dataset list.
        ds_lower = str(dataset).lower()
        if 'male-cns' in ds_lower or 'male_cns' in ds_lower:
            return 'M'
        if 'banc' in ds_lower:
            return 'B'
        if 'fafb' in ds_lower or 'flywire' in ds_lower:
            return 'F'
        if 'hemibrain' in ds_lower:
            return 'H'
        if 'manc' in ds_lower:
            return 'N'
        if 'optic' in ds_lower:
            return 'O'
        return (str(dataset)[:1] or 'X').upper()

    def _get_dataset_short_codes(self, datasets: List[str]) -> List[str]:
        """Return unique display codes for a selected dataset list."""
        base_codes = [self._get_base_dataset_short_code(ds) for ds in datasets]
        return make_unique_dataset_labels(datasets, base_codes)

    def get_dataset_short_code(self, dataset: str, datasets: Optional[List[str]] = None) -> str:
        """
        Get a display code for a dataset.

        With a dataset list, the code is collision-aware.  For example,
        ``['male-cns:v1.0', 'male-cns:v0.9']`` receives ``M_v1_0`` and
        ``M_v0_9``.  Without context, the compact family code is returned for
        backwards compatibility.
        
        Args:
            dataset: Dataset name.
            
        Returns:
            Compact or release-qualified display code.
        """
        if datasets is not None:
            codes = self._get_dataset_short_codes(datasets)
            for selected_dataset, code in zip(datasets, codes):
                if selected_dataset == dataset:
                    return code
        return self._get_base_dataset_short_code(dataset)
    
    def get_dataset_full_name(self, dataset: str) -> str:
        """
        Get the full display name for a dataset.
        
        Args:
            dataset: Dataset name.
            
        Returns:
            Full dataset name for display.
        """
        norm_ds = self._normalize_dataset_name(dataset)
        known_name = self.DATASET_FULL_NAMES.get(norm_ds)
        if known_name:
            return known_name

        ds_lower = str(dataset).lower()
        if 'male-cns' in ds_lower or 'male_cns' in ds_lower:
            family_name = 'male-cns'
        elif 'banc' in ds_lower:
            family_name = 'BANC'
        elif 'fafb' in ds_lower or 'flywire' in ds_lower:
            family_name = 'FlyWire FAFB'
        elif 'hemibrain' in ds_lower:
            family_name = 'hemibrain'
        elif 'manc' in ds_lower:
            family_name = 'MANC'
        elif 'optic' in ds_lower:
            family_name = 'optic-lobe'
        else:
            return str(dataset)

        version = dataset_version(dataset)
        return f"{family_name} {version}" if version else family_name
    
    def get_all_dataset_short_codes(self, datasets: List[str]) -> Dict[str, str]:
        """
        Get short codes for all datasets being compared.
        
        Args:
            datasets: List of dataset names.
            
        Returns:
            Dict mapping short code to full dataset name.
        """
        codes = self._get_dataset_short_codes(datasets)
        return {
            code: self.get_dataset_full_name(dataset)
            for dataset, code in zip(datasets, codes)
        }

    def _get_male_cns_mapping_name(
        self,
        mappings: Dict[str, Optional[str]],
    ) -> Optional[str]:
        """Return the Male-CNS name from a release-aware mapping result."""
        for dataset, mapped_name in mappings.items():
            if mapped_name and self._get_type_mapping_key(dataset) == 'male-cns:v1.0':
                return mapped_name
        return None
    
    def get_display_name(
        self,
        type_name: str,
        datasets: List[str],
        source_dataset: Optional[str] = None,
    ) -> str:
        """
        Get a display name for a type showing mappings across datasets.
        
        Format: {canonical}({alt1}/{alt2}) if names differ, skipping identical names.
        Example: "MeVPLo2(MTe07)" if FAFB uses MTe07 but BANC uses MeVPLo2.
        
        Args:
            type_name: Type name to display.
            datasets: Datasets being compared.
            source_dataset: Source dataset for the type.
            
        Returns:
            Display name with alternative names in parentheses.
        """
        base_name, hemi_suffix = self._split_hemi_suffix(type_name)
        mappings = self.resolve_type_across_datasets(base_name, datasets, source_dataset)
        
        # Get the Male-CNS name as canonical (primary display name).  This
        # works whether the selected Male-CNS release is v0.9 or v1.0.
        mcns_name = self._get_male_cns_mapping_name(mappings)
        
        if not mcns_name:
            # Use the original type name as canonical
            mcns_name = base_name
        canonical_base = mcns_name
        if hemi_suffix:
            mcns_name = f"{mcns_name}{hemi_suffix}"
        
        # Collect unique alternative names (different from canonical)
        alt_names = set()
        for ds, mapped_name in mappings.items():
            if mapped_name and mapped_name != canonical_base:
                alt_names.add(f"{mapped_name}{hemi_suffix}" if hemi_suffix else mapped_name)
        
        if alt_names:
            # Sort for consistent ordering, join with /
            alt_str = '/'.join(sorted(alt_names))
            return f"{mcns_name}({alt_str})"
        
        return mcns_name
    
    def get_display_name_with_dataset_info(
        self,
        type_name: str,
        datasets: List[str],
        source_dataset: Optional[str] = None,
    ) -> Tuple[str, Dict[str, str]]:
        """
        Get display name and dataset->name mapping for hover labels.
        
        Args:
            type_name: Type name to display.
            datasets: Datasets being compared.
            source_dataset: Source dataset for the type.
            
        Returns:
            Tuple of (display_name, {dataset_code: name_in_that_dataset}).
        """
        base_name, hemi_suffix = self._split_hemi_suffix(type_name)
        mappings = self.resolve_type_across_datasets(base_name, datasets, source_dataset)
        display_name = self.get_display_name(type_name, datasets, source_dataset)
        
        # Build dataset code -> name mapping for hover info
        dataset_names = {}
        dataset_codes = self._get_dataset_short_codes(datasets)
        for ds, mapped_name in mappings.items():
            if mapped_name:
                # ``mappings`` uses the original full dataset identifiers as
                # keys.  Resolve the code in the same selected-list context
                # used by the legend so collision-qualified codes cannot be
                # overwritten in the hover dictionary.
                code = next(
                    (
                        code
                        for selected_ds, code in zip(datasets, dataset_codes)
                        if selected_ds == ds
                    ),
                    self.get_dataset_short_code(ds),
                )
                dataset_names[code] = f"{mapped_name}{hemi_suffix}" if hemi_suffix else mapped_name
        
        return display_name, dataset_names
    
    def is_n_to_1_type(self, type_name: str, dataset: str) -> bool:
        """
        Check if a type is involved in an N-to-1 mapping.
        
        These types should not be aggregated across datasets.
        
        Args:
            type_name: Type name to check.
            dataset: Dataset the type belongs to.
            
        Returns:
            True if the type is part of an N-to-1 mapping.
        """
        if not self._loaded:
            self.load()
        
        mapping_key = self._get_type_mapping_key(dataset)
        return type_name in self._n_to_1_types.get(mapping_key, set())
    
    def get_n_to_1_conflicts(self) -> List[TypeMappingConflict]:
        """Get all N-to-1 type mapping conflicts."""
        return [c for c in self._conflicts if c.relationship == 'N-to-1']
    
    def get_1_to_n_conflicts(self) -> List[TypeMappingConflict]:
        """Get all 1-to-N type mapping conflicts."""
        return [c for c in self._conflicts if c.relationship == '1-to-N']
    
    def warn_if_conflicting(self, type_name: str, datasets: List[str]) -> bool:
        """
        Warn if type has conflicting mappings and return True if warned.
        
        Args:
            type_name: Type name to check.
            datasets: Datasets being compared.
            
        Returns:
            True if a warning was issued.
        """
        for dataset in datasets:
            if self.is_n_to_1_type(type_name, dataset):
                # Find the specific conflict
                for conflict in self._conflicts:
                    if conflict.source_type == type_name or type_name in conflict.target_types:
                        msg = (
                            f"Type '{type_name}' is involved in an N-to-1 mapping: "
                            f"{conflict.target_types} in {conflict.target_dataset} all map to "
                            f"'{conflict.source_type}' in {conflict.source_dataset}. "
                            f"Consider using LabelMapper to specify explicit mappings for these types."
                        )
                        warnings.warn(msg, TypeMappingWarning, stacklevel=3)
                        return True
        return False
    
    def check_type_name_conflict(
        self,
        type_name: str,
        datasets: List[str],
    ) -> Optional[Tuple[str, str, str]]:
        """
        Check if a type name exists in multiple datasets but with different mappings.
        
        This catches cases like: 'aMe12' exists in both male-cns and FAFB,
        but the mapping says male-cns:aMe12 should map to FAFB:aMe122.
        
        Args:
            type_name: Type name to check.
            datasets: Datasets to check across.
            
        Returns:
            Tuple of (queried_type, actual_mapped_type, conflict_dataset) if conflict exists,
            None otherwise.
        """
        if not self._loaded:
            self.load()
        
        # Find which type namespaces have this name directly.  The selected
        # dataset may be an older release, but the index is intentionally
        # stored by shared schema namespace (Male-CNS or FlyWire), not by a
        # particular release.
        datasets_with_type = []
        for ds in datasets:
            mapping_key = self._get_type_mapping_key(ds)
            if (
                mapping_key in self._dataset_types
                and type_name in self._dataset_types[mapping_key]
                and mapping_key not in datasets_with_type
            ):
                datasets_with_type.append(mapping_key)
        
        if len(datasets_with_type) <= 1:
            return None
        
        # Check if they're actually the same type (mapped)
        # Use the highest priority dataset as source
        source_ds = None
        for priority_ds in DATASET_PRIORITY:
            mapping_key = self._get_type_mapping_key(priority_ds)
            if mapping_key in datasets_with_type:
                source_ds = mapping_key
                break
        
        if not source_ds:
            return None
        
        # Check mappings to other datasets that have the same type name
        for other_ds in datasets_with_type:
            if other_ds == source_ds:
                continue
            
            mapped_type = self.get_mapped_type(type_name, source_ds, other_ds)
            
            if mapped_type and mapped_type != type_name:
                # Conflict: same type name exists in both datasets,
                # but the mapping says they should be different
                return (type_name, mapped_type, other_ds)
        
        return None
    
    def export_mapping(
        self, 
        output_path: str, 
        filter_types: Optional[Set[str]] = None,
        datasets: Optional[List[str]] = None,
        only_different: bool = True,
    ) -> None:
        """
        Export type mappings to a CSV file.
        
        Args:
            output_path: Path to save the CSV file.
            filter_types: Optional set of type names to include. If provided,
                only exports mappings where ANY column contains a type in this set.
                If None, exports all mappings.
            datasets: Optional list of datasets to include in output columns.
                If None, includes all datasets.
            only_different: If True (default), only exports rows where types differ
                across datasets (i.e., actual mappings, not 1-to-1 identical types).
        """
        if not self._loaded:
            if not self.load():
                raise RuntimeError("Cannot export: mappings not loaded")
        
        rows = []

        # Normalize filter types (strip hemisphere suffixes)
        normalized_filter_types = None
        if filter_types:
            normalized_filter_types = set()
            for t in filter_types:
                base, _ = self._split_hemi_suffix(t)
                if base:
                    normalized_filter_types.add(base)
        
        # Determine which columns/datasets to include
        all_datasets = ['male-cns:v1.0', 'flywire_FAFB_v783', 'banc_v626', 
                        'hemibrain:v1.2.1', 'manc:v1.0', 'manc:v1.2.1']
        if datasets:
            # Column labels keep the REQUESTED (release-specific) names;
            # value lookups resolve through each dataset's mapping-key
            # namespace (banc_v888 -> 'banc_v626', flywire_BANC_* -> the
            # same). Releases sharing a namespace collapse to one column.
            output_datasets = []
            lookup_keys = {}
            for d in datasets:
                key = self._get_type_mapping_key(d)
                lookup_keys[d] = key
                # keep the FIRST requested release per namespace as the column
                if not any(lookup_keys[x] == key for x in output_datasets):
                    output_datasets.append(d)
        else:
            output_datasets = all_datasets
            lookup_keys = {d: self._get_type_mapping_key(d)
                           for d in all_datasets}
        
        if len(output_datasets) < 2:
            self._log(f"Not enough datasets to export mapping (need >= 2, got {len(output_datasets)})")
            return
        
        # Start from male-cns types and export their mappings
        mcns_mappings = self._type_mappings.get('male-cns:v1.0', {})
        
        for mcns_type, target_maps in mcns_mappings.items():
            row = {}
            for ds in output_datasets:
                if ds == 'male-cns:v1.0':
                    row[ds] = mcns_type
                else:
                    row[ds] = target_maps.get(lookup_keys.get(ds, ds), '')
            
            # Filter if filter_types is provided
            if normalized_filter_types is not None:
                # Check if any type in this row is in filter_types
                row_types = {v for v in row.values() if v}
                if not row_types.intersection(normalized_filter_types):
                    continue  # Skip this row
            
            # Filter out identical mappings if only_different is True
            if only_different:
                # Get non-empty values
                non_empty_values = [v for v in row.values() if v]
                if len(non_empty_values) <= 1:
                    continue  # Only one type present, not a meaningful mapping
                # Check if all non-empty values are the same (1-to-1 identical)
                unique_values = set(non_empty_values)
                if len(unique_values) == 1:
                    continue  # All same type name, not a cross-dataset mapping
            
            rows.append(row)

        # Annotation-bridge rows (same-name identity + additional Type(S)
        # ⇄ Alternative Cell Type(s) overlay): pairs without a male-cns
        # anchor row above.  A pair already represented by an anchored
        # row (both endpoint names on one row) is not duplicated.
        # ``mapping_origin`` states how every row was derived; ambiguous
        # (1-to-N) bridges live in the conflicts export only.
        existing_pairs = [
            {lookup_keys.get(ds, ds): row[ds]
             for ds in output_datasets if row.get(ds)}
            for row in rows
        ]
        bridge_rows = []
        for (src_key, src_type, dst_key), prov in getattr(
                self, '_bridge_provenance', {}).items():
            target = prov.get('target')
            if not target:
                continue
            pair = {src_key: src_type, dst_key: target}
            if any(all(existing.get(k) == n for k, n in pair.items())
                   for existing in existing_pairs):
                continue
            row = {ds: '' for ds in output_datasets}
            for key, name in pair.items():
                for ds in output_datasets:
                    if lookup_keys.get(ds, ds) == key:
                        row[ds] = name
                        break
            non_empty = {row[ds] for ds in output_datasets if row[ds]}
            if normalized_filter_types is not None and not (
                    non_empty & normalized_filter_types):
                continue
            if only_different and len(non_empty) <= 1:
                continue
            kind = prov.get('kind', 'annotation bridge')
            via_txt = '; '.join(prov.get('via', ()))
            row['mapping_origin'] = (
                f"{kind} via {via_txt}" if via_txt else kind)
            bridge_rows.append(row)
            # The reverse-direction provenance entry describes the same
            # row from the other side — record the coverage so it is
            # skipped too.
            existing_pairs.append({
                lookup_keys.get(ds, ds): row[ds]
                for ds in output_datasets if row.get(ds)
            })
        for row in rows:
            row['mapping_origin'] = 'crosswalk'
        rows.extend(bridge_rows)

        df = pd.DataFrame(rows, columns=output_datasets + ['mapping_origin'])
        if not df.empty:
            # Sort by first column
            df = df.sort_values(output_datasets[0])
        df.to_csv(output_path, index=False)
        
        filter_parts = []
        if normalized_filter_types:
            filter_parts.append(f"filtered to {len(normalized_filter_types)} result types")
        if only_different:
            filter_parts.append("only different mappings")
        if datasets:
            filter_parts.append(f"{len(output_datasets)} datasets")
        filter_msg = f" ({', '.join(filter_parts)})" if filter_parts else " (complete)"
        self._log(f"Exported {len(rows)} type mappings to {output_path}{filter_msg}")
    
    def export_conflicts(
        self,
        output_path: str,
        filter_types: Optional[Set[str]] = None,
        datasets: Optional[List[str]] = None,
    ) -> None:
        """
        Export conflict information to a CSV file.

        Args:
            output_path: Path to save the CSV file.
            filter_types: Optional set of type names to include. If provided,
                only exports conflicts where source_type or any target_type
                is in this set. If None, exports all conflicts.
            datasets: Optional list of run datasets. If provided, only
                exports conflicts whose source_dataset AND target_dataset
                are both in this list — the global crosswalk knows datasets
                (e.g. banc_v626, hemibrain) that a specific run may not
                include. If None, exports conflicts for all known datasets.
        """
        if not self._conflicts:
            self._log("No conflicts to export")
            return

        rows = []
        for conflict in self._conflicts:
            # Filter if filter_types is provided
            if filter_types is not None:
                conflict_types = {conflict.source_type} | conflict.target_types
                if not conflict_types.intersection(filter_types):
                    continue  # Skip this conflict

            # Scope to the run's datasets (§5 of the report-fixes plan):
            # the global crosswalk covers datasets the run never selected.
            # Datasets arrive as RELEASE names (banc_v888, ...) while the
            # recorded conflicts carry MAPPING keys (banc_v626, ...) —
            # resolve both sides so release-keyed runs keep their
            # conflicts.
            if datasets is not None:
                ds_keys = {self._get_type_mapping_key(d) for d in datasets}
                if (self._get_type_mapping_key(
                        conflict.source_dataset) not in ds_keys
                        or self._get_type_mapping_key(
                            conflict.target_dataset) not in ds_keys):
                    continue
            
            rows.append({
                'source_dataset': conflict.source_dataset,
                'source_type': conflict.source_type,
                'target_dataset': conflict.target_dataset,
                'target_types': ', '.join(sorted(conflict.target_types)),
                'relationship': conflict.relationship,
                'origin': getattr(conflict, 'origin', '') or 'crosswalk',
            })
        
        if not rows:
            self._log("No conflicts to export (all filtered out)")
            return
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        filter_msg = f" (filtered to result types)" if filter_types else " (complete)"
        self._log(f"Exported {len(rows)} conflicts to {output_path}{filter_msg}")
    
    def to_label_mapper(
        self,
        types: List[str],
        datasets: List[str],
        role: str = 'source',
    ) -> 'LabelMapper':
        """
        Convert type mappings to a LabelMapper for specific types.
        
        Args:
            types: List of type names to include.
            datasets: Datasets to include in mapping.
            role: 'source', 'target', or 'intermediate'.
            
        Returns:
            LabelMapper with the type mappings.
        """
        from .label_mapper import LabelMapper
        
        mapping_dict = {}
        labels = []
        
        for type_name in types:
            source_ds = self._detect_type_source(type_name)
            mappings = self.resolve_type_across_datasets(type_name, datasets, source_ds)
            
            # Use the Male-CNS namespace name as label if available.  The
            # selected release may be v0.9 even though the crosswalk source
            # is the v1.0 neuron table.
            mcns_name = self._get_male_cns_mapping_name(mappings) or type_name
            labels.append(mcns_name)
            
            # Build dataset mapping
            for ds in datasets:
                mapped_type = mappings.get(ds)
                if mapped_type:
                    if ds not in mapping_dict:
                        mapping_dict[ds] = []
                    # LabelMapper expects grouped format: [[types_for_label1], [types_for_label2], ...]
                    # We need to align indices
                    idx = len(labels) - 1
                    while len(mapping_dict[ds]) < idx:
                        mapping_dict[ds].append([])
                    if len(mapping_dict[ds]) == idx:
                        mapping_dict[ds].append([mapped_type])
                    else:
                        mapping_dict[ds][idx].append(mapped_type)
        
        # Create LabelMapper based on role
        if role == 'source':
            return LabelMapper(source_mapping_dict=mapping_dict, source_labels=labels)
        elif role == 'target':
            return LabelMapper(target_mapping_dict=mapping_dict, target_labels=labels)
        else:
            return LabelMapper(intermediate_mapping_dict=mapping_dict, intermediate_labels=labels)


    def get_source_target_mapping_summary(
        self,
        neurons: List[str],
        datasets: List[str],
    ) -> Dict[str, any]:
        """
        Generate a summary of source/target neuron mappings for display.
        
        Returns a dict with:
        - 'per_dataset': {dataset: {original_type: mapped_type}}
        - 'different_mappings': [(type, {dataset: mapped_type})] where mapping differs
        - 'n_to_1_warnings': [(type, source_ds, target_ds, conflicting_types)]
        - 'one_to_n_warnings': [(type, source_ds, target_ds, split_types)]
        """
        if not self._loaded:
            self.load()
        
        per_dataset = {ds: {} for ds in datasets}
        different_mappings = []
        n_to_1_warnings = []
        one_to_n_warnings = []
        
        for neuron in neurons:
            # Skip non-string or regex patterns
            if not isinstance(neuron, str):
                for ds in datasets:
                    per_dataset[ds][neuron] = neuron
                continue
            
            if '*' in neuron or ('.' in neuron and '.*' in neuron):
                for ds in datasets:
                    per_dataset[ds][neuron] = neuron
                continue
            
            # Detect source dataset for this type
            source_ds = self._detect_type_source(neuron)
            
            if not source_ds:
                # Type not found, use as-is
                for ds in datasets:
                    per_dataset[ds][neuron] = neuron
                continue
            
            # Resolve to each dataset
            mappings_for_type = {}
            has_different = False
            source_mapping_key = self._get_type_mapping_key(source_ds)
            
            for ds in datasets:
                norm_ds = self._normalize_dataset_name(ds)
                target_mapping_key = self._get_type_mapping_key(ds)
                if norm_ds == source_ds or target_mapping_key == source_mapping_key:
                    per_dataset[ds][neuron] = neuron
                    mappings_for_type[ds] = neuron
                else:
                    mapped = self.get_mapped_type(neuron, source_ds, ds)
                    if mapped and mapped != neuron:
                        per_dataset[ds][neuron] = mapped
                        mappings_for_type[ds] = mapped
                        has_different = True
                    else:
                        per_dataset[ds][neuron] = mapped if mapped else neuron
                        mappings_for_type[ds] = mapped if mapped else neuron
            
            if has_different:
                different_mappings.append((neuron, mappings_for_type))
            
            # Check for N-to-1 conflicts
            for conflict in self._conflicts:
                if conflict.relationship == 'N-to-1':
                    if neuron == conflict.source_type or neuron in conflict.target_types:
                        n_to_1_warnings.append((
                            neuron,
                            conflict.source_dataset,
                            conflict.target_dataset,
                            conflict.target_types,
                        ))
                        break
                elif conflict.relationship == '1-to-N':
                    if neuron == conflict.source_type:
                        one_to_n_warnings.append((
                            neuron,
                            conflict.source_dataset,
                            conflict.target_dataset,
                            conflict.target_types,
                        ))
                        break
        
        return {
            'per_dataset': per_dataset,
            'different_mappings': different_mappings,
            'n_to_1_warnings': n_to_1_warnings,
            'one_to_n_warnings': one_to_n_warnings,
        }
    
    def get_intermediate_mapping_summary(
        self,
        types_used: Set[str],
        datasets: List[str],
    ) -> Dict[str, any]:
        """
        Generate a summary of intermediate neuron type mappings.
        
        Returns counts and file path info for logging.
        """
        if not self._loaded:
            self.load()
        
        mapped_count = 0
        n_to_1_count = 0
        one_to_n_count = 0
        
        for type_name in types_used:
            if not isinstance(type_name, str):
                continue
            if '*' in type_name or ('.' in type_name and '.*' in type_name):
                continue
            
            source_ds = self._detect_type_source(type_name)
            if not source_ds:
                continue
            
            # Check if any dataset has different mapping
            for ds in datasets:
                if self._get_type_mapping_key(ds) != self._get_type_mapping_key(source_ds):
                    mapped = self.get_mapped_type(type_name, source_ds, ds)
                    if mapped and mapped != type_name:
                        mapped_count += 1
                        break
            
            # Check conflicts
            for conflict in self._conflicts:
                if conflict.relationship == 'N-to-1':
                    if type_name == conflict.source_type or type_name in conflict.target_types:
                        n_to_1_count += 1
                        break
                elif conflict.relationship == '1-to-N':
                    if type_name == conflict.source_type:
                        one_to_n_count += 1
                        break
        
        return {
            'total_types': len(types_used),
            'mapped_count': mapped_count,
            'n_to_1_count': n_to_1_count,
            'one_to_n_count': one_to_n_count,
        }

    def build_user_warning_notes(
        self,
        type_names: List[Union[str, int]],
        datasets: List[str],
        max_examples: int = 15,
    ) -> List[str]:
        """Build user-facing warning notes for what auto type mapping changed.

        Covers the three cases that can silently affect a run's results:

        * expanded mappings - a queried type name is different in a target
          dataset (e.g. male-cns SLP249 -> FAFB APDN3 via the additional
          Type(S) annotations), so the target side was matched under the
          mapped name;
        * N-to-1 mappings - several types share one name across datasets;
          they were NOT merged to avoid wrong aggregation;
        * 1-to-N mappings - a type splits into several names in another
          dataset; no automatic mapping was made for it.

        Each note is one plain-text bullet.  When a category affects more
        than ``max_examples`` types it is summarized with count and examples.
        A final note asks the user to double check the automatic mappings.
        Returns an empty list when the mapper is unavailable or nothing
        applies.
        """
        if not self._loaded:
            if not self.load():
                return []
        str_types = [
            t for t in type_names
            if isinstance(t, str) and t and '*' not in t
        ]
        if not str_types or not datasets:
            return []

        # O(1) conflict involvement lookups (a per-type scan over all
        # conflicts would be too slow for result-type sized inputs).
        n_to_1_conflicts = self.get_n_to_1_conflicts()
        one_to_n_conflicts = self.get_1_to_n_conflicts()
        n_to_1_by_source = {c.source_type: c for c in n_to_1_conflicts}
        n_to_1_by_member: Dict[str, TypeMappingConflict] = {}
        for conflict in n_to_1_conflicts:
            for member_type in conflict.target_types:
                n_to_1_by_member.setdefault(member_type, conflict)
        one_to_n_by_source = {c.source_type: c for c in one_to_n_conflicts}

        expanded: List[Tuple[str, Dict[str, str]]] = []
        n_to_1: List[Tuple[str, TypeMappingConflict, bool]] = []
        one_to_n: List[str] = []
        seen_n_to_1: Set[int] = set()
        seen_one_to_n: Set[str] = set()

        for type_name in str_types:
            base_name, _ = self._split_hemi_suffix(type_name)
            mappings = self.resolve_type_across_datasets(base_name, datasets)
            different = {
                ds: mapped
                for ds, mapped in mappings.items()
                if mapped and mapped != base_name
            }
            if different:
                expanded.append((type_name, different))

            conflict = (
                n_to_1_by_source.get(base_name)
                or n_to_1_by_member.get(base_name)
            )
            if conflict is not None and id(conflict) not in seen_n_to_1:
                seen_n_to_1.add(id(conflict))
                n_to_1.append((base_name, conflict, base_name in conflict.target_types))

            conflict = one_to_n_by_source.get(base_name)
            if conflict is not None and base_name not in seen_one_to_n:
                seen_one_to_n.add(base_name)
                one_to_n.append(base_name)

        notes: List[str] = []

        if expanded:
            if len(expanded) <= max_examples:
                for type_name, different in expanded:
                    parts = ', '.join(
                        f"'{mapped}' ({self.get_dataset_full_name(ds)})"
                        for ds, mapped in sorted(different.items())
                    )
                    notes.append(
                        f"Auto type mapping expanded queried type '{type_name}' "
                        f"to {parts}; the queried name may not exist there."
                    )
            else:
                examples = '; '.join(
                    f"'{type_name}' -> '{next(iter(sorted(different.values())))}'"
                    for type_name, different in expanded[:max_examples]
                )
                notes.append(
                    f"Auto type mapping expanded {len(expanded)} type names to "
                    f"their mapped names in other datasets (examples: {examples}; ...)."
                )

        if n_to_1:
            if len(n_to_1) <= max_examples:
                for type_name, conflict, is_member in n_to_1:
                    others = ', '.join(sorted(conflict.target_types))
                    if is_member:
                        notes.append(
                            f"N-to-1 type mapping: '{type_name}' is one of {len(conflict.target_types)} "
                            f"types ({others}) that all correspond to '{conflict.source_type}' in "
                            f"{self.get_dataset_full_name(conflict.source_dataset)}; "
                            "they were NOT merged to avoid wrong aggregation."
                        )
                    else:
                        notes.append(
                            f"N-to-1 type mapping: '{type_name}' corresponds to multiple types "
                            f"({others}) in {self.get_dataset_full_name(conflict.target_dataset)}; "
                            "only exact-name matches were used, so results may be incomplete."
                        )
            else:
                members = sum(1 for _, _, is_member in n_to_1 if is_member)
                examples = ', '.join(
                    f"'{type_name}' ({conflict.source_type})"
                    for type_name, conflict, _ in n_to_1[:max_examples]
                )
                notes.append(
                    f"N-to-1 type mapping involved {len(n_to_1)} type groups "
                    f"({members} queried types are one of several types sharing a "
                    f"name across datasets; examples: {examples}; ...); those were "
                    "NOT merged to avoid wrong aggregation."
                )

        if one_to_n:
            if len(one_to_n) <= max_examples:
                for type_name in one_to_n:
                    conflict = one_to_n_by_source[type_name]
                    split = ', '.join(sorted(conflict.target_types))
                    notes.append(
                        f"1-to-N type mapping: '{type_name}' splits into "
                        f"{len(conflict.target_types)} types ({split}) in "
                        f"{self.get_dataset_full_name(conflict.target_dataset)}; "
                        "no automatic mapping was made for it."
                    )
            else:
                notes.append(
                    f"1-to-N type mapping affected {len(one_to_n)} types "
                    f"(examples: {', '.join(one_to_n[:max_examples])}; ...); "
                    "no automatic mapping was made for them."
                )

        if notes:
            notes.append(
                "These name mappings were applied automatically - please "
                "double check them (against the datasets' type annotations or "
                "the exported mapping files) before interpreting cross-dataset "
                "results."
            )
        return notes

    # Candidate kinds for get_alias_candidates.  Exactly one applies per
    # candidate:
    # - 'same name' vs the rest: identity is checked first.
    # - 'renamed' vs 'splits into': the forward mapping is unique or many.
    # - 'renamed' vs 'one of N': 'renamed' requires a unique reverse
    #   mapping, 'one of N' requires the reverse to be refused (N-to-1).
    # - 'splits into' vs 'one of N': forward conflicts only arise for
    #   male-cns-namespace queries, reverse conflicts only for
    #   non-male-cns queries - the query sits in one namespace.
    ALIAS_KINDS = ('same name', 'renamed', 'splits into', 'one of N')

    def _alias_aggregates(self, name: str, mapping_key: str) -> Optional[List[str]]:
        """Sibling types a match by *name* also covers, if any.

        Non-None exactly when the candidate's own reverse mapping is refused
        (N-to-1): matching by this one name in ``mapping_key`` also matches
        the other listed male-cns types.
        """
        conflict = self._n_to_1_by_source_cache().get(name)
        if conflict is not None and self._get_type_mapping_key(conflict.source_dataset) == mapping_key:
            return sorted(conflict.target_types)
        return None

    def _n_to_1_by_source_cache(self) -> Dict[str, TypeMappingConflict]:
        cache = getattr(self, '_alias_n_to_1_cache', None)
        if cache is None:
            cache = {c.source_type: c for c in self.get_n_to_1_conflicts()}
            self._alias_n_to_1_cache = cache
        return cache

    def get_alias_candidates(
        self,
        type_name: Union[str, int, None],
        datasets: List[str],
    ) -> Dict[str, Dict[str, any]]:
        """Resolve one queried type name into per-dataset alias candidates.

        Built for the neuron-index viewer's expanded search: the local query
        found nothing, so show which names in which datasets correspond to
        the query.  The result never merges datasets - every candidate stays
        attributed to its dataset so callers can keep cross-dataset rows
        strictly informational.

        Returns ``{dataset: outcome_dict}`` where outcome_dict is::

            {'outcome': 'matched', 'candidates': [
                {'name': 'APDN3', 'kind': 'renamed',
                 'aggregates': ['CL125', 'PLP080', 'SLP250']}, ...]}
            # or
            {'outcome': 'no counterpart known', 'candidates': []}
            # or, for non-plain-name queries (bodyId / pattern / empty):
            {'outcome': 'not applicable', 'candidates': []}

        Kinds are mutually exclusive (see ALIAS_KINDS); 'aggregates' is an
        orthogonal annotation naming the sibling types a match by this
        candidate also covers (the candidate's reverse mapping is refused as
        N-to-1).  A 'same name' candidate can carry it too when the name is
        native in both datasets yet aggregates other male-cns types.
        """
        datasets = [str(ds) for ds in (datasets or [])]
        unavailable = {
            ds: {'outcome': 'mapper unavailable', 'candidates': []}
            for ds in datasets
        }
        if not self._loaded:
            if not self.load():
                return unavailable
        if not datasets:
            return {}

        not_applicable = {
            ds: {'outcome': 'not applicable', 'candidates': []}
            for ds in datasets
        }
        if not isinstance(type_name, str):
            return not_applicable
        query = type_name.strip()
        if (
            not query
            or '*' in query
            or query.isdigit()
            or len(query) < 2
        ):
            return not_applicable

        base_name, _ = self._split_hemi_suffix(query)
        query_ns = self._detect_type_source(base_name)

        one_to_n_by_source: Dict[str, List[TypeMappingConflict]] = {}
        for conflict in self.get_1_to_n_conflicts():
            one_to_n_by_source.setdefault(conflict.source_type, []).append(conflict)

        outcomes: Dict[str, Dict[str, any]] = {}
        for dataset in datasets:
            d_key = self._get_type_mapping_key(dataset)
            candidates: List[Dict[str, any]] = []
            seen: Set[str] = set()

            def _add(name: str, kind: str, aggregates: Optional[List[str]] = None):
                if name and name not in seen:
                    seen.add(name)
                    candidates.append({
                        'name': name,
                        'kind': kind,
                        'aggregates': aggregates,
                    })

            # A. identity: the queried name is native in this namespace.
            #    FlyWire primaries come from the dataset's own type column,
            #    so a name can be native even without a crosswalk entry.
            if (
                base_name in self._dataset_types.get(d_key, {})
                or base_name in self._flywire_primaries.get(d_key, ())
            ):
                _add(base_name, 'same name',
                     self._alias_aggregates(base_name, d_key))

            # B. mapping-driven candidates (namespaces must differ; within
            # one namespace the mapping value is the name itself).
            if query_ns is not None and query_ns != d_key:
                mapped = self.get_mapped_type(query, query_ns, dataset)
                if mapped and mapped != base_name:
                    _add(mapped, 'renamed',
                         self._alias_aggregates(mapped, d_key))
                elif mapped is None:
                    if query_ns == 'male-cns:v1.0':
                        # The query splits into several names here.
                        for conflict in one_to_n_by_source.get(base_name, []):
                            if self._get_type_mapping_key(conflict.target_dataset) == d_key:
                                for target in sorted(conflict.target_types):
                                    _add(target, 'splits into')
                    else:
                        # The reverse aggregation is refused: the candidates
                        # are the group members (male-cns namespace only).
                        conflict = self._n_to_1_by_source_cache().get(base_name)
                        if (
                            conflict is not None
                            and self._get_type_mapping_key(conflict.source_dataset) == query_ns
                            and d_key == 'male-cns:v1.0'
                        ):
                            for target in sorted(conflict.target_types):
                                _add(target, 'one of N')

            outcomes[dataset] = {
                'outcome': 'matched' if candidates else 'no counterpart known',
                'candidates': candidates,
            }
        return outcomes

    def _namespace_names(self, key: str) -> set:
        """Every type name known to one namespace (mapping key)."""
        names = set(self._dataset_types.get(key, {}))
        names.update(self._flywire_primaries.get(key, ()))
        for fw_key, alt_table in self._flywire_alt_to_primary.items():
            if self._get_type_mapping_key(fw_key) == key:
                names.update(alt_table)
        return names

    def _flywire_alt_column(self, fw_key: str) -> str:
        """The additional Type(S) column name of one FlyWire namespace."""
        return FLYWIRE_TYPE_SOURCES.get(fw_key, {}).get(
            'alt_column', 'additional_type(s)')

    def _name_neighbors(self, namespace: str, name: str):
        """Derivation neighbors of ``(namespace, name)`` in the name graph.

        Each neighbor is ``(neighbor_namespace, neighbor_name, column, via)``
        where ``column`` names the dataset column that carries the link and
        ``via`` the linking value (when it differs from the neighbor name).
        """
        key = self._get_type_mapping_key(namespace)
        neighbors = []

        # Male-CNS crosswalk cells: every crosswalk edge is licensed by a
        # BRIDGE_SOURCE_MAP entry (§9I) — no hardcoded column/target
        # tuples; the map is the single source of truth for valid bridges.
        for (home, column), targets in BRIDGE_SOURCE_MAP.items():
            if (home != key or column == "type"
                    or column in ANNOTATION_COLUMNS):
                continue
            for target_key in sorted(targets):
                for part in self._crosswalk_parts(name, column):
                    neighbors.append((target_key, part, column, part))

        # FlyWire additional-type columns: an alternative (old) name on
        # rows typed with the primary name.  The edges exist only when
        # the source map licenses this namespace's annotation column.
        alt_column = self._flywire_alt_column(key)
        if source_map_targets(key, alt_column):
            for primary in sorted(
                    self._flywire_alt_to_primary.get(key, {}).get(name, ())
            ):
                neighbors.append((key, primary, alt_column, name))
        # FlyWire reverse annotation edges: a primary type whose rows list
        # additional Type(S) values — the linker bridge walks these BOTH
        # ways (alt -> primary resolves a rename; primary -> alt pools the
        # bodyIds annotated with the linked crosswalk value, which is how
        # BANC/FlyWire type names route into the male-cns crosswalk).
        if source_map_targets(key, alt_column):
            for alt in sorted(
                    self._flywire_primary_to_alts.get(key, {}).get(name, ())
            ):
                neighbors.append((key, alt, alt_column, name))
        # Annotation-reverse edges from the MALE-CNS namespace: when a
        # FlyWire dataset's annotation cells name THIS type (e.g. FAFB
        # rows typed APDN3 carry additional Type(S) 'CL125'), that is
        # registry evidence for the pair and must be walkable from this
        # side too — otherwise the MCNS→FAFB direction loses every
        # mapping that only the FAFB→MCNS direction could see.  This is
        # a male-cns-side bridge form (MCNS --aT-- FAFB, MCNS --ACT--
        # BANC); other namespaces reach the flywire family only through
        # the licensed connector routes, so the edges are gated to the
        # crosswalk home (§bridge rules).  Only namespaces whose
        # annotation column the source map licenses are consulted (§9I).
        if key == CROSSWALK_HOME:
            for fw_key, alt_table in (
                    self._flywire_annotation_primaries.items()):
                if not source_map_targets(
                        fw_key, self._flywire_alt_column(fw_key)):
                    continue
                for primary in sorted(alt_table.get(name, ())):
                    neighbors.append(
                        (fw_key, primary,
                         self._flywire_alt_column(fw_key), name))
        # Reverse-crosswalk legs (§bridge rules, bidirectional): a name
        # of namespace X that appears in a male-cns crosswalk cell hops
        # back to the male-cns primaries carrying it — the reverse of
        # the forward crosswalk leg, licensed by the same source-map
        # entry (the column must route to X).  Without it the reverse
        # direction of every renamed crosswalk pair (HEMI --hT-- MCNS,
        # MANC --mT-- MCNS, MCNS --fT-- FAFB reversed) has no chain.
        if key != CROSSWALK_HOME:
            for (home, column), targets in BRIDGE_SOURCE_MAP.items():
                if (home != CROSSWALK_HOME or column == "type"
                        or column in ANNOTATION_COLUMNS
                        or key not in targets):
                    continue
                for mcns_type in sorted(
                        self._crosswalk_reverse(column).get(name, ())):
                    # §preference: the SAME-NAME arrival keeps its
                    # crosswalk-verification edge too — the evidence chain
                    # (the crosswalk cell naming this very type) must be
                    # walkable, otherwise a same-name pair can never show
                    # metadata verification.  The bare same-name hop is
                    # emitted last (below) and loses the visited race.
                    neighbors.append(
                        (CROSSWALK_HOME, mcns_type, column, name))
        # Cross-namespace annotation-value landing: a token this namespace
        # annotates with may also live in ANOTHER FlyWire namespace's
        # annotation cells (e.g. a FAFB additional_type(s) value that also
        # appears in BANC 'Alternative Cell Type(s)' cells).  The landing
        # hop records the token in that other namespace; the destination's
        # own alt->primary edges then continue to the annotated primaries —
        # the designed two-annotation-column bridge between FAFB and BANC
        # (FAFB type → FAFB additional_types → BANC Alternative Cell
        # Types → BANC type).  Primary names are skipped (the same-name
        # block already covers them) and both namespaces' annotation
        # columns must be licensed (§9I).
        if source_map_targets(key, alt_column):
            for other_key in FLYWIRE_MAPPING_KEYS:
                if other_key == key:
                    continue
                other_alt = self._flywire_alt_column(other_key)
                if not source_map_targets(other_key, other_alt):
                    continue
                if name in self._flywire_primaries.get(other_key, ()):
                    continue
                in_other = (
                    name in self._flywire_annotation_primaries.get(other_key, {})
                    or name in self._flywire_alt_to_primary.get(other_key, {}))
                if in_other:
                    neighbors.append((other_key, name, other_alt, name))
        # Same-name membership in other namespaces (the type columns agree)
        # — EMITTED LAST (§preference, user 2026-09-07): a same-name pair
        # tries to find its metadata bridge FIRST; every evidence edge
        # above (crosswalk, annotation, reverse-crosswalk, landing) wins
        # the walk's visited race, so a same-name pair whose crosswalk
        # cell or annotation cells verify it derives through the EVIDENCE
        # chain (e.g. FAFB DN1a -> MCNS flywireType 'DN1a'), and the bare
        # same-name chain is the LAST choice — reached only when no
        # evidence edge connects the pair.  Sorted so the derivation walk
        # (and every downstream chain order) stays deterministic.
        for other_key in sorted(set(self._dataset_types)
                                | set(self._flywire_primaries)):
            if other_key == key:
                continue
            if name in self._dataset_types.get(other_key, ()) or name in self._flywire_primaries.get(other_key, ()):
                neighbors.append((other_key, name, "type", name))
        return neighbors

    def get_type_bridges(
        self,
        source_type: str,
        source_dataset: str,
        target_dataset: str,
        *,
        max_bridges: int = 6,
    ) -> List[List[Dict[str, str]]]:
        """Derivation chains connecting one type name to a target dataset.

        Every bridge is an ordered list of hops
        ``{'dataset', 'column', 'value'}`` starting at the source type and
        ending at a type of the target dataset — the full evidence chain
        (e.g. male-cns ``type 'CL125'`` → ``flywireType 'LMTe01'`` → FAFB
        ``additional_type(s) 'LMTe01'`` → ``type 'APDN3'``).  Chains are
        searched up to four derivation hops, which covers same-name,
        crosswalk-rename, additional-type-rename (both directions —
        including FlyWire primary types routed through their rows'
        additional Type(S) values, e.g. BANC type names into the male-cns
        crosswalk), and hub-transitive pairs; ambiguous splits return one
        candidate chain per split target.
        """
        if not self._loaded:
            if not self.load():
                return []
        source_type = str(source_type or "").strip()
        if not source_type or self._is_untyped_value(source_type):
            return []
        source_key = self._get_type_mapping_key(source_dataset)
        target_key = self._get_type_mapping_key(target_dataset)
        if source_key == target_key:
            return [
                [{"dataset": source_key, "column": "type", "value": source_type}]
            ] if source_type in self._dataset_types.get(source_key, ()) or source_type in self._flywire_primaries.get(source_key, ()) else []

        start = (source_key, source_type)
        bridges: List[List[Dict[str, str]]] = []
        seen_chains = set()

        def _is_target(ns: str, name: str) -> bool:
            if ns != target_key:
                return False
            if target_key in self._flywire_primaries:
                # FlyWire endpoints must be real primary types, not
                # additional-only names.
                return name in self._flywire_primaries.get(target_key, ())
            return name in self._dataset_types.get(target_key, ())

        def _walk(node, chain, depth, visited, ann_chained=False):
            ns, name = node
            if depth > 0 and _is_target(ns, name):
                # A chain that is NOTHING but type hops through more than
                # one namespace (e.g. MCNS DN1pA -> BANC DN1pA -> FAFB
                # DN1pA) is pure name transitivity — no crosswalk or
                # annotation evidence, and redundant with the direct
                # same-name chain that always exists when both endpoints
                # carry the name.  Never offer it as a derivation.
                if (len(chain) > 2
                        and all(h["column"] == "type"
                                for h in chain[1:])):
                    return
                chain_key = tuple(
                    (h["dataset"], h["column"], h["value"]) for h in chain
                )
                if chain_key not in seen_chains:
                    seen_chains.add(chain_key)
                    bridges.append([dict(h) for h in chain])
                if not has_registry:
                    # §bridge rules: registry-less pairs stop at the
                    # arrival — no annotation continuation (see
                    # has_registry above).
                    return
                # Arrival does NOT end the walk: the reached primary's own
                # annotation edges continue the two-linker registry
                # standard (crosswalk primary -> its annotated siblings,
                # e.g. MCNS FB4A_a -> FAFB FB4A -> FAFB 4I1).  Deeper
                # same-name hops stay blocked by the guard below, so the
                # continuation is annotation-only and bounded by depth.
            if depth >= 5:
                # Depth 5 covers the full registry standard plus one hub
                # routing leg (crosswalk -> additional -> same-name ->
                # annotation); deeper chains carry no new evidence.
                return
            for nns, nname, column, via in self._name_neighbors(ns, name):
                if (nns, nname) in visited:
                    # simple paths only: the bidirectional annotation edges
                    # (alt -> primary and primary -> alt) would otherwise
                    # bounce between a primary and its own additional names.
                    continue
                if self._is_untyped_value(nname):
                    # Untyped labels never become bridge nodes or targets
                    # (mirrors _drop_untyped / overlay semantics; keeps the
                    # BANC 'Unknown' hub out of the derivation walk).
                    continue
                if nns != ns:
                    # Namespace-route license (§bridge rules): a chain
                    # visits at most ONE intermediate namespace, only from
                    # the pair's licensed connector set (ROUTE_MIDS); no
                    # revisits of a namespace already left; the target is
                    # always allowed as the final landing.  This is what
                    # keeps BANC a never-connector for MCNS<->FAFB and
                    # structurally fixes the two-linker order (a flipped
                    # fT/aT chain would have to revisit the source).
                    route_ns = []
                    for h in chain:
                        if h["dataset"] not in route_ns:
                            route_ns.append(h["dataset"])
                    if nns in route_ns:
                        # back into a namespace already left — a revisit
                        continue
                    if nns != target_key and nns not in route_mids:
                        continue
                    if len(route_ns) >= 2 and nns != target_key:
                        # [src, mid, tgt] is the deepest licensed route
                        continue
                prev_column = chain[-1]["column"] if chain else ""
                if (column in ANNOTATION_COLUMNS
                        and prev_column in ANNOTATION_COLUMNS
                        and (nns != target_key or ann_chained)):
                    # No annotation-to-annotation chaining mid-chain:
                    # following a primary's own additional values with
                    # another annotation hop on the same dataset is family
                    # transitivity (T1 -> C2 -> C3 -> L4 ...), not pair
                    # evidence.  The exception is an annotation hop that
                    # ARRIVES in the target namespace (e.g. the BANC LTe71
                    # hub route), which is real routing evidence.
                    continue
                if (column in CROSSWALK_COLUMNS
                        and not crosswalk_route_licensed(
                            column, source_key, target_key)):
                    # Endpoint-family licensing (§9I): a crosswalk hop is
                    # evidence only when the bridge actually connects a
                    # namespace the column routes to — hemibrainType
                    # needs a hemibrain endpoint, mancType a manc
                    # endpoint, flywireType a flywire-family endpoint.
                    continue
                hop = {"dataset": nns, "column": column, "value": nname}
                if via != nname:
                    hop["via"] = via
                chain.append(hop)
                visited.add((nns, nname))
                chained = ann_chained or (
                    column in ANNOTATION_COLUMNS
                    and prev_column in ANNOTATION_COLUMNS
                    and nns == ns)
                _walk((nns, nname), chain, depth + 1, visited, chained)
                visited.discard((nns, nname))
                chain.pop()

        # §bridge rules: this pair's licensed intermediate namespaces
        route_mids = set(ROUTE_MIDS.get(
            frozenset({source_key, target_key}), ()))
        # The post-arrival annotation continuation is the two-linker
        # REGISTRY standard (crosswalk primary -> its annotated siblings).
        # Registry-less pairs (every BANC pair) have no such standard —
        # continuing would wander the target's annotation classes
        # (APDN3 -> R8_unclear -> T1) and land coarse hub names.
        has_registry = bool(
            BRIDGE_STANDARD.get((source_dataset, target_dataset))
            or BRIDGE_STANDARD.get((target_dataset, source_dataset)))

        chain0 = [{"dataset": source_key, "column": "type", "value": source_type}]
        _walk(start, chain0, 0, {(source_key, source_type)})

        # Registry scoping (§9F): for a pair with a BRIDGE_STANDARD
        # registry, a chain's metadata hops must be the pair's own
        # registry columns — a hemibrainType hop on a MCNS~FAFB chain
        # (or a FAFB-annotation hop on a MCNS~HEMI chain) describes a
        # THIRD dataset's naming and must not be offered as evidence.
        # Registry-less (hub) pairs keep all chains, flagged indirect.
        # Registry scoping is DIRECTION-SYMMETRIC: BRIDGE_STANDARD keys
        # are ordered (male-cns first by convention), but a reverse-order
        # call (FAFB -> male-cns) must scope identically — otherwise the
        # unscoped reverse direction offers hub detours through third
        # datasets (e.g. FAFB -> BANC -> male-cns annotation chains),
        # which describe a THIRD dataset's naming, not the pair's.
        registry = (BRIDGE_STANDARD.get((source_dataset, target_dataset))
                    or BRIDGE_STANDARD.get((target_dataset, source_dataset))
                    or ())
        if registry:
            allowed = {column for column, _home in registry} | {"type"}
            endpoints = {source_dataset, target_dataset}
            within = []
            for chain in bridges:
                if not all(hop["column"] in allowed for hop in chain[1:]):
                    continue
                # Namespace containment: every hop must stay within the
                # pair's two namespaces (a crosswalk hop records the
                # namespace it lands in). A detour through a third
                # dataset's naming — e.g. a flywireType hop landing in
                # BANC on a MCNS~FAFB bridge — is not evidence for this
                # pair.
                if not all(hop.get("dataset", source_dataset) in endpoints
                           for hop in chain[1:]):
                    continue
                # the two-linker standard is a hard cap: chained renames
                # (vDeltaB -> vDeltaA -> vDelta ...) exceed the
                # standardizable shape and are not offered
                direct = sum(
                    1 for l in standardize_bridge(
                        chain, source_dataset, target_dataset)
                    if l["kind"] == "linker" and not l["indirect"])
                if direct <= 2:
                    within.append(chain)
            bridges = within

        # Source-map validity (§9I): every hop licensed by
        # BRIDGE_SOURCE_MAP, the crosswalk endpoint rule, and the
        # ping-pong suppression (consecutive identical standardized
        # linkers) in one map-derived check — valid bridges are derived
        # from the source map, not by guessing.
        bridges = [
            chain for chain in bridges
            if bridge_is_valid(chain, source_dataset, target_dataset,
                               key_of=self._get_type_mapping_key)
        ]

        # Evidence subsumption: a bare same-name chain is IMPLIED by any
        # linker-bearing chain to the same target type (the verification
        # proves the pair; the name echo adds nothing).  When a target
        # type has verified derivation(s), keep only those; the bare
        # chain survives only for pairs with no corroboration at all.
        by_end: Dict[str, List[List[Dict[str, str]]]] = {}
        for chain in bridges:
            by_end.setdefault(chain[-1]["value"], []).append(chain)
        filtered: List[List[Dict[str, str]]] = []
        for _end_value, group in by_end.items():
            verified = [
                chain for chain in group
                if any(l["kind"] == "linker" for l in standardize_bridge(
                    chain, source_dataset, target_dataset))
            ]
            filtered.extend(verified or group)
        bridges = filtered
        return bridges

    def get_mapping_origins(
        self,
        local_type: str,
        foreign_type: str,
        foreign_dataset: str,
    ) -> List[Dict[str, str]]:
        """Condensed origin descriptors for one mapped pair.

        Thin wrapper over :meth:`get_type_bridges`: every derivation hop
        except the chain's endpoints becomes one
        ``{'source': <column>, 'via': <value>}`` descriptor, matching the
        historical format (e.g. ``additional_type(s) via 'LMTe01'``).
        """
        source_dataset = (
            self._detect_type_source(local_type) or "male-cns:v1.0"
        )
        bridges = self.get_type_bridges(
            local_type, source_dataset, foreign_dataset
        )
        origins: List[Dict[str, str]] = []
        seen = set()
        for bridge in bridges:
            if len(bridge) == 2 and bridge[1]["column"] == "type":
                # Same name in both datasets (primary on the foreign side).
                if ("type",) not in seen:
                    seen.add(("type",))
                    origins.append({"source": "type"})
                continue
            if len(bridge) < 3:
                continue
            for hop in bridge[1:-1]:
                key = (hop["column"], hop["value"])
                if key not in seen:
                    seen.add(key)
                    origins.append({"source": hop["column"], "via": hop["value"]})
            last = bridge[-1]
            if last.get("via"):
                key = (last["column"], last["via"])
                if key not in seen:
                    seen.add(key)
                    origins.append({"source": last["column"], "via": last["via"]})
        return origins

    def _crosswalk_parts(self, local_type: str, crosswalk_col: str) -> List[str]:
        """Distinct raw crosswalk-cell parts of one male-cns type (cached)."""
        cache = getattr(self, '_crosswalk_parts_cache', None)
        if cache is None:
            cache = {}
            self._crosswalk_parts_cache = cache
        key = (crosswalk_col, local_type)
        if key not in cache:
            parts: set = set()
            neuron_df = getattr(self, '_neuron_df', None)
            if (
                neuron_df is not None
                and crosswalk_col in neuron_df.columns
                and 'type' in neuron_df.columns
            ):
                rows = neuron_df.loc[
                    neuron_df['type'] == local_type, crosswalk_col
                ]
                for cell in rows.dropna().astype(str):
                    parts.update(self._split_type_cell(cell))
            cache[key] = parts
        return sorted(cache[key])

    def get_canonical_type(self, type_name: str, source_dataset: Optional[str] = None) -> str:
        """
        Get the canonical (male-cns) type name for cross-dataset merging.
        
        This is used by NeuronBridge to merge types across datasets.
        Types from different datasets (e.g., 'MTe07' from FAFB, 'MeVPLo2' from male-cns)
        are unified to a single canonical name (the male-cns name).
        
        Args:
            type_name: Type name from any dataset.
            source_dataset: Optional source dataset hint.
            
        Returns:
            Canonical type name (male-cns name if mapping exists, else original).
        """
        if not self._loaded:
            self.load()
        
        # Skip empty or pattern types
        if not type_name or not isinstance(type_name, str):
            return type_name
        if '*' in type_name or ('.' in type_name and '.*' in type_name):
            return type_name
        
        # Detect source dataset
        if source_dataset is None:
            source_dataset = self._detect_type_source(type_name)
        
        if not source_dataset:
            return type_name
        
        # If already from male-cns, return as-is
        if self._get_type_mapping_key(source_dataset) == 'male-cns:v1.0':
            return type_name
        
        # Get male-cns mapping
        mapped = self.get_mapped_type(type_name, source_dataset, 'male-cns:v1.0')
        return mapped if mapped else type_name
    
    def get_merge_mapping_for_types(
        self,
        prefixed_types: List[str],
        queried_name: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, str]:
        """
        Build a merge mapping for prefixed type names (e.g., 'MCNS_aMe12', 'FAFB_MTe07').
        
        This is used by NeuronBridge to properly merge types across datasets,
        accounting for cases where the same neuron has different type names
        in different datasets.
        
        The merged type name format is:
          {queried_name}({datasetA_name}/{datasetB_name})
        
        If no queried_name is provided, uses male-cns name as the main name:
          {mcns_name}({datasetA_name}/{datasetB_name})
        
        Args:
            prefixed_types: List of prefixed type names (e.g., 'MCNS_aMe12', 'FAFB_MTe07').
            queried_name: Optional queried name to use as main display name.
            verbose: Print merge info.
            
        Returns:
            Dict mapping prefixed_type -> merged_display_name.
            E.g., {'MCNS_aMe12': 'aMe12', 'FAFB_aMe12': 'aMe12', 
                   'MCNS_MeVPLo2': 'MeVPLo2(MTe07)', 'FAFB_MTe07': 'MeVPLo2(MTe07)'}
        """
        if not self._loaded:
            self.load()
        
        merge_map = {}
        aggregation_warnings = []  # Track N-to-1 and 1-to-N aggregations
        
        # Group prefixed types by their canonical name to build display names
        canonical_groups: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)  # canonical -> [(prefixed, prefix, base_type)]
        
        # Parse prefixed types and map to canonical
        for prefixed in prefixed_types:
            parts = prefixed.split('_', 1)
            if len(parts) < 2:
                merge_map[prefixed] = prefixed
                continue
            
            prefix, base_type = parts
            
            # Determine source dataset from prefix
            source_ds = None
            prefix_upper = prefix.upper()
            if prefix_upper in ['MCNS', 'MALECNS', 'MALE-CNS']:
                source_ds = 'male-cns:v1.0'
            elif prefix_upper in ['FAFB', 'FLYWIRE', 'FW']:
                source_ds = 'flywire_FAFB_v783'
            elif prefix_upper in ['BANC']:
                source_ds = 'banc_v626'
            elif prefix_upper in ['HEMI', 'HEMIBRAIN', 'HB']:
                source_ds = 'hemibrain:v1.2.1'
            elif prefix_upper in ['MANC']:
                source_ds = 'manc:v1.0'
            
            if source_ds is None:
                merge_map[prefixed] = base_type
                continue
            
            # Get canonical name (male-cns name)
            canonical = self.get_canonical_type(base_type, source_ds)
            
            # Group by canonical
            canonical_groups[canonical].append((prefixed, prefix, base_type))
            
            # Track aggregation warnings
            if canonical != base_type:
                # Check if this is an N-to-1 or 1-to-N case
                for conflict in self._conflicts:
                    if base_type == conflict.source_type or base_type in conflict.target_types:
                        aggregation_warnings.append((prefixed, base_type, canonical, conflict))
                        break
        
        # Build merged display names: {main_name}({other_names})
        for canonical, group_items in canonical_groups.items():
            # Collect all distinct type names (excluding canonical if present)
            all_names = set()
            has_canonical_as_base = False
            for prefixed, prefix, base_type in group_items:
                all_names.add(base_type)
                if base_type == canonical:
                    has_canonical_as_base = True
            
            # Determine main name: queried_name if provided, else male-cns canonical
            main_name = queried_name if queried_name else canonical
            
            # Collect "other" names (names that differ from main_name)
            other_names = {name for name in all_names if name != main_name and name != canonical}
            
            # Build display name
            if other_names:
                # Format: main_name(name1/name2)
                others_str = '/'.join(sorted(other_names))
                display_name = f"{main_name}({others_str})"
            else:
                display_name = main_name
            
            # Assign display name to all prefixed types in this group
            for prefixed, prefix, base_type in group_items:
                merge_map[prefixed] = display_name
        
        if verbose and aggregation_warnings:
            self._log(f"Type merging found {len(aggregation_warnings)} cross-dataset mappings")
            for prefixed, orig, canonical, conflict in aggregation_warnings[:5]:
                if conflict.relationship == 'N-to-1':
                    self._log(f"  {prefixed}: '{orig}' → '{canonical}' (N-to-1 aggregation)")
                else:
                    self._log(f"  {prefixed}: '{orig}' → '{canonical}' (1-to-N aggregation)")
            if len(aggregation_warnings) > 5:
                self._log(f"  ... and {len(aggregation_warnings) - 5} more")
        
        return merge_map
    
    def standardize_partner_types(
        self,
        partner_types: Dict[str, float],
        source_dataset: str,
    ) -> Dict[str, float]:
        """
        Standardize partner type names for cross-dataset comparison.
        
        Maps all type names to their canonical (male-cns) names.
        Weights from types that map to the same canonical name are summed.
        
        This is used during cross-dataset homolog finding to ensure that
        partner types like 'MTe07' (FAFB) and 'MeVPLo2' (male-cns) are
        recognized as the same type.
        
        Args:
            partner_types: Dict[type_name -> weight] of partner connections.
            source_dataset: Dataset the types come from.
            
        Returns:
            Dict[canonical_type -> weight] with standardized type names.
        """
        if not self._loaded:
            self.load()
        
        # If already in the Male-CNS namespace, return as-is.  This includes
        # male-cns:v0.9, whose native type names share the v1.0 namespace.
        if self._get_type_mapping_key(source_dataset) == 'male-cns:v1.0':
            return partner_types.copy()
        
        standardized: Dict[str, float] = {}
        
        for type_name, weight in partner_types.items():
            if not type_name or not isinstance(type_name, str):
                # Keep empty/invalid types as-is
                standardized[type_name] = standardized.get(type_name, 0.0) + weight
                continue
            
            # Skip 2hop prefix handling
            if type_name.startswith('2hop:'):
                base_type = type_name[5:]
                canonical = self.get_canonical_type(base_type, source_dataset)
                canonical_key = f"2hop:{canonical}"
                standardized[canonical_key] = standardized.get(canonical_key, 0.0) + weight
            else:
                canonical = self.get_canonical_type(type_name, source_dataset)
                standardized[canonical] = standardized.get(canonical, 0.0) + weight
        
        return standardized


# Module-level singleton for easy access
_global_type_mapper: Optional[CrossDatasetTypeMapper] = None


def get_type_mapper(workspace_path: Optional[str] = None, force_reload: bool = False) -> CrossDatasetTypeMapper:
    """
    Get the global CrossDatasetTypeMapper instance.
    
    Args:
        workspace_path: Optional workspace path for initialization.
        force_reload: Force reloading of mappings.
        
    Returns:
        CrossDatasetTypeMapper instance.
    """
    global _global_type_mapper
    
    if _global_type_mapper is None or force_reload:
        _global_type_mapper = CrossDatasetTypeMapper(workspace_path=workspace_path, verbose=False)
        _global_type_mapper.load()
    
    return _global_type_mapper
