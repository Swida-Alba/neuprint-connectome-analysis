"""
Canonical applied-threshold / bottleneck provenance for DROCAT
pathfinding runs.

Single source of truth for the applied-threshold contract, shared by
``FindNeuronConnection`` (all modes + replay folders), the exported run
guides, and Cross-Dataset Comparison's summary exports
(``ComparisonAnalyzer._applied_state_for``). Pure scalars in, dict out —
no heavy imports, so any subsystem can compute the same numbers.
"""

def applied_threshold_provenance(
    requested_threshold,
    strongest_first_tau=None,
    strongest_first_budget_bitten=False,
    strongest_dropped_bottleneck=None,
    tau_canonical=None,
    edge_weight_floor=None,
    edge_budget_landing=None,
    edge_budget=None,
    strongest_retained_bottleneck=None,
):
    """Canonical threshold/bottleneck provenance for one pathfinding run.

    Single source for the applied-threshold contract shared by
    parameters.txt, all_attributes.json, data_details/parameters.csv,
    the replay folders, and the run guides:

    - ``requested_threshold``: the user-entered Min Synapse Count before
      any budget effect.
    - ``strongest_first_tau``: the StrongestFirst LANDING tau — all intact
      paths with bottleneck >= tau are retained when the path budget
      bites. For a complete run it is the natural weakest emitted-path
      bottleneck.
    - ``tau_canonical`` / applied threshold: the MINIMAL threshold that
      reproduces the materialized output set (``w2 + 1`` when the budget
      bite leaves a gap; the landing tau otherwise).
    - ``strongest_dropped_bottleneck`` (w2): the strongest path NOT
      emitted after the StrongestFirst budget bites.
    - ``edge_weight_floor`` (w0) / ``edge_budget_landing`` (w1): the Edge
      Budget floor and the tier that determined it; a floored run is
      exactly a complete run at ``max(requested, w0)``.
    - ``strongest_retained_bottleneck`` (W*): the widest-path ceiling
      after lossless pruning; lossless pruning must not change it.

    ``applied_threshold`` is the requested threshold for an
    unbounded/complete run (the natural tau is reported separately);
    when a lossy budget affects the output it is the canonical minimal
    threshold describing the materialized set, and
    ``applied_threshold_source`` names the contributing mechanism(s):
    'requested', 'strongest_first_budget', 'edge_budget', or
    'strongest_first_budget+edge_budget'.
    """
    requested_threshold = int(requested_threshold)
    bitten = bool(strongest_first_budget_bitten)
    floor_applied = edge_weight_floor is not None
    sources = []
    if bitten:
        sources.append('strongest_first_budget')
    if floor_applied:
        sources.append('edge_budget')

    if not sources:
        applied_threshold = requested_threshold
        applied_source = 'requested'
        paths_complete = True
        # Complete runs: the natural tau IS the canonical (minimal)
        # threshold for this set.
        canonical_tau = (
            int(strongest_first_tau)
            if strongest_first_tau is not None else tau_canonical)
    else:
        if tau_canonical is not None:
            canonical_tau = int(tau_canonical)
        elif strongest_dropped_bottleneck is not None:
            canonical_tau = int(strongest_dropped_bottleneck) + 1
        elif strongest_first_tau is not None:
            canonical_tau = int(strongest_first_tau)
        else:
            canonical_tau = None
        if canonical_tau is None and floor_applied:
            canonical_tau = int(edge_weight_floor)
        if canonical_tau is not None and floor_applied:
            # Invariant: the floored graph cannot emit a path weaker than
            # the floor, so canonical >= w0. Keep the max as a guard for
            # degenerate inputs rather than emitting a contradiction.
            canonical_tau = max(canonical_tau, int(edge_weight_floor))
        applied_threshold = canonical_tau
        applied_source = '+'.join(sources)
        paths_complete = False

    return {
        'requested_threshold': requested_threshold,
        'applied_threshold': applied_threshold,
        'applied_threshold_source': applied_source,
        'strongest_first_budget_bitten': bitten,
        'strongest_first_tau': strongest_first_tau,
        'tau_canonical': canonical_tau,
        'strongest_dropped_bottleneck': strongest_dropped_bottleneck,
        'edge_budget': (int(edge_budget) if edge_budget else None),
        'edge_budget_applied': floor_applied,
        'edge_budget_landing': edge_budget_landing,
        'edge_weight_floor': edge_weight_floor,
        'strongest_retained_bottleneck': strongest_retained_bottleneck,
        'paths_complete': paths_complete,
    }
