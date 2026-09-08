# Issue and traceability schema

Use this reference to keep findings consistent across the human-readable
report and machine-readable exports.

## Finding record

Every actionable finding should contain these fields:

| Field | Requirement |
|---|---|
| `id` | Stable audit-local ID such as `F-CODE-001`; never recycle an ID |
| `severity` | `P0` blocking/correctness/data-loss, `P1` material behavior/performance, `P2` important drift/usability/maintainability, or `P3` minor polish |
| `confidence` | `confirmed`, `likely`, `inferred`, or `unverified` |
| `status` | `open`, `needs-reproduction`, `accepted-behavior`, `fix-proposed`, `fixed-pending-validation`, `verified`, or `deferred` |
| `area` | Code, test, config, data/schema, docs, UI, HTML, export, template, skill, build/release, performance, or cross-cutting |
| `title` | Short statement of the issue, not the fix |
| `statement` | One precise description of observed behavior or mismatch |
| `evidence` | Absolute or repository-resolvable paths with line ranges, commands, test names, or artifact identifiers |
| `impact` | User, data, correctness, performance, reproducibility, support, or maintenance consequence |
| `affected_artifacts` | All code/docs/UI/templates/exports/skills implicated |
| `reproduction` | Minimal command, fixture, interaction, or comparison; write `not reproduced` with reason when applicable |
| `proposal` | Smallest safe fixation direction, clearly separated from implementation |
| `validation` | Test/check/rendering needed to verify the proposal |
| `first_seen` / `last_checked` | Dates for audit history |

Use one finding for one causal issue. Link related findings instead of merging
unrelated symptoms into a large paragraph. A performance observation and a
correctness issue may share evidence but should have separate IDs when their
fixes or priorities differ.

## Evidence standard

Use the strongest available evidence in this order:

1. deterministic test or minimal fixture;
2. direct code path with line-specific behavior;
3. generated artifact compared with its source/template;
4. reproducible command output or phase timing;
5. documentation/UI comparison;
6. reasoned inference, explicitly labeled.

Do not call a suspected issue “confirmed” solely because a document looks
stale. Confirm the implementation path or label the item as likely. Do not
call a passing test proof of correctness if it asserts a narrower behavior than
the public contract; record that distinction.

## Traceability matrix

The matrix maps a behavior or contract through its artifact chain:

```text
source symbol/config
  -> test/fixture
  -> UI/API instruction
  -> template/schema/generator
  -> generated/exported artifact
  -> consumer/user-facing claim
```

Recommended columns:

| Column | Meaning |
|---|---|
| `trace_id` | Stable row ID such as `TR-001` |
| `behavior_or_contract` | Plain-language behavior being traced |
| `source_artifact` | Implementing symbol/path and lines |
| `test_artifact` | Test/fixture or `none found` |
| `ui_or_api_artifact` | Control, endpoint, payload, or `not applicable` |
| `instruction_artifact` | README/guide/docstring/skill path and lines |
| `template_or_schema` | Template/schema/generator path or `not applicable` |
| `generated_or_exported_artifact` | Output path/pattern or `not observed` |
| `consumer` | Downstream reader, renderer, user, or service |
| `state` | `matched`, `partial`, `unmatched`, `conflicting`, `unverified` |
| `gap_or_note` | Exact missing, stale, or conflicting information |
| `finding_ids` | Related finding IDs |
| `last_checked` | Audit date |

## Unmatched-information taxonomy

Use one or more of these labels to make gaps searchable:

- `CODE_UNDOCUMENTED`: behavior exists but no adequate user/developer
  description was found;
- `DOC_NOT_IMPLEMENTED`: instructions claim behavior not reachable in code;
- `UI_NOT_PROPAGATED`: UI label/control/default is not reflected in the
  payload or backend behavior;
- `EXPORT_NOT_EMITTED`: documented/template field is absent from generated
  output;
- `EXPORT_UNDOCUMENTED`: output exists without a guide/schema/provenance
  description;
- `TEMPLATE_UNREACHABLE`: template or variant is not selected by any writer;
- `GENERATOR_DRIFT`: generated artifact differs from the current source or
  cannot be reproduced;
- `SKILL_DRIFT`: skill metadata/instructions/resources do not match the
  available workflow;
- `DEFAULT_CONFLICT`: code, UI, docs, template, or example use different
  defaults;
- `NAME_OR_PATH_DRIFT`: symbols, filenames, anchors, or paths disagree;
- `SCHEMA_DRIFT`: readers, writers, templates, and exports disagree on fields;
- `LIMIT_SEMANTICS`: two limits have different scopes but are described as
  interchangeable;
- `VALIDATION_GAP`: an important behavior has no focused test/check; and
- `OBSERVABILITY_GAP`: a failure or performance problem cannot be diagnosed
  from recorded metadata.

## Severity and disposition

- `P0`: incorrect, missing, or destructive output; security/data integrity;
  blocks a core supported workflow.
- `P1`: material performance, reliability, compatibility, or user-facing
  behavior issue; significant workaround or degraded result.
- `P2`: important documentation, maintainability, usability, or consistency
  issue that does not usually invalidate core output.
- `P3`: low-risk polish or minor inconsistency.

Use `accepted-behavior` only when the project owner or authoritative contract
explicitly confirms the observed behavior. Use `deferred` when the issue is
valid but intentionally postponed, and retain the evidence and proposal.

## Checkpoint record

At the end of each completed audit section, add a compact checkpoint to the
report:

```markdown
### Audit checkpoint — <section> — <date>

- Inspected: <paths, symbols, artifact classes>
- Evidence: <commands, tests, renders, line ranges>
- Findings added/updated: <IDs and status changes>
- Unmatched information: <gaps or “none found”>
- Limitations: <fixtures, network, binary rendering, or scope limits>
- Next: <next section or completion statement>
```

The checkpoint is part of the audit trail; do not replace it with a silent
edit to an earlier narrative.
