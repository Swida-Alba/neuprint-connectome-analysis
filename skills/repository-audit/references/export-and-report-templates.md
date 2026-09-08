# Audit report and export templates

Use the assets under `assets/templates/` as starting points. They are schemas
and skeletons, not evidence. Populate them only with observed project facts
and approved output paths.

## Human-readable Markdown report

Use `audit-report.md.tmpl` for a new report or mirror its sections when
updating an existing project report. The report should be answer-first but
retain enough detail for another engineer to reproduce the findings:

1. project, audit date, scope, exclusions, report status;
2. executive summary and highest-priority findings;
3. inventory and source-of-truth map;
4. completed audit sections with checkpoints;
5. issue details with evidence, impact, and fixation proposal;
6. code/docs/UI/template/export/skill traceability gaps;
7. validation commands and results;
8. residual uncertainty, blocked checks, and assumptions; and
9. no-implementation-changes statement.

When a report already exists, preserve its headings and history where
possible. Append a dated audit section or checkpoint rather than replacing
earlier conclusions. If a new inspection supersedes a finding, keep the old
finding ID and record the disposition and new evidence.

## CSV issue register

Use `issue-register.csv`. Keep one row per finding and quote fields according
to RFC 4180. Preserve line breaks inside quoted fields when they improve
reproduction. Required columns are:

```text
id,severity,confidence,status,area,title,statement,evidence,impact,affected_artifacts,reproduction,proposal,validation,first_seen,last_checked
```

Use semicolons inside a field when multiple paths or IDs must be represented;
do not create ambiguous comma-separated sublists. Keep the Markdown report as
the narrative source of truth if the CSV would become unwieldy.

## CSV traceability matrix

Use `traceability-matrix.csv`. Required columns are:

```text
trace_id,behavior_or_contract,source_artifact,test_artifact,ui_or_api_artifact,instruction_artifact,template_or_schema,generated_or_exported_artifact,consumer,state,gap_or_note,finding_ids,last_checked
```

Add a row even when one link is absent; use `none found`, `not applicable`, or
`not observed` explicitly. Do not leave a blank cell that could be mistaken
for an unchecked field.

## JSON audit summary

Use `audit-summary.json` for automation or downstream dashboards. Preserve the
top-level keys in the asset. Recommended rules:

- `audit.status` is `planned`, `in_progress`, `blocked`, or `complete`;
- `scope.roots` and `scope.excluded` contain normalized project-relative paths
  unless an external consumer requires absolute paths;
- `counts` is derived from the issue register, not hand-maintained prose;
- each `findings` item has at least `id`, `severity`, `status`, `title`, and
  `evidence`; and
- `validation.commands` stores the exact command while `validation.results`
  stores its outcome, date, and limitations.

Do not put secrets, access tokens, private data, or full external API payloads
into the exports. Store only the minimum identifiers and paths needed to
reproduce the observation.

## Other export formats

If the user asks for HTML, PDF, DOCX, PPTX, XLSX, or a project-specific format:

1. start from the Markdown report and structured registers;
2. use the project's existing renderer/template when one exists;
3. verify required headings, tables, links, filenames, metadata, and data
   provenance;
4. render/inspect layout when the format is layout-sensitive; and
5. record the renderer version, source template, output path, and validation
   result in the report.

Do not silently create a second incompatible schema for the same audit. If a
format requires a different representation, link it to the canonical finding
IDs and trace IDs.

## Export completion checklist

- [ ] report path is user-approved and existing history is preserved;
- [ ] audit date, project root, scope, and status are present;
- [ ] every issue has evidence and a stable ID;
- [ ] unmatched fields are explicit rather than blank;
- [ ] report, CSV, JSON, and generated formats agree on counts and statuses;
- [ ] paths and filenames point to real artifacts or are labeled not observed;
- [ ] no secrets or unnecessary raw payloads are exported; and
- [ ] the report records what was not inspected and why.
