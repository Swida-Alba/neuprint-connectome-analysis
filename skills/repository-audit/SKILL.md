---
name: repository-audit
description: Audit code, docs, exports, templates & skills systematically.
---

# Repository Audit

Use this skill for a systematic, evidence-backed audit of a project when the
user wants code, documentation, UI instructions, generated/exported files,
templates, configuration, tests, or skills compared for omissions,
contradictions, drift, and operational risk.

The default output is a durable audit record, not a code change. Preserve
existing worktree changes and do not modify implementation files, tests, UI,
documentation sources, templates, or generated outputs unless the user
explicitly expands the task to include fixes. Writing or incrementally updating
the user-designated audit report and audit data exports is in scope.

## Operating contract

- Start by resolving the project root, report path, requested scope, and any
  explicit exclusions. Inspect `git status --short` before reading deeply so
  pre-existing changes are not mistaken for audit changes.
- Inventory first with `rg --files` and targeted `rg` searches. Classify
  generated, vendored, cached, binary, and ignored paths before deciding what
  to inspect.
- Follow behavior end to end: entry point or UI control -> configuration ->
  public API -> internal implementation -> persistence/cache -> export or
  visualization -> documentation/example/skill.
- Treat code as the observed behavior and documentation as the claimed
  contract. Record a mismatch only when evidence supports it; label inference,
  uncertainty, and unmeasured behavior explicitly.
- Update the report when each audit section is complete. Each checkpoint must
  state what was inspected, evidence locations, findings added or closed,
  unresolved questions, and the next section. Never wait until the final
  synthesis to record all findings.
- Use tight file and line references. Keep a finding separate from its
  proposed fix, and do not silently convert a proposal into an implementation.
- Run focused validation proportional to the risk. Report commands, pass/fail
  results, warnings, fixture limitations, and tests that encode current
  behavior but may conflict with the documented contract.

## Workflow

Read the relevant supporting reference before acting:

1. Read [audit-scope-and-routing.md](references/audit-scope-and-routing.md)
   for artifact discovery and format-specific routing. For a broad project
   audit, read it completely.
2. Read [issue-and-traceability-schema.md](references/issue-and-traceability-schema.md)
   before assigning finding IDs or creating machine-readable registers.
3. Read [export-and-report-templates.md](references/export-and-report-templates.md)
   when a report, issue register, traceability matrix, JSON summary, or other
   export is requested. Reuse the templates in `assets/templates/` rather than
   inventing incompatible schemas.

Then perform these stages, adapting depth to the repository:

### 1. Establish scope and inventory

Identify source roots, tests, configuration, schemas/data, docs, UI assets,
generators, templates, build/publish scripts, export directories, project-local
skills, and CI/release definitions. Record excluded or generated areas and the
reason for exclusion. Find existing audit reports and append to the requested
one rather than overwriting it.

### 2. Audit code and executable behavior

Inspect public entry points, call paths, defaults, validation, error handling,
caching, persistence, schemas, output writers, and feature flags. Include
docstrings, type hints, comments that define behavior, and tests that establish
contracts. For performance concerns, separate discovery, network/database,
graph/algorithm, materialization, rendering, and export cost; do not attribute
total latency to an algorithm without phase evidence.

### 3. Audit documentation and instructions

Check README files, technical guides, UI guides, Markdown/HTML companions,
API docs, examples, inline help, release notes, user guides, and generated
documentation. Compare names, signatures, defaults, allowed values, file
patterns, paths, screenshots, links, warnings, and stated limits against the
observed implementation. Verify internal links and code examples where
practical.

### 4. Audit UI-linked and exported artifacts

Trace labels, controls, defaults, payload fields, progress messages, linked
HTML, JavaScript/CSS assets, export buttons, generated guides, and actual
output files. Verify that templates, renderers, schemas, and representative
exports agree. For DOCX/PDF/PPTX/XLSX or other layout-sensitive artifacts,
inspect extracted content and render a representative artifact when visual
layout is part of the contract; record tool or fixture limitations.

### 5. Audit skills and reusable instructions

Inspect project-local and relevant installed skills: frontmatter, trigger
description, invocation policy, references, scripts, assets, paths, examples,
and claimed outputs. Check that references are discoverable, scripts are
called with valid arguments, assets exist, and instructions do not describe
stale filenames or behavior. Do not modify a skill merely because it is
inconsistent; record the evidence and proposed repair.

### 6. Build the traceability view

For every important behavior, map source implementation, test, UI/instruction,
template/schema, generated/exported artifact, and user-facing claim. Record
unmatched information in both directions:

- implemented but undocumented;
- documented but not implemented or unreachable;
- UI control not propagated to behavior;
- export/template field not emitted, consumed, or documented;
- generated artifact not reproducible from its source;
- skill/reference/script/asset not reachable or stale; and
- conflicting defaults, names, schemas, or version assumptions.

### 7. Validate and synthesize

Run focused tests, static checks, link/path checks, schema checks, and small
fixture probes appropriate to the findings. Then write an executive summary,
prioritized issue register, evidence-backed fixation proposal, validation
record, residual uncertainty, and explicit “no implementation changes” status.

## Report update protocol

Use the report path supplied by the user. If none is supplied and a durable
report is requested, use an existing project audit location; otherwise propose
one before creating it. At every completed section, append or update a
checkpoint using the report template. Keep earlier findings intact, preserve
the report's existing structure, and mark superseded findings rather than
deleting their history.

Use the report as the human-readable source of truth and export the structured
issue register/traceability matrix/summary only when requested or useful for
the project. All exports must include the audit date, project root, scope,
status, evidence references, and validation limitations.

## Safety and stopping rules

This is an inspection workflow. Do not run destructive commands, rewrite
generated files, install dependencies, make network mutations, or “fix” code
as an implied part of auditing. Read-only network/API calls require the user's
scope and are optional; prefer local fixtures and caches when they answer the
question. Stop and report a blocker when a required external system, private
artifact, or missing user decision prevents a reliable conclusion. Continue
with clearly labeled partial findings when safe.

## Reusable exports

The templates in `assets/templates/` provide a compatible starting point for:

- the incrementally updated Markdown audit report;
- a CSV issue register;
- a CSV traceability matrix; and
- a machine-readable JSON audit summary.

Copy or adapt them only into the user-approved output location. Read the export
reference for required fields and escaping rules before populating them.
