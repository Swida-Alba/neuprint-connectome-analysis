# Audit scope and routing

Use this reference to decide what to inspect and how deeply to inspect it.
The goal is coverage with evidence, not an indiscriminate dump of every file.

## Initial inventory

Run safe, read-only checks from the resolved project root:

```sh
pwd
git status --short
rg --files -g '!node_modules' -g '!vendor' -g '!dist' -g '!build' -g '!coverage' | sed -n '1,240p'
```

Then locate likely audit surfaces with targeted searches. Adapt terms to the
project rather than copying every command:

```sh
rg -n "TODO|FIXME|deprecated|experimental|default|export|template|guide|help|docstring" .
rg -n "def |class |function |export |public |@param|@returns" src app lib tests
rg -n "\.html|\.md|\.rst|\.docx|\.pdf|\.pptx|\.xlsx|\.csv|skill|SKILL\.md" .
```

Identify source-of-truth and derived paths before reporting duplicates. Look
for build/generation commands, manifests, package metadata, CI jobs, release
scripts, documentation generators, export writers, and template loaders.

## Artifact coverage matrix

| Artifact family | Inspect | Cross-check against |
|---|---|---|
| Source code | entry points, public APIs, call graph, defaults, validation, errors, caching, persistence, output writers | tests, docstrings, UI, docs, schemas, exports |
| Tests and fixtures | assertions, fixtures, expected files, snapshots, performance tests, skipped tests | current behavior, public contract, generated outputs |
| Configuration | defaults, environment variables, CLI flags, feature flags, version pins | UI defaults, examples, deployment/CI, docs |
| Data/schema | migrations, serialized fields, indexes, metadata, compatibility | readers/writers, templates, exports, fixtures |
| Markdown/RST/text docs | claims, examples, links, file names, limits, version statements | implementation and generated docs |
| HTML/UI instructions | labels, controls, defaults, linked assets, anchors, embedded examples | UI code, payloads, CSS/JS, exported guide |
| Generated guides | source template, generator, output path, content and links | actual UI/API behavior, release artifact |
| Export templates | placeholders, required fields, escaping, naming, versioning | renderer/writer and representative export |
| Exported files | presence, names, schemas, metadata, data provenance, readable layout | template, writer, user guide, consumers |
| Skills/instructions | frontmatter, trigger scope, policy, references, scripts, assets, examples | available paths, actual workflow, claimed output |
| Build/release/CI | generation order, packaging, ignored files, publish paths | reproducibility, artifact inventory, version metadata |

## Code and behavior routing

For each user-visible capability, follow this chain:

```text
UI/CLI/API entry point
  -> parameter/default normalization
  -> validation and routing
  -> core algorithm or business logic
  -> data/cache/network access
  -> filtering/materialization
  -> writer/renderer/export
  -> documentation/example/skill
```

At each arrow, record the actual symbol/file and the contract it establishes.
Pay special attention to:

- values that are renamed, dropped, coerced, or given a hidden default;
- flags that appear in the UI but never reach the backend;
- source/body/type/ID mappings and loss of provenance;
- cache completeness and direction-specific assumptions;
- limits that affect visualization only versus limits that affect computation;
- output files created conditionally or only in one representation; and
- tests that pass because they encode an implementation detail contrary to
  public documentation.

For performance audits, capture separate phases and counters where available:
discovery, cache hits/misses, API/database calls, graph construction,
enumeration, materialization, rendering, and export. A microbenchmark that
omits a real API or file format must be labeled as a core-only benchmark.

## Document and format routing

### Markdown, RST, plain text, and docstrings

Check headings, examples, signatures, defaults, file names, links, code fences,
warnings, output schemas, and terminology. Search both the documented symbol
name and the implementation name; stale aliases often reveal drift.

### HTML and UI-linked instructions

Inspect source HTML and the code that generates it. Verify IDs/anchors, links,
form controls, labels, default values, payload keys, referenced JS/CSS/assets,
and whether the file is source, generated, or manually maintained. Compare the
rendered user flow with the actual backend dispatch.

### DOCX, PDF, PPTX, XLSX, CSV, and other binary/structured exports

First inspect metadata and extractable text/schema. Render only when layout,
pagination, charts, formulas, or visual hierarchy are part of the contract.
Compare a representative generated file with its template and writer. Record
tool availability, skipped visual checks, and whether the sample is current.
Use format-specific document/spreadsheet skills when they are available and
the audit requires their specialized render/verify workflow.

### Templates and generators

Trace template -> renderer/generator -> output directory -> consumer/user
guide. Verify required placeholders, conditional sections, escaping,
character encoding, filenames, version stamping, and failure behavior when a
field is absent. A template that exists but is never selected is an issue even
if it renders correctly in isolation.

### Skills

For each relevant skill folder, inspect `SKILL.md`, optional `agents/openai.yaml`,
references, scripts, and assets. Check frontmatter name/description, trigger
boundaries, implicit invocation, relative paths, referenced resources, script
arguments, and expected outputs. Distinguish a missing optional resource from
a broken required reference.

## Search and evidence tactics

- Prefer `rg`/`rg --files` for text and file discovery.
- Use `nl -ba` with a focused range for line-precise evidence.
- Use `git diff`, `git log`, and file metadata only to explain provenance or
  pre-existing changes; do not rewrite history.
- Search for both sides of every mapping: field name, filename, CLI/UI label,
  class/method, schema key, and output consumer.
- Confirm absence with a scoped search and state the search scope; do not claim
  a repository-wide absence after checking only one directory.
- Prefer a small synthetic fixture to a live external query when testing a
  contract. If a live query is necessary, record date, source, parameters,
  and whether the query can mutate external state.

## Completion check

Before closing a section, confirm that its report checkpoint lists:

1. paths and symbols inspected;
2. searches/tests/rendering performed;
3. confirmed, likely, and unverified issues;
4. unmatched or contradictory information;
5. proposed fixation and priority; and
6. the next audit section or a reason the audit is complete.
