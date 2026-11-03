---
name: kb-add-topic
description: >-
  Add new material (a chat, transcript, paper, doc, or finished task) to the
  general-knowledge GPU kernel knowledge base in this repository. Sorts it into
  the existing areas, keeps only general lessons, and runs the link, frontmatter,
  and leak gate. Use when the user says "add this to the KB", "capture what I
  learned", "add a topic", or "organize these notes into the KB".
---

# kb-add-topic

Put new knowledge into the KB. The KB is the git repository that contains this
skill: the folder with the root `README.md`, the area folders, and `skills/`.
Below, `<kb-root>` means that folder. Read its root `README.md` first so you know
what already exists.

The KB holds **general knowledge only**. Every path in this skill is relative to
`<kb-root>`. Never write a home-directory path, a host name, or a user name into
the KB, including in this skill.

## Layout

Match the existing tree. The sketch shows its shape:

```
<kb-root>/
├── README.md                  area table + conventions + "What belongs here"
├── hardware/                  facts about chips; true for any kernel
│   ├── README.md
│   ├── amd-cdna/              CU/SIMD/wave, registers, gfx942, gfx950, MI350
│   └── nvidia-blackwell/      TMEM, 2-CTA MMA, scaling
├── kernel-optimization/       DSL-neutral method
│   ├── README.md
│   ├── method/                the loop, baselines, measurement, diagnosis
│   ├── levers/                one note per lever (memory, pipeline, MFMA, epilogue, XCD remap...)
│   ├── autotuning/
│   └── failure-modes/         failure modes, heuristics, dead ends
├── papers/                    one folder per paper
│   ├── README.md
│   └── <paper>/               README.md, glossary.md, sequential 01-/02- section folders
├── dsl/                       tool-specific mappings only
│   ├── README.md
│   └── <dsl>/                 README.md + notes (idioms, primitive map, commands)
├── skills/                    agent skills that maintain this KB (tooling, not knowledge)
│   ├── README.md
│   └── <skill>/               SKILL.md + optional scripts/
└── _sources/                  the author's own research captures, public links only
```

Reading order lives in each folder's `README.md`. Folders have no number prefixes
(a paper may use sequential `01-`, `02-` sections). Every folder has a `README.md`
index. Every note is linked from one README.

## The general test

A note or bullet stays only if **all five** are true:

1. It is true without a specific machine, ticket, branch, or chat.
2. It has no host, container, user name, or home-directory path.
3. Its subject is a concept, not an incident or a date.
4. It cites a public primary source.
5. Task numbers (shapes, µs, speedups) are short examples, never the topic.

If material fails the test, **extract the lesson, drop the log.** Never import a
task log, ticket thread, or tuning diary as a note.

## Where does new material go?

| The material is about... | It goes to... |
|---|---|
| A chip fact (registers, caches, ISA behavior, a SKU) | `hardware/<vendor-arch>/<name>.md` |
| A way to find or fix slowness that works on any kernel | `kernel-optimization/` (method → `method/`, a knob → `levers/`, a bad result or trap → `failure-modes/`) |
| A published paper or public blog | `papers/<paper>/` |
| How one DSL or library spells a concept | `dsl/<dsl>/` |
| A finished task (optimization, port, debug) | **Lessons only**, into the lever/failure-mode notes above. The log stays in the task repo. |
| A host runbook, driver recovery, personal setup | Not the KB. Say so, and suggest a dotfiles or ops repo. |

A new area folder is rare. Add one only if nothing above fits, and add its README
and a row in the root `README.md`.

## Skills (`skills/`)

`skills/` holds agent skills that maintain this KB, such as this one. They are
tooling, not knowledge, so the general test does not apply to them. The gate still
checks their links, frontmatter, and leaks.

- Each skill is `skills/<name>/SKILL.md`, with optional `scripts/`.
- The frontmatter `name` equals the folder name. The gate checks this.
- Link each skill from `skills/README.md`. The gate checks this too.
- A skill has no personal paths, names, or emails either. Write `<kb-root>`, not a path.

## Workflow

### 1. Read and classify
- Read all inputs fully. Classify each part: **general** (keep), **task-specific**
  (extract lessons or drop), **personal/infra** (leave out), **redundant** (drop).
- Read the target notes first. If a note already states the idea, skip it (same
  idea in other words counts as covered). Report what you skipped.

### 2. Propose, then confirm only real forks
Show the file list: new note path, what it covers, which source it came from. Ask
only about choices you cannot infer (a new area; a lesson that could live in two
places). Do not ask about things you can default.

### 3. Write

**New note.** Frontmatter, then body, then a footer:

```markdown
---
name: <kebab-slug equal to the filename without .md>
description: <one specific line, quote it if it has ": ">
source: <public URL or paper; for own captures, _sources/<file>>
---

# Title
...
---
Related: [note](relative/path.md) · [note](relative/path.md)
```

- Quote any frontmatter value that contains `: ` (YAML breaks otherwise).
- Cite public primary sources. Never strip an existing citation.
- Paper and vendor-doc content: keep accurate prose, do not paraphrase for style.
  Add a `Sources:` footer with links, and a "verify specifics against the primary
  source" caveat if numbers are secondhand.
- A note that cites live code paths gets a "paths drift; verify before use" line.

**Lesson in an existing note.** Append one bullet to that note's
`## Lessons from worked examples` section (create it at the end, before any
trailing `---` footer). Format: `- **Short title.** Rule, then the mechanism
(why), 1 to 4 lines.` One short numeric example in parentheses is fine. Do not
change other text in the note. Do not name the task, the ticket, or the shape.

**DSL-neutral rule.** `kernel-optimization/` and `hardware/` name no DSL
primitives, source paths, or probe scripts. Put those in the DSL's folder under
`dsl/<dsl>/`, under a heading named after the general note, and add one line to the
general note that links to it: `<dsl> specifics: [primitive-map](...#<note>)`.
ISA mnemonics (`ds_read_b128`, `s_waitcnt`) and rocprofv3 counter names are not
DSL names; keep them.

**New paper.** Create `papers/<paper>/` with `README.md` (what it is, sources,
reading order, one-sentence thesis), `glossary.md` if useful, and area folders
sequential `01-`, `02-` section folders. Hardware facts the paper depends on go to
`hardware/` and the paper README links there. A mechanism that works beyond the
paper goes to a `kernel-optimization/levers/` note, linked both ways.

**Own research capture.** Raw material that backs a note goes in `_sources/`.
It must also pass the general test. Never store a chat export or usage report.

### 4. Index
- Add each new note to its folder `README.md` as
  `- [name](path.md) — <one line, 25 words max>`, in reading order.
- A new area or paper: add a row to the area table in the root `README.md`.
- Edit only files inside `<kb-root>`.

### 5. Check (do not skip)
Run the gate from `<kb-root>`. It checks broken links, frontmatter, unindexed notes
and skills, and leaks (personal paths, internal hosts, container names, ticket ids,
emails, credentials):

```bash
cd <kb-root>
python3 skills/kb-add-topic/scripts/check_kb.py
```

It must print `OK`. Fix every line it lists. A flagged item that is a real public
term (not a leak) means the pattern is too broad: say so in your report, do not
silence it. Pass another KB root as an argument to check a different KB.

Also read each new note once for task bleed: shape names, ticket words, "we",
"my machine", "this session", dates.

### 6. Commit
- `cd <kb-root>`; one focused commit per topic.
- **Identity:** the repo-local `user.email` must be a GitHub noreply address. Check
  it with `git config user.email` first. Never commit with a work email.
- **Push:** only when the user asks. The remote is `origin`. If a push fails on
  auth, give the user the exact command instead of retrying.

## Anti-patterns
- Importing a task log, ticket thread, or tuning diary as a note.
- Naming a DSL primitive, repo path, or probe script in `kernel-optimization/` or `hardware/`.
- A lesson that restates an existing note in other words.
- Verbatim chat exports or usage reports in `_sources/`.
- Skipping the gate, or loosening it to get `OK`.
- A note not linked from any README, or a frontmatter `name` that differs from the filename.
- A skill whose `name` differs from its folder, or that is not linked from `skills/README.md`.
- Writing a home-directory path, a user name, or an email into any file in the KB.
- Committing with a work email, or pushing unasked.
