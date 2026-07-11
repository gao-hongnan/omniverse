# Content-Modernization Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rescue at-risk book content, bump the repo to Python 3.14, and modernize + extend the type-theory series (8 articles → 13 spine entries) per `docs/superpowers/specs/2026-07-11-omniverse-content-modernization-pilot-design.md`.

**Architecture:** Three stages, strictly ordered: (1) Phase-0 rescue commits protect uncommitted/broken content before any tooling churn; (2) the Python 3.14 spike moves the toolchain (4 pyproject pin edits + lock refresh, with a defined failure ladder); (3) the typing series is rewritten one article per task in spine order, each independently shippable with author review between tasks. All existing URLs stay stable except one deliberate rename (07) covered by a redirect.

**Tech Stack:** Jupyter Book 1.0.4 (Sphinx 7.4.7, MyST), uv, Python 3.14 (fallback 3.13), pyright + mypy for snippet verification, sphinx-reredirects.

## Global Constraints

- Spec of record: `docs/superpowers/specs/2026-07-11-omniverse-content-modernization-pilot-design.md`. Its §5.2 errata register and §6 standard are authoritative; task work orders below restate the relevant items.
- Python target: `requires-python = ">=3.14"`; fallback `">=3.13"` only if Gate 1/2 of Task 3 fails on a genuinely needed dependency.
- Article code idiom (spec §6): builtins generics (`list[int]`, `X | Y`); PEP 695 syntax for all taught generics (`class Pair[S, T]:`, `def first[T](...)`); legacy `TypeVar(...)`/`Generic[T]`/`covariant=True` appear only inside explicitly-labeled "legacy" sidebars; `typing.reveal_type` (3.11+), `typing.override` (3.12+), `typing.Self`; ABC imports from `collections.abc`; type aliases via `type` statement.
- Checker outputs in articles: show **pyright and mypy** where they differ; never write "compile time" for static analysis — write "static-analysis time" or "type-check time".
- House style to preserve on every article: jupytext MyST frontmatter; hidden `:tags: [remove-cell]` setup cell importing only what the page uses; badge block; `{contents}` with `:local:`; executed `{code-cell} ipython3` for runtime truths (failures wrapped in try/except so the book builds); static ```python / ```bash blocks for checker-rejected code and checker output; named footnotes; `{prf:*}` for formal content. Keep all existing `{prf:*}` labels stable (06 cross-references 03's criterion label).
- Snippet verification commands (run for EVERY code example before pasting its claimed checker verdict; use the scratchpad dir, never commit harness files):
  ```bash
  # write snippet to /tmp scratch file first, then:
  uvx pyright --pythonversion 3.14 <snippet>.py
  uvx --python 3.14 mypy --python-version 3.14 --strict <snippet>.py
  ```
  For "checker rejects this" examples: run, confirm the error appears, paste the REAL output (trim paths). After Task 4 lands, `uv run mypy` / repo pyright also work.
- Build gates: `make ci` (lint, security, typecheck, test, coverage) and `make docs` (`cd omniverse && uv run jupyter-book build .`; ~8–15 min, cached execution). A task is not done if either regresses.
- Commits: conventional-commit style enforced by commitizen (`docs: ...`, `build: ...`, `fix: ...`, `chore: ...`), lowercase, trailing period, plus `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Author review checkpoint:** after each article task (6–18), STOP and ask the author to review the rendered page before starting the next article (spec §8).
- Verify-at-writing-time items (spec §10): current PEP 661 / `typing_extensions.Sentinel` status (Task 17) and PEP 728 status (Task 16) must be checked against peps.python.org / typing docs via web before writing those sections.

---

### Task 1: Rescue — commit memory_internals and wire it into the TOC

**Files:**
- Commit (untracked): `omniverse/software_engineering/python/memory_internals/` (7 md files + `c_examples/`)
- Modify: `omniverse/_toc.yml:261-266`
- Modify: `omniverse/software_engineering/python/memory_internals/06_optimization_techniques.md:2199`

**Interfaces:**
- Consumes: nothing.
- Produces: memory_internals series fully published; later tasks assume a clean `git status` for this directory.

- [ ] **Step 1: Commit the untracked series exactly as-is (protect first, edit after)**

```bash
git add omniverse/software_engineering/python/memory_internals/
git commit -m "docs: add python memory internals series.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 2: Fix the broken "coming soon" link in 06**

In `06_optimization_techniques.md` (line ~2199), replace:

```markdown
- **Next:** [07 - Advanced Topics](07_advanced_topics.md) *(coming soon)*
```

with:

```markdown
- **Next:** 07 - Advanced Topics *(planned)*
```

- [ ] **Step 3: Wire 05 and 06 into the TOC**

In `omniverse/_toc.yml`, the block currently ends at `04_string_internals.md`:

```yaml
          - file: software_engineering/python/memory_internals/00_introduction.md
            sections:
              - file: software_engineering/python/memory_internals/01_c_memory_basics.md
              - file: software_engineering/python/memory_internals/02_python_object_model.md
              - file: software_engineering/python/memory_internals/03_integer_internals.md
              - file: software_engineering/python/memory_internals/04_string_internals.md
```

Append two lines inside the same `sections:` list:

```yaml
              - file: software_engineering/python/memory_internals/05_ctypes_inspection.md
              - file: software_engineering/python/memory_internals/06_optimization_techniques.md
```

- [ ] **Step 4: Build and verify**

Run: `make docs`
Expected: build completes; `omniverse/_build/html/software_engineering/python/memory_internals/05_ctypes_inspection.html` and `06_optimization_techniques.html` exist; no new warnings referencing `memory_internals`.

- [ ] **Step 5: Commit**

```bash
git add omniverse/_toc.yml omniverse/software_engineering/python/memory_internals/06_optimization_techniques.md
git commit -m "docs: wire memory internals 05-06 into toc and fix dangling next link.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 2: Rescue — fix live TODO bodies and the stale copyright year

**Files:**
- Modify: `omniverse/probability_theory/03_discrete_random_variables/0303_probability_mass_function.md:107-109`
- Modify: `omniverse/probability_theory/03_discrete_random_variables/binomial/0309_binomial_distribution_concept.md:125-129`
- Modify: `omniverse/_config.yml:10`

**Interfaces:**
- Consumes: nothing.
- Produces: zero `**TODO**` bodies in TOC'd probability pages (Phase-0 acceptance).

- [ ] **Step 1: Write the missing normalization proof**

In `0303_probability_mass_function.md`, replace:

````markdown
```{prf:proof}
**TODO**
```
````

with (uses the book's `_config.yml` macros `\S`, `\pmf`, `\P`, `\lset`, `\rset`, `\lsq`, `\rsq`, `\st`):

````markdown
```{prf:proof}
Since $X$ is a discrete random variable, its range $X(\S)$ is countable.
The events $\lset X = x \rset$ for $x \in X(\S)$ partition $\S$: every
sample point $\xi \in \S$ is mapped by the function $X$ to exactly one
state $x = X(\xi)$, so the pre-images
$X^{-1}\lpar \lset x \rset \rpar = \lset \xi \in \S \st X(\xi) = x \rset$
are pairwise disjoint, and their union over all $x \in X(\S)$ recovers
$\S$.

By countable additivity of the probability law $\P$,

$$
\sum_{x \in X(\S)} \pmf(x)
= \sum_{x \in X(\S)} \P \lsq X = x \rsq
= \P \lsq \bigcup_{x \in X(\S)} \lset X = x \rset \rsq
= \P \lsq \S \rsq
= 1
$$

where the last equality is the normalization axiom of $\P$.
```
````

- [ ] **Step 2: Remove the unfulfilled CDF/ECDF promise in the binomial page**

In `0309_binomial_distribution_concept.md`, delete this paragraph and the TODO (no CDF plotting helper exists in `omnivault.utils.probability_theory.plot`; the section reads cleanly without it):

```markdown
The below plot shows the CDF and its Empirical ECDF distribution for parameters
$n=10$ and $p=0.5$, with the latter consisting of 5000 samples drawn from a
binomial distribution.

**TODO**.
```

- [ ] **Step 3: Update the copyright year**

In `omniverse/_config.yml` line 10: `copyright: "2024"` → `copyright: "2026"`.

- [ ] **Step 4: Verify no TODO bodies remain in TOC'd probability pages**

Run: `grep -rn '\*\*TODO\*\*' omniverse/probability_theory/`
Expected: no output.

- [ ] **Step 5: Build, then commit**

Run: `make docs` — expected: build completes; the PMF page renders the proof.

```bash
git add omniverse/probability_theory/03_discrete_random_variables/0303_probability_mass_function.md omniverse/probability_theory/03_discrete_random_variables/binomial/0309_binomial_distribution_concept.md omniverse/_config.yml
git commit -m "docs: write pmf normalization proof, drop dangling binomial todo, bump copyright.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 3: Python 3.14 spike — Gates 1–2 (resolve and install)

**Files:**
- Modify: `pyproject.toml:11` (requires-python), `:21-40` (dependencies)
- Modify: `.python-version`
- Regenerate: `uv.lock`

**Interfaces:**
- Consumes: probe evidence in spec §7.1.
- Produces: an installed 3.14 environment; Task 4 runs the quality gates on it.

- [ ] **Step 1: Apply the version edits**

In `pyproject.toml`:

```toml
requires-python = ">=3.14"
```

In `dependencies`, change these four lines and delete one:

```toml
    "matplotlib>=3.10",          # was: "matplotlib>=3.8.0,<3.9.1"
    "numpy>=2.1",                # was: "numpy>=1.26.0,<2.0.0"
    "scikit-learn>=1.6",         # was: "scikit-learn==1.5.0"
    "torch>=2.9",                # was: "torch>=2.1.0"  (2.9 = first with cp314 wheels)
```

Delete the line `    "torchtext",` (imported by nothing; archived upstream, capped at Python 3.12).

Update `.python-version`: `3.12` → `3.14`.

Also update the classifiers list: replace `"Programming Language :: Python :: 3.12",` with `"Programming Language :: Python :: 3.14",` (keep the 3.13 classifier).

- [ ] **Step 2: Gate 1 — resolve with fresh metadata**

Run: `uv lock --refresh`
Expected: `Resolved ~207 packages`; `grep -A1 'name = "torch"' uv.lock | head -2` shows `version = "2.9.0"` or newer; `grep -A1 'name = "numpy"' uv.lock | head -2` shows 2.x.
**On failure:** the error names the offending package → raise/remove that pin and re-run. Only if a needed dependency has no 3.14-compatible release: set `requires-python = ">=3.13"` and `.python-version` to `3.13`, re-run, and record the blocker in the spec's §10.

- [ ] **Step 3: Gate 2 — install**

Run: `uv sync --all-extras --all-groups`
Expected: completes; `uv run python -V` prints `Python 3.14.x`.
**On failure (sdist build):** the failing package's chosen version predates 3.14 → add a floor pin for it (newest release) and return to Step 2.

- [ ] **Step 4: Smoke-import the package**

Run: `uv run python -c "import omnivault, torch, numpy, sklearn, matplotlib; print(torch.__version__, numpy.__version__)"`
Expected: prints versions, no ImportError.

Do NOT commit yet — Task 4 completes the spike and commits it atomically.

### Task 4: Python 3.14 spike — Gate 3 (toolchain behaves) and commit

**Files:**
- Modify: `pyproject.toml` dependency-groups (`mypy==1.13.0` → floor)
- Modify: `pyrightconfig.json:2`
- Check: `.github/workflows/cicd_deploy_jupyterbook.yaml`, `Makefile`

**Interfaces:**
- Consumes: installed 3.14 env from Task 3.
- Produces: green `make ci` + `make docs` on 3.14 — the baseline every article task builds on.

- [ ] **Step 1: Bump type-checking toolchain**

In `pyproject.toml` `[dependency-groups] type`: `"mypy==1.13.0"` → `"mypy>=1.16"` (1.13 predates full 3.13/3.14 support). Run `uv lock && uv sync --all-extras --all-groups`.
In `pyrightconfig.json`: `"pythonVersion": "3.13"` → `"pythonVersion": "3.14"`.

- [ ] **Step 2: Check CI has no hardcoded interpreter**

Run: `grep -n "3\.1[0-3]" .github/workflows/*.yaml Makefile`
Expected: no interpreter pins (setup-uv reads `.python-version`). If a pin appears, update it to 3.14 in the same style.

- [ ] **Step 3: Run the quality gate**

Run: `make ci`
Expected: lint, security, typecheck, test, coverage all pass.
**On mypy failures in `omnivault/`:** fix only what the new mypy flags — smallest change that keeps behavior (add annotations, replace deprecated aliases). Do not retype the package beyond what `make ci` requires (spec §10).
**On test failures:** these are numpy-2.x/torch behavioral — fix per module; if a fix is non-obvious, record it and consult the author before hacking around it.

- [ ] **Step 4: Run the book build**

Run: `make docs`
Expected: completes. Executed notebooks re-run on the 3.14 kernel.
**On a notebook that resists (numpy 2.x behavior):** fix the page if local; otherwise add its path to `_config.yml` `execute:exclude_patterns` (existing pattern — cached outputs still render) and record it.

- [ ] **Step 5: Commit the spike atomically**

```bash
git add pyproject.toml uv.lock .python-version pyrightconfig.json
# plus any files fixed in steps 3-4 and _config.yml if exclude_patterns changed
git commit -m "build: bump to python 3.14, numpy 2, modern torch stack; drop unused torchtext.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 5: Rename chapter 07 with a redirect

**Files:**
- Rename: `omniverse/computer_science/type_theory/07-pep-3124-overloading.md` → `07-overload.md`
- Modify: `pyproject.toml` docs group, `omniverse/_config.yml` (extensions + redirects), `omniverse/_toc.yml:290`

**Interfaces:**
- Consumes: nothing.
- Produces: `07-overload.md` (Task 15 rewrites its content); old URL redirects.

- [ ] **Step 1: Rename and rewire**

```bash
git mv omniverse/computer_science/type_theory/07-pep-3124-overloading.md omniverse/computer_science/type_theory/07-overload.md
```

In `omniverse/_toc.yml`, change `- file: computer_science/type_theory/07-pep-3124-overloading.md` → `- file: computer_science/type_theory/07-overload.md`.

- [ ] **Step 2: Add sphinx-reredirects**

`pyproject.toml` docs group: add `"sphinx-reredirects",`. Run `uv lock && uv sync --group docs`.

In `omniverse/_config.yml` under `sphinx:`, add to `extra_extensions:` the line `- sphinx_reredirects`, and under `sphinx: config:` add:

```yaml
    redirects:
      "computer_science/type_theory/07-pep-3124-overloading": "07-overload.html"
```

- [ ] **Step 3: Build and verify the redirect**

Run: `make docs`
Expected: `omniverse/_build/html/computer_science/type_theory/07-pep-3124-overloading.html` exists and contains `http-equiv="refresh"` pointing at `07-overload.html`; `07-overload.html` renders the (still-old) content.
**If sphinx-reredirects fails against Sphinx 7.4.7:** fall back per spec §10 — remove the extension and instead append a post-build step to `.github/workflows/cicd_deploy_jupyterbook.yaml` that writes the stub html before deploy; verify the stub locally with the same grep.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "build(docs): rename overload chapter with redirect from pep-3124 slug.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 6: Article — intro.md

**Files:**
- Modify: `omniverse/computer_science/type_theory/intro.md` (51 lines)

**Interfaces:**
- Consumes: spine order (Global Constraints; spec §5.1 table).
- Produces: roadmap section other chapters may link to.

- [ ] **Step 1: Rewrite per work order**

Keep: the type-theory lineage framing (set theory → Russell's paradox → static analysis) and the humility disclaimer. Change:

1. Add the house badge block at the top (copy the shields.io block verbatim from `01-subtypes.md` lines 1–20, adjust the page-specific Tag badge to `Structured_Musings`).
2. Replace the "skim PEP 483 first" advice with: the maintained typing specification (`https://typing.python.org/en/latest/spec/`) as the canonical reference, PEP 483/484 repositioned as historical design records.
3. Add a "How this series is organized" roadmap: one line per spine entry in TOC order (13 entries, titles from spec §5.1) with `{doc}` links, and one sentence on the 3.14-first convention: examples target Python 3.14 syntax and are verified with pyright and mypy; legacy forms appear in labeled sidebars.
4. Either cite the two dangling references (Luo; Muñoz) from a sentence in the lineage paragraph or delete them.

- [ ] **Step 2: Build and check**

Run: `make docs` — expected: intro renders with badges + roadmap; `{doc}` links resolve (new-chapter links added only as those files land — link ONLY the 9 existing files now; list unwritten chapters as plain text with "(upcoming)" and convert to links in Tasks 11/14/16/18).

- [ ] **Step 3: Commit and pause for author review**

```bash
git add omniverse/computer_science/type_theory/intro.md
git commit -m "docs: rewrite type theory intro with roadmap and typing spec reference.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

STOP: ask the author to review the rendered page before Task 7.

### Task 7: Article — 01-subtypes.md (light touch)

**Files:**
- Modify: `omniverse/computer_science/type_theory/01-subtypes.md` (619 lines)

**Interfaces:**
- Consumes: Global Constraints idiom rules.
- Produces: the canonical `Animal`/`Dog`/`Cat` fixture cell that 02/03 reuse by reference (`{doc}` link back), and section labels for cross-refs.

- [ ] **Step 1: Apply errata and modernization**

1. L113–115 admonition: "when defining via **`mypy`'s `Protocol`**" → `typing.Protocol` ([PEP 544], stdlib since 3.8); checked by any type checker.
2. L190: fix "Robit" → "Robot".
3. L40 setup cell: remove unused imports; replace `typing.Sized` → `collections.abc.Sized`; `Type[Sized]` → `type[Sized]` (keep the reproduced CPython `.pyi` excerpt verbatim — it is quoted source; annotate it as such).
4. L230: `T = TypeVar("T")` on the non-generic `Dataset` → make it honest PEP 695: `class Dataset[T]:` with `def __init__(self, elements: Sequence[T]) -> None:`.
5. Retitle "Pros and Cons" → "When Structural Subtyping Backfires: LSP" (content already is that discussion).
6. The `Flyable` cell (L338–355) whose comments claim "Error" while executing silently: keep the executed cell error-free, and move the checker rejection into a static block with REAL verified pyright + mypy output.
7. Compress "Inclusive vs. Coercive Implementations" (∀/∃ formalism for `5 + 2.5`) to ~2 paragraphs + one `{prf:remark}`.
8. Add a closing "Summary" section: nominal vs structural in one table, subtyping = substitutability, forward pointer to 02/03.
9. Mark the Animal/Dog/Cat definition cell with a comment `# Canonical fixture for this series — reused by later chapters` (02/03 will link here instead of redefining).

- [ ] **Step 2: Verify snippets**

Every modified code example through the Global Constraints commands; paste real outputs for every checker claim.

- [ ] **Step 3: Build, commit, pause**

Run: `make docs` — page renders, no new warnings.

```bash
git add omniverse/computer_science/type_theory/01-subtypes.md
git commit -m "docs: modernize subtypes article to pep 695 idiom and fix protocol attribution.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

STOP for author review.

### Task 8: Article — 02-type-safety.md (heavy rewrite)

**Files:**
- Modify: `omniverse/computer_science/type_theory/02-type-safety.md` (381 lines)

**Interfaces:**
- Consumes: 01's canonical fixture (link, don't redefine); 03's `{prf:theorem}` LSP label for cross-reference.
- Produces: the series' authoritative "int/float is promotion, not subtyping" treatment (06 links to it).

- [ ] **Step 1: Apply errata and restructure**

1. Collapse the L49–72 triple-restatement of substitutability into one paragraph; keep the `{prf:definition}` of type safety verbatim (label stable).
2. De-dupe LSP: replace the re-explanation with a `{prf:ref}` to 03's theorem.
3. **Fix the false claim (L236–243)**: rewrite the int/float passage as its own subsection "Why `int` is *not* a subtype of `float`": PEP 484's numeric tower is a checker special-case (*promotion*) — `isinstance(3.0, int)` is `False`, there is no truncation on assignment (names rebind); show `x: float = 3` accepted by both checkers (verified output) next to `isinstance` evidence in an executed cell.
4. **Fix the broken example (L225–229)**: `entities = [Dog, Cat, Robot]` passes classes — change to instances `[Dog(), Cat(), Robot()]` and show the REAL checker error for `Robot` (verified).
5. Rename the duplicate sections: "Violating Type Safety" stays; "Further Violation of Type Safety" content merges into the int/float subsection (its two bare cells are that topic).
6. Add a setup cell (house style — the article currently imports mid-page at L264).
7. Keep the dynamic-vs-static (Siek/gradual typing) section but retitle "Static, Dynamic, and Gradual Typing" and replace the link-rot-prone `wphomes.soic.indiana.edu` link with the archived paper reference; keep the Java `if (false)` example.
8. mypy-only framing → pyright + mypy per Global Constraints; add Summary section.

- [ ] **Step 2: Verify snippets** (Global Constraints commands; real outputs).

- [ ] **Step 3: Build, commit, pause**

```bash
git add omniverse/computer_science/type_theory/02-type-safety.md
git commit -m "docs: rewrite type safety article, correct int-float promotion story.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

STOP for author review.

### Task 9: Article — 03-subsumption.md (medium)

**Files:**
- Modify: `omniverse/computer_science/type_theory/03-subsumption.md` (466 lines)

**Interfaces:**
- Consumes: 01 fixture; 02's promotion subsection (link).
- Produces: `{prf:criterion}` and `{prf:theorem}` labels UNCHANGED (06 references them).

- [ ] **Step 1: Apply errata and tighten**

1. **Fix the sign error**: L301–302 says "$\mathcal{T}_1$ as `Circle` and $\mathcal{T}_2$ as `Shape`" while the criterion is stated for $\mathcal{T}_2 <: \mathcal{T}_1$ and L359 uses the opposite — make the whole worked example consistently $\mathcal{T}_1$ = `Shape` (supertype), $\mathcal{T}_2$ = `Circle` (subtype). Keep the criterion's wording and label verbatim.
2. Soften antisymmetry (L207–212): subtyping per PEP 483 is reflexive + transitive — a *preorder*; in structural systems two distinct protocols can mutually subsume (equivalent, not equal). Reframe the section as "Reflexivity and Transitivity (and why not Antisymmetry)".
3. Fix "Transivity" → "Transitivity" in both headings (L191, L397).
4. Fix `PoliceDog.search(self) -> str` (L417–419): body prints and returns `None` — make it `return f"{self.name} is searching."` and show it type-checks.
5. Compress the ℤ/ℝ walkthrough to a short `{prf:example}`, replacing the "In python, ℝ can be denoted as float and ℤ as int" mapping with a link to 02's promotion subsection; keep Circle/Shape as the full worked example.
6. Setup cell: remove the entirely-unused import list (L38).
7. pyright + mypy framing; add Summary.

- [ ] **Step 2: Verify snippets.** Confirm with `grep -n "prf:ref" omniverse/computer_science/type_theory/06-*.md` that 06's reference target label still exists unchanged.

- [ ] **Step 3: Build, commit, pause**

```bash
git add omniverse/computer_science/type_theory/03-subsumption.md
git commit -m "docs: fix subsumption criterion orientation and preorder framing.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

STOP for author review.

### Task 10: Article — 04-generics.md (heavy rewrite)

**Files:**
- Modify: `omniverse/computer_science/type_theory/04-generics.md` (911 lines)

**Interfaces:**
- Consumes: idiom rules; 01 fixture.
- Produces: PEP 695-first generics chapter; ends with forward pointers to the new PEP 695 deep-dive (Task 11) and bounds chapter (Task 12). The `Pair` and `Stack` examples in PEP 695 form are referenced by Task 11.

- [ ] **Step 1: Restructure**

New section order: (1) single Motivation — keep the misleading-`Pair`-with-`Any` and silent-`'16' + '4'` demos, delete the second and third "Any is bad" passes (the Employee-list AttributeError section keeps ONE compact appearance inside Containers); (2) the Glossary (generic type / generic type constructor / type variable / type parameter / type argument) PROMOTED here from ~L500; (3) Containers are Generics; (4) Writing Generic Classes — PEP 695-first; (5) Generic Functions and Methods — DELETE the unconstrained `add(x: T, y: T)` example and its mypy output (L729-755; chapter 05 owns that example as its motivation) and end the section with a one-line forward pointer to "Bounds, Constraints, and Defaults"; (6) Summary.

- [ ] **Step 2: Modernize all declarations**

Every taught declaration flips to PEP 695; the legacy form appears once in a labeled sidebar admonition. Core transformations:

```python
# OLD (delete everywhere as the taught form)
S = TypeVar("S")
T = TypeVar("T")
class Pair(Generic[S, T]): ...

# NEW (taught form)
class Pair[S, T]:
    def __init__(self, first: S, second: T) -> None:
        self.first = first
        self.second = second
```

```python
# OLD
T = TypeVar("T")
def append_and_return_list(item: T, items: List[T]) -> List[T]: ...
# NEW
def append_and_return_list[T](item: T, items: list[T]) -> list[T]: ...
```

`Stack(Generic[T])` → `class Stack[T]:`. Delete `from typing_extensions import reveal_type` (stdlib `typing.reveal_type`); delete all `reveal_locals()` uses (mypy-only, never standardized) — replace each with targeted `reveal_type(...)` calls; fix the latent `sys.maxsize`-without-import (L82) by importing `sys` in the cell; delete hardcoded "line 27"-style references to invisible line numbers; replace the "I came all the way just for a Moot Example" `{prf:example}` with a straight `Pair[int, str]` swap-method example showing inferred parameter types (verified).

- [ ] **Step 3: Verify snippets** (every claim, both checkers, real output).

- [ ] **Step 4: Build, commit, pause**

```bash
git add omniverse/computer_science/type_theory/04-generics.md
git commit -m "docs: rebuild generics article pep 695 first with single motivation.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

STOP for author review.

### Task 11: New article — pep-695-type-parameters.md

**Files:**
- Create: `omniverse/computer_science/type_theory/pep-695-type-parameters.md`
- Modify: `omniverse/_toc.yml` (insert after `04-generics.md` line), `intro.md` roadmap (text → link)

**Interfaces:**
- Consumes: 04's `Pair`/`Stack` PEP 695 examples (links back).
- Produces: the mechanics reference that 12 (bounds syntax) and 18 (lazy evaluation) link to.

- [ ] **Step 1: Author the chapter**

Full house-style scaffold (jupytext frontmatter, badges `Structured_Musings`, `{contents}`, setup cell). Outline with required content:

1. **From ceremony to syntax** — the exact before/after 04 established, plus what the syntax *means*: type params are declared, scoped, and lazily evaluated.
2. **The three declaration sites** — `def f[T](...)`, `class C[T]: ...`, `type Alias[T] = ...`; executed cells showing each works at runtime on 3.14.
3. **Scoping rules** — type params visible in the whole definition (signature, body, bases); shadowing behavior; a verified checker error for using a class type param in a `@staticmethod` without redeclaring.
4. **The `type` statement** — `type UserId = str`, generic aliases `type Cache[K, V] = dict[K, tuple[V, float]]`, recursive `type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]` (verified with both checkers); contrast with legacy `X: TypeAlias = ...`.
5. **Lazy evaluation** — bounds/defaults/alias values evaluate on access, not definition: executed cell with `class C[T: Undefined]: ...` failing only when the bound is introspected; introspection via `C.__type_params__` and `T.__bound__`.
6. **When legacy `TypeVar` survives** — supporting ≤3.11, dynamic TypeVar creation, explicit-variance protocols; one labeled legacy sidebar.
7. **Summary** + footnotes to PEP 695 and the typing spec's generics chapter.

- [ ] **Step 2: Verify every snippet; executed cells must run on the 3.14 kernel.**

- [ ] **Step 3: Wire TOC + intro link**

`_toc.yml`: insert `          - file: computer_science/type_theory/pep-695-type-parameters.md` immediately after the `04-generics.md` line. `intro.md`: convert the roadmap plain-text entry to a `{doc}` link.

- [ ] **Step 4: Build, commit, pause**

```bash
git add omniverse/computer_science/type_theory/pep-695-type-parameters.md omniverse/_toc.yml omniverse/computer_science/type_theory/intro.md
git commit -m "docs: add pep 695 type parameter syntax chapter.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

STOP for author review.

### Task 12: Article — 05-typevar-bound-constraints.md → "Bounds, Constraints, and Defaults"

**Files:**
- Modify: `omniverse/computer_science/type_theory/05-typevar-bound-constraints.md` (611 lines; filename unchanged, page title changes)

**Interfaces:**
- Consumes: 04's forward pointer; Task 11's syntax chapter (links).
- Produces: PEP 696 defaults coverage (no standalone chapter, per spec).

- [ ] **Step 1: Apply errata**

1. L154: "type binding happens at function call time (runtime)" → binding is solved by the static checker per call *expression*; nothing binds at runtime. Keep the substitution walkthrough (it is good) under the corrected framing.
2. L364–367: DELETE the false claim that `bound=list[int]` is an error — a concrete parameterized bound is legal; keep only the genuinely illegal TypeVar-parameterized bound (`[T: list[S]]`) with verified checker output.
3. L605 footnote: `mypy run <file>.py` → `mypy <file>.py`.
4. L384–401: examples call `compare_lengths` but define `longer` — unify on `longer`.
5. L583–592: the constraints demo named `function_with_bound` → `function_with_constraints`.
6. L313–332: replace the "I guess the checker..." speculation with the typing spec's constraint-solving reference (link) and verified behavior from both checkers.

- [ ] **Step 2: Recast syntax and absorb defaults**

All declarations to PEP 695: bounds `def longer[T: Sized](x: T, y: T) -> T:`, constraints `def add[T: (int, float, str)](x: T, y: T) -> T:`; legacy `TypeVar(bound=..., ...)` once in a labeled sidebar. Modernize the `Addable` protocol case study (annotate `self`, PEP 695 form, show the passing checker run it currently omits). ADD a new section "Defaults (PEP 696)": syntax `class Registry[T = str]: ...`, semantics (default applies when unparameterized), rules (defaults after non-defaulted params; default must satisfy the bound), one realistic example (a `Result[OkT, ErrT = Exception]` container), when NOT to use defaults (hiding ambiguity), all verified. Retitle the page "Bounds, Constraints, and Defaults". Add `{prf:remark}` styling for the formal statements (house-style gap), Summary, pyright+mypy outputs.

- [ ] **Step 3: Verify snippets; build; commit; pause**

```bash
git add omniverse/computer_science/type_theory/05-typevar-bound-constraints.md
git commit -m "docs: recast bounds and constraints in pep 695 form, absorb pep 696 defaults.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

STOP for author review.

### Task 13: Article — 06-invariance-covariance-contravariance.md (flagship rewrite)

**Files:**
- Modify: `omniverse/computer_science/type_theory/06-invariance-covariance-contravariance.md` (640 lines)

**Interfaces:**
- Consumes: 03's criterion label (`{prf:ref}` must keep resolving); 02's promotion subsection.
- Produces: closing bridge sentence to Task 14 (narrowing) and Task 16 (ReadOnly).

- [ ] **Step 1: Restructure around inference**

1. Keep the Guido `append_pi` motivation but trim the ~60-line typeshed dump to the three carrying lines (`append`, `pop`, `__iter__`) with a link to typeshed.
2. Keep the three `{prf:definition}`s verbatim (labels stable).
3. Merge the duplicated list-invariance argument (motivation + section 3) into one pass.
4. **New core section "Variance is inferred"**: with PEP 695 generics the checker infers variance from usage — `class ImmutableList[T]:` exposing only read operations is covariant automatically; demonstrate by REAL checker outputs for assigning `ImmutableList[Manager]` to `ImmutableList[Employee]` (accepted) vs the mutable version (rejected). The old `_T_co = TypeVar("_T_co", covariant=True)` + `Generic[_T_co]` moves to a labeled legacy sidebar including `infer_variance=True` as the transitional form.
5. L54: delete the PEP 484 type comment (`# type: List[int]`) → normal annotation.
6. L472–473: fix the function analogy — x² is NOT idempotent; either call it simply "non-monotonic" or cut the invariance leg of the analogy (recommended: cut; keep the covariant/monotone-increasing and contravariant/decreasing legs).
7. Fix the setup cell: import `Callable` from `collections.abc` (currently used but never imported).
8. Keep the Callable/contravariance walkthrough (Employee/Manager/CEO, both runtime demos, PEP 483 salary example) with cosmetic idiom updates only.
9. End with the bridge: mutability ⇒ invariance; read-only interfaces ⇒ covariance — "PEP 705's `ReadOnly` makes this operational for dict-shaped data (chapter upcoming)" and a pointer to narrowing next. Add Summary.

- [ ] **Step 2: Verify snippets (inference claims especially — both checkers); build; commit; pause**

```bash
git add omniverse/computer_science/type_theory/06-invariance-covariance-contravariance.md
git commit -m "docs: pivot variance chapter to pep 695 inferred variance.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

STOP for author review.

### Task 14: New article — type-narrowing.md

**Files:**
- Create: `omniverse/computer_science/type_theory/type-narrowing.md`
- Modify: `omniverse/_toc.yml` (insert after 06 line), `intro.md` roadmap link

**Interfaces:**
- Consumes: 01–03 subtype formalism (links); 06's bridge.
- Produces: narrowing reference that Tasks 15 and 17 link to.

- [ ] **Step 1: Author the chapter**

House scaffold; outline with required content:

1. **What narrowing is** — the checker refining a variable's type along control flow; executed + verified examples of built-in narrowing: `isinstance`, `is None`, `assert`, `match`, literal comparison. Note this is what 07's motivation and 08's sentinel checks have been doing implicitly.
2. **User-defined predicates: `TypeGuard` (PEP 647)** — signature contract, only-true-branch narrowing, verified example where the false branch stays wide; why that is sound (the guard may narrow to a non-subtype, e.g. `list[str]` from `list[object]`).
3. **`TypeIs` (PEP 742)** — narrows BOTH branches; requirement that the narrowed type be a *subtype* of the declared input — connect explicitly to 03's subsumption criterion via `{prf:ref}` (this is the series' differentiator: TypeIs soundness is a subtype condition).
4. **Choosing** — decision rule: `TypeIs` when the narrowed type is a subtype and the negative branch is meaningful; `TypeGuard` otherwise; table + two verified contrasting examples (checker outputs where they diverge).
5. **`assert_type` and `reveal_type`** as the tools used throughout the series to *test* narrowing claims.
6. **Summary** + footnotes (PEP 647, PEP 742, typing spec narrowing chapter).

- [ ] **Step 2: Verify; wire TOC (`- file: computer_science/type_theory/type-narrowing.md` after the 06 line) + intro link; build; commit; pause**

```bash
git add omniverse/computer_science/type_theory/type-narrowing.md omniverse/_toc.yml omniverse/computer_science/type_theory/intro.md
git commit -m "docs: add type narrowing chapter covering typeis and typeguard.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

STOP for author review.

### Task 15: Article — 07-overload.md content (medium rewrite)

**Files:**
- Modify: `omniverse/computer_science/type_theory/07-overload.md` (373 lines; renamed in Task 5)

**Interfaces:**
- Consumes: Task 14 (narrowing links); Task 5 (rename done).
- Produces: overload chapter that 17 cross-links.

- [ ] **Step 1: Reframe and fix**

1. New title "Function Overloading with `@overload`". Reframe: `typing.overload` comes from PEP 484 and is specified in the typing spec's overload chapter; PEP 3124 (Deferred, runtime generic functions) is removed from framing and references.
2. Keep the motivation sequence beat-for-beat (Union return → `reveal_type` → rejected call → narrowing workaround → overload payoff); update the narrowing workaround paragraph to link the new narrowing chapter.
3. Dispatch section: name `functools.singledispatch` / `singledispatchmethod` with one executed example; keep the compile-vs-runtime dispatch prose compressed.
4. Runtime behavior: correct "Python ignores these variants" with `typing.get_overloads()` (3.11+) — executed cell.
5. `SimpleList(Sequence[T])` → PEP 695 `class SimpleList[T](Sequence[T]):`; stale `builtins.int*` mypy outputs → REAL current outputs from both checkers.
6. Replace the "Not So Good" estimator example with the classic None-default overload pattern (verified):

```python
@overload
def fit(X: np.ndarray, y: np.ndarray) -> SupervisedResult: ...
@overload
def fit(X: np.ndarray, y: None = None) -> UnsupervisedResult: ...
def fit(X: np.ndarray, y: np.ndarray | None = None) -> SupervisedResult | UnsupervisedResult: ...
```

7. Keep the unsafe-overlap section (good) with refreshed outputs; cross-link 08 for the sentinel alternative to `None`; add `{prf:remark}` styling and Summary.

- [ ] **Step 2: Verify; build; commit; pause**

```bash
git add omniverse/computer_science/type_theory/07-overload.md
git commit -m "docs: reframe overload chapter around pep 484 and modern dispatch.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

STOP for author review.

### Task 16: New article — typeddict-readonly.md

**Files:**
- Create: `omniverse/computer_science/type_theory/typeddict-readonly.md`
- Modify: `omniverse/_toc.yml` (insert after `07-overload.md` line), `intro.md` roadmap link

**Interfaces:**
- Consumes: 01 (structural subtyping), 06 (immutability ⇒ covariance bridge).
- Produces: completes the ReadOnly payoff promised by 13.

- [ ] **Step 1: Verify current status, then author**

FIRST: web-check PEP 728 (closed TypedDicts) status on peps.python.org — include a section only if Accepted/Final; otherwise one forward-looking footnote.

Outline: (1) TypedDict as *structural typing of data* — dict-shaped values with a checked schema; callback to ch. 1's nominal/structural distinction; (2) `Required`/`NotRequired`/`total=False` with verified examples; (3) structural assignability between TypedDicts (which widening/narrowing is allowed and why — tie to the subsumption criterion); (4) **`ReadOnly` (PEP 705)** — read-only keys, and the payoff: a TypedDict with all-ReadOnly keys behaves covariantly where mutable dicts cannot — demonstrate with verified checker outputs, closing the loop 06 opened; note it is checker-enforced only, not runtime; (5) when to choose TypedDict vs dataclass vs Pydantic (one honest paragraph, JSON-boundary framing); (6) Summary + footnotes (PEP 589, 655, 705, spec chapter).

- [ ] **Step 2: Verify; wire TOC + intro link; build; commit; pause**

```bash
git add omniverse/computer_science/type_theory/typeddict-readonly.md omniverse/_toc.yml omniverse/computer_science/type_theory/intro.md
git commit -m "docs: add typeddict and readonly chapter.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

STOP for author review.

### Task 17: Article — 08-pep-661-sentinel-values.md (medium rewrite)

**Files:**
- Modify: `omniverse/computer_science/type_theory/08-pep-661-sentinel-values.md` (282 lines)

**Interfaces:**
- Consumes: narrowing chapter (Task 14), overload chapter (Task 15).
- Produces: final pre-capstone chapter.

- [ ] **Step 1: Verify current status, then rewrite**

FIRST: web-check PEP 661 status and `typing_extensions.Sentinel` availability/checker support (knowledge dates to mid-2025) — the chapter's framing depends on it.

1. Fix mechanical errata: duplicate `from __future__ import annotations` (L35/L37 — and per the capstone, remove entirely if repo is 3.14); `from typing_extensions import override` → `from typing import override`; `Type[_NotGiven]` → `type[_NotGiven]`; `super(_NotGiven, cls)` → `super()`; "NOTGIVEN" (L271) → `NOT_GIVEN`.
2. Keep the OpenAI `NotGiven` case study and the timeout walkthrough (good motivation), but add the missing punchline the article currently works around silently: **`is NOT_GIVEN` on a hand-rolled singleton does not narrow** — show the verified checker error, explain via the narrowing chapter (link), THEN present the fixes:
   - the single-member `Enum` sentinel (`class _NotGivenType(Enum): NOT_GIVEN = auto()`) which checkers DO narrow via `is` (verified both checkers);
   - `typing_extensions.Sentinel` / PEP 661 state as verified in the web check.
3. "Use Case 1" promises more use cases that never arrive → either add the config-`Missing` walkthrough as "Use Case 2" (the material is already in the article) or renumber to a single unnumbered section.
4. Move the `_Missing` explanation bullets out of the docstring into prose; cross-link 07's `None`-default overload as the alternative design; Summary.

- [ ] **Step 2: Verify; build; commit; pause**

```bash
git add omniverse/computer_science/type_theory/08-pep-661-sentinel-values.md
git commit -m "docs: modernize sentinel chapter with enum idiom and narrowing story.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

STOP for author review.

### Task 18: New article — annotations-at-runtime.md (capstone)

**Files:**
- Create: `omniverse/computer_science/type_theory/annotations-at-runtime.md`
- Modify: `omniverse/_toc.yml` (insert after the 08 line), `intro.md` roadmap link
- Modify: setup cells of `01,02,03,04,05,06,07,08` + the three new chapters (future-import removal sweep)

**Interfaces:**
- Consumes: every prior chapter's setup cell.
- Produces: series complete; pilot acceptance check.

- [ ] **Step 1: Author the capstone**

Outline: (1) why annotations were strings — the old eager-evaluation problem and the `from __future__ import annotations` era (every chapter in this series carried it); (2) **PEP 649/749**: deferred evaluation by default in 3.14 — annotations are compiled to a lazy `__annotate__` function; forward references without quotes; executed cell proving a forward reference works with no future import on the 3.14 kernel; (3) **`annotationlib`** — `get_annotations()` with `Format.VALUE` / `FORWARDREF` / `STRING`, one executed example each, when each format is right; why reading `__annotations__` directly is now a trap; (4) `typing.get_type_hints` vs `annotationlib` guidance; (5) what stays: `if TYPE_CHECKING:` for import-cost/cycles; (6) **the series changelog** — this rewrite removed the future import from every chapter's setup cell (the sweep below); (7) Summary + footnotes (PEP 649, PEP 749, annotationlib docs).

If Task 3 landed on 3.13 instead of 3.14: demos in (2)–(3) become static blocks with pasted 3.14 output (obtained via `uv run --python 3.14 python ...`), per spec §7.3.

- [ ] **Step 2: The sweep (amended 2026-07-11: verification, not removal)**

Per-chapter removal is now the policy — each rewrite task drops `from __future__ import annotations` from its own setup cell as it lands. This step VERIFIES none remain anywhere in the series (`grep -rn "from __future__" omniverse/computer_science/type_theory/` → no output) and the capstone's changelog section documents the per-chapter removals. Rebuild; every executed cell must still pass.

- [ ] **Step 3: Verify; wire TOC + intro link; build**

Pilot acceptance (spec §8): all 13 spine entries render in TOC order; `07-pep-3124-overloading.html` still redirects; `make ci` green.

- [ ] **Step 4: Commit and close out**

```bash
git add omniverse/computer_science/type_theory/ omniverse/_toc.yml
git commit -m "docs: add annotations-at-runtime capstone and remove future imports across series.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

STOP: final author review of the whole series; then update the spec's status line to "implemented".
