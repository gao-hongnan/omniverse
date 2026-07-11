# Omniverse content-modernization pilot — design

Date: 2026-07-11
Status: approved (pending final spec review)
Scope: Phase 0 (content rescue) + Phase 1 (type-theory series modernization)

## 1. Context and motivation

The `omniverse/` Jupyter Book (~272 files, ~529K words, published at
gaohongnan.com) was audited on 2026-07-10/11 across content health, platform,
and structure. The author chose a contained pilot over a book-wide campaign:
modernize the highest-traffic content first, produce a reusable standard,
then reassess.

Why the type-theory series: site analytics identify it as the most-read
section. A deep read of all 9 files (~4,325 lines) found:

- **Idiom age**: uniformly ~2023 / Python 3.8–3.10 era — capital
  `List/Dict/Union/Optional`, `TypeVar(...)` + `Generic[T]`,
  `covariant=True` flags, `typing_extensions` imports now in stdlib, one
  PEP 484 type comment, mypy-only framing. Zero mentions of PEP 695/696/
  705/742/649, pyright, or the maintained typing spec.
- **Factual errors** live on the most-read pages (itemized per chapter in
  §5.2 below), including a sign error in the series' central formal
  criterion and a chapter misattributed to an unrelated PEP.
- **The differentiator is real**: the type-theory formalism layer
  (subsumption criterion, variance definitions in type-constructor
  notation) is more rigorous than mainstream typing tutorials and must be
  preserved.

## 2. Decisions log (2026-07-10/11)

| Decision | Choice |
| --- | --- |
| Platform | Stay on Jupyter Book 1 (no MyST/JB2 migration now); all current URLs remain canonical |
| Book restructure | Deferred entirely (audit sketches recorded in §9 backlog) |
| Program shape | Approach 3: rescue + typing-series pilot only, then reassess |
| Series scope | Modernize existing 8 articles + extend with 4 new chapters |
| Chapters 02/03 overlap | Keep both files, de-duplicate content (no merge) |
| Chapter 07 misattribution | Rename `07-pep-3124-overloading.md` → `07-overload.md`; old URL gets a redirect |
| Snippet verification | Authoring-time only (scratch harness; no new repo tooling) |
| Python version | Spike `requires-python >=3.14`; fall back to `>=3.13` if core deps resist |

## 3. Scope and non-goals

**In scope**

- Phase 0: protect at-risk content; fix live embarrassments (§4).
- Phase 1: the 13-chapter target spine for
  `omniverse/computer_science/type_theory/` (§5), the modernization
  standard (§6), and the Python-version spike (§7).

**Non-goals (deferred, see §9)**

- JB2/MyST migration; theme work.
- Book-wide restructure or TOC re-parting.
- Modernizing any other chapter (GPT series, playbook, DSA, …).
- Rescuing the commented-out Docker/REST-API sections.
- CI changes beyond what the Python bump forces.

## 4. Phase 0 — rescue punch list

1. **Commit `omniverse/software_engineering/python/memory_internals/`**
   (untracked; ~8.4K lines) as its own commit, isolated from unrelated
   working-tree changes.
2. **Wire into `_toc.yml`**: `05_ctypes_inspection.md`,
   `06_optimization_techniques.md` (TOC currently stops at 04).
3. **Fix broken link** at `06_optimization_techniques.md:2199` pointing to
   nonexistent `07_advanced_topics.md` (remove or convert to a
   non-linking "planned" note).
4. **Fix `**TODO**`-bodied blocks live in the TOC**:
   `probability_theory/03_discrete_random_variables/0303_probability_mass_function.md:109`
   (a `{prf:proof}` whose body is `**TODO**`) and
   `binomial/0309_binomial_distribution_concept.md:127`. Write the short
   proof or convert to a labeled exercise.
5. **`_config.yml:10`** `copyright: "2024"` → `"2026"`.

Acceptance: `jupyter-book build` passes; memory_internals visible in the
published sidebar through 06; no `**TODO**` proof bodies remain in TOC'd
probability pages.

## 5. Phase 1 — type-theory series

### 5.1 Target spine (TOC order)

Existing files keep their names (= URLs) except 07 (renamed by decision).
New chapters get unnumbered kebab slugs — numeric prefixes would lie about
TOC order, which `_toc.yml` alone controls. Sidebar shows titles, not
filenames.

| # | File | Status | Treatment |
| --- | --- | --- | --- |
| 0 | `intro.md` | light | Add per-chapter roadmap + badge block (house style); replace "skim PEP 483 first" with the maintained typing spec (typing.python.org/spec); keep the type-theory lineage framing |
| 1 | `01-subtypes.md` | light | Fix errata; retitle mislabeled "Pros and Cons" section (it is an LSP-violation discussion); compress inclusive/coercive ∀/∃ math; prune dead imports |
| 2 | `02-type-safety.md` | heavy | Fix errata (int/float, truncation story, classes-vs-instances cell); cut triple-restated lede; de-dupe LSP with 03; absorb dynamic-vs-static (Siek) material properly; add summary |
| 3 | `03-subsumption.md` | medium | Fix criterion sign error + typos + PoliceDog bug; antisymmetry → preorder framing; drop one of two near-identical worked examples; keep `{prf:criterion}` wording and label (`06` cross-references it) |
| 4 | `04-generics.md` | heavy | PEP 695-first rebuild (`class Pair[S, T]:`); single motivation; glossary promoted forward; delete "Moot Example" apology; drop `reveal_locals`/`typing_extensions.reveal_type`; end on forward pointer to 05 (no duplicated `add` example); add summary |
| 5 | `pep-695-type-parameters.md` | **new** | Owns the mechanics 04 only uses: scoping rules, `type` alias statement, lazy bound evaluation, `__type_params__`, when legacy `TypeVar` is still required |
| 6 | `05-typevar-bound-constraints.md` | medium-heavy | Recast as PEP 695 `[T: Sized]` / `[T: (int, float, str)]`; fix errata; replace guessed solver behavior with typing-spec citation; **absorb PEP 696 defaults**; retitle "Bounds, Constraints, and Defaults"; add summary |
| 7 | `06-invariance-covariance-contravariance.md` | heavy | Pivot to **inferred variance** (PEP 695) with `covariant=True` as legacy sidebar (incl. `infer_variance=True` transitional form); trim 60-line typeshed dump to the 3 carrying lines; fix "idempotent" mislabel; kill the type comment; keep the three `{prf:definition}`s and the Callable/contravariance walkthrough intact; close on "read-only ⇒ covariant" as the bridge to TypedDict/ReadOnly |
| 8 | `type-narrowing.md` | **new** | `TypeIs` (PEP 742) vs `TypeGuard`: TypeIs demands the narrowed type be a subtype of the input and narrows both branches — connect to ch. 1–3 formalism; `isinstance`/`assert` narrowing; motivates 07's overloads and 08's sentinel-narrowing problem |
| 9 | `07-overload.md` (renamed) | medium | Reframe around PEP 484 `@overload` + the typing spec's overload chapter (PEP 3124 is Deferred and unrelated — remove as reference); name `functools.singledispatch`/`singledispatchmethod`; note `typing.get_overloads()` (3.11+); replace the "not so good" estimator example (e.g. None-default overload pattern); keep the motivation sequence beat-for-beat; cross-link 08 |
| 10 | `typeddict-readonly.md` | **new** | TypedDict as structural subtyping of data (callback to ch. 1); `ReadOnly` (PEP 705) as the operational payoff of "immutability ⇒ covariance"; `Required`/`NotRequired`; note PEP 728 closed TypedDicts if landed (verify) |
| 11 | `08-pep-661-sentinel-values.md` | medium | Add the single-member-Enum sentinel idiom (checkers narrow it via `is`); add `typing_extensions.Sentinel` + current PEP 661 status (**verify at writing time**); explain why `is NOT_GIVEN` on a hand-rolled singleton does not narrow (the buried `assert isinstance`); fix duplicate future-import, `typing.override` import; cross-link 07 |
| 12 | `annotations-at-runtime.md` | **new** | Capstone: PEP 649/749 deferred evaluation; `annotationlib` and its `VALUE`/`FORWARDREF`/`STRING` formats; why `from __future__ import annotations` becomes deletable in 3.14-only code; doubles as changelog for the series-wide setup-cell cleanup |

### 5.2 Errata register (must-fix false claims)

- `01-subtypes.md:113` — "mypy's `Protocol`": it is `typing.Protocol`
  (PEP 544). `:190` "Robit" typo. Dead imports in setup cell;
  `typing.Sized`/`typing.Type` deprecated aliases.
- `02-type-safety.md:236-243` — "int <: float" and "old will truncate to 3
  silently": int is **not** a subtype of float (numeric-tower *promotion*;
  `isinstance(3.0, int)` is False), and assignment rebinds — nothing
  truncates. `:225-229` — example passes classes where instances are
  meant.
- `03-subsumption.md:301/359` — T₁/T₂ assignment contradicts the stated
  criterion direction. `:207-212` — antisymmetry overstated (subtyping is
  a preorder; structural systems have equivalent-but-distinct types).
  `:191/397` — "Transivity" typos. `:417-419` — `PoliceDog.search`
  annotated `-> str` but returns None.
- `04-generics.md:82-83` — `sys.maxsize` used, `sys` never imported
  (latent). Hardcoded mypy output line numbers that match nothing visible.
- `05-typevar-bound-constraints.md:154` — "type binding happens at function
  call time (runtime)": binding is solved statically per call expression.
  `:364-367` — claims `bound=List[int]` is an error; **false** (only
  TypeVar-parameterized bounds are illegal). `:605` — `mypy run file.py`
  is not a command. `:384-401` — defines `longer`, calls
  `compare_lengths`. `:583-592` — constraints demo named
  `function_with_bound`.
- `06-invariance...md:54` — PEP 484 type comment. `:379` —
  `covariant=True` taught as *the* mechanism (now inference).
  `:472-473` — x² called "idempotent" (it is not; f(f(x)) ≠ f(x)).
  `Callable` used but never imported (survives only via future import).
- `07-…overloading.md` — entire PEP 3124 framing wrong (Deferred, runtime
  generic functions; `@overload` is PEP 484). `:149` — mypy `builtins.int*`
  output format is years gone. Estimator example forces X and y to one
  TypeVar.
- `08-…sentinel-values.md:35/37` — duplicate future import. `:41` —
  `override` from `typing_extensions` (stdlib since 3.12). `:271` —
  "NOTGIVEN" spelling drift. PEP 661 status framing stale.

### 5.3 Cross-cutting rules

- One canonical `Animal/Dog/Cat` fixture module-pattern, introduced once in
  ch. 1 and reused (today it is redefined five times; 06 keeps its
  Employee/Manager/CEO family for variance).
- The unconstrained `add(x: T, y: T)` example lives only in 05; 04 ends on
  a one-line forward pointer.
- Every chapter gains a closing summary section.
- `{prf:*}` house style extended to 05/07/08 (currently bare).
- Checker outputs shown as **pyright and mypy** wherever they differ;
  "compile time" phrasing → "static-analysis time".
- All existing `{prf:ref}` labels stay stable (06 → 03's criterion label).
- House conventions preserved: jupytext MyST frontmatter, hidden
  `remove-cell` setup cell, badge block, `{contents}`, executed
  `{code-cell}` for runtime truths vs static blocks for checker-rejected
  code, named footnotes.

### 5.4 URL and redirect policy

- Files `01`–`06`, `08`, `intro` keep names — zero URL breakage.
- `07-pep-3124-overloading.md` → `07-overload.md`. Old URL preserved via
  the `sphinx-reredirects` extension (added to the `docs` dependency
  group, configured in `_config.yml` `sphinx.extra_extensions` +
  `redirects` mapping). One redirect entry; mechanism generalizes if the
  rollout later needs more.
- New chapters are new URLs; no constraint.

## 6. The modernization standard (reusable checklist)

Applied per article; this is the artifact that survives the pilot.

**Code idiom**

- `List/Dict/Tuple/Set/Type/Optional/Union` → builtins + `X | Y`/`| None`.
- `TypeVar(...)`/`Generic[T]`/`covariant=` → PEP 695 syntax; legacy forms
  appear only in explicitly-labeled "legacy" sidebars.
- `typing_extensions` → stdlib where landed (`reveal_type` 3.11,
  `override` 3.12, `Self` 3.11, `TypeIs` 3.13).
- ABC generics from `collections.abc`, not deprecated `typing` aliases.
- Type aliases via the `type` statement; `Self` for fluent/classmethod
  returns; `@override` on overrides.
- Setup cells import only what the article uses; no
  `from __future__ import annotations` in 3.14-only executed code (per
  the capstone chapter; keep if the repo lands on 3.13).

**Factual verification**

- Every "checker says X" claim reproduced against current pyright and mypy
  before it is written; outputs pasted, not remembered.
- Every PEP status claim checked against peps.python.org at writing time;
  prefer the maintained typing spec over historical PEPs as the cited
  authority.

**Flow**

- One motivation per article; lede in the first screenful; no apologizing
  for weak examples — replace them; closing summary required; formal
  content uses `{prf:*}`; internal cross-links where topics touch.

**House style** — badges, `{contents}`, executed-vs-static convention,
footnotes, label stability (§5.3).

## 7. Verification and Python-version mechanics

### 7.1 Why the bump is safe to attempt (probe evidence, 2026-07-11)

The version dependence lives in **four `pyproject.toml` lines, not in the
code**:

- Code pre-cleared by grep: zero imports of stdlib modules removed in
  3.12/3.13 (PEP 594 sweep) and zero numpy-1.x-only APIs
  (`np.float_`, `np.NaN`, …) across `omnivault/`, `omnixamples/`,
  `tests/`.
- `torchtext` is imported by **nothing** (no .py file, no notebook) and is
  archived upstream with wheels capped at Python 3.12 → delete the
  dependency.
- Era caps force uninstallable versions on 3.13+: `numpy<2.0.0` (1.26.4
  has no 3.13/3.14 wheels), `scikit-learn==1.5.0`, `matplotlib<3.9.1`.
- Throwaway-worktree probe: with those four edits, the full 207-package
  set **resolves under `requires-python >=3.14`** → numpy 2.5.1,
  scikit-learn 1.9.0, matplotlib 3.11.0, pandas 2.3.2,
  reproducibility 8.0.0.
- Wheel reality check (a lock can resolve versions that cannot install):
  torch 2.8.0 — picked from stale local cache — ships **0** cp314 wheels;
  torch 2.9.0 and 2.13.0 (current) ship them. The spike therefore sets
  `torch>=2.9` and locks with `--refresh`.

### 7.2 Spike procedure and failure ladder

1. **Edits**: `requires-python = ">=3.14"`; delete `torchtext`;
   `numpy>=2.1`; `scikit-learn>=1.6`; `matplotlib>=3.10`; `torch>=2.9`.
2. **Gate 1 — resolve**: `uv lock --refresh`. A failure names the
   offending pin → bump or remove it. Only if a genuinely needed
   dependency has no 3.14-compatible release: fall back to `">=3.13"`
   (every version above already satisfies 3.13).
3. **Gate 2 — install**: `uv sync --all-groups`. An sdist-build failure
   means the chosen version predates the interpreter → raise that
   dependency's floor and re-lock.
4. **Gate 3 — behave**: `make all` (mypy 1.13.0 upgraded for the 3.13/3.14
   target and full PEP 695/696; `pyrightconfig.json` `pythonVersion`
   bumped to match) + `jupyter-book build` (executes non-excluded
   notebooks — numpy 2.x behavioral differences surface here; fixes are
   per-page and localized).
5. **Containment valve**: a notebook that resists is added to
   `_config.yml` `execute:exclude_patterns` (an existing, already-used
   pattern — its cached outputs still render) instead of blocking the
   bump.
6. **Decoupling valve (last resort)**: the typing series can execute on
   its own registered 3.14 ipykernel — myst-nb honors each page's
   jupytext `kernelspec` — even if the repo-wide bump stalls entirely.
   The pilot never blocks on the package upgrade.

### 7.3 Authoring verification

- Executed `{code-cell}`s must run on the book's kernel. If the repo lands
  on 3.13: PEP 695/696 cells execute (3.12/3.13 features); deferred-
  annotation (3.14-only) demos use static blocks + pasted output.
- Snippet verification is authoring-time: scratch harness (uv-run pyright/
  mypy on extracted snippets) during writing; no repo tooling committed.

## 8. Process and acceptance

- Phase 0 lands first as isolated commits.
- Phase 1 proceeds one article per iteration, spine order (§5.1): draft →
  author reviews rendered page (`jupyter-book build`) → commit. Each
  article independently shippable.
- Article acceptance: errata for it (§5.2) resolved; standard (§6)
  checklist satisfied; page builds and renders; checker outputs verified
  against the landed Python version.
- Pilot acceptance: all 13 spine entries (intro + 12 chapters) live in the
  TOC in spine order; 07 redirect works on the published site;
  modernization standard usable as-is for a future rollout.

## 9. Deferred backlog (recorded, not scheduled)

- **Orphan rescue beyond Phase 0**: `influential/generative_pretrained_transformer/train_phase.md`
  (1,298 ln) + `unit_integration_tests.ipynb`; binary-search problems
  `74`/`153` + `template.md`; `big-O.md`; `consumer_producer.md`;
  `reading_list.md` (stale, orphaned); 5 zero/one-line DSA files (delete
  or write); jupytext twin `dsa/queue/concept.md`.
- **Hidden sections**: containerization (~2,742 ln incl. a 1,037-line
  `temp.md`) and RESTful serving (~2,014 ln) — rescue-and-modernize or
  delete; decide when reached.
- **Structure**: audit produced three re-part sketches (learner's arc /
  audience shelves + genre tags / minimal relabel) — all pure `_toc.yml`
  rewrites; revisit after the pilot.
- **Platform**: JB2/MyST migration evidence is strong (the repo's own Aug
  2025 mystmd build in `omniverse/_build/site/` converted all 592
  `{prf:*}`, 42 tabs, 163 citations, 59 TeX macros cleanly; JB2 GA
  Nov 2025, v2.1.6 Jul 2026). Costs when revisited: all URLs change
  (redirect stubs), custom footer template fork-or-drop, one autodoc
  page, Node in CI. Also: the stale
  `docker/documentation/jupyterbook.Dockerfile` (pip/requirements drift,
  Python 3.13 vs project pin) should be retired or regenerated.
- **Hygiene**: ~80 timestamped checkpoint dirs + raw MNIST inside book
  source; `010_` mis-numbering in ML-lifecycle; `citations.md` vs
  `references.bib` duplication; missing probability ch. 07 (tracks Chan's
  textbook — decide to renumber or note).

## 10. Risks and open questions

- **PEP 661 / `typing_extensions.Sentinel` status** — verify current state
  when writing ch. 11 (knowledge here dates to mid-2025).
- **PEP 728 (closed TypedDicts)** — verify landed status for ch. 10.
- **numpy 2.x behavior** — code greps clean, but executed notebooks may
  hit behavioral differences (dtype promotion, reprs); caught at Gate 3,
  fixed per page or contained via `exclude_patterns`.
- **Stale uv cache** — the probe's resolver chose torch 2.8.0 while PyPI
  is at 2.13.0; always lock with `--refresh` during the spike.
- **mypy upgrade ripple** — a newer mypy may flag existing `omnivault/`
  code; scope any fixes to what `make all` requires, not a package-wide
  retype.
- **Redirect extension** — `sphinx-reredirects` compatibility with the
  pinned Sphinx 7.4.7 to be confirmed at implementation (fallback: a
  post-build copy step in CI).
