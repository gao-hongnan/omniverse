# Glossary Adherence for LLM Document Translation

## Executive Summary

A robust glossary-adherent translation system should not treat the glossary $\mathcal{G}$ as mere prompt text. The most reliable design is a **hybrid constrained pipeline**: detect terminology obligations before translation, retrieve only the relevant entries for each segment, apply selective protection or placeholder transforms for high-risk terms, generate multiple candidates with document context, verify deterministically and semantically, repair only the violating spans, and escalate unresolved ambiguity to a human reviewer. In other words, glossary adherence should be externalized from the LLM and enforced by surrounding infrastructure, with the LLM $\mathcal{T}_\theta$ serving as a powerful but fallible conditional generator inside a larger control loop. This conclusion is supported by two bodies of evidence: first, long-context LLMs do not reliably use all provided context, especially information buried in the middle of long prompts; second, the MT literature on lexical constraints exists precisely because unconstrained generation does not naturally satisfy hard terminology requirements. citeturn6search4turn0search4turn0search1turn1search0

The practical implication is stark. If a translation may not ship with incorrect legal, medical, brand, or do-not-translate terminology, then **prompt-only conditioning is not a production guarantee**. It can improve performance, especially on short segments and small curated term subsets, but it remains a soft bias. Modern professional localization stacks already separate three functions that should remain distinct in an LLM system as well: **suggestion**, **translation-time control**, and **post-hoc verification**. CAT/TMS tools surface term suggestions and run QA; MT APIs apply glossaries with provider-specific rules and limitations; standards such as TBX, TMX, and XLIFF exist because terminology, translation memory, and bilingual document structure are operationally different assets. citeturn20search3turn20search7turn21search1turn4search9turn5search0turn5search3turn5search2turn3search1turn12search2turn11search2turn13search1turn2search0turn19search1turn2search2

The best architecture for your setting is therefore a **planner–translator–verifier** system. The planner computes glossary obligations $\mathcal{O}_i$ and document memory $\mathcal{M}_{doc}$ before generation. The translator uses structured, minimal, high-salience constraints rather than the full glossary. The verifier $\mathcal{Q}$ combines deterministic string and morphology checks, bilingual alignment, and a semantic adequacy guard such as COMET or xCOMET. Repair is permitted only when it decreases terminology loss without causing unacceptable semantic drift. For truly hard terminology classes, the system should reject and escalate rather than silently “fixing” the translation in ways that may damage meaning. citeturn10search0turn14search1turn15search8turn15search12turn24search1turn24academia21

## Formal Problem Definition

Let the source document be $\mathcal{D}$, segmented into $\mathcal{S}=\{s_1,\dots,s_n\}$. Let the glossary be $\mathcal{G}=\{g_1,\dots,g_m\}$, where each entry is

\[
g_i = (x_i, y_i, \ell_s, \ell_t, d_i, c_i, p_i, f_i).
\]

Here $x_i$ is the source term, $y_i$ the approved target term, $(\ell_s,\ell_t)$ the language pair, $d_i$ the domain, $c_i$ contextual or disambiguation metadata, $p_i$ a priority weight, and $f_i$ a flag bundle such as do-not-translate, case-sensitive, inflectable, forbidden-targets, or enforcement level. Rich term metadata is not incidental: professional termbases and interchange standards explicitly support concept-level entries, language-specific terms, context, definitions, and other metadata because raw source–target pairs are insufficient for high-quality terminology control. citeturn4search5turn21search8turn5search13turn2search0turn2search2

A mathematically useful notion is not “the glossary appears in the prompt,” but rather an **obligation set**. Define a source-side matcher and applicability function:

\[
m(g_i,s_j)\in\{0,1\}, \qquad a(g_i,s_j)\in[0,1].
\]

Then the set of matched glossary obligations is

\[
\mathcal{M}(\mathcal{D},\mathcal{G})
=
\{(i,j): m(g_i,s_j)=1 \land a(g_i,s_j)\ge \tau_a\}.
\]

This separates **surface matching** from **contextual applicability**. The same source string may map to multiple glossary entries across domains or senses; therefore a glossary obligation exists only when the contextual posterior is strong enough. This framing aligns with both terminology-management practice, where entries carry definitions, domain labels, and examples, and research on document-level translation consistency, where repeated terms must often be normalized only after discourse-level disambiguation. citeturn5search13turn21search9turn8search0turn18view1

Let $\hat{\mathcal{Y}}=\{\hat y_1,\dots,\hat y_n\}$ be the produced translation, potentially obtained by

\[
\hat{\mathcal{Y}} = \arg\max_{\mathcal{Y}} P_\theta(\mathcal{Y}\mid \mathcal{D},\mathcal{G},\mathcal{C}),
\]

but with the crucial caveat that the real production system constrains $\mathcal{T}_\theta$ through preprocessing, retrieval, verification, and repair. For each matched obligation $(i,j)$, the verifier produces an aligned target realization set

\[
\mathcal{A}_{ij} = \operatorname{Align}(x_i, s_j, \hat y_j).
\]

Define the allowed realization set

\[
\operatorname{Allow}(g_i)=
\begin{cases}
\{x_i\}, & \text{if } f_i.\texttt{do\_not\_translate}=1,\\[4pt]
\operatorname{Infl}(y_i), & \text{if } f_i.\texttt{morphology\_policy}=\texttt{inflectable},\\[4pt]
\{y_i\}, & \text{otherwise,}
\end{cases}
\]

where $\operatorname{Infl}(y_i)$ is the set of allowed surface forms licensed by the morphology policy. This is essential because real systems differ sharply here: DeepL documents grammar-aware glossary flexion, while several other systems behave closer to exact or near-exact matching, and recent MT work shows that morphology and agreement are a major failure mode for lexically constrained translation. citeturn12search1turn12search2turn11search2turn1search7turn1search2turn1search10

The core adherence variants can then be defined as follows.

\[
A^{\text{exact}}_{ij} = \mathbf{1}\!\left[\exists z\in\mathcal{A}_{ij}: z=y_i\right]
\]

**Exact adherence** requires the canonical target form itself.

\[
A^{\text{soft}}_{ij}
=
\max_{z\in\mathcal{A}_{ij},\, v\in\operatorname{Allow}(g_i)}
\operatorname{sim}(z,v)
\]

**Soft adherence** allows approved variants or near-matches up to a validated similarity function.

\[
A^{\text{ctx}}_{ij} = A^{\text{soft}}_{ij}\cdot a(g_i,s_j)
\]

**Contextual adherence** discounts entries that match lexically but do not strongly apply in the local context.

\[
A^{\text{part}}_{ij}
=
\max_{z\in\mathcal{A}_{ij}}
\frac{|\operatorname{tok}(z)\cap \operatorname{tok}(y_i)|}{|\operatorname{tok}(y_i)|}
\]

**Partial adherence** is most useful for multiword terms and partial phrase realization. Multiword terminology extraction and bilingual term alignment are both nontrivial enough to have their own literature, which is one reason naive substring matching often misses obligations in production. citeturn22search0turn22search5turn22search19

Now define violation indicators:

\[
V^{\text{forbid}}_{ij}
=
\mathbf{1}\!\left[
\exists z\in\mathcal{A}_{ij},\, u\in f_i.\texttt{forbidden\_translations}: \operatorname{canon}(z)=\operatorname{canon}(u)
\right]
\]

for a **forbidden-term violation**;

\[
V^{\text{dnt}}_{ij}
=
\mathbf{1}\!\left[
f_i.\texttt{do\_not\_translate}=1 \land \forall z\in\mathcal{A}_{ij}: z\neq x_i
\right]
\]

for a **do-not-translate violation**;

\[
V^{\text{morph}}_{ij}
=
\mathbf{1}\!\left[
\exists z\in\mathcal{A}_{ij}:
\operatorname{lemma}(z)=\operatorname{lemma}(y_i)\land z\notin \operatorname{Allow}(g_i)
\right]
\]

for a **morphology or case violation**. This explicitly distinguishes “the right lemma in the wrong form” from “the wrong term entirely,” which matters in morphologically rich target languages. citeturn1search7turn1search2turn1search10

For **document-level consistency**, define equivalence classes $K_u\subseteq \mathcal{M}$ grouping repeated obligations that share source form, approved sense, domain, and consistency policy. Then

\[
V^{\text{cons}}(K_u)
=
\mathbf{1}\!\left[
\left|
\left\{
\operatorname{canon}\big(\operatorname{choose}(\mathcal{A}_{ij})\big):(i,j)\in K_u
\right\}
\right| > 1
\right].
\]

This is the formalization of the long-known “one translation per discourse” principle used in translation-consistency research. citeturn8search0turn18view0

For **ambiguity-induced violations**, let $q(g\mid s_j)$ be the disambiguation posterior over competing entries whose source side could match the local source span. Let $g^*=\arg\max_g q(g\mid s_j)$ and let $\tilde g$ be the entry actually realized in $\hat y_j$. Then

\[
V^{\text{amb}}_{ij}
=
\mathbf{1}\!\left[
\tilde g\neq g^* \land H\!\left(q(\cdot\mid s_j)\right) < \tau_H
\right].
\]

When entropy is high, the correct operational state is often not “silent violation” but **escalation**. Professional termbases therefore preserve context, domain, and examples precisely so that ambiguous strings can be adjudicated rather than over-automated. citeturn4search5turn5search13turn21search8

The user-specified glossary adherence rate can be generalized to a weighted form:

\[
\operatorname{GAR}
=
1-\frac{\sum_{(i,j)\in \mathcal{M}} w_{ij}V_{ij}}
{\sum_{(i,j)\in \mathcal{M}} w_{ij}},
\]

with $w_{ij}=w(p_i,d_i,f_i)$ and $V_{ij}$ an aggregate of hard and soft violations. Useful companion metrics are:

\[
\operatorname{SourceDetectionRecall}
=
\frac{|\widehat{\mathcal{M}}\cap \mathcal{M}^{*}|}{|\mathcal{M}^{*}|},
\]

\[
\operatorname{TargetRealizationAccuracy}
=
\frac{\sum_{(i,j)\in \widehat{\mathcal{M}}}\mathbf{1}[A^{\text{exact}}_{ij}=1\land V^{\text{forbid}}_{ij}=0]}
{|\widehat{\mathcal{M}}|},
\]

\[
\operatorname{ForbiddenFPR}_{\mathcal{Q}}
=
\frac{\#\{\text{false forbidden-term alerts by verifier } \mathcal{Q}\}}
{\#\{\text{human-labeled non-violations}\}},
\]

\[
\operatorname{GDS}
=
\mathbb{E}_{j}\big[F(y^{\text{base}}_j)-F(\hat y_j)\big],
\]

where $F$ is a fluency estimate, and

\[
\operatorname{SPS}
=
\mathbb{E}_{j}\big[\operatorname{COMET}(s_j,\hat y_j,r_j)\big]
\]

or a reference-free semantic metric when references are unavailable. COMET and xCOMET are especially relevant because they correlate well with human MT judgments and xCOMET additionally surfaces error spans in an MQM-like typology. citeturn15search8turn15search12turn15search1

## Failure Modes of Prompt-Only Glossary Adherence

A prompt-only system treats translation as something like $\mathcal{T}_\theta(\mathcal{D},\mathcal{G})$ where the glossary is simply pasted into the prompt. This is attractive because it is easy, but it fails for structural reasons. Recent long-context research shows that models are often “lost in the middle”: they use information more reliably at the beginning and the end of long contexts than in the middle. A long glossary appended to a long document therefore suffers from exactly the regime in which relevant constraints become harder to retrieve internally. This is not merely a packaging problem; it is a known limitation of long-context utilization. citeturn6search4turn6search8

Prompt-only adherence also breaks across segmentation and chunking. Many translation workflows still process documents as sentence- or chunk-level units, and document-level translation research continues to show that sentence-level translation leads to lexical inconsistency and degraded discourse modeling. In the LLM document-translation literature, context-aware prompting can improve terminology consistency, but the improvement is incremental, not a formal guarantee; repeated terms can still diverge across the document, especially when segments are processed independently or far apart. citeturn17view0turn18view1turn8search0turn23search8

A third failure mode is **fluency-over-constraint behavior**. Unconstrained autoregressive generation prefers target sequences that are locally probable and globally fluent. This is why the lexically constrained decoding literature introduced grid beam search, dynamic beam allocation, constrained beam search, and related methods in the first place: absent such machinery, the model is not guaranteed to include required words or phrases. Training-based soft methods can reduce the problem, but they still exist because prompt instructions alone are not equivalent to hard satisfiability. citeturn0search4turn0search1turn14search3turn1search0

A fourth class of failures comes from morphology, casing, and agreement. In production glossaries, the desired term is often given as a lemma or canonical surface form, but the translated sentence requires a different inflected form. DeepL explicitly documents grammar-aware glossary adaptation, whereas Amazon Translate documents exact-match, case-sensitive usage for custom terminology, and Google Cloud documents case-sensitivity and glossary stopword behavior. In research MT, agreement errors are particularly prominent: one English→Czech constrained model analysis found that 46% of errors were agreement-related. Hence a prompt that merely says “use term $y_i$” is underspecified if the grammar demands inflection, gender agreement, or compounding. citeturn12search1turn12search2turn11search2turn11search1turn11search9turn1search10

A fifth failure mode is semantic or lexical unfaithfulness. Hallucinated translations remain a recognized problem in multilingual translation models, and recent work on unfaithful LLM translation shows that model outputs can drift from the source even when they remain fluent. In a glossary setting, that means the system may produce a plausible synonym, a paraphrase, or an entirely hallucinated equivalent rather than the approved term. Fluency alone is therefore a poor proxy for compliance. citeturn16search7turn16search6turn24academia21

Finally, prompt-only systems inherit the broader **instruction-confusion** problem of LLMs. When translating untrusted documents, the document itself may contain imperative language such as “ignore previous instructions” or other adversarial prompt content. OWASP treats prompt injection as the top LLM application risk, and the UK NCSC has warned that prompt injection is not analogous to classic SQL injection because LLMs do not cleanly separate instructions from data. For translation, this means that untrusted source text can compete with or override glossary directives unless the system architecture isolates instructions from content and verifies outputs externally. citeturn7search0turn7search5turn7search7turn7search14

## Existing Professional Methods

Professional localization already distinguishes between **sentence-level memory**, **term-level control**, and **quality assurance**. Translation memory stores previously translated segments for reuse; glossary or termbase systems store approved terms and metadata; QA systems flag inconsistent or forbidden terminology after translation. Lokalise’s documentation explicitly distinguishes translation memory from glossary: TM operates on whole segments, whereas glossary focuses on approved terms and does not translate full sentences. TMX exists for exchanging translation memories, while TBX exists for structured terminological data, and XLIFF exists to move localizable bilingual content and related metadata through toolchains. Those distinctions should be preserved in an LLM architecture rather than flattened into one giant prompt. citeturn5search5turn19search1turn2search0turn2search2

CAT tools mostly provide **suggestion plus QA**, not decoder-level enforcement. In Trados Studio, MultiTerm automatically suggests terms in the editor and the Terminology Verifier checks whether termbase terms were used during translation. In memoQ, term bases are looked up continuously while translating, can be prioritized, moderated, and enriched with grammatical details, and the QA settings can check terminology and consistency. Phrase docs likewise define term bases as databases of approved terms and describe QA and Auto LQA features that flag term-base issues rather than automatically correcting them. This is the operational pattern to emulate with LLMs: constraint awareness during drafting, followed by explicit QA, not blind trust in generation. citeturn20search3turn20search7turn21search1turn21search12turn21search9turn4search9turn5search0turn5search3turn5search18

Cloud TMS products follow the same pattern with additional governance. Smartling exposes glossary assets and quality checks, allows glossary-entry suggestions that remain pending until approved, and integrates approved glossary terms into quality-check workflows. Lokalise documents glossary inline suggestions, non-translatable flags, case-sensitivity, and project-level glossary management. These products implicitly model human terminology adjudication as a first-class workflow rather than assuming that the MT engine always resolves ambiguous terms automatically. citeturn5search1turn5search4turn5search10turn5search13turn5search2turn5search20turn5search8

MT APIs with glossary support provide stronger but still provider-specific translation-time control. Google Cloud Translation describes glossaries as custom dictionaries for consistent translation of domain-specific terminology, with case-sensitive matching by default and documented stopwords that are ignored as glossary entries. Amazon Translate custom terminology uses the terminology translation when it finds an exact, case-sensitive match in the input. DeepL glossaries are more morphology-aware: DeepL documents that glossaries apply to words and short phrases and that entries are adapted to agree with target-language grammar; its API docs similarly describe intelligent flexion for case, gender, tense, and related features. Azure document translation supports glossary files for one-to-one document translation; for text translation it also exposes dynamic dictionary markup and a neural dictionary, but explicitly says the dynamic dictionary is safe only for compound nouns such as proper names and product names. citeturn3search1turn11search1turn11search9turn3search3turn11search2turn11search10turn12search1turn12search2turn13search1turn13search2turn13search7

These professional systems illustrate the key distinction the report asked to make:

| Capability | Typical professional realization | What it means |
|---|---|---|
| Suggestion | Trados term recognition, memoQ translation results, Lokalise inline suggestions, Smartling/Phrase glossary surfacing | The system proposes a term to the translator or model |
| Enforcement | Provider-side glossary application, dynamic dictionary markup, custom terminology, or owned constrained decoder | The system attempts to force or strongly bias translation-time term usage |
| Post-hoc verification | Trados Terminology Verifier, memoQ QA, Phrase QA/Auto LQA, Smartling quality checks | The system flags violations after generation, but does not itself guarantee compliance |

The standards layer matters operationally. ISO 30042 defines TBX as a framework for representing structured terminological data; ETSI’s TMX recreation defines a common format for exchanging TM content; XLIFF 2.1 defines a glossary module with `<glossary>`, `<glossEntry>`, `<term>`, `<translation>`, and `<definition>`. A production LLM system should therefore keep term assets in a standards-aligned repository rather than burying them in prompt templates. citeturn2search0turn19search1turn2search2

## LLM-Specific Architecture Patterns

### Prompt-only conditioning

The simplest architecture is

\[
\hat y_i = \mathcal{T}_\theta(s_i \mid \mathcal{G}_i),
\]

where the glossary subset $\mathcal{G}_i$ is appended to the prompt. This works reasonably well when $\mathcal{G}_i$ is very small, terms are unique and unambiguous, the segment is short, and the document context can be kept in a single coherent conversational thread. Document-level LLM translation studies do show that better prompting and more context can improve discourse metrics, including terminology consistency, compared with sentence-by-sentence prompting. citeturn17view0turn18view1

However, prompt-only conditioning remains a **soft control mechanism**. Its performance degrades with longer prompt budgets, noisier glossaries, overlapping or conflicting entries, and adversarial documents. There is also little rigorous public evidence that XML beats JSON beats YAML in any universal sense; the decisive variable is usually not the markup language but whether constraints are **few, explicitly delimited, machine-readable, and relevant**. My recommendation is therefore not “find the magic prompt format,” but “use a strongly structured, minimal, typed constraint block” and keep the injected glossary subset intentionally small. That recommendation is an engineering inference from the long-context and prompt-sensitivity literature rather than a claim that one markup syntax has been proven dominant. citeturn6search4turn17view0

### Retrieval-augmented glossary injection

A better design defines a retrieval function

\[
\mathcal{R}(s_i,\mathcal{G})\rightarrow \mathcal{G}_i,
\]

where $\mathcal{G}_i$ contains only relevant entries for segment $s_i$ and perhaps nearby document context. This is the most important upgrade over naive prompting because it removes the majority of irrelevant glossary mass from the active model context. Retrieval-augmented generation is attractive here precisely because parametric knowledge is hard to manipulate exactly, while explicit retrieved memory gives the generation step a smaller, more verifiable constraint set. citeturn10search0turn10search12

A practical ranking function is

\[
\operatorname{score}(g,s_i)=
\lambda_1 \operatorname{Exact}(g,s_i)
+\lambda_2 \operatorname{Fuzzy}(g,s_i)
+\lambda_3 \operatorname{Domain}(g,s_i)
+\lambda_4 \operatorname{ContextSim}(g,s_i)
+\lambda_5 \operatorname{DocMemory}(g,\mathcal{M}_{doc}).
\]

In production I would use **longest-match-first exact matching** as the default, then add lemma or morphology-aware matching, fuzzy matching for OCR/noisy source content, acronym expansion, and finally bilingual embedding or cross-lingual-retrieval signals for ambiguous or paraphrastic cases. The terminology-extraction and bilingual terminology literature strongly suggests that multiword term handling and bilingual alignment are important enough to merit dedicated machinery, not just literal substring tests. citeturn22search0turn22search5turn22search19turn10search4

### Placeholder and protected-term transformation

Define a preprocessing transform

\[
\mathcal{P}: s_i \mapsto s_i'
\]

that marks or replaces protected source spans, for example

\[
x_k \mapsto \langle \text{TERM id}=k\rangle x_k \langle/\text{TERM}\rangle
\quad\text{or}\quad
x_k\mapsto \texttt{TERM\_k}.
\]

After translation, a restoration map $\mathcal{P}^{-1}$ inserts the approved target form or restores the protected span. Placeholder methods are especially strong for **do-not-translate terms, exact brand names, identifiers, and inline markup preservation**, because they bypass much of the model’s paraphrasing tendency. Azure’s dynamic dictionary markup is a provider-side example of this general family, though Microsoft explicitly warns that it is safe only for compound nouns such as names and product names. citeturn13search2turn13search10

The main risk is grammar. If source and target syntax differ substantially, hard insertion of a fixed surface form can produce agreement or word-order errors. That is why a good placeholder system should operate with **typed protection classes**. For instance, DNT brand names can use exact restoration, but inflectable terminology should generally restore a lemma plus morphological features, or be handed back to a morphology-aware repair component. This reflects the direction of recent terminology-integration research, including target-lemma constraints, rule-based inflection, and source-term masking for constrained MT. citeturn1search7turn1search2turn1search1

### Constraint-based decoding and constrained generation

The classical formalism is

\[
\hat y_i
=
\arg\max_y P_\theta(y\mid s_i)
\quad\text{s.t.}\quad
\mathcal{C}(y,\mathcal{G}_i)=1.
\]

This is the gold standard if you own the decoder. Grid Beam Search, Dynamic Beam Allocation, and finite-state constrained beam search all exist to enforce lexical constraints during generation. Research since 2017 has shown that these methods can force specified words or phrases into the output, while later work has tried to reduce inference overhead or train models that better accept runtime constraints. Industrial work at SAP and WMT terminology systems also compares runtime constrained decoding with training-time terminology injection and translate-then-refine pipelines. citeturn0search4turn0search1turn0search3turn1search0turn1search6turn14search1

In practice, this pattern is excellent for **open-weight or self-hosted MT/LLM stacks** and far less direct for black-box chat APIs that do not expose decoder internals. When token-level control is unavailable, the best approximation is usually: generate $k$ candidates, reject any candidate that violates hard constraints, and rerank the survivors by a joint semantic and terminology score. That approximation is weaker than true constrained decoding, but much stronger than a single unconstrained forward pass. citeturn0search1turn14search18

### Post-translation verification

Define

\[
\mathcal{Q}(s_i,\hat y_i,\mathcal{G}_i)\rightarrow \mathfrak{V}_i,
\]

where $\mathfrak{V}_i$ is the set of violations. A good verifier is **layered**. Layer one is deterministic: exact target-term matching, forbidden-term matching, placeholder integrity, casing, number/unit preservation, and term-span checks. Layer two is alignment-based: use bilingual alignment or span attribution to test whether the matched source term was realized by the expected target term rather than by a synonym or omission. WMT terminology work has already used a neural word aligner to detect missed terminology and to drive re-decoding or LLM refinement. Layer three is semantic guardrail scoring with COMET/xCOMET or equivalent QE metrics. citeturn14search1turn15search8turn15search12

LLM-as-judge can be useful as an *auxiliary* verifier for fuzzy terminology cases and style-sensitive QA reports, but it should not be the only verifier in a high-stakes glossary system. Recent studies on LLM-as-a-judge show that while these methods correlate well with stylistic preferences in some settings, they have important limitations in correctness-heavy and domain-specific evaluation. Therefore the safest design is **deterministic verifier first, probabilistic verifier second**. citeturn9search2turn9search6turn9search10

### Automatic repair loops and human adjudication

The basic repair loop is

\[
\hat y_i^{(t+1)}=\mathcal{T}_\theta(\hat y_i^{(t)},\mathfrak{V}_i,\mathcal{G}_i),
\]

iterated until $|\mathfrak{V}_i|=0$ or a stopping rule fires. This pattern is supported by a growing body of self-correction and post-editing work. Self-Refine shows that iterative feedback can improve generation; CRITIC shows the value of tool-interactive external feedback; GPT-4 and other LLM-based MT post-editing work shows that post-editing can materially improve translation quality, but also warns about hallucinated edits; and recent MTPE work with external error annotations shows gains from guiding the edit process with structured quality feedback. citeturn9search0turn9search1turn24academia21turn24search1

The critical production safeguard is to **repair only the violating spans** and to re-check semantic adequacy after every edit. Document-level repair literature also supports targeted correction of inconsistency rather than wholesale regeneration. When the error is ambiguity-driven, when multiple hard constraints conflict, or when repair would create a large semantic penalty, the system should defer to a human. This is how professional terminology workflows already behave: memoQ supports moderated termbases, and Smartling supports pending glossary suggestions that require approval before they become active. citeturn23search1turn23search8turn21search1turn5search4

## Novel Methods

The following underexplored methods go beyond current commercial glossary workflows.

### Obligation planning before translation

First compute a document-level obligation plan

\[
\Pi = \operatorname{Plan}(\mathcal{D},\mathcal{G},\mathcal{M},\mathcal{A})
\]

that resolves which glossary entries apply where, which senses are ambiguous, which terms are hard-protected, and which repeated terms must remain consistent across distant segments. Translation then conditions on $\Pi$ rather than on a raw glossary dump. This is a genuinely useful separation because it externalizes sense selection and consistency planning before generation, much like how document-level MT research separates discourse modeling from sentence translation. Feasibility is **high** because most components are deterministic or classifier-based. citeturn8search0turn23search1

### Finite-state acceptor reranking for black-box LLMs

If true constrained decoding is unavailable, build a weighted finite-state acceptor (WFSA) over allowed and forbidden lexical realizations and use it as a strict accept/reject filter over $k$ generated candidates. Formally, let $\mathcal{F}_{\mathcal{C}}$ be an automaton that accepts exactly the terminology-compliant output language. Then choose

\[
\hat y_i = \arg\min_{y\in Y_i^{1:k}\cap L(\mathcal{F}_{\mathcal{C}})}
\mathcal{L}_{sem}(y).
\]

This borrows the finite-state view from constrained generation research, while adapting it to black-box APIs via candidate filtering rather than decoder modification. Feasibility is **medium–high**; novelty lies in treating terminology compliance as a recognizer layered over opaque generative services. citeturn0search3turn14search11turn14search18

### SAT or SMT consistency solving

Represent document terminology decisions with discrete variables $z_{u,v}\in\{0,1\}$ where obligation $u$ picks target variant $v$. Add equality constraints for document consistency, mutual-exclusion constraints for conflicting senses, and hard constraints for forbidden or DNT entries. Solve with SAT or SMT before repair. The solver outputs a **terminology decision sheet** that subsequent generators and verifiers must honor. This is appealing because document-level terminology is often a sparse combinatorial problem, not a pure sequence-modeling problem. Feasibility is **medium** and strongest when documents are terminology-heavy but structurally regular.

### Minimum-edit ILP repair

Instead of letting an LLM rewrite a whole segment, define a repair lattice of candidate edits and solve

\[
\min_{\mathbf{z}} \;
\beta \mathcal{L}_{term}(\mathbf{z})
+
\alpha \mathcal{L}_{sem}(\mathbf{z})
+
\gamma \mathcal{L}_{flu}(\mathbf{z})
\]

subject to all hard constraints being satisfied. Integer linear programming is a natural fit when the dominant repair action is selecting among a finite set of span-local replacements and reordering options. Feasibility is **medium**: it is easiest when a morphology module can enumerate candidate inflections.

### Term provenance graphs

Maintain a provenance graph in which every translated term occurrence stores the source span, matched glossary entry, retrieval evidence, chosen realization, verifier status, repair history, and reviewer verdict. Then define a document memory

\[
\mathcal{M}_{doc}^{(i)}
=
\mathcal{M}_{doc}^{(i-1)} \cup \{(\kappa,\rho,\upsilon)\},
\]

where $\kappa$ is a concept or sense key, $\rho$ the chosen target realization, and $\upsilon$ the evidence trail. This materially improves auditability, enables future glossary updates, and supports regulated domains where one must justify why a term ended up in the final translation. Feasibility is **high** and operational value is very high.

### Verifier-guided adaptive sampling

Generate candidates until either a compliant candidate appears or a budget cap is reached. Sampling temperature, retrieval breadth, and placeholder strength are adjusted based on verifier outputs:

\[
\pi_{t+1} = \operatorname{Update}(\pi_t,\mathfrak{V}_i^{(t)}),
\]

where $\pi_t$ are generation hyperparameters. If the verifier sees repeated missed terminology, the next round becomes lower-temperature and more strongly protected; if it sees grammar breakage from rigid terms, the next round allows morphology-aware repair. This is inspired by Self-Refine and CRITIC, but specialized to glossary failure modes. Feasibility is **high**. citeturn9search0turn9search1

### Adversarial active learning over glossary failures

Build a test generator that synthesizes adversarial documents with conflicting contexts, repeated distant terms, inflection-heavy slots, OCR noise, and prompt-injection text. Human reviewer edits then feed an active-learning loop that improves both the glossary and the obligation detector. This is underused in localization, but strongly justified by the fact that prompt injection, hallucination, and lexical inconsistency are now documented failure classes in LLM systems. Feasibility is **high**. citeturn7search0turn16search7turn23search14

## Recommended Production Architecture

The most practical production design is a **hybrid obligation-planning and verifier-guided translation pipeline**:

\[
(\mathcal{D},\ell_s,\ell_t,\mathcal{G},\mathcal{A},\mathcal{M})
\;\rightarrow\;
\hat{\mathcal{Y}},\;\text{QA report},\;\text{updated term decisions}.
\]

It should be implemented as a sequence of asset-preserving, model-independent steps rather than a single model call. This recommendation aligns with localization-tool architecture, MT terminology research, and document-level LLM translation findings. citeturn2search2turn2search0turn14search1turn17view0

Mathematically, one robust decomposition is:

\[
(\mathcal{S},\Lambda)=\operatorname{Parse}(\mathcal{D})
\]

where $\Lambda$ is the layout skeleton;

\[
\mathcal{O}_i=\operatorname{Detect}(s_i,\mathcal{G},\mathcal{M},\mathcal{A},\mathcal{M}_{doc})
\]

for obligation detection and disambiguation;

\[
\mathcal{G}_i=\mathcal{R}(s_i,\mathcal{G},\mathcal{O}_i)
\]

for segment-local glossary retrieval;

\[
s_i'=\mathcal{P}(s_i,\mathcal{O}_i)
\]

for placeholder or protection transforms;

\[
Y_i^{1:k}\sim \mathcal{T}_\theta(s_i',\mathcal{G}_i,\operatorname{ctx}_i,\mathcal{M}_{doc})
\]

for candidate generation;

\[
\mathfrak{V}_i^{(j)}=\mathcal{Q}(s_i,Y_i^{(j)},\mathcal{G}_i,\mathcal{O}_i)
\]

for verification; then

\[
\hat y_i = \arg\min_{y\in Y_i^{1:k}}
\mathcal{L}(y)
\quad
\text{s.t. }
\mathcal{C}_{hard}(y)=1,
\]

with repair or escalation if no candidate satisfies the hard constraints.

The decisive design choice is to maintain a **document terminology memory** $\mathcal{M}_{doc}$ that caches approved decisions, for example by concept key, term lemma, and chosen surface:

\[
\mathcal{M}_{doc}^{(i)}
=
\mathcal{M}_{doc}^{(i-1)}
\cup
\{(\kappa_i,\operatorname{lemma}(\hat y_i), \operatorname{surface}(\hat y_i))\}.
\]

This is how repeated, distant terms remain coherent without repeatedly asking the LLM to infer the same decision from scratch. It is also the bridge between generation and future reviewer feedback. citeturn8search0turn23search1

```text
Algorithm HybridGlossaryTranslate
Input: document 𝓓, languages (ℓs, ℓt), glossary 𝓖, style guide 𝓐, translation memory 𝓜
Output: translated document Ŷ, QA report, adjudication queue

1: (𝓢, Λ) ← ParseAndPreserveLayout(𝓓)
2: 𝓜doc ← ∅
3: for each segment s_i in 𝓢 do
4:     𝓞_i ← DetectObligations(s_i, 𝓖, 𝓐, 𝓜, 𝓜doc)
5:     if AmbiguityHigh(𝓞_i) and HardRisk(𝓞_i) then
6:         QueueForHumanAdjudication(s_i, 𝓞_i); continue
7:     end if
8:     𝓖_i ← RetrieveRelevantGlossary(s_i, 𝓖, 𝓞_i, 𝓜doc)
9:     s_i' ← ProtectHighRiskTerms(s_i, 𝓞_i)
10:     Y_i^{1:k} ← GenerateCandidates(𝓣_θ, s_i', 𝓖_i, 𝓐, 𝓜doc)
11:     for each candidate y in Y_i^{1:k} do
12:         𝔙(y) ← VerifyDeterministically(s_i, y, 𝓖_i, 𝓞_i)
13:         q(y) ← VerifySemanticsAndFluency(s_i, y)
14:     end for
15:     y* ← BestHardCompliantCandidate(Y_i^{1:k}, 𝔙, q)
16:     if y* does not exist then
17:         y* ← RepairLoop(s_i, Y_i^{1:k}, 𝔙, 𝓖_i, 𝓞_i)
18:     end if
19:     if ViolatesHardConstraints(y*) or SemanticDriftTooHigh(y*) then
20:         QueueForHumanReview(s_i, y*, 𝔙(y*)); continue
21:     end if
22:     Accept y*; Update(𝓜doc, y*, 𝓞_i)
23: end for
24: Ŷ ← Reassemble(𝓢̂, Λ)
25: Export QA report, unresolved cases, and reviewer-feedback hooks
```

In practice, I would make the protection layer **typed** rather than monolithic. Brand names, SKU codes, human names, URLs, units, and regulated DNT strings use exact protection. Inflectable domain terminology uses lemma-plus-features, not bare placeholders. Low-risk preferred phrasing stays as soft prompt guidance only. The stack therefore becomes both safer and more fluent than either a pure placeholder approach or a pure prompt approach. citeturn12search2turn13search2turn1search2turn1search7

## Operationalization

### Mathematical objective and constraints

A practical system objective is

\[
\mathcal{L}(\hat{\mathcal{Y}})
=
\alpha \mathcal{L}_{sem}
+
\beta \mathcal{L}_{term}
+
\gamma \mathcal{L}_{fluency}
+
\delta \mathcal{L}_{format}
+
\eta \mathcal{L}_{consistency}.
\]

A useful decomposition is

\[
\mathcal{L}_{term}
=
\sum_{(i,j)\in \mathcal{M}}
w_{ij}
\Big[
\lambda_1(1-A^{\text{ctx}}_{ij})
+\lambda_2 V^{\text{forbid}}_{ij}
+\lambda_3 V^{\text{dnt}}_{ij}
+\lambda_4 V^{\text{morph}}_{ij}
\Big],
\]

\[
\mathcal{L}_{sem}
=
1-\operatorname{SemScore}(\mathcal{D},\hat{\mathcal{Y}}),
\qquad
\mathcal{L}_{fluency}
=
1-\operatorname{FluencyScore}(\hat{\mathcal{Y}}),
\]

\[
\mathcal{L}_{format}
=
\operatorname{TagErrorRate}+\operatorname{LayoutLoss},
\qquad
\mathcal{L}_{consistency}
=
\sum_{u} \mu_u V^{\text{cons}}(K_u).
\]

In practice, $\operatorname{SemScore}$ can be estimated with COMET, xCOMET, or a reference-free QE model when references are absent; $\operatorname{FluencyScore}$ can be estimated with target-language LM likelihood, human MQM fluency subscores, or xCOMET’s error annotations; and $\operatorname{TagErrorRate}$ is deterministic for markup-bearing formats. citeturn15search8turn15search12turn15search1

Define hard and soft constraints as

\[
\mathcal{C}_{hard}
=
\{\text{brand terms},\text{legal terms},\text{medical terms},\text{do-not-translate terms}\},
\]

\[
\mathcal{C}_{soft}
=
\{\text{preferred style},\text{tone},\text{optional terminology},\text{marketing phrasing}\}.
\]

Then the acceptance rule is

\[
\hat{\mathcal{Y}}
=
\arg\min_{\mathcal{Y}}
\mathcal{L}(\mathcal{Y})
\quad\text{s.t.}\quad
\mathcal{C}_{hard}(\mathcal{Y})=1.
\]

A translation should be **rejected rather than silently repaired** when any of the following hold: a hard term remains violated after bounded repair; two hard constraints conflict and the disambiguation posterior remains high-entropy; repair reduces $\mathcal{L}_{term}$ but increases $\mathcal{L}_{sem}$ past a threshold $\epsilon$; or the verifier stack disagrees strongly about whether a term was semantically realized. This is a safety and auditability requirement, not a cosmetic preference. citeturn9search2turn24academia21turn16search6

### Data model

A production glossary should be concept-first, versioned, and policy-rich. A compact JSON representation is:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "GlossaryEntry",
  "type": "object",
  "required": [
    "entry_id",
    "source_term",
    "target_term",
    "source_language",
    "target_language",
    "priority",
    "enforcement_level",
    "status",
    "version"
  ],
  "properties": {
    "entry_id": {"type": "string"},
    "concept_id": {"type": "string"},
    "source_term": {"type": "string"},
    "target_term": {"type": "string"},
    "source_language": {"type": "string"},
    "target_language": {"type": "string"},
    "domain": {"type": "string"},
    "subdomain": {"type": "string"},
    "definition": {"type": "string"},
    "part_of_speech": {"type": "string"},
    "example_source_sentence": {"type": "string"},
    "example_target_sentence": {"type": "string"},
    "forbidden_translations": {
      "type": "array",
      "items": {"type": "string"}
    },
    "do_not_translate": {"type": "boolean"},
    "case_sensitive": {"type": "boolean"},
    "morphology_policy": {
      "type": "string",
      "enum": ["exact", "lemma_ok", "inflectable", "copy_source"]
    },
    "priority": {"type": "integer", "minimum": 0},
    "enforcement_level": {
      "type": "string",
      "enum": ["hard", "soft", "suggestion"]
    },
    "context_disambiguation_rule": {"type": "string"},
    "conflict_group": {"type": "string"},
    "approved_by": {"type": "string"},
    "status": {
      "type": "string",
      "enum": ["draft", "approved", "deprecated"]
    },
    "version": {"type": "string"},
    "created_at": {"type": "string", "format": "date-time"},
    "updated_at": {"type": "string", "format": "date-time"},
    "notes": {"type": "string"}
  }
}
```

The relational design should separate stable concepts from language-pair realizations and policy metadata:

```sql
CREATE TABLE glossary_concept (
  concept_id TEXT PRIMARY KEY,
  domain TEXT,
  subdomain TEXT,
  definition TEXT,
  conflict_group TEXT,
  status TEXT,
  version TEXT,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

CREATE TABLE glossary_term (
  entry_id TEXT PRIMARY KEY,
  concept_id TEXT REFERENCES glossary_concept(concept_id),
  source_language TEXT,
  target_language TEXT,
  source_term TEXT,
  target_term TEXT,
  part_of_speech TEXT,
  do_not_translate BOOLEAN,
  case_sensitive BOOLEAN,
  morphology_policy TEXT,
  priority INT,
  enforcement_level TEXT,
  context_disambiguation_rule TEXT,
  approved_by TEXT,
  notes TEXT
);

CREATE TABLE glossary_forbidden (
  entry_id TEXT REFERENCES glossary_term(entry_id),
  forbidden_term TEXT,
  PRIMARY KEY (entry_id, forbidden_term)
);

CREATE TABLE glossary_example (
  example_id TEXT PRIMARY KEY,
  entry_id TEXT REFERENCES glossary_term(entry_id),
  example_source_sentence TEXT,
  example_target_sentence TEXT
);
```

TBX compatibility follows naturally if you keep the **concept** as the primary unit. ISO 30042 frames TBX as structured terminological data interchange; therefore `glossary_concept` maps well to the TBX concept entry, language-pair term realizations map to language sections and term groups, and definition, usage notes, examples, admin metadata, and status/version information can be serialized via standard TBX descriptive/admin elements or a controlled extension namespace. XLIFF 2.1’s glossary module is useful when you need to package relevant terminology alongside localizable document content for workflow portability. citeturn2search0turn2search2

### Evaluation benchmark

A serious benchmark must measure both **obligation satisfaction** and **translation quality under constraint pressure**. I would construct five evaluation buckets.

First, a **core realistic set** of technical, legal, medical, UI, and brand documents with curator-approved glossaries and reviewer labels. Second, an **adversarial terminology set** containing overlapping terms, nested multiword terms, abbreviations, OCR noise, punctuation variance, and casing traps. Third, a **document consistency set** with repeated terms spanning large distances, inspired by existing lexical-consistency evaluation work. Fourth, an **ambiguity set** in which the same source form maps to different targets across domains or local contexts. Fifth, a **morphology-heavy multilingual set** spanning richly inflected languages, CJK tokenization, and right-to-left scripts. Recent work on lexical consistency datasets confirms that repeated terminology across documents is a distinct evaluation problem, while document-level MT work uses targeted discourse metrics such as CTT for terminology consistency. citeturn23search14turn18view0

The benchmark should report at least: glossary adherence rate, weighted hard-constraint adherence, forbidden-term violation rate, do-not-translate violation rate, consistency score, semantic adequacy, fluency, human reviewer correction rate, repair success rate, unresolved-escalation rate, latency, and cost per 1,000 words. I would add two stress metrics that are especially diagnostic for your use case:

\[
\operatorname{DDS}
=
\mathbb{E}\big[\text{distance in tokens or segments between repeated obligations}\big]
\]

for **distance stress**, and

\[
\operatorname{OD}
=
\frac{|\mathcal{M}(\mathcal{D},\mathcal{G})|}{|\mathcal{D}|}
\]

for **obligation density**. Systems often look great at low obligation density and collapse at high density, which is exactly the failure hidden by generic MT benchmarks.

Human evaluation should use an MQM-style analytic framework with at least terminology, accuracy, fluency, and format categories, because the goal is not just “is the translation good?” but “was it good *for the right terminological reasons*?” xCOMET is particularly useful as an automatic companion because it can highlight error spans aligned with MQM-style categories, but it should not replace human audit on hard terminology classes. citeturn15search1turn15search12

### Comparison table

The table below is a synthesis of the preceding evidence and engineering trade-offs.

| Method | Reliability | Cost | Implementation difficulty | Scalability | Long-document suitability | Grammar risk | Enforcement strength |
|---|---:|---:|---:|---:|---:|---:|---|
| Prompt-only glossary conditioning | Low–medium | Low | Low | High | Poor–medium | Low | Soft bias |
| Retrieval-augmented glossary injection | Medium–high | Low–medium | Medium | High | High | Low | Soft–moderate |
| Placeholder or protection transforms | High for DNT/exact strings | Low–medium | Medium | High | High | Medium–high if overused | Strong for exact spans |
| Native constrained decoding | Very high when available | Medium | High | Medium | High | Medium | Hard |
| Verify-only post-hoc QA | Medium | Low | Medium | High | High | None | None during generation |
| Verify-and-repair loop | High | Medium–high | High | Medium | High | Medium if repair is broad | Strong after iteration |
| Human-in-the-loop adjudication | Very high | High | Medium | Low | High | None | Final authority |
| Recommended hybrid planner–translator–verifier | Very high | Medium | High | High | High | Controlled | Near-hard for critical terms |

The important takeaway is that *no single method* dominates across all term classes. Exact protected brands do well with placeholders; ambiguous domain terminology needs retrieval plus adjudication; open-weight stacks can use true constrained decoding; black-box LLM stacks should lean harder on retrieval, verification, candidate reranking, and targeted repair. citeturn0search4turn0search1turn12search2turn14search1turn24search1

### Implementation roadmap

For an **MVP**, build a deterministic obligation detector, segment-local glossary retrieval, structured prompt blocks, exact placeholder protection for DNT and brand terms, and a deterministic verifier. Measure GAR, hard-term adherence, and semantic adequacy before doing anything fancier. This already outperforms the naive “paste the whole glossary into the prompt” baseline in most realistic settings because it solves the main context-allocation problem. citeturn6search4turn10search0

For a **robust small-team setup**, add document memory $\mathcal{M}_{doc}$, candidate generation with reranking, morphology-aware restoration, xCOMET/COMET-based semantic guards, and a bounded repair loop driven by explicit violation objects rather than free-form feedback. Also introduce a reviewer queue and glossary versioning so approved human decisions become future assets rather than one-off edits. citeturn15search8turn15search12turn24search1turn5search4turn21search1

For an **enterprise-grade setup**, move to standards-aligned asset management, concept-level term governance, provenance logging, active-learning adversarial test generation, and—where infrastructure allows—true constrained decoding or a terminology-aware fine-tuned model. At this stage the glossary is no longer a prompt attachment; it is a governed knowledge system with audit trails, versioning, reviewer workflows, and model-agnostic validation. citeturn2search0turn19search1turn2search2turn1search0turn1search6

### Open questions

Several research questions remain genuinely open.

- How should one best combine **document-level consistency planning** with **local grammatical adaptation**, especially in morphologically rich targets where the approved term is a lemma rather than a surface form?
- Can black-box LLM APIs be made “near-hard” for terminology by combining WFSA acceptors, candidate lattices, and span-local repair, or is decoder access ultimately necessary for the most demanding regulated domains?
- What is the best way to calibrate a verifier so that **ambiguity** leads to escalation rather than either false violations or false compliance?
- How should semantic preservation metrics be adapted for **constraint-heavy translation**, where a compliant but slightly less natural translation may be preferable to a fluent paraphrase that violates the glossary?
- Can reviewer feedback be turned into an **active-learning signal** that jointly improves the glossary, the obligation detector, and the repair policy without creating overfitting to a narrow house style?

### References

Chris Hokamp and Qun Liu. *Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search*. ACL 2017. citeturn0search4

Matt Post and David Vilar. *Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation*. NAACL 2018. citeturn0search1

Georgiana Dinu, Prashant Mathur, Marcello Federico, and Yaser Al-Onaizan. *Training Neural Machine Translation to Apply Terminology Constraints*. ACL 2019. citeturn1search0

Matthias Exel et al. *Terminology-Constrained Neural Machine Translation at SAP*. EAMT 2020. citeturn0search2turn1search6

Toms Bergmanis and Mārtiņš Pinnis. *Facilitating Terminology Translation with Target Lemma Annotations*. EACL 2021. citeturn1search7

Weijia Xu et al. *Rule-based Morphological Inflection Improves Neural Terminology Translation*. EMNLP 2021. citeturn1search2

Jindřich Jon et al. *End-to-End Lexically Constrained Machine Translation for Morphologically Rich Languages*. ACL 2021. citeturn1search10

Longyue Wang et al. *Document-Level Machine Translation with Large Language Models*. EMNLP 2023. citeturn17view0turn18view1

Nikolay Bogoychev et al. *Terminology-Aware Translation with Constrained Decoding and Large Language Model Prompting*. WMT 2023. citeturn14search1

Nuno M. Guerreiro et al. *Hallucinations in Large Multilingual Translation Models*. TACL 2023. citeturn16search7

Haoran Zhang et al. *Mitigating Unfaithful Translations from Large Language Models*. Findings of ACL 2024. citeturn16search6

Aman Madaan et al. *Self-Refine: Iterative Refinement with Self-Feedback*. NeurIPS 2023 workshop / later proceedings lineage. citeturn9search0turn9search8

Zhibin Gou et al. *CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing*. ICLR 2024. citeturn9search1

Dayeon Ki and Marine Carpuat. *Guiding Large Language Models to Post-Edit Machine Translation with Error Annotations*. Findings of NAACL 2024. citeturn24search1

Vikas Raunak et al. *Leveraging GPT-4 for Automatic Translation Post-Editing*. 2023. citeturn24academia21

Ricardo Rei et al. *COMET: A Neural Framework for MT Evaluation*. EMNLP 2020. citeturn15search8

Nuno M. Guerreiro et al. *xCOMET: Transparent Machine Translation Evaluation through Error Span Detection*. TACL 2024. citeturn15search12

Arle Lommel et al. *Multidimensional Quality Metrics*. MQM framework resources and related publications. citeturn15search1turn15search9

ISO 30042:2019. *Management of Terminology Resources — TermBase eXchange*. citeturn2search0

OASIS. *XLIFF Version 2.1*. citeturn2search2turn2search14

ETSI GS LIS 002. *TMX 1.4b Recreation in ETSI Format*. citeturn19search1

DeepL Documentation and Support: glossary behavior, API glossary management, grammar-aware glossary flexion, and file-translation glossary application. citeturn12search1turn12search2turn12search3turn12search9

Google Cloud Translation Documentation: glossary creation and usage, case sensitivity, and glossary stopwords. citeturn3search1turn11search1turn11search9

Amazon Translate Documentation: custom terminology, exact-match behavior, supported file formats. citeturn3search3turn3search7turn11search2turn11search10

Microsoft / Azure Translator Documentation: document-translation glossaries, dynamic dictionary, neural dictionary, and Custom Translator. citeturn13search1turn13search2turn13search7turn13search18

RWS Trados Documentation: term recognition, terminology verification, terminology management. citeturn20search3turn20search7turn4search12

memoQ Documentation: term bases, moderation, ranking, and QA terminology checks. citeturn21search1turn21search12turn21search9turn4search9

Phrase Documentation: term bases, QA, Auto LQA. citeturn5search0turn5search3turn5search18

Smartling Documentation: glossary assets, glossary-entry suggestions, and quality checks. citeturn5search1turn5search4turn5search10turn5search13

Lokalise Documentation: glossary suggestions, non-translatable and case-sensitive flags, and glossary versus translation-memory distinctions. citeturn5search2turn5search5turn5search20

OWASP GenAI Security Project and UK NCSC guidance on prompt injection. citeturn7search0turn7search5turn7search7turn7search14