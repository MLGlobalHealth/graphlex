# graphlex — paper plan (draft for review, 2026-06-14)

Status: **plan only.** No paper writing, no large experiments yet. This document
sharpens the contribution against the full prior-work reads in
`RELATED_WORK_NOTES.md`, proposes a venue + framing, lists the experiments needed
to make the claims defensible, and drafts an outline + risk register. Read
`RELATED_WORK_NOTES.md` first.

> **VENUE UPDATE (2026-06-14): target is a top general-science journal (PNAS /
> Nature / Nature-family).** This raises the bar from "novel ML method / useful
> tool" to "**finding of broad scientific significance, rigorously established.**"
> Consequences threaded through this doc: (i) the contribution must be framed as a
> *finding about LLMs and structure*, not as the graphlex toolkit (the toolkit
> becomes the enabling method + a resource, not the headline); (ii) generality is
> mandatory — multiple LLM families and real networks from multiple scientific
> domains, not one model on synthetic data; (iii) the memorization confound goes
> from "address it" to **existential** — a top-journal reviewer will desk-question
> it first; (iv) the GraphText overlap becomes an editor-screening risk on novelty,
> so the headline must rest on what GraphText demonstrably did NOT do. See the
> frank feasibility assessment in §8. The earlier §2 venue options are superseded
> by §2′ below.

---

## 1. The contribution, sharpened against prior work

The original handoff listed four candidate novelties. After reading GraphText,
Talk-like-a-Graph, NLGraph, and both surveys in full, here is the honest verdict.

| # | Candidate claim | Verdict | Why |
|---|---|---|---|
| 1 | Deterministic **computed-FEATURE** verbalization (not raw-structure encoding), no LLM in rendering | **SURVIVES — the core wedge** | GraphText/Talk-like-a-Graph/NLGraph all verbalize *raw topology* (edge lists, adjacency, ego-tree attribute labels) and ask the LLM to compute. None compute-and-verbalize centralities, modularity/communities, assortativity/homophily, or **null-model comparisons**. Neither survey names "verbalize computed network statistics" as a category. |
| 2 | The controlled permutation experiment (raw → chance, verbalized → 0.92) | **SURVIVES as a clean instance, NOT as first observation** | NLGraph already showed LLMs exploit surface shortcuts (degree/mention frequency). Our permutation control is a *cleaner, isolated* demonstration of the node-ordering artifact, but must be framed as "consistent with / sharpening NLGraph," not novel-in-kind. |
| 3 | Social-science structure×attributes + null-model framing | **SURVIVES as positioning, not as method novelty** | Homophily/assortativity/null-models are standard network science; the novelty is *verbalizing them for LLM ICL* + serving an audience (small attributed social nets) the ML-benchmark work skips. Claim it as application + framing, not as a new metric. |
| 4 | Interpretable NetworkX-native toolkit + agent skills (MCP) | **SURVIVES as a software/position contribution** | The inspectable, versioned thresholds table (numbers→words, no generative model) is a concrete artifact GraphText's anecdotal NL-trace interpretability lacks. This is a tools/position contribution, not an empirical ML claim. |

**Does NOT survive (must NOT be claimed):**
- "Deterministic verbalization, no LLM in rendering, makes ICL GNN-competitive" → **GraphText**.
- "How you present a graph to an LLM matters" → **Talk like a Graph**.
- "LLMs read shortcuts instead of structure" → **NLGraph** (we sharpen it; we didn't find it).
- "Interpretability via NL graph reasoning" → GraphText + both surveys list it.

**One-sentence contribution — TOOLS-VENUE version (now demoted to a secondary claim):**
> *We show that deterministically verbalizing **computed** graph-theoretic
> statistics — centralities, community structure, assortativity/homophily, and
> explicit null-model contrasts — rather than raw topology, lets a frozen LLM do
> interpretable in-context analysis of small attributed networks; we ship this as
> an inspectable, training-free, NetworkX-native toolkit.*

**One-sentence contribution — GENERAL-SCIENCE version (the new headline; see §2′):**
> *Large language models do not perceive the structure of relational data: shuffle
> the labels of a network and an LLM's read of it collapses to chance, revealing
> it was exploiting surface artifacts, not topology. Yet the competence is
> recoverable — when the structure is **computed and stated in language**, a frozen,
> untrained model reasons about networks across the sciences at a level competitive
> with classical analysis, and does so interpretably. We establish this across N
> model families and M real networks from K scientific domains, and provide the
> deterministic protocol that makes it reproducible.*

The general-science version leads with a **finding about the models** (a clean,
surprising limit + its resolution), which is what a PNAS/Nature editor screens for;
the protocol/toolkit is the method that operationalizes and democratizes it. The
honesty hedge ("exploiting surface artifacts, not topology"; separating recovered
competence from reused priors) is not just defensible — at this venue it is the
*spine* of the result, because the permutation control is the evidence the model
truly lacks an internal structural model.

---

## 2′. Target: PNAS / Nature — framing for a general-science journal

A general-science editor triages on one question: *is there a finding here that a
biologist, a physicist, and a social scientist would all find significant and
trust?* The toolkit is not that; a finding about **what LLMs do and don't
understand about structure**, demonstrated to generalize and rigorously
controlled, can be. The toolkit becomes the **method + a community resource**, not
the headline.

### The headline finding (what makes it general-science)
A two-part claim, both halves needed:
1. **The limit (the surprise):** frozen LLMs have *no internal model of relational
   structure*. Under label permutation — a transformation that leaves the graph
   identical — their accuracy on structural questions collapses to chance. They
   were reading node-ordering/naming artifacts, not topology. (Clean, counter-
   intuitive, matters to everyone now feeding structured data to LLMs.)
2. **The resolution (the constructive part):** the competence is *recoverable
   without any training* — compute the structure with classical network science and
   state it in language, and the same frozen model reasons about networks
   competitively with classical methods, **interpretably and reproducibly**, across
   scientific domains. The deterministic protocol (graphlex) is what makes #2 a
   method anyone can apply, not a Claude-specific trick.

Why this clears the bar where a tools paper would not: it is a *property of a
technology now used across all of science*, it is **surprising** (people assume
LLMs "see" the graph they're given), it is **actionable** (a concrete, training-
free protocol), and the permutation control gives it the kind of crisp, decisive
evidence top journals reward.

### Two framing variants to choose between (THE decision I need from you)
- **Variant L — "limits-led" (Nature / Nature Human Behaviour / Nature Machine
  Intelligence).** Lead with the cognitive-science-flavored finding ("LLMs are
  blind to raw relational structure"); the protocol is the resolution. Highest
  general-interest ceiling; competes on *insight about the models*. Most exposed to
  "is this just memorization?" and to NLGraph (which already showed shortcut
  reliance — we must show our permutation control is the decisive, general version).
- **Variant R — "resource/enabling-method-led" (PNAS).** Lead with "a training-
  free, interpretable instrument that lets any scientist analyze networks with an
  LLM," the permutation finding as the *why it works / why naive use fails*
  section. PNAS is friendlier to methods-that-enable-science; lower novelty-screen
  risk vs GraphText because the pitch is breadth-of-application + rigor, not a new
  ML idea. Slightly lower "wow," higher acceptance realism.

**My recommendation:** **PNAS via Variant R**, written so the limits finding (L) is
still a prominent, self-contained result inside it. Rationale: (a) the GraphText
overlap is least damaging when the pitch is "enabling instrument + cross-domain
evidence + rigor," not "new method"; (b) PNAS's contributed/standard track and
breadth-friendly scope fit a cross-domain demonstration; (c) Variant L's ceiling is
higher but its desk-reject risk on novelty/memorization is also higher, and Nature
Machine Intelligence/Nature Human Behaviour reviewers will be the harshest on the
GraphText/NLGraph overlap. We can decide L-vs-R *after* the cross-domain + multi-
model results (§3) tell us how strong and how general the effect actually is — if
the permutation collapse is dramatic and universal across models, escalate to L.

> Reality check (see §8): with PNAS/Nature as the target, the current evidence base
> (single-run, one model, mostly synthetic, two confounds baked into the eval code)
> is **not within striking distance** — it is the pilot. The plan below is the real
> study. Be prepared for this to be a 6–12-month empirical program, and for an
> honest outcome where the finding is real but the venue settles at PNAS/Nature
> Machine Intelligence rather than Nature main.

---

## 3. Experiments needed to make the claims solid

Current results (`eval/RESULTS.md`) are **single-run, one model (Claude), mostly
synthetic, tiny n** (24/36/40 queries). For a general-science venue this is a
**pilot**, not a result set. The bar adds two non-negotiable generality axes on top
of everything in the tools-venue plan: **(G1) multiple LLM families** (the finding
must be about LLMs, not Claude) and **(G2) real networks from multiple scientific
domains** (the finding must be about networks-in-science, not one dataset).
Required work, in priority order:

### P0 — Generality across model families (G1) — MANDATORY for "LLMs", not "Claude"
- Run every headline experiment on **≥4 model families**, spanning vendors and
  scales: e.g. Claude (Opus + a smaller tier), GPT-4-class, Gemini, and ≥1 strong
  **open-weight** model (Llama-3.x / Qwen) — the open model matters because it lets
  reviewers reproduce without paid APIs and lets us probe training data.
- The permutation collapse and the verbalization-recovery must hold **across all of
  them** (directionally) for the finding to be "about LLMs." Report per-model curves;
  a single dissenting model is a finding in itself, not something to hide.
- Subagent ICL runs need no key; cross-vendor runs need their respective API access.

### P0 — Multi-seed replication of existing results (do first; cheap)
- Re-run all three studies over **≥20 seeds** (vary graph generation, shot
  sampling, query split). Currently `make_verbalize_eval.py` uses a single
  `RandomState(7)`, `fair_node_pred.py` a single `RandomState(1)`.
- Report **mean ± 95% CI** (bootstrap or seed-wise), not point estimates. With 24
  queries a single run's CI is ≈ ±0.19 — the headline gaps may not survive.
- Increase query counts (≥150/condition) so CIs are usable.
- Keep facts()/verbalize() LLM-free (unchanged); only the harness/seed loop changes.

### P0 — Fix two confounds the eval code currently bakes in (must do before claiming)
1. **Named-family prior leak (synthetic study).** `make_verbalize_eval.py` puts the
   true family names in the prompt ("SCALE-FREE (preferential attachment)"), so the
   verbal arm can use Claude's *textbook prior* about scale-free/small-world degree
   profiles instead of learning from shots. **Fix:** add an **anonymized-label
   arm** (families relabeled Class A/B/C, no generative names) and a **no-shot
   arm**. The honest claim needs: verbalized-features ICL beats raw *even when the
   model cannot name the family*. This directly answers the memorization risk (§6).
2. **Neighbor-count tautology (fair node study).** `fair_node_pred.py`'s
   `describe()` literally prints "known colleagues by department: CS:3, Math:1…" —
   i.e. it hands the model the exact statistic neighbor-majority uses. So
   "verbalize+Claude = neighbor-majority = 0.750" is near-tautological, not evidence
   of structural reasoning. **Fix:** add a harder variant where the target's
   neighbor labels are *not* directly tabulated (e.g. give 2-hop structure, or
   withhold direct neighbor dept counts and force the model to infer from position),
   so the LLM has to do something a lookup can't.

### P0/P1 — REAL networks from MULTIPLE scientific domains (G2) — was "one real net"
- Generality across **≥3 scientific domains** is now mandatory, not optional. The
  pitch "across the sciences" must be earned with data: e.g.
  - **Social/behavioral:** Lazega lawyers, SocioPatterns face-to-face (school/
    hospital, role attributes), Copenhagen Networks Study, Add Health (restricted —
    check access early). Avoid Zachary karate club.
  - **Biological:** protein–protein interaction subnetworks, gene co-expression,
    food webs, connectomics — with node attributes (function/module/taxon).
  - **Physical/infrastructure or information:** power grid, transport, or a
    non-citation information network with attributes.
- **Leakage constraint (now existential, see §6):** prefer networks that are (a)
  not canonical textbook datasets and ideally (b) **post-training-cutoff or newly
  constructed**, so the model cannot have memorized them. For each, also run an
  **entity-anonymized** version (strip recognizable names/labels); if accuracy
  drops, the model was recognizing, not reasoning — that delta is a key reported
  quantity, not a nuisance.
- Tasks per domain: node-attribute prediction + structure-as-finding (modular /
  core-periphery vs configuration-model null) + a homophily/role read the
  verbalizer produces. Keep the *same protocol* across domains — uniformity is the
  point ("one instrument, many sciences").

### P1 — A FAIR KumoRFM comparison with the CORRECT schema (OPEN QUESTION — flag)
- Current `eval/RESULTS.md` native-relational KumoRFM = 0.306 ≈ chance. The
  controlled `kumo_linktest.py` (0.139) shows neighbor-feature aggregation isn't
  happening across the self-edges table. **This is almost certainly our schema
  misconfiguration, NOT KumoRFM's true ability.** Do **not** publish 0.306 as
  Kumo's performance — it would be a misrepresentation of a competitor's system.
- **Action:** the only defensible Kumo number today is the tabular-features 0.639.
  Before any comparison ships, **email/ask the Kumo team** for the correct
  multi-table schema for homogeneous-graph node/collective classification, then
  re-run. If they confirm a correct schema and it still underperforms, report that;
  if it does better, report that. Either way the comparison must be one Kumo would
  endorse as fair. (API key at `/home/scratch/.kumo_api_key`; free tier 1000/day;
  env-only, never commit.)
- Fallback if Kumo can't be made fair in time: drop the head-to-head, cite Kumo as
  "enterprise relational FM, out of graphlex's small-network scope" (consistent
  with the project memory positioning) and lean on logreg / SGC+logreg / neighbor-
  majority / untrained-GNN baselines, which we control and trust.

### P1 — Ablations (these ARE the scientific core; they make claim #1 land)
- **Verbalization detail ladder:** raw edge list → raw+permuted → counts only →
  + centralities → + community/modularity → + assortativity/homophily → + null-
  model deltas. Show the accuracy/interpretability curve as computed content is
  added. This is the figure that distinguishes graphlex from GraphText.
- **Shot count:** 0/1/3/6/12-shot, per arm. Tests whether wins are ICL or prior.
- **Model family/size:** at least one non-Claude (e.g. an open model) to show the
  effect isn't Claude-specific; and a small-vs-large within a family.
- **Threshold-table sensitivity:** vary `thresholds.py` cutoffs, show wording
  changes are stable / the downstream LLM answer is robust to reasonable cutoffs
  (interpretability claim #4).

### P2 — Baselines to always report alongside (cheap, credibility-critical)
- logreg on classical features; SGC+logreg (per memory, often SOTA on small nets);
  neighbor-majority; **untrained random GNN** (the killer control from prior
  project work). Verbalize+LLM must be shown *against* these, not in isolation.

---

## 4. Paper outline — general-science (PNAS/Nature) format, finding-led

Top journals use a compact main text (~3–5k words, ~4 display items) with methods
and the bulk of evidence in Supplementary. Structure leads with the finding, not
the tool.

**Main text**
1. **Framing paragraph(s)** — networks are how science represents relational data
   (cells, brains, societies, ecosystems); LLMs are now used to interpret them; do
   they actually read structure? State both halves of the finding up front.
2. **Result 1 — LLMs are blind to raw relational structure.** Permutation collapse
   to chance, across model families and domains (F1). The decisive control.
3. **Result 2 — competence is recoverable, training-free, by stating computed
   structure in language.** The verbalization-detail ladder (F2): accuracy climbs
   as classical network-science content is added; competitive with classical
   analysis; holds across models and domains.
4. **Result 3 — it generalizes and it is interpretable.** Cross-domain panel (F3);
   the worked structure×attributes + null-model read with its inspectable threshold
   table (F4). One protocol, many sciences.
5. **What the LLM does and does not add** — priors vs ICL; where classical methods
   (logreg / SGC+logreg / untrained-GNN) already match it; the honest boundary.
6. **Discussion** — implications for using LLMs on structured scientific data; the
   deterministic protocol as a reusable instrument; limits.

**Methods + Supplementary** — graphlex internals (facts() exact NetworkX →
verbalize() templates + versioned thresholds, no LLM; two modes; agent/MCP layer);
full related-work matrix (T1); all multi-seed tables with CIs; per-model and
per-domain breakdowns; leakage/anonymization analysis; fair-Kumo (or scope-out);
ablations (shots, verbalization detail, threshold sensitivity).

### Key display items (≤4 main, rest in SI)
- **F1 (headline)** Permutation control across model families × domains: verbal
  arm flat vs raw arm collapsing to chance. The single most decisive figure.
- **F2** Verbalization-detail ladder: accuracy vs amount of *computed* content
  (raw → raw-permuted floor → counts → +centrality → +community → +homophily →
  +null-model), curves per model family.
- **F3** Cross-domain generalization panel: protocol vs classical baselines across
  ≥3 scientific domains, mean±CI.
- **F4** Worked interpretability example: prose output + the threshold table that
  produced it (the "no generative model in the loop" artifact).
- **T1 (SI)** Related-work matrix: raw vs computed × training-free × attributes ×
  null × cross-domain × multi-model — graphlex is the empty cell.
- **T-models / T-domains (SI)** Full per-model and per-domain results with CIs and
  leakage deltas.

---

## 5. Risks to the contribution (ranked — re-ranked for PNAS/Nature)

0. **General-science significance / desk-reject (now highest).** An editor decides
   in minutes whether this is "of broad significance" or "a specialized ML/methods
   result." A toolkit, or "computed verbalization beats raw," reads as the latter
   and never reaches review. *Mitigation:* lead with the finding (the permutation
   limit + training-free recovery, §2′), demonstrate it across model families and
   scientific domains (§3 G1/G2), and write the abstract for a general scientist,
   not an ML audience. If after the pilot the effect is not large, clean, and
   general, **down-target honestly** (PNAS→Nature Machine Intelligence→LoG/comp-
   social-science) rather than overclaim — see §8.
1. **GraphText overlap (now an editor novelty-screen, not just a reviewer note).**
   At a top venue, "prior work already verbalized graphs for LLMs" can trigger a
   desk reject on novelty. *Mitigation:* the headline is the *finding* (LLMs lack a
   structural model; competence is recoverable by computed-feature verbalization),
   which GraphText does not establish; GraphText/Talk-like-a-Graph become *support*
   that verbalization matters. Concede explicitly; the related-work matrix (T1) must
   make the empty cell (computed stats + null models, cross-domain, multi-model)
   obvious. Never imply we are first to verbalize graphs.
2. **Memorization / prior leak (now EXISTENTIAL at this venue).** A top-journal
   reviewer's first question: "did the model just recognize a graph/distribution it
   memorized?" If we can't rule it out decisively, the finding is dead. The headline
   synthetic win already partly reflects Claude's textbook knowledge of "scale-
   free"; the IJCAI survey warns LLMs memorized common benchmarks. *Mitigation
   (must be airtight):* anonymized-label arm + no-shot arm; leakage-controlled,
   ideally post-cutoff/newly-constructed real networks; entity-anonymization deltas
   reported as data; the permutation control itself is the strongest anti-
   memorization argument (same graph, shuffled labels → collapse means it wasn't
   recall). Never use karate/Cora/Citeseer.
3. **Effect evaporates under multi-seed / fair baselines (high).** Tiny n; gaps may
   not be significant; SGC+logreg/untrained-GNN may match the LLM (consistent with
   the project's own prior negative results). *Mitigation:* P0 first; be willing to
   report a null/scoped result — that is still a contribution given the framing.
4. **Unfair Kumo number (reputational).** Publishing 0.306 as Kumo's ability would
   be wrong and is a misrepresentation risk. *Mitigation:* §3 P1 — ask Kumo, or
   scope out.
5. **Neighbor-count tautology (medium).** The fair-node win may be a lookup.
   *Mitigation:* harder variant in §3 P0.
6. **"Just a wrapper around NetworkX" (medium).** *Mitigation:* lead with the
   science (the ladder result + what-the-LLM-adds section), ship the tool as a
   *second* contribution (JOSS), not the only one.

---

## 6. Honesty checklist (carry into the writing)
- [ ] No "first to verbalize graphs for LLMs." (We are not.)
- [ ] No "verbalize→ICL matches GNNs" as our novelty. (GraphText.)
- [ ] Permutation framed as sharpening NLGraph, not discovering the effect.
- [ ] Kumo native-mode 0.306 never reported as Kumo's true ability.
- [ ] Every LLM number paired with logreg / SGC+logreg / untrained-GNN baselines.
- [ ] Memorization addressed empirically (anonymized arm), not just hand-waved.
- [ ] facts()/verbalize() remain LLM-free; LLM only in agent/eval layer.
- [ ] All headline numbers multi-seed with CIs.

---

## 7. Immediate next steps (after review)
1. Confirm framing direction **L vs R** within the PNAS/Nature target (§2′) — the
   one decision I need from you.
2. Land the strengthened pilot: multi-seed harness + the two confound fixes +
   anonymized/no-shot arms, **on ≥2 model families** (Claude + one open model) to
   sanity-check generality before scaling. This tells us if the effect is real and
   general enough to justify the full program.
3. Identify + acquire ≥3 leakage-safe real networks spanning scientific domains
   (prefer post-cutoff/newly-constructed); secure any restricted-data access early.
4. Contact Kumo team re: correct schema; decide fair-comparison vs scope-out.
5. **Decision gate:** review pilot results → confirm L vs R, confirm PNAS/Nature vs
   honest down-target (§8) → only then scale to full multi-model × multi-domain run
   and start drafting.

---

## 8. Honest feasibility assessment (read this before committing)

The user's target is PNAS/Nature. I will not pretend the current state is close.

**Where we are:** a promising *pilot* — one model, single seed, mostly synthetic,
two confounds baked into the eval code, one SBM as the only "relational" test, and a
core idea that overlaps a published paper (GraphText). As an ML/tools result it is
respectable; as a general-science submission it is **pre-pilot**.

**What it would take to be a credible PNAS/Nature submission:**
- A finding that is **surprising, clean, and general** — the permutation collapse +
  training-free recovery, shown across **≥4 model families** and **≥3 scientific
  domains** with real (leakage-safe) data and tight CIs.
- **Airtight** memorization controls (the reviewer's first attack).
- Honest, strong classical baselines that the LLM is shown *against*, not instead of.
- This is realistically a **6–12 month empirical program**, not a write-up of the
  existing numbers.

**The two genuine paths to "broad significance":**
- The finding is robust and universal → a real shot at PNAS, possibly Nature-family
  (esp. if the limit is dramatic and the recovery striking).
- The finding is real but partial/model-dependent → still a strong paper, but the
  honest home is PNAS *Nexus* / Nature Machine Intelligence / Nature Human
  Behaviour / PNAS at best, or a top ML venue (LoG) — and that is a *good* outcome.

**My recommendation:** commit to the PNAS/Nature *standard of evidence* (it forces
the right experiments regardless of where it lands), run the strengthened pilot
across 2 models first, and **decide the final venue at the §7 decision gate from
data, not aspiration.** The fastest way to waste a year is to write toward Nature
before the cross-domain × multi-model evidence exists. Build the evidence; let it
choose the journal. The biggest single threat is not rigor — it's that GraphText
already verbalized graphs for LLMs, so the paper *must* live or die on the
finding-about-models (limit + recovery + generality), never on the method.
