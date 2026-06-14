# graphlex — strengthened pilot results + venue verdict (2026-06-14)

Run after the PNAS/Nature retarget. Fixes the two confounds found in the original
eval code, adds multi-seed + multi-model generality, and answers the question:
**PNAS territory or PNAS Nexus territory?**

Method: the "LLM" arm is Claude Code subagents doing **pure in-context reasoning**
— each subagent's only tool uses were Read (load the prompt) and Write (save its
answers): **2 tool uses each, verified, zero computation tools**. Models via the
Claude tiers (Opus 4.8, Sonnet 4.6, Haiku 4.5). Classical baselines (logreg,
neighbor-majority) computed locally in the fmsn-dev venv. Harness:
`eval/synth_multiseed.py`, `eval/score_synth.py`, `eval/fair_node_hard.py`.

---

## Experiment 1 — synthetic graph-family ID (ER vs BA vs WS)
5 seeds × 30 queries/seed (chance 0.333). logreg-on-features ref = **0.920 ± 0.078**.

| arm | what it tests | Haiku 4.5 | Sonnet 4.6 | Opus 4.8 |
|---|---|---|---|---|
| `raw` | edge list, original node labels | 0.787 ± 0.100 | 0.940 ± 0.044 | 0.953 ± 0.027 |
| `raw_perm` | edge list, **node labels permuted** (artifact removed) | **0.387 ± 0.034** | 0.620 ± 0.105 | 0.553 ± 0.086 |
| `counts_only` | verbalized n/m/density only (ladder floor) | — | — | 0.707 ± 0.188 |
| `verbal` | full computed-feature verbalization, **named** families | — | — | 0.920 ± 0.034 |
| `verbal_anon` | full verbalization, **anonymized** families A/B/C | 0.867 ± 0.037 | 0.907 ± 0.049 | 0.940 ± 0.039 |

**What it says (clean and robust):**
1. **The verbalization win is genuine ICL, not memorization.** `verbal_anon`
   (families relabeled A/B/C per seed — the model *cannot* use any "scale-free"
   prior) scores 0.87–0.94 across all three models, ≈ the named version and ≈
   logreg. The memorization confound that threatened the original headline is
   **ruled out** for this task. ✓ (This is the single most important new result.)
2. **Raw-edge prompting leans heavily on node-ordering artifacts.** Permuting
   labels drops accuracy by 0.32–0.40 for every model.
3. **The collapse is capacity-dependent.** Haiku falls essentially to chance
   (0.387 ≈ 0.333) — no genuine structural reading at all. Opus/Sonnet retain
   *partial* structure-reading (0.55–0.62). So the pilot's "collapses to chance"
   was true for a small model but **overstated for frontier models** — they read
   some real structure from raw edges, just badly degraded.
4. **Verbalized-feature ICL ≫ permuted-raw ICL** for every model (gap +0.29 to
   +0.48). Stating computed structure in language recovers competence that raw
   edges do not provide.
5. **Honest deflation:** `counts_only` (just n/m/density) already gets 0.707 —
   much of the signal is cheap size/density, not deep structure. The richer
   features add ~+0.21 on top. And the LLM only **matches** logreg (0.92), never
   beats it.

## Experiment 2 — fair relational node prediction (SBM org net), tautology fixed
3 seeds × 60 held-out/seed, 50% labeled (chance 0.25). Original confound: the
prompt handed the model a pre-aggregated department tally — the exact statistic
neighbor-majority argmaxes. Two arms isolate it.

| method | acc |
|---|---|
| neighbor-majority over known neighbors (trivial baseline) | **0.756 ± 0.021** |
| logreg (struct + neighbor counts) | 0.706 ± 0.016 |
| Opus `tally` (handed the aggregated counts) [confounded] | 0.722 ± 0.034 |
| Opus `nolist` (unaggregated list, some neighbors unlabeled) [de-confounded] | 0.728 ± 0.044 |

**What it says (a genuine weakness, reported honestly):**
- **The LLM does not beat — is slightly below — a one-line heuristic.** On
  relational node prediction, verbalize+LLM ≈ neighbor-majority ≈ logreg ≈
  0.70–0.76.
- **Removing the tautology barely changed the LLM (0.722 → 0.728):** even when not
  handed the tally, Opus aggregates the raw neighbor list itself just as well — but
  it caps at the neighbor-majority ceiling and does not exceed it. The LLM has no
  special relational-reasoning edge here.

---

## VERDICT: PNAS Nexus territory, not PNAS-main / Nature — *as it currently stands*

**What is solid (and genuinely nice):**
- A clean, multi-seed, multi-model dissociation: **raw-structure reading is
  artifact-driven and permutation-fragile; computed-structure-in-language is
  permutation-invariant and recovers competence — and this is not memorization**
  (the anonymized control is the proof). The capacity-scaling of the collapse
  (Haiku→chance, frontier→partial) is a crisp, quantitative bonus.

**What caps it below PNAS-main / Nature today:**
1. **No performance win over classical.** The LLM only *matches* logreg /
   neighbor-majority everywhere, and is beaten by a trivial heuristic on node
   prediction. A general-science "LLMs unlock network analysis" claim needs the
   LLM to *enable something classical methods can't* — not yet shown.
2. **All-synthetic.** ER/BA/WS + one SBM. No real attributed network, no
   cross-domain evidence — both mandatory for PNAS/Nature breadth.
3. **The "limit" half is partly known** (NLGraph showed shortcut reliance) and is
   **model-dependent** (frontier models don't fully collapse), which softens the
   "LLMs are blind to structure" headline.
4. **GraphText overlap** still constrains the novelty of the constructive half.

**Bottom line for the author:** what you have is a real, well-controlled finding
that belongs in **PNAS Nexus, a strong ML venue (LoG / NeurIPS D&B), or a
computational-social-science journal** — a confident submission *today* after
writing up. It is **not yet** a PNAS-main or Nature paper, and the gap is not
"more polish" — it is two specific things that would have to come back positive:

### What would move it INTO PNAS-main / Nature territory (the bet)
- **A real cross-domain demonstration** (≥3 scientific fields, leakage-safe / post-
  cutoff networks) where the permutation-invariance dissociation **holds** and the
  capacity-scaling trend **replicates** — ideally including non-Claude models
  (GPT/Gemini/open-weight) so the claim is about *LLMs*, not Claude.
- **At least one place the verbalize-LLM does something classical methods cannot**
  — e.g. zero-shot structure-as-finding-vs-null with a correct natural-language
  interpretation a practitioner would act on, or transfer to a task with no
  training labels where logreg can't run. Matching logreg is a Nexus result;
  *enabling a new analysis* is a PNAS result.

If both come back positive → escalate to PNAS-main, possibly Nature-family on the
"what LLMs (don't) understand about relational structure" framing. If only the
first → PNAS Nexus is the honest, strong home. If neither → still a good ML / comp-
soc-sci paper.

---

## Caveats on this pilot (don't over-read)
- Multi-model = Claude tiers only; a true "about LLMs" claim needs other vendors.
- Synthetic graphs; small n per seed (30 / 60 queries).
- `counts_only` and `verbal`(named) were run on Opus only (ladder/upper-bound
  anchors); the cross-model rows use the 3 decisive arms.
- Subagent runs are one draw per (model, seed, arm); ICL has its own sampling
  noise not captured by the seed-wise std.
