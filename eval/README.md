# graphlex eval/ — experiment harnesses & results (lab notes)

Reproducible experiment scripts behind the result docs. **These are lab notes /
methods records, not manuscript text.** Two non-negotiable rules learned the hard
way (see SWEEP_RESULTS / LABEL_CURVE_RESULTS):

- **Always report BALANCED accuracy** (macro-recall) for the cross-domain
  classification comparison — raw accuracy rewards always-predict-majority on
  imbalanced sets, which the LLM (given balanced shots) can't do.
- **Always ≥3 seeds.** Single-seed Opus produced *false* failures (DBLP/NCI1/
  Fingerprint) that all vanished with 3 seeds.

## How to run
All scripts use the fmsn venv + graphlex on PYTHONPATH:
```
cd /home/scratch/fmsn-dev && source .venv/bin/activate
PYTHONPATH=/home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex python <eval/script.py>
```
The LLM arm is run either as Claude Code subagents (pure ICL, no key) or via Qwen
on clpc35 (`run_qwen.py`; Ollama at clpc35:11434, drive over ssh — `/home/scratch`
is NOT shared, so build JSON locally and pipe over ssh).

## Pipeline / what produces what
| script | produces | result doc |
|---|---|---|
| `synth_multiseed.py` + `score_synth.py` | raw→permuted→verbalized→anon arms (multi-seed) | PILOT_RESULTS.md |
| `fair_node_hard.py` | de-confounded relational node prediction | PILOT_RESULTS.md |
| `crossdomain_graphcls.py` | graph-cls vs classical + FM embeddings (IMDB/PROTEINS/NCI1) | CROSSDOMAIN_RESULTS.md |
| `mutag_elements.py` | element-naming + substructure ablation (MUTAG) | CROSSDOMAIN_RESULTS.md |
| `zero_label.py` | zero-/few-label capability curves (family, MUTAG) | ZEROLABEL_RESULTS.md |
| `label_curve.py` + `score_labelcurve.py` | label-efficiency crossover (family/PROTEINS/IMDB, 8 seeds) | LABEL_CURVE_RESULTS.md |
| `probe_datasets.py` | dataset inventory (35 TUDatasets, 9 domains) | results/probe_datasets.json |
| `sweep.py` + `score_sweep.py` + `balanced_rescore.py` | broad sweep, 30 datasets / 8 sciences | SWEEP_RESULTS.md |
| `run_qwen.py` | drive any Ollama model on clpc35 over a prompt dir (env QMODEL/OUTSUB/DOMAINS) | — |
| `make_figures.py` | regret heatmap, label-efficiency, zero-label PNGs | figures/ |
| `_common.py` | shared helpers: tolerant `parse_ans`, `bal_acc`, `fvec`, node/composition | (imported) |

## Notes
- Answer files live under `/home/scratch/bench_out/<exp>/...` (outside the repo);
  small manifests are snapshotted to `eval/results/`.
- The classical baseline = logreg on the `graphlex.facts()` feature vector (verified
  comparable to Jess's NetworkStatsEncoder). FM embeddings (graphpfn/gmn/kumorfm) are
  precomputed at `/home/scratch/real_fm_embeddings/`.
- `parse_ans` is tolerant of model format drift ("Query N CLASS", "0: CLASS", …) —
  use it everywhere; the old strict regex silently dropped whole seeds.
