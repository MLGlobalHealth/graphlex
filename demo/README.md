# graphlex demo

A lightweight localhost web demo that shows the core graphlex thesis **visually**:

> **NetworkX + graphlex compute graph structure DETERMINISTICALLY (no LLM)**,
> graphlex **verbalizes** that structure into prose via versioned threshold tables,
> and only **then** does an LLM reason over the verbalized facts.

The LLM never sees the raw graph — only the verbalized prose. The demo shows the
exact prompt sent so this is impossible to miss.

## Run

```bash
cd /home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex
/home/scratch/fmsn-dev/.venv/bin/python demo/app.py
```

Then open <http://127.0.0.1:5000>.

If Flask is missing from the venv:

```bash
/home/scratch/fmsn-dev/.venv/bin/python -m pip install flask
```

## What it does

1. **Input panel** — pick a built-in example graph (office network, Erdos-Renyi,
   Barabasi-Albert / scale-free, Watts-Strogatz / small-world, or a tiny benzene +
   nitro molecule), **or** paste your own edge list (`0-1, 1-2, 2-0`) with optional
   node attributes (`0:A, 1:A, 2:B`). A pasted edge list takes priority.
2. **Analyze** — builds the `nx.Graph`, runs `facts()` then `verbalize(focus="all")`,
   and shows side by side:
   - **LEFT**: a NetworkX `spring_layout` drawing (matplotlib → base64 PNG),
     colored by node attribute if present.
   - **RIGHT**: the deterministic verbalization, labeled *"Computed by NetworkX +
     graphlex — deterministic, no LLM"*, with the raw `facts()` dict (pretty JSON)
     in a collapsible `<details>` and a note that wording comes from the versioned
     thresholds table (`graphlex/thresholds.py`).
3. **Ask the LLM** — sends `verbalization + question` to an LLM and shows **both**
   the exact prompt (transparency: the LLM only sees the verbalized facts) and the
   answer.

## LLM backend (auto-selected, no API key needed for the default)

The backend is chosen automatically and is configurable via environment variables:

| Condition | Backend |
|---|---|
| `ANTHROPIC_API_KEY` set **and** `anthropic` SDK installed | Claude API (`CLAUDE_MODEL`, default `claude-3-5-sonnet-20241022`) |
| otherwise | **Qwen on `clpc35`** over ssh → ollama (`QMODEL`, default `qwen2.5:14b-instruct`) |
| neither reachable | no live call — the app returns the prompt with a note to copy it into your own LLM |

Qwen is reached exactly like the eval harness (`eval/run_qwen.py`): the request JSON
is piped over ssh into ollama's HTTP API:

```
ssh clpc35 'curl -s http://localhost:11434/api/generate -d @-'
```

This relies on passwordless ssh to `clpc35` (where ollama runs). If ollama is down:

```bash
ssh clpc35 'bash /home/scratch/ollama/start_ollama.sh'
```

Override env vars: `QHOST`, `QMODEL`, `CLAUDE_MODEL`.

## Notes

- The graphlex package itself is **not** modified; the repo root is added to
  `sys.path` at import time.
- Synthetic graphs and the spring layout use fixed seeds, so the demo is reproducible.
