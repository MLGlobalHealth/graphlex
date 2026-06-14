"""graphlex demo — a lightweight localhost web app for showing colleagues the
core thesis VISUALLY:

    NetworkX + graphlex compute graph STRUCTURE deterministically (no LLM),
    graphlex VERBALIZES that structure into prose via versioned threshold tables,
    and only THEN does an LLM reason over the verbalized facts.

The LLM never sees the raw graph — only the deterministic verbalization. The app
shows the exact prompt sent so that point is impossible to miss.

Run:
    cd graphlex && /home/scratch/fmsn-dev/.venv/bin/python demo/app.py
Then open http://127.0.0.1:5000

LLM backend (auto-selected, no API key needed for the default):
    * if ANTHROPIC_API_KEY is set and the `anthropic` SDK is installed -> Claude API
    * else -> Qwen on clpc35 over ssh (ollama, model qwen2.5:14b-instruct)
    * else -> no live LLM; the app just returns the prompt to copy into your own LLM
"""
import base64
import io
import json
import os
import shutil
import subprocess
import sys

# --- make graphlex importable without installing it (do NOT modify the package) ---
GRAPHLEX_ROOT = "/home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex"
sys.path.insert(0, GRAPHLEX_ROOT)

import networkx as nx           # noqa: E402
import matplotlib               # noqa: E402
matplotlib.use("Agg")           # headless: render to PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from flask import Flask, request, jsonify  # noqa: E402
from graphlex import facts, verbalize       # noqa: E402

app = Flask(__name__)

# --- LLM backend configuration -------------------------------------------------
QWEN_HOST = os.environ.get("QHOST", "clpc35")
QWEN_MODEL = os.environ.get("QMODEL", "qwen2.5:14b-instruct")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")


# =============================================================================
# Example graph builders. Each returns (nx.Graph, attr_name_or_None).
# Synthetic graphs use a fixed seed so the demo is reproducible.
# =============================================================================
def build_office():
    """The structure x attributes worked example: a 10-person office network,
    siloed by department, bridged only by senior staff (graphlex/examples/office.py)."""
    people = {
        "Alice": ("Eng", "snr"), "Bob": ("Eng", "jr"), "Carol": ("Eng", "jr"),
        "Dan": ("Eng", "jr"), "Eve": ("Eng", "snr"),
        "Frank": ("Sales", "snr"), "Grace": ("Sales", "jr"), "Hank": ("Sales", "jr"),
        "Iris": ("Sales", "jr"), "Jack": ("Sales", "snr"),
    }
    edges = [
        ("Alice", "Bob"), ("Alice", "Carol"), ("Alice", "Dan"), ("Bob", "Carol"),
        ("Carol", "Dan"), ("Dan", "Eve"), ("Bob", "Eve"),
        ("Frank", "Grace"), ("Frank", "Hank"), ("Grace", "Iris"), ("Hank", "Jack"),
        ("Iris", "Jack"), ("Grace", "Jack"),
        ("Alice", "Frank"),  # the single cross-department tie
    ]
    G = nx.Graph()
    for p, (dept, sen) in people.items():
        G.add_node(p, dept=dept, seniority=sen)
    G.add_edges_from(edges)
    return G, "dept"


def build_erdos_renyi():
    """Random graph: edges placed independently at random (G(n,p))."""
    return nx.erdos_renyi_graph(40, 0.12, seed=7), None


def build_barabasi_albert():
    """Scale-free graph: preferential attachment -> a few high-degree hubs."""
    return nx.barabasi_albert_graph(40, 2, seed=7), None


def build_watts_strogatz():
    """Small-world graph: a ring lattice with a few rewired shortcuts -> high
    clustering but short paths."""
    return nx.watts_strogatz_graph(40, 4, 0.1, seed=7), None


def build_molecule():
    """A tiny molecule: a benzene ring (6 carbons) plus a nitro group (-NO2),
    with each atom's element stored as the 'atom' node attribute."""
    G = nx.Graph()
    # benzene ring: C0..C5
    ring = [f"C{i}" for i in range(6)]
    for c in ring:
        G.add_node(c, atom="C")
    for i in range(6):
        G.add_edge(ring[i], ring[(i + 1) % 6])
    # nitro group attached to C0: C0 - N, N = O, N - O
    G.add_node("N", atom="N")
    G.add_node("O1", atom="O")
    G.add_node("O2", atom="O")
    G.add_edge("C0", "N")
    G.add_edge("N", "O1")
    G.add_edge("N", "O2")
    return G, "atom"


EXAMPLES = {
    "office": ("Office network (10 people, dept + seniority)", build_office),
    "erdos_renyi": ("Erdos-Renyi (random)", build_erdos_renyi),
    "barabasi_albert": ("Barabasi-Albert (scale-free)", build_barabasi_albert),
    "watts_strogatz": ("Watts-Strogatz (small-world)", build_watts_strogatz),
    "molecule": ("Molecule (benzene + nitro group)", build_molecule),
}

# Graphs without node attributes get the structure question; attributed graphs
# get the attribute question.
DEFAULT_Q_SYNTH = "Is this network small-world, scale-free, or random, and why?"
DEFAULT_Q_ATTR = "Which nodes are the brokers and is it homophilous?"


# =============================================================================
# Free-text input parsing
# =============================================================================
def parse_edge_list(text):
    """Parse '0-1, 1-2, 2-0' (comma/newline/semicolon separated) into edges."""
    edges = []
    for chunk in text.replace("\n", ",").replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        # accept '0-1', '0 1', '0,1' already split; try '-' then whitespace
        if "-" in chunk:
            a, b = chunk.split("-", 1)
        else:
            parts = chunk.split()
            if len(parts) != 2:
                continue
            a, b = parts
        a, b = a.strip(), b.strip()
        if a and b:
            edges.append((a, b))
    return edges


def parse_node_attrs(text):
    """Parse '0:A, 1:A, 2:B' into {node: value}."""
    attrs = {}
    for chunk in text.replace("\n", ",").replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        node, val = chunk.split(":", 1)
        node, val = node.strip(), val.strip()
        if node and val:
            attrs[node] = val
    return attrs


def build_from_text(edge_text, attr_text):
    """Build an nx.Graph from pasted edge list + optional node attributes."""
    edges = parse_edge_list(edge_text)
    if not edges:
        raise ValueError("No valid edges parsed. Use a format like '0-1, 1-2, 2-0'.")
    attrs = parse_node_attrs(attr_text) if attr_text else {}
    G = nx.Graph()
    G.add_edges_from(edges)
    attr_name = None
    if attrs:
        attr_name = "group"
        for node, val in attrs.items():
            if node in G:                       # only attributes for nodes that exist
                G.nodes[node][attr_name] = val
        # nodes without a provided value get a placeholder so the attr is well-defined
        for node in G.nodes:
            G.nodes[node].setdefault(attr_name, "?")
    return G, attr_name


# =============================================================================
# Drawing: spring layout -> matplotlib -> base64 PNG, colored by attribute.
# =============================================================================
def draw_graph_png(G, attr_name):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    pos = nx.spring_layout(G, seed=42)  # fixed seed: same picture every run

    # Light, colorblind-friendly palette (ColorBrewer "Paired" light members) so
    # BLACK node labels stay readable. First two = light blue / light orange.
    PALETTE = ["#a6cee3", "#fdbf6f", "#b2df8a", "#cab2d6", "#fb9a99",
               "#ffff99", "#fccde5", "#ccebc5"]
    legend_handles = None
    if attr_name:
        vals = nx.get_node_attributes(G, attr_name)
        groups = sorted({str(v) for v in vals.values()})
        color_of = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
        node_colors = [color_of[str(vals.get(node, "?"))] for node in G.nodes]
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w", label=g,
                       markerfacecolor=color_of[g], markeredgecolor="#555",
                       markersize=10)
            for g in groups
        ]
    else:
        node_colors = "#a6cee3"   # light blue

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.45, width=1.0, edge_color="#888")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=440, linewidths=1.0, edgecolors="#555")
    # only label small graphs to keep the picture readable
    if G.number_of_nodes() <= 25:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="#000")
    if legend_handles:
        ax.legend(handles=legend_handles, title=attr_name, fontsize=8,
                  loc="best", framealpha=0.9)
    ax.set_axis_off()
    fig.tight_layout(pad=0.2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def facts_to_jsonable(f):
    """facts() includes a '_communities' key holding python sets, which aren't
    JSON-serializable. Convert sets to sorted lists for pretty display."""
    def conv(o):
        if isinstance(o, set):
            return sorted(o, key=str)
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        return o
    return conv(f)


# =============================================================================
# LLM backends. Each returns (answer_text, backend_label).
# =============================================================================
def ask_claude(prompt):
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=800,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return text.strip(), f"Claude API ({CLAUDE_MODEL})"


def ask_qwen(prompt):
    """Call Qwen on QWEN_HOST via ollama, piping the request JSON over ssh —
    the same pattern the eval harness uses (eval/run_qwen.py)."""
    body = json.dumps({
        "model": QWEN_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192},
    })
    p = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", QWEN_HOST,
         "curl -s -m 240 http://localhost:11434/api/generate -d @-"],
        input=body, capture_output=True, text=True, timeout=300,
    )
    if p.returncode != 0:
        raise RuntimeError(f"ssh/curl rc={p.returncode}: {p.stderr[:300]}")
    resp = json.loads(p.stdout).get("response", "")
    return resp.strip(), f"Qwen via {QWEN_HOST} ({QWEN_MODEL})"


CLAUDE_CLI = os.environ.get("CLAUDE_CLI", "claude")


def ask_claude_cli(prompt):
    """Use the installed Claude Code CLI in headless print mode (`claude -p`).
    No API key needed — uses the existing Claude Code auth. The frontier model,
    so the best demo backend when available."""
    p = subprocess.run(
        [CLAUDE_CLI, "-p", prompt],
        capture_output=True, text=True, timeout=300,
    )
    if p.returncode != 0:
        raise RuntimeError(f"claude -p rc={p.returncode}: {p.stderr[:300]}")
    return p.stdout.strip(), "Claude Code CLI (claude -p)"


def run_llm(prompt):
    """Pick a backend by availability. Returns (answer, backend_label, is_live).
    Preference: Claude Code CLI (`claude -p`, no key) -> Claude API (if key) ->
    Qwen on clpc35 -> copy-paste. Set GRAPHLEX_NO_CLAUDE_CLI=1 to skip the CLI."""
    errs = []

    if not os.environ.get("GRAPHLEX_NO_CLAUDE_CLI") and shutil.which(CLAUDE_CLI):
        try:
            ans, label = ask_claude_cli(prompt)
            if ans:
                return ans, label, True
            raise RuntimeError("empty response")
        except Exception as e:
            errs.append(f"claude-cli: {str(e)[:160]}")

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            ans, label = ask_claude(prompt)
            return ans, label, True
        except Exception as e:
            errs.append(f"claude-api: {str(e)[:160]}")

    try:
        ans, label = ask_qwen(prompt)
        if ans:
            return ans, label, True
        raise RuntimeError("empty response")
    except Exception as e:
        errs.append(f"qwen: {str(e)[:160]}")

    note = ("No live LLM reachable — copy the prompt above into your own LLM. ("
            + "; ".join(errs) + ")")
    return note, "none (no backend reachable)", False


# =============================================================================
# HTML (single inline template, no JS framework — just fetch()).
# =============================================================================
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>graphlex demo</title>
<style>
  :root { --blue:#2b6cb0; --green:#2f855a; --bg:#f7fafc; --card:#fff; --line:#e2e8f0; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0; background: var(--bg); color: #1a202c; line-height: 1.5; }
  header { background: #1a202c; color: #fff; padding: 18px 26px; }
  header h1 { margin: 0; font-size: 21px; }
  header p { margin: 6px 0 0; color: #cbd5e0; font-size: 14px; max-width: 900px; }
  .pipeline { font-size: 13px; color: #a0aec0; margin-top: 8px; }
  .pipeline b { color: #fff; }
  main { max-width: 1180px; margin: 0 auto; padding: 22px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
          padding: 18px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.05); }
  .card h2 { margin: 0 0 12px; font-size: 16px; }
  label { font-weight: 600; font-size: 13px; display: block; margin: 10px 0 4px; }
  select, textarea, input[type=text] { width: 100%; padding: 8px; border: 1px solid var(--line);
          border-radius: 6px; font-size: 13px; font-family: inherit; }
  textarea { resize: vertical; }
  .row { display: flex; gap: 18px; flex-wrap: wrap; }
  .row > div { flex: 1; min-width: 240px; }
  button { background: var(--blue); color: #fff; border: 0; padding: 10px 20px;
           border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; margin-top: 14px; }
  button:hover { filter: brightness(1.08); }
  button.ask { background: var(--green); }
  button:disabled { opacity: .55; cursor: progress; }
  .results { display: flex; gap: 20px; flex-wrap: wrap; }
  .results > div { flex: 1; min-width: 320px; }
  img.graph { max-width: 100%; border: 1px solid var(--line); border-radius: 8px; background:#fff; }
  .badge { display:inline-block; background:#ebf8ff; color:#2b6cb0; border:1px solid #bee3f8;
           font-size:11px; font-weight:700; padding:3px 8px; border-radius:999px; margin-bottom:8px;
           text-transform:uppercase; letter-spacing:.04em; }
  .badge.det { background:#f0fff4; color:#276749; border-color:#c6f6d5; }
  .verb { white-space: pre-wrap; background:#f8fafc; border:1px solid var(--line);
          border-radius:8px; padding:12px; font-size:14px; }
  .note { font-size:12px; color:#718096; margin-top:8px; }
  pre { white-space: pre-wrap; word-break: break-word; background:#1a202c; color:#e2e8f0;
        padding:12px; border-radius:8px; font-size:12px; overflow:auto; max-height:380px; }
  details summary { cursor:pointer; font-weight:600; font-size:13px; margin-top:10px; }
  .promptwrap { position:relative; }
  .prompt { white-space: pre-wrap; background:#fffaf0; border:1px solid #feebc8;
            border-radius:8px; padding:12px; padding-top:34px; font-size:13px; }
  .copybtn { position:absolute; top:6px; right:6px; font-size:12px; cursor:pointer;
             background:#fff; color:#c05621; border:1px solid #feb2a8;
             border-radius:6px; padding:3px 9px; }
  .copybtn:hover { background:#fffaf0; }
  .answer { white-space: pre-wrap; background:#f0fff4; border:1px solid #c6f6d5;
            border-radius:8px; padding:12px; font-size:14px; }
  .hidden { display:none; }
  .err { color:#c53030; font-weight:600; }
  .muted { color:#718096; font-size:13px; }
  .ctxlabel { font-size:12px; font-weight:700; color:#dd6b20; margin:14px 0 4px; }
</style>
</head>
<body>
<header>
  <h1>graphlex demo &mdash; deterministic structure, then LLM reasoning</h1>
  <p>NetworkX + graphlex compute graph structure and verbalize it
     <b>deterministically (no LLM)</b>. The LLM only ever sees that verbalized prose.</p>
  <div class="pipeline">Pipeline:&nbsp;
     <b>NetworkX</b> &rarr; <b>graphlex.facts()</b> (exact numbers) &rarr;
     <b>graphlex.verbalize()</b> (thresholds &rarr; prose) &rarr; <b>LLM</b> reasons</div>
</header>

<main>
  <div class="card">
    <h2>1. Choose an input graph</h2>
    <div class="row">
      <div>
        <label for="example">Built-in example</label>
        <select id="example" onchange="analyze()">__OPTIONS__</select>
        <p class="muted">Or paste your own graph on the right (used instead of the
           example if non-empty).</p>
      </div>
      <div>
        <label for="edges">Edge list (e.g. <code>0-1, 1-2, 2-0</code>)</label>
        <textarea id="edges" rows="3" placeholder="0-1, 1-2, 2-0, 2-3"></textarea>
        <label for="attrs">Optional node attributes (e.g. <code>0:A, 1:A, 2:B</code>)</label>
        <textarea id="attrs" rows="2" placeholder="0:A, 1:A, 2:B, 3:B"></textarea>
      </div>
    </div>
    <button id="analyzeBtn" onclick="analyze()">Analyze</button>
  </div>

  <div id="analyzeCard" class="card hidden">
    <h2>2. Deterministic analysis</h2>
    <div class="results">
      <div>
        <div class="badge">NetworkX spring layout</div>
        <img id="graphImg" class="graph" alt="graph drawing">
      </div>
      <div>
        <div class="badge det">Computed by NetworkX + graphlex &mdash; deterministic, no LLM</div>
        <div id="verb" class="verb"></div>
        <p class="note">Wording (e.g. &ldquo;strongly homophilous&rdquo;, &ldquo;scale-free&rdquo;
           cues) comes from the <b>versioned thresholds table</b>
           (<code>graphlex/thresholds.py</code>) &mdash; edit a number there and the prose
           changes everywhere, reproducibly.</p>
        <details>
          <summary>Show raw facts() structure (JSON)</summary>
          <pre id="factsJson"></pre>
        </details>
      </div>
    </div>
  </div>

  <div id="askCard" class="card hidden">
    <h2>3. Ask the LLM</h2>
    <p class="muted">The LLM <b>only sees the verbalized facts</b>, never the raw
       graph structure. The prompt that will be sent is shown live below and updates
       as you edit your question.</p>
    <label for="question">Question</label>
    <input type="text" id="question" oninput="updatePreview()">
    <button id="askBtn" class="ask" onclick="ask()">Ask the LLM</button>
    <div id="askOut" class="hidden">
      <div class="ctxlabel" id="promptLabel"></div>
      <div class="promptwrap">
        <button class="copybtn" onclick="copyPrompt(this)">Copy</button>
        <div id="prompt" class="prompt"></div>
      </div>
      <div class="ctxlabel" id="answerLabel">LLM answer:</div>
      <div id="answer" class="answer"></div>
    </div>
  </div>
</main>

<script>
let lastVerbalization = "";
const PREVIEW_LABEL = "Prompt preview — this exact text is sent when you click “Ask the LLM” (verbalized facts + your question; no raw structure):";
const SENT_LABEL = "✓ Exact prompt sent to the LLM (verbalized facts + your question; no raw structure):";

// Mirrors the server-side prompt built in /ask EXACTLY — keep in sync.
function buildPrompt(verb, q) {
  return "You are analyzing a graph. Here is a factual, deterministically-computed " +
    "description of its structure (you do NOT have the raw graph, only this " +
    "verbalized summary):\n\n" + verb + "\n\nQuestion: " + (q || "") + "\n\n" +
    "Answer concisely, citing the specific numbers above that justify your reasoning.";
}
function copyPrompt(btn) {
  const txt = document.getElementById('prompt').textContent;
  const done = () => { const o = btn.textContent; btn.textContent = 'Copied!';
                       setTimeout(() => { btn.textContent = o; }, 1200); };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(txt).then(done).catch(() => fallbackCopy(txt, done));
  } else { fallbackCopy(txt, done); }
}
function fallbackCopy(txt, done) {
  const ta = document.createElement('textarea');
  ta.value = txt; ta.style.position = 'fixed'; ta.style.opacity = '0';
  document.body.appendChild(ta); ta.select();
  try { document.execCommand('copy'); done(); } catch (e) {}
  document.body.removeChild(ta);
}
function updatePreview() {
  if (!lastVerbalization) return;
  document.getElementById('prompt').textContent =
    buildPrompt(lastVerbalization, document.getElementById('question').value);
  document.getElementById('promptLabel').textContent = PREVIEW_LABEL;
}

// Minimal, safe markdown for LLM answers: escape HTML first, then render
// **bold**, *italic*, and `code`. Newlines are preserved by CSS pre-wrap.
function mdLite(s) {
  s = String(s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  s = s.replace(/\*\*([^*]+?)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^*])\*([^*\n]+?)\*(?!\*)/g, '$1<em>$2</em>');
  s = s.replace(/`([^`]+?)`/g, '<code>$1</code>');
  return s;
}

async function analyze() {
  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true; btn.textContent = 'Analyzing…';
  try {
    const body = {
      example: document.getElementById('example').value,
      edges: document.getElementById('edges').value,
      attrs: document.getElementById('attrs').value,
    };
    const r = await fetch('/analyze', {method:'POST', headers:{'Content-Type':'application/json'},
                                       body: JSON.stringify(body)});
    const d = await r.json();
    if (d.error) { alert('Error: ' + d.error); return; }
    document.getElementById('graphImg').src = 'data:image/png;base64,' + d.image;
    document.getElementById('verb').textContent = d.verbalization;
    document.getElementById('factsJson').textContent = d.facts_json;
    lastVerbalization = d.verbalization;
    document.getElementById('question').value = d.default_question;
    document.getElementById('analyzeCard').classList.remove('hidden');
    document.getElementById('askCard').classList.remove('hidden');
    // show the live prompt preview immediately; clear any prior answer
    document.getElementById('askOut').classList.remove('hidden');
    updatePreview();
    document.getElementById('answerLabel').textContent = 'LLM answer (click “Ask the LLM”):';
    document.getElementById('answer').textContent = '';
  } catch (e) {
    alert('Request failed: ' + e);
  } finally {
    btn.disabled = false; btn.textContent = 'Analyze';
  }
}

async function ask() {
  const btn = document.getElementById('askBtn');
  btn.disabled = true; btn.textContent = 'Asking the LLM…';
  document.getElementById('askOut').classList.remove('hidden');
  updatePreview();
  document.getElementById('answerLabel').textContent = 'LLM answer:';
  document.getElementById('answer').textContent = '…';
  try {
    const body = { verbalization: lastVerbalization,
                   question: document.getElementById('question').value };
    const r = await fetch('/ask', {method:'POST', headers:{'Content-Type':'application/json'},
                                   body: JSON.stringify(body)});
    const d = await r.json();
    document.getElementById('prompt').textContent = d.prompt || '(none)';
    document.getElementById('promptLabel').textContent = SENT_LABEL;
    document.getElementById('answerLabel').textContent =
        'LLM answer  —  backend: ' + (d.backend || 'unknown');
    document.getElementById('answer').innerHTML = mdLite(d.answer || '(empty)');
  } catch (e) {
    document.getElementById('answer').textContent = 'Request failed: ' + e;
  } finally {
    btn.disabled = false; btn.textContent = 'Ask the LLM';
  }
}

// Show the default example's graph immediately on page load (no click needed).
window.addEventListener('DOMContentLoaded', analyze);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    options = "\n".join(
        f'<option value="{key}">{label}</option>' for key, (label, _) in EXAMPLES.items()
    )
    return PAGE.replace("__OPTIONS__", options)


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True, silent=True) or {}
    edge_text = (data.get("edges") or "").strip()
    attr_text = (data.get("attrs") or "").strip()
    example = data.get("example") or "office"
    try:
        # Pasted edge list takes priority if provided; else the chosen example.
        if edge_text:
            G, attr_name = build_from_text(edge_text, attr_text)
        else:
            if example not in EXAMPLES:
                raise ValueError(f"Unknown example: {example}")
            G, attr_name = EXAMPLES[example][1]()

        if G.number_of_nodes() == 0:
            raise ValueError("Graph is empty.")

        # --- the deterministic core: facts() then verbalize(). No LLM. ---
        node_attrs = [attr_name] if attr_name else None
        f = facts(G, node_attrs=node_attrs)
        verbalization = verbalize(f, focus="all")

        image = draw_graph_png(G, attr_name)
        facts_json = json.dumps(facts_to_jsonable(f), indent=2, default=str)
        default_q = DEFAULT_Q_ATTR if attr_name else DEFAULT_Q_SYNTH

        return jsonify({
            "verbalization": verbalization,
            "facts_json": facts_json,
            "image": image,
            "default_question": default_q,
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True, silent=True) or {}
    verbalization = (data.get("verbalization") or "").strip()
    question = (data.get("question") or "").strip() or DEFAULT_Q_SYNTH
    if not verbalization:
        return jsonify({"error": "Run Analyze first."})

    # The prompt is exactly: verbalized facts + the question. Nothing else — and
    # crucially, NOT the raw graph. This is the whole point of the demo.
    prompt = (
        "You are analyzing a graph. Here is a factual, deterministically-computed "
        "description of its structure (you do NOT have the raw graph, only this "
        "verbalized summary):\n\n"
        f"{verbalization}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely, citing the specific numbers above that justify your reasoning."
    )

    answer, backend, _is_live = run_llm(prompt)
    return jsonify({"prompt": prompt, "answer": answer, "backend": backend})


if __name__ == "__main__":
    # Port is configurable via PORT (default 5000). On macOS, port 5000 is taken
    # by the AirPlay Receiver (ControlCenter), so set e.g. PORT=5001 there.
    port = int(os.environ.get("PORT", "5000"))
    url = f"http://127.0.0.1:{port}"
    print("=" * 60)
    print("  graphlex demo running at:  " + url)
    if not os.environ.get("GRAPHLEX_NO_CLAUDE_CLI") and shutil.which(CLAUDE_CLI):
        _backend = "Claude Code CLI (claude -p)"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        _backend = "Claude API"
    else:
        _backend = f"Qwen via {QWEN_HOST} ({QWEN_MODEL})"
    print("  LLM backend: " + _backend)
    print("=" * 60, flush=True)
    app.run(host="127.0.0.1", port=port, debug=False)
