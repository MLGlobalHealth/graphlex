# graphlex web demo (static, GitHub Pages)

A **backend-free** version of the demo: graphlex's `facts()` + `verbalize()` run
**in the browser** via [Pyodide](https://pyodide.org) (the real `graphlex` wheel +
NetworkX, compiled to WebAssembly). The user pastes/selects a graph, gets the exact
deterministic verbalization + a ready-to-paste prompt, and runs it in their own
ChatGPT / Claude / Gemini. No server, no API key, no data leaves the browser.

## Files
- `index.html` — the whole app (UI + embedded Python + Pyodide bootstrap).
- `graphlex-0.0.1-py3-none-any.whl` — the graphlex package, installed by micropip.
- `.nojekyll` — tell GitHub Pages to serve files (incl. the `.whl`) untouched.

## Test locally
Pyodide fetches the wheel over HTTP, so serve the folder (don't open via `file://`):
```
cd docs && python -m http.server 8000
# open http://localhost:8000   (first load ~10–15s while Pyodide + NetworkX download)
```

## Publish on GitHub Pages
1. Push the repo (the `docs/` folder must be on the default branch).
2. Repo → **Settings → Pages** → *Source*: **Deploy from a branch** →
   *Branch*: `main`, *Folder*: `/docs` → **Save**.
3. Live at **https://mlglobalhealth.github.io/graphlex/** (first deploy takes ~1 min).

## When graphlex changes
Rebuild the wheel so the site uses the latest code:
```
rm docs/graphlex-*.whl
python -m pip wheel . --no-deps -w docs/
```
