"""Drive the local Qwen (Ollama) on clpc35 from THIS host (clpc95).

/home/scratch is NOT shared between clpc95 and clpc35 — only NFS home is. So we keep
the prompt files local and use clpc35 purely as a remote inference endpoint: build
the request JSON locally, pipe it over ssh to the Ollama API (`curl -d @-`), and
write the model's raw response to <root>/<domain>/ans/qwen/<stem>.ans. Scorers
regex-extract the '<id> <TOKEN>' lines, so extra chatter is harmless. temp=0.
Resumable (skips non-empty .ans). Sequential keeps the model warm in VRAM.

Usage: /home/scratch/fmsn-dev/.venv/bin/python eval/run_qwen.py [ROOT]
  ROOT default /home/scratch/bench_out/labelcurve ; env QMODEL to change model.
"""
import os, sys, glob, json, subprocess

MODEL = os.environ.get('QMODEL', 'qwen2.5:14b-instruct')
ROOT = sys.argv[1] if len(sys.argv) > 1 else '/home/scratch/bench_out/labelcurve'
NUM_CTX = int(os.environ.get('NUM_CTX', '16384'))
HOST = os.environ.get('QHOST', 'clpc35')
OUTSUB = os.environ.get('OUTSUB', 'qwen')          # ans/<OUTSUB>/ output subdir
DOMAINS = os.environ.get('DOMAINS', '')             # comma-sep domain filter, '' = all


def gen(prompt):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "num_ctx": NUM_CTX}})
    p = subprocess.run(
        ["ssh", HOST, "curl -s http://localhost:11434/api/generate -d @-"],
        input=body, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        raise RuntimeError(f"ssh/curl rc={p.returncode}: {p.stderr[:200]}")
    return json.loads(p.stdout)["response"]


def main():
    pats = ["seed*_k*.txt", "seed*_shot*.txt", "seed*.txt"]
    files = sorted({f for pat in pats for f in glob.glob(f"{ROOT}/*/{pat}")})
    if DOMAINS:
        keep = set(DOMAINS.split(','))
        files = [f for f in files if f.split('/')[-2] in keep]
    print(f"model={MODEL} host={HOST} root={ROOT} out={OUTSUB} files={len(files)}", flush=True)
    done = ok = 0
    for f in files:
        dom = f.split('/')[-2]
        stem = os.path.basename(f)[:-4]
        outdir = f"{ROOT}/{dom}/ans/{OUTSUB}"
        os.makedirs(outdir, exist_ok=True)
        outf = f"{outdir}/{stem}.ans"
        if os.path.exists(outf) and os.path.getsize(outf) > 0:
            done += 1
            continue
        try:
            resp = gen(open(f).read())
        except Exception as e:
            print(f"ERR {dom}/{stem}: {e}", flush=True)
            continue
        open(outf, 'w').write(resp)
        done += 1
        ok += 1
        if ok <= 3 or ok % 10 == 0:
            print(f"ok {dom}/{stem} ({done}/{len(files)})", flush=True)
    print(f"DONE {done}/{len(files)} (new {ok})", flush=True)


if __name__ == "__main__":
    main()
