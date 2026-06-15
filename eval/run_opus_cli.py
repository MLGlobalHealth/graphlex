"""Drive Opus over prompt files via the local `claude -p` CLI (no API key needed).

Mirrors run_qwen.py but uses the Claude Code CLI as a pure-ICL endpoint: each
prompt file is sent to a FRESH `claude -p` invocation (single turn, no tools), so
the model sees only that prompt — equivalent to the pure-ICL subagent arm. The raw
stdout is written to <root>/<rep>/ans/<OUTSUB>/<stem>.ans; the tolerant
_common.parse_ans extracts the '<id> <CLASS>' lines, so extra chatter is harmless.
Resumable (skips non-empty .ans). Sequential.

Usage: python eval/run_opus_cli.py <ROOT>
  ROOT e.g. /home/scratch/bench_out/node_icl/Cora  (globs ROOT/<rep>/seed*_k*.txt)
  env: REP (comma-sep rep filter e.g. 'readable'), OUTSUB (default 'opus'),
       CMODEL (default 'opus'), REDO_UNPARSED.
"""
import os, sys, glob, subprocess

ROOT = sys.argv[1]
MODEL = os.environ.get('CMODEL', 'opus')
OUTSUB = os.environ.get('OUTSUB', 'opus')
REPS = os.environ.get('REP', '')                  # comma-sep rep filter, '' = all
REDO_UNPARSED = os.environ.get('REDO_UNPARSED')
CLAUDE = os.environ.get('CLAUDE_CLI', '/users/setman/.local/bin/claude')

if REDO_UNPARSED:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _common import parse_ans

NOTOOLS = ("\n\nRespond with ONLY the answer lines, one per query, each exactly "
           "'<id> <CLASS>'. No explanation, no tools, no other text.")


def gen(prompt):
    p = subprocess.run(
        [CLAUDE, "-p", "--model", MODEL, "--allowed-tools", ""],
        input=prompt + NOTOOLS, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        raise RuntimeError(f"claude rc={p.returncode}: {p.stderr[:300]}")
    return p.stdout


def main():
    files = sorted(glob.glob(f"{ROOT}/*/seed*_k*.txt"))
    if REPS:
        keep = set(REPS.split(','))
        files = [f for f in files if f.split('/')[-2] in keep]
    print(f"model={MODEL} root={ROOT} out={OUTSUB} files={len(files)}", flush=True)
    done = ok = 0
    for f in files:
        rep = f.split('/')[-2]
        stem = os.path.basename(f)[:-4]
        outdir = f"{ROOT}/{rep}/ans/{OUTSUB}"
        os.makedirs(outdir, exist_ok=True)
        outf = f"{outdir}/{stem}.ans"
        if os.path.exists(outf) and os.path.getsize(outf) > 0:
            if not REDO_UNPARSED or parse_ans(outf):
                done += 1
                continue
        try:
            resp = gen(open(f).read())
        except Exception as e:
            print(f"ERR {rep}/{stem}: {e}", flush=True)
            continue
        open(outf, 'w').write(resp)
        done += 1
        ok += 1
        print(f"ok {rep}/{stem} ({done}/{len(files)})", flush=True)
    print(f"DONE {done}/{len(files)} (new {ok})", flush=True)


if __name__ == "__main__":
    main()
