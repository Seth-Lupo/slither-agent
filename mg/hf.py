"""Hugging Face transfers. Node /tmp is wiped when the job ends, so everything durable round-trips through HF.
Data goes to the model repo once. Checkpoints and dashboards go to a private bucket and are overwritten in place:
a repo keeps every checkpoint version in its history, which filled the private storage quota (2026-10-17).

python hf.py fetch     # source videos + labels -> ROOT/src
python hf.py restore   # tokens, tokenizer, checkpoints, dashboard from a previous job (if any)
python hf.py loop      # back up every MG_HF_EVERY s (2 h) (data once, checkpoints + dashboard each time)
"""
import fnmatch, json, os, sys, time
from huggingface_hub import HfApi, snapshot_download
import common as C

DATASET = "mundoamundo/slither-wam-video-actions"
REPO = "mundoamundo/slither-maskgit"          # private model repo for this project (data)
BUCKET = "mundoamundo/slither-maskgit-live"   # private bucket, no history (checkpoints + dashboards)
DASH_AT = os.environ.get("MG_HF_DASH", "dash")  # a second job backs its dashboard and checkpoints up elsewhere,
RUNS_AT = os.environ.get("MG_HF_RUNS", "runs")  # so the main job's stale copies can't overwrite them
RESTORE = os.environ.get("MG_HF_RESTORE", "data/*,runs/*,dash/**").split(",")
EVERY = float(os.environ.get("MG_HF_EVERY", 2 * 3600))
SKIP = set(filter(None, os.environ.get("MG_HF_SKIP", "").split(",")))   # runs another job owns (stale copies here)
api = HfApi()


def fetch():
    snapshot_download(DATASET, repo_type="dataset", local_dir=C.SRC, max_workers=16,
                      allow_patterns=["prepared/*.mp4", "raw/*.mp4", "actions/*", "reports/*", "metadata/*", "heads_v1/tracks/*"])


def restore():
    try: files = api.list_repo_files(REPO)
    except Exception: print("no backup repo yet"); return
    if "data/tokens.npy" in files:                  # data, plus pre-bucket checkpoints/dashboards as a fallback
        snapshot_download(REPO, local_dir=C.ROOT, max_workers=8, allow_patterns=RESTORE)
        print("restored", len(files), "repo files")
    try: live = [f for f in api.list_bucket_tree(BUCKET, recursive=True) if hasattr(f, "size")]
    except Exception: print("no bucket yet"); return
    want = [f for f in live if any(fnmatch.fnmatch(f.path, p) for p in RESTORE)]   # newer than the repo copies
    api.download_bucket_files(BUCKET, [(f, f"{C.ROOT}/{f.path}") for f in want])
    print("restored", len(want), "bucket files")


def backup():
    api.create_repo(REPO, private=True, exist_ok=True); api.create_bucket(BUCKET, private=True, exist_ok=True)
    have = set(api.list_repo_files(REPO))
    data = ["tokens.npy", "acts.npy", "cut.npy", "index.json", "meta.json", "evalclips.npz", "tokenizer.pt"] + [f for f in ("deaths.npz",) if os.path.exists(f"{C.DATA}/{f}")]
    if os.path.exists(f"{C.DATA}/tokens.npy") and not {f"data/{f}" for f in data} <= have:
        api.upload_folder(repo_id=REPO, folder_path=C.DATA, path_in_repo="data", allow_patterns=data, commit_message="data")
    up = []                                         # (local, bucket path) for files changed since the last backup
    for r in sorted(os.listdir(f"{C.ROOT}/runs")) if os.path.isdir(f"{C.ROOT}/runs") else []:
        if r in SKIP: continue
        up += [(f"{C.ROOT}/runs/{r}/{f}", f"{RUNS_AT}/{r}/{f}") for f in ("ckpt.pt", "DONE") if os.path.exists(f"{C.ROOT}/runs/{r}/{f}")]
    for d, _, fs in os.walk(C.DASH):
        rel = os.path.relpath(d, C.DASH)
        if rel.split("/")[0] == "dyn" and len(rel.split("/")) > 1 and rel.split("/")[1] in SKIP: continue
        up += [(f"{d}/{f}", f"{DASH_AT}/{os.path.normpath(f'{rel}/{f}')}") for f in fs if not (fnmatch.fnmatch(f, "*_0*.gif") or ".tmp" in f)]
    sf = f"{C.ROOT}/.bucket_state.json"; done = json.load(open(sf)) if os.path.exists(sf) else {}
    up = [(l, b) for l, b in up if done.get(b) != os.path.getmtime(l)]
    if up:
        api.batch_bucket_files(BUCKET, add=up)
        done.update({b: os.path.getmtime(l) for l, b in up}); json.dump(done, open(sf, "w"))
    C.stage("backup", repo=BUCKET, ok=True, files=len(up))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fetch": fetch()
    elif cmd == "restore": restore()
    elif cmd == "backup": backup()
    elif cmd == "loop":
        while True:
            time.sleep(EVERY)
            try: backup()
            except Exception as e: print("backup failed:", e, flush=True); C.stage("backup", repo=REPO, ok=False, error=str(e)[:300])
