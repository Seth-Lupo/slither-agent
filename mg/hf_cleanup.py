"""One-off HF cleanup: remove past-architecture WAM repos/buckets and dead MaskGIT runs, then squash slither-maskgit's
history (old checkpoint versions are what fill the private storage quota). Dry run by default; --yes deletes, permanently.

python3 hf_cleanup.py          # list what would go, with sizes
python3 hf_cleanup.py --yes    # do it
"""
import os, sys
from huggingface_hub import HfApi

U = "mundoamundo"
MODELS = ["slither-wam-pretrain", "slither-miniworld-v5-fast-20261002", "slither-miniworld-v5-20261001",   # OpenWAM, MiniWorld,
          "slither-residual-v3-20261001", "slither-wan-v3-20261001", "slither-spatial-v2-20261001",          # Wan/residual/spatial,
          "slither-wam-xm8-paused-20261001-step600", "slither-wam-xm8", "slither-wam-rl"]                   # XM and RL attempts
BUCKETS = ["slither-miniworld-v5-fast-20261002", "slither-miniworld-v5-20261001", "slither-wan-v3-20261001",
           "slither-residual-v3-20261001", "slither-spatial-v2-20261001", "slither-wam-xm8-paused-20261001-step600",
           "slither-wam-xm8-checkpoints", "slither-wam-rl-checkpoints", "slither-wam-training-latest"]
MASKGIT, DEAD = f"{U}/slither-maskgit", ["runs/s150ev", "runs/pilot50", "dash_ev"]                        # abandoned runs
DATASET, DATASET_DEAD = f"{U}/slither-wam-video-actions", ["ae_v1"]                                          # old autoencoder

tok = os.environ.get("HF_TOKEN") or next((l.split("=", 1)[1].strip().strip('"') for l in open(os.path.expanduser("~/slither/.env"))
                                          if l.startswith("HF_TOKEN=")), None)
api, go = HfApi(token=tok), "--yes" in sys.argv


def gb(n): return f"{n / 1e9:7.1f} GB"


total = 0
for m in MODELS:
    try: s = sum(f.size or 0 for f in api.model_info(f"{U}/{m}", files_metadata=True).siblings); total += s; print(f"model   {gb(s)}  {m}")
    except Exception as e: print(f"model   (missing) {m}: {str(e)[:60]}")
for b in BUCKETS:
    try: s = api.bucket_info(f"{U}/{b}").size or 0; total += s; print(f"bucket  {gb(s)}  {b}")
    except Exception as e: print(f"bucket  (missing) {b}: {str(e)[:60]}")
for rid, kind, dead in [(MASKGIT, "model", DEAD), (DATASET, "dataset", DATASET_DEAD)]:
    files = (api.model_info if kind == "model" else api.dataset_info)(rid, files_metadata=True).siblings
    for d in dead:
        s = sum(f.size or 0 for f in files if f.rfilename.startswith(d + "/")); total += s; print(f"folder  {gb(s)}  {rid}/{d}")
print(f"current files to remove: {gb(total)}; plus every old checkpoint version in {MASKGIT}'s history (squash)")
if not go: print("dry run: nothing deleted. Re-run with --yes to delete permanently."); sys.exit()

for m in MODELS:
    try: api.delete_repo(f"{U}/{m}", repo_type="model"); print("deleted model", m)
    except Exception as e: print("skip model", m, str(e)[:80])
for b in BUCKETS:
    try: api.delete_bucket(f"{U}/{b}"); print("deleted bucket", b)
    except Exception as e: print("skip bucket", b, str(e)[:80])
for rid, kind, dead in [(MASKGIT, "model", DEAD), (DATASET, "dataset", DATASET_DEAD)]:
    for d in dead:
        try: api.delete_folder(d, repo_id=rid, repo_type=kind, commit_message=f"remove {d}"); print("deleted folder", rid, d)
        except Exception as e: print("skip folder", rid, d, str(e)[:80])
api.super_squash_history(MASKGIT, repo_type="model"); print("squashed history of", MASKGIT)
