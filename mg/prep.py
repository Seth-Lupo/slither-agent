"""Pass 1: decode every video once (CPU, all cores), write per-video actions, keep a random ~1M-frame subset.

Outputs in DATA: acts/<mid>.npz, subset.u8 [N,128,128,3] (memmap), meta.json (turn bin edges, counts).
"""
import json, os, sys, time, zlib
from multiprocessing import Pool
import numpy as np
from PIL import Image
import common as C

TARGET = 1_000_000


def work(args):
    v, p = args
    out = f"{C.DATA}/acts/{v['mid']}.npz"
    if os.path.exists(out): return v["mid"], 0
    fr = C.decode(v["path"], v["src"])
    src = v["src"][:len(fr)]
    a = C.actions(v["mid"], src)
    rng = np.random.default_rng(abs(hash(v["mid"])) % 2**32)
    pick = np.flatnonzero(rng.random(len(fr)) < p)
    np.save(f"{C.DATA}/sub/{v['mid']}.npy", fr[pick])
    np.savez(out, src=src, **a)
    return v["mid"], len(fr)


def main():
    os.makedirs(f"{C.DATA}/acts", exist_ok=True); os.makedirs(f"{C.DATA}/sub", exist_ok=True)
    vs = C.videos()
    total = sum(len(v["src"]) for v in vs); p = min(1.0, TARGET / total)
    print(f"{len(vs)} videos, {total} frames at {C.FPS} fps ({total / C.FPS / 3600:.0f} h game time), subset p={p:.3f}", flush=True)
    vs.sort(key=lambda v: -len(v["src"]))                       # longest first for load balance
    t0, done, nf = time.time(), 0, 0
    with Pool(max(1, (os.cpu_count() or 8) // 2)) as pool:
        for mid, n in pool.imap_unordered(work, [(v, p) for v in vs]):
            done += 1; nf += n
            if done % 10 == 0 or done == len(vs):
                el = time.time() - t0
                C.stage("prep", done=done, total=len(vs), frames=nf, fps=nf / max(el, 1), eta_min=el / done * (len(vs) - done) / 60)
                print(f"prep {done}/{len(vs)} {nf / max(el, 1):.0f} fr/s", flush=True)

    # turn bin edges: symmetric quantiles of |turn| over valid frames -> NB bins with equal mass per side
    turns, nval, ntot = [], 0, 0
    for v in vs:
        a = np.load(f"{C.DATA}/acts/{v['mid']}.npz"); turns.append(a["turn"][a["valid"]]); nval += a["valid"].sum(); ntot += len(a["valid"])
    t = np.abs(np.concatenate(turns))
    q = np.quantile(t, np.arange(1, C.NB // 2) / (C.NB // 2))
    edges = np.concatenate([-q[::-1], [0.0], q])                  # NB-1 edges -> NB bins
    # subset -> one memmap
    files = sorted(os.listdir(f"{C.DATA}/sub")); n = sum(np.load(f"{C.DATA}/sub/{f}", mmap_mode="r").shape[0] for f in files)
    mm = np.lib.format.open_memmap(f"{C.DATA}/subset.npy", "w+", np.uint8, (n, C.RES, C.RES, 3)); k = 0
    for f in files:
        x = np.load(f"{C.DATA}/sub/{f}"); mm[k:k + len(x)] = x; k += len(x); os.remove(f"{C.DATA}/sub/{f}")
    mm.flush()
    meta = dict(videos=len(vs), frames=int(ntot), valid_frac=float(nval / ntot), subset=int(n), edges=edges.tolist(),
                turn_abs_quantiles={str(k): float(np.quantile(t, k)) for k in (0.5, 0.9, 0.99)})
    C.jdump(meta, f"{C.DATA}/meta.json")
    # preview: 32 random subset frames for the dashboard
    idx = np.random.default_rng(0).choice(n, 32, replace=False)
    grid = mm[np.sort(idx)].reshape(4, 8, C.RES, C.RES, 3).transpose(0, 2, 1, 3, 4).reshape(4 * C.RES, 8 * C.RES, 3)
    os.makedirs(f"{C.DASH}/tok", exist_ok=True); Image.fromarray(grid).save(f"{C.DASH}/tok/subset.png")
    C.stage("prep", done=len(vs), total=len(vs), frames=int(ntot), finished=True, meta=meta)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
