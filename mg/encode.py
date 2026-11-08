"""Pass 2: decode every video again and encode all frames to tokens (one process per GPU, CPU decode pool each).

Outputs in DATA: tokens.npy [N,256] uint16, acts.npy [N,2] uint8, cut.npy [N] bool, index.json (per-video ranges, split),
evalclips.npz (raw frames of fixed validation clips for three-panel videos).
torchrun --standalone --nproc_per_node 4 encode.py
"""
import json, os, time
from multiprocessing import Pool
import numpy as np, torch, torch.distributed as dist
import common as C, tokenizer as T

CLIP = 24


def dec(v):
    out = f"{C.DATA}/tmpfr/{v['mid']}.npy"
    np.save(out, C.decode(v["path"], v["src"])); return v, out


def main():
    dist.init_process_group("nccl", timeout=__import__("datetime").timedelta(hours=3)); rank, world = dist.get_rank(), dist.get_world_size(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    os.makedirs(f"{C.DATA}/tok", exist_ok=True); os.makedirs(f"{C.DATA}/tmpfr", exist_ok=True); os.makedirs(f"{C.DATA}/clips", exist_ok=True)
    tok = T.load(); meta = json.load(open(f"{C.DATA}/meta.json")); edges = np.array(meta["edges"])
    vs = sorted(C.videos(), key=lambda v: -len(v["src"]))[rank::world]
    vs = [v for v in vs if not os.path.exists(f"{C.DATA}/tok/{v['mid']}.npy")]
    t0, nf = time.time(), 0
    with Pool(max(1, (os.cpu_count() or 8) // world - 1)) as pool:
        for i, (v, path) in enumerate(pool.imap_unordered(dec, vs)):
            fr = np.load(path); os.remove(path)
            codes = []
            for j in range(0, len(fr), 1024):
                x = torch.from_numpy(fr[j:j + 1024]).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16): codes.append(tok.encode(x).view(len(x), -1).cpu())
            codes = torch.cat(codes).numpy().astype(np.uint16)
            if C.is_val(v["mid"]):                               # two fixed eval clips per validation video
                a = np.load(f"{C.DATA}/acts/{v['mid']}.npz"); cut = a["cut"][:len(fr)]
                ok = [s for s in range(2, len(fr) - CLIP) if not cut[s + 1:s + CLIP].any() and a["valid"][s + 1:s + CLIP].mean() > 0.8]
                if ok:
                    r = np.random.default_rng(len(fr)); pick = sorted(r.choice(ok, min(2, len(ok)), replace=False))
                    np.savez(f"{C.DATA}/clips/{v['mid']}.npz", starts=np.array(pick), frames=np.stack([fr[s:s + CLIP] for s in pick]))
            np.save(f"{C.DATA}/tok/{v['mid']}.npy", codes); nf += len(fr)
            if rank == 0 and i % 5 == 0:
                el = time.time() - t0
                C.stage("encode", done=i + 1, total=len(vs), frames=nf * world, fps=nf * world / el, eta_min=el / (i + 1) * (len(vs) - i - 1) / 60)
    dist.barrier()
    if rank == 0: assemble(edges)
    dist.barrier(); dist.destroy_process_group()


def assemble(edges):
    vs = sorted(C.videos(), key=lambda v: v["mid"]); index, n = [], 0
    lens = {v["mid"]: len(np.load(f"{C.DATA}/tok/{v['mid']}.npy", mmap_mode="r")) for v in vs}
    N = sum(lens.values())
    tk = np.lib.format.open_memmap(f"{C.DATA}/tokens.npy", "w+", np.uint16, (N, 256))
    ac = np.zeros((N, 2), np.uint8); cut = np.ones(N, bool); clips = []
    for v in vs:
        m, L = v["mid"], lens[v["mid"]]
        tk[n:n + L] = np.load(f"{C.DATA}/tok/{m}.npy")
        a = np.load(f"{C.DATA}/acts/{m}.npz")
        ac[n:n + L] = C.action_tokens(a["turn"][:L], a["boost"][:L], a["valid"][:L], edges)
        cut[n:n + L] = a["cut"][:L]; cut[n] = True                   # first frame of each video starts a new segment
        if os.path.exists(f"{C.DATA}/clips/{m}.npz"):
            c = np.load(f"{C.DATA}/clips/{m}.npz")
            for s, f in zip(c["starts"], c["frames"]): clips.append((n + int(s), f))
        index.append(dict(mid=m, start=n, len=L, val=C.is_val(m))); n += L
    tk.flush(); np.save(f"{C.DATA}/acts.npy", ac); np.save(f"{C.DATA}/cut.npy", cut)
    C.jdump(index, f"{C.DATA}/index.json")
    clips = clips[:64]
    np.savez(f"{C.DATA}/evalclips.npz", starts=np.array([s for s, _ in clips]), frames=np.stack([f for _, f in clips]))
    hours = dict(train=sum(i["len"] for i in index if not i["val"]) / C.FPS / 3600, val=sum(i["len"] for i in index if i["val"]) / C.FPS / 3600)
    C.stage("encode", done=len(vs), total=len(vs), frames=N, finished=True, hours=hours, eval_clips=len(clips),
            action_valid=float((ac[:, 0] < C.NB).mean()))


if __name__ == "__main__":
    main()
