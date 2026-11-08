"""Dyna imagination worker (one per GPU): take real states from the actors' episodes, pick a random action, let the world
model (s150p MaskGIT) generate the next 0.3 s (3 frames), and hand (f(s), a, f(s')) to the learner via RL/imag.

The per-frame action tokens for an imagined action are drawn from what actually happened when the actors took that
action (the game limits turn rate, so "turn 90 deg" is not 30 deg every frame). Rank 2 also renders RL/dash/imagine.jpg:
real context frame, then the imagined frame 0.3 s later under hard-left / straight / hard-right.
"""
import argparse, glob, os, random, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rlcore as R
M = R.M
OUT = R.RL
for d in ("imag", "dash"): os.makedirs(f"{OUT}/{d}", exist_ok=True)


class Pool:
    def __init__(s, cap):
        s.cap, s.n, s.i = cap, 0, 0
        s.tok = np.zeros((cap, 8, 256), np.uint16); s.act = np.zeros((cap, 8, 2), np.uint8); s.f = np.zeros((cap, 1024), np.float16)
        s.happened = {a: [] for a in range(R.NA)}; s.seen = set()
    def ingest(s):
        files = sorted(set(glob.glob(f"{OUT}/real/ep_*.npz")) - s.seen)
        for p in files[-400:]:                                        # newest first matter most; skip a huge backlog
            s.seen.add(p)
            try: z = np.load(p); tok, act, f, a, at = z["ctx_tok"], z["ctx_act"], z["f"], z["a"], z["a_tok"]
            except Exception: continue
            for j in range(len(a)):
                k = s.i; s.tok[k], s.act[k], s.f[k] = tok[j], act[j], f[j]; s.i = (s.i + 1) % s.cap; s.n = min(s.cap, s.n + 1)
                h = s.happened[int(a[j])]; h.append(at[j])
                if len(h) > 5000: del h[:1000]
        s.seen.update(files)
    def triple(s, a):
        h = s.happened[a]
        if h: return h[random.randrange(len(h))]
        t, b = R.ACTIONS[a]; return np.array([R.act_token(t / 3, b)] * 3, np.uint8)


@torch.no_grad()
def imagine(wm, tok, act, trip, steps, cfg):
    acts = torch.cat([act, trip], 1)                                   # [B, 8+3, 2]
    gen = M.rollout(wm, tok, acts, 3, steps=steps, cfg=cfg)            # [B,3,256]
    ctx2, act2 = torch.cat([tok[:, 3:], gen], 1), acts[:, 3:]
    return R.features(wm, ctx2, act2), gen


def preview(wm, tokz, pool, steps, cfg):
    from PIL import Image, ImageDraw
    idx = np.random.randint(0, pool.n, 4)
    tok = torch.from_numpy(pool.tok[idx].astype(np.int64)).cuda(); act = torch.from_numpy(pool.act[idx].astype(np.int64)).cuda()
    cols = [tokz.decode(tok[:, -1].view(-1, 16, 16))]
    for a in (R.ACTIONS.index((90, 0)), R.ACTIONS.index((0, 0)), R.ACTIONS.index((-90, 0))):
        trip = torch.from_numpy(np.stack([pool.triple(a) for _ in idx]).astype(np.int64)).cuda()
        _, gen = imagine(wm, tok, act, trip, steps, cfg); cols.append(tokz.decode(gen[:, -1].view(-1, 16, 16)))
    x = torch.stack(cols, 1).mul(255).byte().permute(0, 3, 1, 4, 2).reshape(4 * 128, 4 * 128, 3).cpu().numpy()
    im = Image.fromarray(x).resize((1024, 1024), Image.NEAREST); dr = ImageDraw.Draw(im)
    for i, s in enumerate(("real now", "imagined: left", "imagined: straight", "imagined: right")): dr.text((i * 256 + 6, 4), s, fill=(255, 255, 0))
    im.save(f"{OUT}/dash/imagine.jpg.tmp.jpg", quality=85); os.replace(f"{OUT}/dash/imagine.jpg.tmp.jpg", f"{OUT}/dash/imagine.jpg")


def main(a):
    wm = R.load_wm(); tokz = R.load_tokenizer() if a.rank == 2 else None
    pool = Pool(a.pool); buf = []; t_ing = 0.0; t_prev = 0.0; n_done = 0; t0 = time.time()
    while True:
        if time.time() - t_ing > 20: pool.ingest(); t_ing = time.time()
        backlog = len(glob.glob(f"{OUT}/imag/im_*.npz"))
        if pool.n < 2000 or backlog > 200: time.sleep(2); continue
        idx = np.random.randint(0, pool.n, a.bs); acts_ = np.random.randint(0, R.NA, a.bs)
        tok = torch.from_numpy(pool.tok[idx].astype(np.int64)).cuda(); act = torch.from_numpy(pool.act[idx].astype(np.int64)).cuda()
        trip = torch.from_numpy(np.stack([pool.triple(int(x)) for x in acts_]).astype(np.int64)).cuda()
        f2, _ = imagine(wm, tok, act, trip, a.steps, a.cfg)
        buf.append((pool.f[idx], acts_.astype(np.int16), f2.half().cpu().numpy())); n_done += a.bs
        if len(buf) >= 4:
            f, aa, ff = (np.concatenate(x) for x in zip(*buf)); buf = []
            p = f"{OUT}/imag/im_{a.rank}_{int(time.time() * 1000)}.npz"
            np.savez(p + ".tmp.npz", f=f, a=aa, f2=ff); os.replace(p + ".tmp.npz", p)
            R.jdump(dict(t=time.time(), rank=a.rank, imagined=n_done, per_s=n_done / (time.time() - t0), pool=pool.n), f"{OUT}/dash/imag_{a.rank}.json")
        if tokz is not None and time.time() - t_prev > 300:
            try: preview(wm, tokz, pool, a.steps, a.cfg)
            except Exception as e: print("preview failed", e, flush=True)
            t_prev = time.time()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=2); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--steps", type=int, default=4, help="MaskGIT decoding steps per frame"); p.add_argument("--cfg", type=float, default=1.5)
    p.add_argument("--pool", type=int, default=300_000)
    main(p.parse_args())
