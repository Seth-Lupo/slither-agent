"""Zoom calibration: which harness crop makes live frames look like the training videos?

For each crop factor, play a few live games (cluster IP, no proxy) with straight-ish random steering, record 10 Hz
frames, and score them with the pretrained models: tokenizer reconstruction PSNR, and the world model's next-frame
cross-entropy (all target tokens masked, 8-frame context, real action tokens). Lower CE / higher PSNR = closer to the
training distribution.   MG_ROOT=/tmp/slither/mg CUDA_VISIBLE_DEVICES=4 python calib.py --crops 0.3 0.4 0.5 0.7 1.0
"""
import argparse, asyncio, math, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rlcore as R
M = R.M
sys.path.insert(0, os.environ.get("HARNESS", "/tmp/slither/harness"))
from slither_env.env import SlitherEnv, launch_browser   # noqa: E402


async def collect(crop, n_sessions, seconds):
    from playwright.async_api import async_playwright
    clips = []
    async with async_playwright() as pw:
        async def one(i):
            b = await launch_browser(pw); e = SlitherEnv(b, nick="calib", render_fps=20, crop=crop, fpd=1, actions=R.ACTIONS)
            await e.open(); obs = await e.reset(); fr = [x for x in obs["samples"] if x.get("frame") is not None]
            t0 = time.time(); a = R.NA // 2
            while time.time() - t0 < seconds:
                if random.random() < 0.1: a = random.randrange(R.NA) & ~1           # occasional turn change, no boost
                o, r, d, info = await e.step(a); fr += [x for x in info["samples"] if x.get("frame") is not None]
                if d: clips.append(fr); obs = await e.reset(); fr = [x for x in obs["samples"] if x.get("frame") is not None]
            clips.append(fr); await b.close()
        await asyncio.gather(*[one(i) for i in range(n_sessions)])
    return [c for c in clips if len(c) >= 9]


@torch.no_grad()
def score(clips, tokz, wm):
    psnr, ce = [], []
    for c in clips:
        fr = torch.from_numpy(np.stack([x["frame"] if isinstance(x, dict) else x for x in c])).cuda()
        act = torch.tensor([M.NULL] * len(c)).cuda()          # "no action" tokens: same for live and training clips
        tok = torch.cat([tokz.encode(fr[i:i + 64]) for i in range(0, len(fr), 64)]).view(len(fr), 256)
        rec = torch.cat([tokz.decode(tok[i:i + 64].view(-1, 16, 16)) for i in range(0, len(tok), 64)])
        x = fr.permute(0, 3, 1, 2).float() / 255
        psnr.append(float((-10 * torch.log10(((rec - x) ** 2).mean((1, 2, 3)).clamp_min(1e-10))).mean()))
        idx = list(range(8, len(tok), 4))[:32]
        if not idx: continue
        ctx = torch.stack([tok[i - 8:i + 1] for i in idx]).clone(); acts = torch.stack([act[i - 8:i + 1] for i in idx])
        tgt = ctx[:, -1].clone(); ctx[:, -1] = M.MASK
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = wm(ctx, acts).float()
        ce.append(float(F.cross_entropy(lg.reshape(-1, M.NCODE), tgt.reshape(-1))))
    return np.mean(psnr), np.mean(ce), len(clips)


def main(a):
    tokz, wm = R.load_tokenizer(), R.load_wm()
    ev = np.load(f"{R.MG}/data/evalclips.npz")["frames"]
    p, c, n = score([list(x) for x in ev], tokz, wm)
    print(f"TRAINING clips (reference): tokenizer PSNR {p:.2f} dB   world-model CE {c:.3f}   clips {n}", flush=True)
    for crop in a.crops:
        clips = asyncio.run(collect(crop, a.sessions, a.seconds))
        p, c, n = score(clips, tokz, wm)
        print(f"crop {crop:.2f}: tokenizer PSNR {p:.2f} dB   world-model CE (fully masked) {c:.3f}   clips {n}", flush=True)
        import PIL.Image as I
        I.fromarray(np.concatenate([clips[0][min(20, len(clips[0]) - 1)]["frame"], clips[-1][min(20, len(clips[-1]) - 1)]["frame"]], 1)).save(f"/tmp/slither/calib_{crop:.2f}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--crops", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.7, 1.0])
    p.add_argument("--sessions", type=int, default=4); p.add_argument("--seconds", type=float, default=45)
    main(p.parse_args())
