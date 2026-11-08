"""Is the harness's turn convention the same as the training data's?

Play live games (cluster IP) alternating hard-left / hard-right / straight blocks, record frames and the measured
per-frame action tokens. Then score each real next frame with the world model given (a) the measured tokens and
(b) the same tokens mirrored (turn bin k -> 31-k). If the conventions agree, (a) must predict the real frames better.
Also reports it separately for frames where the snake was turning hard (where the action matters most).
  MG_ROOT=/tmp/slither/mg CUDA_VISIBLE_DEVICES=4 python signtest.py
"""
import asyncio, math, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rlcore as R
M = R.M
sys.path.insert(0, os.environ.get("HARNESS", "/tmp/slither/harness"))
from slither_env.env import SlitherEnv, launch_browser   # noqa: E402


async def collect(n_sessions=4, seconds=90):
    from playwright.async_api import async_playwright
    clips = []
    async with async_playwright() as pw:
        async def one(i):
            b = await launch_browser(pw); e = SlitherEnv(b, nick="sign", render_fps=20, fpd=1, actions=R.ACTIONS)
            await e.open(); obs = await e.reset(); fr = [x for x in obs["samples"] if x.get("frame") is not None]
            t0 = time.time(); k = 0
            while time.time() - t0 < seconds:
                a = [R.NA - 2, 0, 32][(k // 8) % 3]; k += 1                # hard left 0.8 s, hard right 0.8 s, straight 0.8 s
                o, r, d, info = await e.step(a); fr += [x for x in info["samples"] if x.get("frame") is not None]
                if d: clips.append(fr); obs = await e.reset(); fr = [x for x in obs["samples"] if x.get("frame") is not None]
            clips.append(fr); await b.close()
        await asyncio.gather(*[one(i) for i in range(n_sessions)])
    return [c for c in clips if len(c) >= 12]


@torch.no_grad()
def main():
    tokz, wm = R.load_tokenizer(), R.load_wm()
    clips = asyncio.run(collect())
    res = {"measured": [], "mirrored": [], "null": []}; hard = {"measured": [], "mirrored": [], "null": []}
    for c in clips:
        fr = torch.from_numpy(np.stack([x["frame"] for x in c])).cuda()
        ang = np.array([x["ang"] for x in c]); d = (np.diff(ang, prepend=ang[:1]) + math.pi) % (2 * math.pi) - math.pi
        deg = -np.degrees(d)                                               # harness convention: + = left
        meas = torch.tensor([R.act_token(t, 0) for t in deg]).cuda()
        mirr = meas.clone(); mirr[:, 0] = (R.NB - 1) - mirr[:, 0]
        null = torch.tensor([M.NULL] * len(c)).cuda()
        tok = torch.cat([tokz.encode(fr[i:i + 64]) for i in range(0, len(fr), 64)]).view(len(fr), 256)
        for i in range(8, len(tok)):
            ctx = tok[i - 8:i + 1][None].clone(); tgt = ctx[0, -1].clone(); ctx[0, -1] = M.MASK
            for name, act in (("measured", meas), ("mirrored", mirr), ("null", null)):
                with torch.autocast("cuda", dtype=torch.bfloat16): lg = wm(ctx, act[i - 8:i + 1][None]).float()
                ce = float(F.cross_entropy(lg[0], tgt)); res[name].append(ce)
                if abs(deg[i]) > 5: hard[name].append(ce)
    print(f"{len(clips)} clips, {len(res['measured'])} frames ({len(hard['measured'])} with |turn| > 5 deg/frame)")
    for name in res: print(f"  {name:9s} tokens: CE all {np.mean(res[name]):.4f}   hard turns {np.mean(hard[name]) if hard[name] else float('nan'):.4f}")
    m, w = np.array(hard["measured"]), np.array(hard["mirrored"])
    if len(m): print(f"  hard turns: measured better on {np.mean(m < w) * 100:.0f}% of frames (mirrored better on {np.mean(w < m) * 100:.0f}%)")
    print("max turn seen deg/frame:", float(np.max(np.abs(np.concatenate([np.diff(np.unwrap([x['ang'] for x in c])) for c in clips])) * 180 / math.pi)))


if __name__ == "__main__":
    main()
