"""One session, scripted actions: checks frames, turn sign, boost, reward and death. Writes smoke.gif.
    .venv/bin/python scripts/smoke.py [--out smoke_out] [--episodes 1] [--max_steps 300]"""
import argparse, asyncio, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from playwright.async_api import async_playwright
from slither_env.env import SlitherEnv, launch_browser, ACTIONS
from PIL import Image

def wrap(d): return (d + math.pi) % (2 * math.pi) - math.pi

async def main(a):
    os.makedirs(a.out, exist_ok=True)
    async with async_playwright() as pw:
        b = await launch_browser(pw); env = SlitherEnv(b); await env.open()
        for ep in range(a.episodes):
            obs = await env.reset(); frames = list(obs["frames"]); ang = obs["state"]["ang"]; ret = 0
            print(f"episode {ep}: start score {obs['state']['score']}")
            for t in range(a.max_steps):
                # scripted: 10 decisions hard left, 10 hard right, 10 straight boosting, then random
                k = t // 10
                act = ACTIONS.index((90, 0)) if k == 0 else ACTIONS.index((-90, 0)) if k == 1 else ACTIONS.index((0, 1)) if k == 2 else np.random.randint(len(ACTIONS))
                obs, r, done, info = await env.step(act); ret += r; frames += list(obs["frames"])
                st = obs["state"]
                if st and t < 30 and t % 2 == 0:
                    turned = -math.degrees(wrap(st["ang"] - ang))          # positive = left
                    print(f"  t={t:3d} act={ACTIONS[act]} turned={turned:+6.1f}deg sp={st['sp']:.2f} score={info['score']} r={r:+.4f} late={info['late']:+.2f}s")
                if st: ang = st["ang"]
                if done: print(f"  died at decision {t}, return {ret:.3f} (log final/start)"); break
        Image.fromarray(frames[0]).save(f"{a.out}/frame0.png")
        ims = [Image.fromarray(f).resize((256, 256)) for f in frames[::2]]
        ims[0].save(f"{a.out}/smoke.gif", save_all=True, append_images=ims[1:], duration=200, loop=0)
        print("frames:", len(frames), "->", f"{a.out}/smoke.gif"); await env.close(); await b.close()

p = argparse.ArgumentParser(); p.add_argument("--out", default="smoke_out"); p.add_argument("--episodes", type=int, default=1); p.add_argument("--max_steps", type=int, default=300)
asyncio.run(main(p.parse_args()))
