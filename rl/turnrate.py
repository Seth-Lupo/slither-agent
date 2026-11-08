"""Can the agent turn as fast as a human? Hold a sustained turn with different commanded targets and measure the
actual heading change per 0.1 s frame (cluster IP, 2 sessions). Target 26 deg = the agent's hardest action;
90/150 deg = a mouse far to the side (what a human does for a hard turn)."""
import asyncio, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("HARNESS", "/tmp/slither/harness"))
from slither_env.env import SlitherEnv, launch_browser   # noqa: E402

TARGETS = [(26, 0), (60, 0), (90, 0), (150, 0), (26, 1), (150, 1)]


async def main():
    from playwright.async_api import async_playwright
    res = {t: [] for t in TARGETS}
    async with async_playwright() as pw:
        async def one(i):
            b = await launch_browser(pw); e = SlitherEnv(b, nick="turn", render_fps=20, fpd=1, actions=TARGETS)
            await e.open(); await e.reset()
            for rep in range(6):
                for k, t in enumerate(TARGETS):
                    angs = []
                    for _ in range(12):                      # hold 1.2 s
                        o, r, d, info = await e.step(k)
                        angs += [x["ang"] for x in info["samples"] if x.get("alive")]
                        if d: await e.reset(); break
                    if len(angs) > 6:
                        dd = np.abs((np.diff(angs[3:]) + math.pi) % (2 * math.pi) - math.pi)   # skip the ramp-up
                        res[t] += list(np.degrees(dd))
            await b.close()
        await asyncio.gather(one(0), one(1))
    for t in TARGETS:
        x = np.array(res[t]); print(f"target {t[0]:3d} deg{' + boost' if t[1] else ''}: actual turn per 0.1 s median {np.median(x):5.1f}  p90 {np.percentile(x, 90):5.1f}  (n {len(x)})")


asyncio.run(main())
