"""Run N live sessions with a policy and record every episode.
    .venv/bin/python scripts/run.py --sessions 8 --seconds 300 --out episodes/   (random policy)
The real agent passes its own batched policy to slither_env.runner.run()."""
import argparse, asyncio, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slither_env.runner import run, random_policy

p = argparse.ArgumentParser()
p.add_argument("--sessions", type=int, default=8); p.add_argument("--browsers", type=int, default=None)
p.add_argument("--seconds", type=float, default=None); p.add_argument("--out", default="episodes")
p.add_argument("--crop", type=float, default=1.0); p.add_argument("--stagger", type=float, default=1.0)
p.add_argument("--render_fps", type=int, default=0, help="cap the game render loop (0 = uncapped)")
p.add_argument("--proxies", default=None, help="proxy file; up to --per_proxy sessions per proxy")
p.add_argument("--per_proxy", type=int, default=4)
a = p.parse_args()
eps = asyncio.run(run(random_policy, a.sessions, a.out, a.browsers, a.seconds, crop=a.crop, stagger=a.stagger, render_fps=a.render_fps, proxies=a.proxies, per_proxy=a.per_proxy))
json.dump(eps, open(f"{a.out}/summary_{int(time.time())}.json", "w"), indent=1)
print(f"{len(eps)} episodes, mean return {sum(e['return'] for e in eps) / max(1, len(eps)):+.3f}")
