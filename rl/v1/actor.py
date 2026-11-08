"""Actors: live slither.io games (one headless Chromium per session, up to 4 sessions per proxy IP), acting with the
learner's latest Q head on frozen world-model features. Ape-X exploration: session i is epsilon-greedy with its own
epsilon 0.4^(1 + 7 i / (N-1)), so some sessions explore a lot and some almost not at all.

Each finished episode -> RL/real/ep_*.npz (per decision: 8-frame context tokens + action tokens, features, action,
reward, done, the action tokens that actually happened during it) and one line in RL/dash/episodes.jsonl.
Live view: RL/dash/live.jpg (mosaic of the sessions' latest frames), RL/dash/actor.json (per-session status).
"""
import argparse, asyncio, collections, json, math, os, random, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rlcore as R
sys.path.insert(0, os.environ.get("HARNESS", "/tmp/slither/harness"))
from slither_env.env import SlitherEnv, launch_browser, load_proxies, ACTIONS   # noqa: E402

assert [tuple(a) for a in ACTIONS] == R.ACTIONS
OUT = R.RL
for d in ("real", "dash"): os.makedirs(f"{OUT}/{d}", exist_ok=True)


class Policy:
    def __init__(s, n):
        s.tokz, s.wm, s.heads = R.load_tokenizer(), R.load_wm(), R.Heads().cuda().eval()
        s.ver, s.last_load = -1, 0.0
        s.eps = [0.4 ** (1 + 7 * i / max(1, n - 1)) for i in range(n)]

    def reload(s):
        if time.time() - s.last_load < 15: return
        s.last_load = time.time()
        try: ck = torch.load(f"{OUT}/heads.pt", map_location="cuda")
        except Exception: return
        if ck["ver"] != s.ver: s.heads.load_state_dict(ck["heads"]); s.ver = ck["ver"]

    @torch.no_grad()
    def __call__(s, batch):
        s.reload(); B = len(batch)
        fr = torch.from_numpy(np.stack([o["frames"] for o in batch])).cuda()
        tok = s.tokz.encode(fr.view(-1, 128, 128, 3)).view(B, 8, 256)
        act = torch.from_numpy(np.stack([o["act"] for o in batch])).cuda().long()
        f = R.features(s.wm, tok, act)
        q = s.heads.q(f) if s.ver >= 0 else torch.zeros(B, R.NA, device="cuda")
        tok, f, q = tok.cpu().numpy().astype(np.uint16), f.half().cpu().numpy(), q.cpu().numpy()
        out = []
        for i, o in enumerate(batch):
            greedy = s.ver >= 0 and random.random() >= s.eps[o["sid"]]
            a = int(q[i].argmax()) if greedy else random.randrange(R.NA)
            out.append((a, dict(tok=tok[i], f=f[i], q=float(q[i].max()), qs=q[i].round(3).tolist(), greedy=greedy)))
        return out


class Batcher:
    def __init__(s, policy): s.policy, s.pending, s.n, s.t = policy, [], 0, 0.0
    async def act(s, obs):
        fut = asyncio.get_running_loop().create_future(); s.pending.append((obs, fut)); return await fut
    async def loop(s):
        while True:
            await asyncio.sleep(0.01)
            if not s.pending: continue
            batch, s.pending = s.pending, []
            t = time.time()
            try: res = await asyncio.to_thread(s.policy, [o for o, _ in batch])
            except Exception as e:
                for _, f in batch: f.set_exception(e)
                continue
            s.n += len(batch); s.t += time.time() - t
            for (_, f), r in zip(batch, res): f.set_result(r)


def stream(sid, frame, info):
    from PIL import Image
    d = f"{OUT}/dash/s"; os.makedirs(d, exist_ok=True)
    Image.fromarray(frame).save(f"{d}/{sid}.tmp.jpg", quality=90); os.replace(f"{d}/{sid}.tmp.jpg", f"{d}/{sid}.jpg")
    R.jdump(info, f"{d}/{sid}.json")


def wrap(x): return (x + math.pi) % (2 * math.pi) - math.pi


async def episode(env, sid, batcher, stop, S):
    obs = await asyncio.wait_for(env.reset(), 150)                 # a frozen tab must not hang the session forever
    hist, acts = collections.deque(maxlen=8), collections.deque(maxlen=8); prev = [None]
    def push(x, boost):
        t = 0.0 if prev[0] is None else -math.degrees(wrap(x["ang"] - prev[0]))
        prev[0] = x["ang"]; hist.append(x["frame"]); acts.append(R.act_token(t, boost))
        S["frame"] = x["frame"]
    for x in [x for x in obs["samples"] if x.get("frame") is not None][-8:]: push(x, 0)
    while len(hist) < 8: hist.appendleft(hist[0]); acts.appendleft(acts[0])
    E = collections.defaultdict(list); t0 = time.time(); start = env.score; best = env.score
    S.update(alive=True, score=env.score, ep_t0=t0)
    while True:
        ctx_act = np.array(acts, np.uint8)
        a, ex = await batcher.act(dict(sid=sid, frames=np.stack(hist), act=ctx_act))
        obs, r, done, info = await asyncio.wait_for(env.step(a), 20)
        new = [x for x in info["samples"] if x.get("frame") is not None]
        for x in new: push(x, ACTIONS[a][1])
        happened = [R.act_token(0, ACTIONS[a][1])] * 3 if not new else list(acts)[-min(3, len(new)):]
        happened = (happened + happened[-1:] * 3)[:3]
        for k, v in dict(ctx_tok=ex["tok"], ctx_act=ctx_act, f=ex["f"], q=ex["q"], greedy=ex["greedy"], a=a, r=r, done=done,
                         a_tok=np.array(happened, np.uint8), score=info["score"]).items(): E[k].append(v)
        best = max(best, info["score"]); S.update(score=info["score"], decisions=S["decisions"] + 1)
        if new:                                                   # per-session live stream for the dashboard (click a game)
            try: stream(sid, new[-1]["frame"], dict(t=time.time(), sid=sid, proxy=S["proxy"], score=info["score"], best=best,
                        a=a, turn=ACTIONS[a][0], boost=ACTIONS[a][1], greedy=ex["greedy"], q=ex["qs"], eps=S["eps"],
                        alive=not done, ep_s=round(time.time() - t0, 1), ver=batcher.policy.ver))
            except Exception: pass
        if done or stop.is_set(): break
    S["alive"] = False
    n = len(E["a"])
    if n == 0: return
    path = f"{OUT}/real/ep_{sid:03d}_{int(time.time() * 1000)}.npz"
    np.savez(path + ".tmp.npz", **{k: np.array(v) for k, v in E.items()}); os.replace(path + ".tmp.npz", path)
    rec = dict(t=time.time(), sid=sid, proxy=S["proxy"], decisions=n, seconds=round(time.time() - t0, 1), start=start,
               final=E["score"][-1], best=best, ret=float(sum(E["r"])), died=bool(E["done"][-1]), eps=round(S["eps"], 4),
               greedy_frac=float(np.mean(E["greedy"])), q_mean=float(np.mean(E["q"])), ver=batcher.policy.ver)
    with open(f"{OUT}/dash/episodes.jsonl", "a") as fh: fh.write(json.dumps(rec) + "\n")
    S["episodes"] += 1; S["last_ret"] = rec["ret"]; S["last_final"] = rec["final"]


async def actor(env, sid, batcher, stop, S):
    while not stop.is_set():
        try: await episode(env, sid, batcher, stop, S); S["fails_row"] = 0
        except BaseException as e:
            if isinstance(e, asyncio.CancelledError) and stop.is_set(): raise
            S["fails"] += 1; S["fails_row"] += 1; S["alive"] = False; S["err"] = f"{type(e).__name__}: {str(e)[:80]}"
            print(f"s{sid:03d}: {S['err']}; reloading", flush=True)
            await asyncio.sleep(min(120, 5 * S["fails_row"]))         # back off a proxy that keeps failing
            try: await asyncio.wait_for(env.close(), 30); await asyncio.wait_for(env.open(), 180)
            except Exception: await asyncio.sleep(10)


def ensure_web():
    """Keep the public dashboard (web.sh: http server + Cloudflare tunnel) running; started from here so it lives in the
    job's process tree (processes started by a short srun step are killed when that step ends)."""
    import subprocess
    if subprocess.run(["pgrep", "-f", "[c]loudflared tunnel"], capture_output=True).returncode != 0 and os.path.exists(f"{HERE}/web.sh"):
        subprocess.Popen(["bash", f"{HERE}/web.sh"], stdout=open(f"{OUT}/logs/websh.log", "a"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)


async def monitor(states, batcher, stop):
    from PIL import Image, ImageDraw
    t0, n0, t_web = time.time(), 0, 0.0
    while not stop.is_set():
        await asyncio.sleep(5)
        if time.time() - t_web > 120:
            t_web = time.time()
            try: ensure_web()
            except Exception as e: print("ensure_web:", e, flush=True)
        el = time.time() - t0; dps = (batcher.n - n0) / el; t0, n0 = time.time(), batcher.n
        cols = 8; rows = math.ceil(len(states) / cols); W = 160
        im = Image.new("RGB", (cols * W, rows * W), (12, 16, 22)); dr = ImageDraw.Draw(im)
        for i, S in enumerate(states):
            x, y = (i % cols) * W, (i // cols) * W
            if S.get("frame") is not None: im.paste(Image.fromarray(S["frame"]).resize((W - 4, W - 4)), (x + 2, y + 2))
            dr.rectangle([x + 1, y + 1, x + W - 2, y + W - 2], outline=(60, 200, 120) if S["alive"] else (200, 70, 70))
            dr.text((x + 6, y + 4), f"s{i} p{S['proxy']} len {S['score']}", fill=(255, 255, 255))
        im.save(f"{OUT}/dash/live.jpg.tmp.jpg", quality=80); os.replace(f"{OUT}/dash/live.jpg.tmp.jpg", f"{OUT}/dash/live.jpg")
        R.jdump(dict(t=time.time(), dps=dps, policy_ms=1000 * batcher.t / max(1, batcher.n), ver=batcher.policy.ver,
                     sessions=[{k: v for k, v in S.items() if k != "frame"} for S in states]), f"{OUT}/dash/actor.json")


async def main(a):
    from playwright.async_api import async_playwright
    px = load_proxies(a.proxies); n = min(a.sessions, len(px) * a.per_proxy)
    policy = Policy(n); batcher = Batcher(policy); stop = asyncio.Event()
    states = [dict(sid=i, proxy=i // a.per_proxy, alive=False, score=0, decisions=0, episodes=0, fails=0, fails_row=0,
                   eps=policy.eps[i], err="", frame=None) for i in range(n)]
    async with async_playwright() as pw:
        envs = []
        for i in range(n):
            b = await launch_browser(pw, proxy=px[i // a.per_proxy])
            e = SlitherEnv(b, nick=a.nick, render_fps=20)
            try: await e.open()
            except Exception as ex: print(f"s{i} open failed: {ex}", flush=True)
            envs.append(e); await asyncio.sleep(a.stagger)
        print(f"{n} sessions on {len(px)} proxies", flush=True)
        tasks = [asyncio.create_task(actor(e, i, batcher, stop, states[i])) for i, e in enumerate(envs)]
        tasks += [asyncio.create_task(batcher.loop()), asyncio.create_task(monitor(states, batcher, stop))]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--proxies", default="/tmp/slither/proxies.txt"); p.add_argument("--per_proxy", type=int, default=4)
    p.add_argument("--sessions", type=int, default=80); p.add_argument("--stagger", type=float, default=1.0)
    p.add_argument("--nick", default="dyna")
    asyncio.run(main(p.parse_args()))
