"""Live RL actors: the self-play policy (sp/train.py Net) plays real slither.io, one decision per 0.1 s.

One process per GPU serves a slice of the proxies (4 games each). Sessions with sid % 4 == 3 play greedily (evaluation).
Trajectories -> OUT/frag/*.npz (64-step fragments) for learn.py; finished games -> OUT/dash/episodes.jsonl;
each game's picture (game frame | what the policy sees) -> OUT/dash/s/<sid>.jpg + .json every 0.5 s.
Reward (same as self-play): +1 * dlog length, +KILL per kill, DEATH on death.
"""
import argparse, asyncio, json, math, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("SP_CODE", "/tmp/slither/sp/code"))
sys.path.insert(0, os.environ.get("HARNESS", "/tmp/slither/harness"))
import obs as O                                                                      # noqa: E402
from train import Net                                                                # noqa: E402
from slither_env.env import SlitherEnv, launch_browser, load_proxies                 # noqa: E402

OUT = os.environ.get("LIVE_ROOT", "/tmp/slither/live")
for d in ("frag", "dash", "dash/s", "logs"): os.makedirs(f"{OUT}/{d}", exist_ok=True)
T_FRAG = 64
GROW, KILL, DEATH = 1.0, 3.0 * math.log(400.0), -6.0     # victim length unknown live: credit a typical (400) snake


def jdump(o, p):
    with open(p + ".tmp", "w") as f: json.dump(o, f)
    os.replace(p + ".tmp", p)


class Policy:
    def __init__(s):
        s.net = Net().cuda().eval(); s.ver = -1; s.t = 0.0; s.reload(force=True)
    def reload(s, force=False):
        if not force and time.time() - s.t < 10: return
        s.t = time.time()
        for p in (f"{OUT}/policy.pt", f"{OUT}/init_policy.pt"):
            try: ck = torch.load(p, map_location="cuda", weights_only=False)
            except Exception: continue
            ver = ck.get("ver", 0) if isinstance(ck, dict) and "sd" in ck else 0
            sd = ck["sd"] if isinstance(ck, dict) and "sd" in ck else ck
            if ver != s.ver: s.net.load_state_dict(sd); s.ver = ver
            return
    @torch.no_grad()
    def __call__(s, batch):
        s.reload()
        o = torch.from_numpy(np.stack([b["obs"] for b in batch])).cuda().float()
        sc = torch.from_numpy(np.stack([b["sca"] for b in batch])).cuda()
        lg, v = s.net(o, sc); lp = torch.log_softmax(lg, -1)
        samp = torch.multinomial(lp.exp(), 1)[:, 0]; greedy = lg.argmax(-1)
        out = []
        for i, b in enumerate(batch):
            a = int(greedy[i] if b["greedy"] else samp[i]); out.append((a, float(lp[i, a]), float(v[i])))
        return out


class Batcher:
    def __init__(s, policy): s.policy, s.pending, s.n, s.t = policy, [], 0, 0.0
    async def act(s, x):
        f = asyncio.get_running_loop().create_future(); s.pending.append((x, f)); return await f
    async def loop(s):
        while True:
            await asyncio.sleep(0.005)
            if not s.pending: continue
            batch, s.pending = s.pending, []; t = time.time()
            try: res = await asyncio.to_thread(s.policy, [x for x, _ in batch])
            except Exception as e:
                for _, f in batch: f.set_exception(e)
                continue
            s.n += len(batch); s.t += time.time() - t
            for (_, f), r in zip(batch, res): f.set_result(r)


def write_frag(sid, OB, SC, A, LP, V, R, D, ver, last_v):
    if not A: return
    name = f"f_{sid:03d}_{time.time_ns()}.npz"; p = f"{OUT}/frag/{name}"
    np.savez_compressed(f"{OUT}/frag/tmp_{name}", obs=np.stack(OB), sca=np.stack(SC), a=np.array(A, np.int16), logp=np.array(LP, np.float32),
                        v=np.array(V, np.float32), r=np.array(R, np.float32), done=np.array(D, bool), last_v=np.float32(last_v), ver=ver)
    os.replace(f"{OUT}/frag/tmp_{name}", p)


def stream(sid, frame, ob, info):
    from PIL import Image
    im = Image.new("RGB", (512, 256))
    if frame is not None: im.paste(Image.fromarray(frame).resize((256, 256)), (0, 0))
    im.paste(Image.fromarray(O.view_rgb(ob)).resize((256, 256), Image.NEAREST), (256, 0))
    d = f"{OUT}/dash/s"; im.save(f"{d}/{sid}.tmp.jpg", quality=85); os.replace(f"{d}/{sid}.tmp.jpg", f"{d}/{sid}.jpg")
    jdump(info, f"{d}/{sid}.json")


async def world(env):
    return await env.page.evaluate(f"() => window.__rlw ? window.__rlw({O.SPAN}) : null")


async def episode(env, sid, B, stop, S, greedy):
    obs0 = await asyncio.wait_for(env.reset(), 60)
    await env.page.evaluate(O.PAGE_JS)
    frame = next((x["frame"] for x in reversed(obs0["samples"]) if x.get("frame") is not None), None)
    t0 = time.time(); start = env.score; best = env.score; last_a = NB_MID; n = 0; t_stream = 0.0; kills = 0
    OB, SC, A, LP, V, R, D = [], [], [], [], [], [], []
    w = await world(env)
    if w is None or not any(o["me"] for o in w["sn"]): raise RuntimeError("no world state")
    k0 = w["kills"]; S.update(alive=True, score=env.score, last_ok=time.time())
    died = False
    while not stop.is_set():
        ob, sca, me = O.build(w, env.score, last_a)
        a, lp, v = await B.act(dict(obs=ob, sca=sca, greedy=greedy))
        if len(A) == T_FRAG:
            write_frag(sid, OB, SC, A, LP, V, R, D, B.policy.ver, v); OB, SC, A, LP, V, R, D = [], [], [], [], [], [], []
        dturn, boost = O.turn_of(a, me["sc"])
        env.actions = [(-math.degrees(dturn), boost)]            # harness convention: positive = left = decreasing angle
        _, rlog, done, info = await asyncio.wait_for(env.step(0), 10)
        if info.get("stall"): continue
        fr = [x["frame"] for x in info["samples"] if x.get("frame") is not None]
        if fr: frame = fr[-1]
        w2 = None if done else await world(env)
        if not done and (w2 is None or not any(o["me"] for o in w2["sn"])): done = True   # vanished: treat as death
        dk = 0 if done else max(0, w2["kills"] - k0)
        rw = DEATH if done else GROW * rlog + KILL * dk
        kills += dk
        OB.append(ob); SC.append(sca); A.append(a); LP.append(lp); V.append(v); R.append(rw); D.append(done)
        n += 1; last_a = a; best = max(best, info["score"]); S.update(score=info["score"], decisions=S["decisions"] + 1, last_ok=time.time())
        if time.time() - t_stream > 0.5:
            t_stream = time.time()
            try: stream(sid, frame, ob, dict(t=time.time(), sid=sid, score=info["score"], best=best, kills=kills, greedy=greedy, alive=not done,
                                             ep_s=round(time.time() - t0, 1), ver=B.policy.ver, a=a, boost=boost))
            except Exception: pass
        if done:
            died = True; write_frag(sid, OB, SC, A, LP, V, R, D, B.policy.ver, 0.0); break
        w = w2; k0 = w["kills"]
    S["alive"] = False
    rec = dict(t=time.time(), sid=sid, decisions=n, seconds=round(time.time() - t0, 1), start=start, final=env.score, best=best,
               kills=kills, died=died, greedy=greedy, ver=B.policy.ver)
    with open(f"{OUT}/dash/episodes.jsonl", "a") as fh: fh.write(json.dumps(rec) + "\n")
    S["episodes"] += 1


NB_MID = O.NB // 2


async def actor(env, sid, B, stop, S, greedy):
    while not stop.is_set():
        try: await episode(env, sid, B, stop, S, greedy); S["fails_row"] = 0
        except BaseException as e:
            if isinstance(e, asyncio.CancelledError): raise
            S["fails"] += 1; S["fails_row"] += 1; S["alive"] = False; S["err"] = f"{type(e).__name__}: {str(e)[:80]}"
            print(f"s{sid:03d}: {S['err']}; reloading", flush=True)
            await asyncio.sleep(min(30, 3 * S["fails_row"]))
            try: await asyncio.wait_for(env.close(), 30); await asyncio.wait_for(env.open(), 180)
            except Exception: await asyncio.sleep(10)


async def monitor(states, B, stop, rank):
    t0, n0 = time.time(), 0
    while not stop.is_set():
        await asyncio.sleep(5)
        el = time.time() - t0; dps = (B.n - n0) / el; t0, n0 = time.time(), B.n
        jdump(dict(t=time.time(), rank=rank, dps=dps, policy_ms=1000 * B.t / max(1, B.n), ver=B.policy.ver, sessions=states), f"{OUT}/dash/actor_{rank}.json")


async def watchdog(pw, a, px, sids, envs, browsers, tasks, states, B, stop):
    while not stop.is_set():
        await asyncio.sleep(10)
        for k, sid in enumerate(sids):
            S = states[k]
            if time.time() - S.get("last_ok", time.time()) < a.stale_s: continue
            print(f"watchdog: s{sid} stuck -> rebuilding browser", flush=True)
            S["last_ok"] = time.time(); S["rebuilds"] = S.get("rebuilds", 0) + 1; S["alive"] = False
            tasks[k].cancel()
            try: await asyncio.wait_for(asyncio.gather(tasks[k], return_exceptions=True), 15)
            except Exception: pass
            try: await asyncio.wait_for(browsers[k].close(), 20)
            except Exception: pass
            try:
                browsers[k] = await asyncio.wait_for(launch_browser(pw, proxy=px[k // a.per_proxy]), 60)
                envs[k] = SlitherEnv(browsers[k], nick=a.nick, render_fps=20, fpd=1, actions=[(0, 0)])
                await asyncio.wait_for(envs[k].open(), 120)
            except Exception as ex: print(f"watchdog: s{sid} reopen failed ({type(ex).__name__})", flush=True)
            tasks[k] = asyncio.create_task(actor(envs[k], sid, B, stop, S, S["greedy"]))


async def main(a):
    from playwright.async_api import async_playwright
    px = load_proxies(a.proxies)[a.proxy_lo:a.proxy_hi]
    sids = [(a.proxy_lo + i) * a.per_proxy + j for i in range(len(px)) for j in range(a.per_proxy)]
    B = Batcher(Policy()); stop = asyncio.Event()
    states = [dict(sid=sid, alive=False, score=0, decisions=0, episodes=0, fails=0, fails_row=0, greedy=(sid % 4 == 3), err="") for sid in sids]
    async with async_playwright() as pw:
        envs, browsers = [], []
        for k, sid in enumerate(sids):
            b = await launch_browser(pw, proxy=px[k // a.per_proxy]); browsers.append(b)
            e = SlitherEnv(b, nick=a.nick, render_fps=20, fpd=1, actions=[(0, 0)])
            try: await e.open()
            except Exception as ex: print(f"s{sid} open failed: {ex}", flush=True)
            envs.append(e); await asyncio.sleep(a.stagger)
        print(f"rank {a.rank}: {len(sids)} sessions", flush=True)
        for S in states: S["last_ok"] = time.time()
        tasks = [asyncio.create_task(actor(e, sid, B, stop, states[k], states[k]["greedy"])) for k, (e, sid) in enumerate(zip(envs, sids))]
        await asyncio.gather(B.loop(), monitor(states, B, stop, a.rank), watchdog(pw, a, px, sids, envs, browsers, tasks, states, B, stop))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--proxies", default="/tmp/slither/proxies.txt"); p.add_argument("--per_proxy", type=int, default=4)
    p.add_argument("--proxy_lo", type=int, default=0); p.add_argument("--proxy_hi", type=int, default=20)
    p.add_argument("--rank", type=int, default=0); p.add_argument("--stagger", type=float, default=1.0)
    p.add_argument("--nick", default="sp"); p.add_argument("--stale_s", type=float, default=60)
    asyncio.run(main(p.parse_args()))
