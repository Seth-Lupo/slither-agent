"""Actors (v2): live slither.io games, one decision per 0.1 s, sampled from the PPO policy (which starts as the
pretrained s150p behaviour-cloning head). One process per GPU serves a slice of the proxies, 4 sessions per proxy.
Sessions with sid % 4 == 3 act greedily (argmax per factor) as an evaluation of the current policy.

Trajectories go to the learner as 64-step fragments RL/frag/*.npz (state features, action, behaviour log-prob, reward,
done, plus the state after the last step for bootstrapping). Finished episodes -> RL/dash/episodes.jsonl.
"""
import argparse, asyncio, collections, json, math, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rlcore as R
sys.path.insert(0, os.environ.get("HARNESS", "/tmp/slither/harness"))
from slither_env.env import SlitherEnv, launch_browser, load_proxies   # noqa: E402

OUT = R.RL
for d in ("frag", "dash", "dash/s", "logs"): os.makedirs(f"{OUT}/{d}", exist_ok=True)
T_FRAG = 64


class Policy:
    def __init__(s):
        s.tokz, s.wm = R.load_tokenizer(), R.load_wm().to(torch.bfloat16)
        s.pi = R.Pi(s.wm.pol).cuda().eval(); s.ref = R.Pi(s.wm.pol).cuda().eval()     # ref = frozen pretrained policy
        s.vf = R.Vf().cuda().eval()
        s.ver, s.pupd, s.last_load = 0, 0, 0.0

    def reload(s):
        if time.time() - s.last_load < 5: return
        s.last_load = time.time()
        try: ck = torch.load(f"{OUT}/policy.pt", map_location="cuda")
        except Exception: return
        if ck["ver"] != s.ver:
            s.pi.load_state_dict(ck["pi"]); s.ver, s.pupd = ck["ver"], ck["pupd"]
            if "vf" in ck: s.vf.load_state_dict(ck["vf"])
            if "bb" in ck: s.wm.load_state_dict({k: v.to(torch.bfloat16) for k, v in ck["bb"].items()}, strict=False)   # fine-tuned layers

    @torch.no_grad()
    def __call__(s, batch):
        s.reload(); B = len(batch)
        newf = [o["new"] for o in batch]; cnt = [len(x) for x in newf]           # only frames not yet tokenised
        enc = s.tokz.encode(torch.from_numpy(np.concatenate(newf)).cuda()).view(-1, 256) if sum(cnt) else None
        toks, k = [], 0
        for o, c in zip(batch, cnt):
            old = torch.from_numpy(o["tok"].astype(np.int64)).cuda() if len(o["tok"]) else torch.zeros(0, 256, dtype=torch.long, device="cuda")
            t = torch.cat([old, enc[k:k + c]])[-8:]; k += c
            while len(t) < 8: t = torch.cat([t[:1], t])
            toks.append(t)
        tok = torch.stack(toks)
        act = torch.from_numpy(np.stack([o["act"] for o in batch])).cuda().long()
        feat, g, _ = R.encode_state(s.wm, tok, act); v = s.vf(g, feat)
        lp = R.bc_logp(s.pi(g, feat)); lp_ref = R.bc_logp(s.ref(g, feat))
        samp = torch.multinomial(lp.exp(), 1)[:, 0]
        lg = s.pi(g, feat); greedy_a = lg[:, :R.NB].argmax(-1) * 2 + lg[:, R.NB:].argmax(-1)
        rl = s.ref(g, feat)
        dist = torch.cat([torch.softmax(lg[:, :R.NB], -1), torch.softmax(lg[:, R.NB:], -1)[:, 1:],
                          torch.softmax(rl[:, :R.NB], -1), torch.softmax(rl[:, R.NB:], -1)[:, 1:]], 1).cpu().numpy().round(4)
        v_np, lp_np, lr_np = v.float().cpu().numpy(), lp.cpu().numpy(), lp_ref.cpu().numpy()
        out = []
        for i, o in enumerate(batch):
            a = int(greedy_a[i] if o["greedy"] else samp[i])
            d = dist[i]; n = R.NB
            out.append((a, dict(v=float(v_np[i]), logp=float(lp_np[i][a]), tok=tok[i].cpu().numpy().astype(np.uint16),
                                dist=dict(turn=d[:n].tolist(), boost=float(d[n]), turn0=d[n + 1:2 * n + 1].tolist(), boost0=float(d[2 * n + 1])))))
        return out


class Batcher:
    def __init__(s, policy): s.policy, s.pending, s.n, s.t = policy, [], 0, 0.0
    async def act(s, obs):
        fut = asyncio.get_running_loop().create_future(); s.pending.append((obs, fut)); return await fut
    async def loop(s):
        while True:
            await asyncio.sleep(0.005)
            if not s.pending: continue
            batch, s.pending = s.pending, []
            t = time.time()
            try: res = await asyncio.to_thread(s.policy, [o for o, _ in batch])
            except Exception as e:
                for _, f in batch: f.set_exception(e)
                continue
            s.n += len(batch); s.t += time.time() - t
            for (_, f), r in zip(batch, res): f.set_result(r)


def wrap(x): return (x + math.pi) % (2 * math.pi) - math.pi


def stream(sid, frame, info):
    from PIL import Image
    d = f"{OUT}/dash/s"
    Image.fromarray(frame).save(f"{d}/{sid}.tmp.jpg", quality=90); os.replace(f"{d}/{sid}.tmp.jpg", f"{d}/{sid}.jpg")
    R.jdump(info, f"{d}/{sid}.json")


def save_death(sid, clip, score):
    """Last 3 s before a confirmed death: frames, actions, danger, policy distributions -> RL/deaths (for analysis)."""
    if not clip: return
    d = f"{OUT}/deaths"; os.makedirs(d, exist_ok=True)
    if len(os.listdir(d)) > 3000: return
    fr, a, dz, pt, pb = zip(*clip)
    np.savez_compressed(f"{d}/d_{sid:03d}_{time.time_ns()}.npz", frames=np.stack(fr), a=np.array(a, np.int16), danger=np.array(dz, np.float32),
                        turn=np.array(pt, np.float32), boost=np.array(pb, np.float32), score=score)


def write_frag(sid, S, A, LP, RW, D, DZ, ver, terminal):
    """S: len(A)+1 states (dicts g, feat); the last is the bootstrap state (zeros if terminal)."""
    if not A: return
    p = f"{OUT}/frag/f_{sid:03d}_{time.time_ns()}.npz"
    np.savez(p + ".tmp.npz", tok=np.stack([x["tok"] for x in S]), act=np.stack([x["act"] for x in S]), v=np.array([x["v"] for x in S], np.float32), a=np.array(A, np.int16),
             logp=np.array(LP, np.float32), r=np.array(RW, np.float32), done=np.array(D), danger=np.array(DZ, np.float32), ver=ver, terminal=terminal)
    os.replace(p + ".tmp.npz", p)


async def episode(env, sid, batcher, stop, S, greedy):
    obs = await asyncio.wait_for(env.reset(), 60)
    newf, acts = [], collections.deque(maxlen=8); prev = [None]; tokh = np.zeros((0, 256), np.uint16)
    def push(x, boost):
        t = 0.0 if prev[0] is None else -math.degrees(wrap(x["ang"] - prev[0]))
        prev[0] = x["ang"]; newf.append(x["frame"]); acts.append(R.act_token(t, boost)); S["frame"] = x["frame"]
    for x in [x for x in obs["samples"] if x.get("frame") is not None][-8:]: push(x, 0)
    while len(acts) < 8: acts.appendleft(acts[0])
    St, A, LP, RW, D, DZ = [], [], [], [], [], []
    t0 = time.time(); start = env.score; best = env.score; t_stream = 0.0; n = 0; ret = 0.0; died = False
    dz = float(obs["state"].get("danger", 0.0)) if obs.get("state") else 0.0
    clip = collections.deque(maxlen=30)                                  # last 3 s, saved if the snake really dies
    S.update(alive=True, score=env.score, ep_t0=t0, last_ok=time.time()); pupd0 = batcher.policy.pupd
    while True:
        ctx_act = np.array(acts, np.uint8)
        a, ex = await batcher.act(dict(sid=sid, greedy=greedy, new=np.stack(newf[-8:]) if newf else np.zeros((0, 128, 128, 3), np.uint8),
                                       tok=tokh, act=ctx_act))
        tokh = ex["tok"]; newf = []
        St.append(dict(tok=ex["tok"], act=ctx_act, v=ex["v"]))
        if len(A) == T_FRAG:
            write_frag(sid, St, A, LP, RW, D, DZ, batcher.policy.ver, False); St, A, LP, RW, D, DZ = St[-1:], [], [], [], [], []
        obs, r, done, info = await asyncio.wait_for(env.step(a), 10)
        S.update(stalls=getattr(env, "stalls", 0), false_deaths=getattr(env, "false_deaths", 0), continued=getattr(env, "continued", 0))
        if info.get("stall"): St.pop(); continue                 # page stalled: no transition recorded
        new = [x for x in info["samples"] if x.get("frame") is not None]
        for x in new: push(x, a % 2)
        dz2 = float(info["samples"][-1].get("danger", dz)) if info["samples"] else dz
        rw = R.DEATH if done else R.ALIVE + R.GROW_W * r + R.SHAPE_K * (dz - R.GAMMA * dz2)
        A.append(a); LP.append(ex["logp"]); RW.append(rw); D.append(done); DZ.append(dz); n += 1; ret += r
        if new: clip.append((new[-1]["frame"], a, dz, ex["dist"]["turn"], ex["dist"]["boost"]))
        dz = dz2
        S["last_ok"] = time.time()
        best = max(best, info["score"]); S.update(score=info["score"], decisions=S["decisions"] + 1)
        if new and time.time() - t_stream > 0.25:
            t_stream = time.time()
            try: stream(sid, new[-1]["frame"], dict(t=time.time(), sid=sid, proxy=S["proxy"], score=info["score"], best=best, a=a,
                        greedy=greedy, dist=ex["dist"], danger=round(dz2, 3), alive=not done,
                        ep_s=round(time.time() - t0, 1), ver=batcher.policy.ver, pupd=batcher.policy.pupd))
            except Exception: pass
        if done:
            died = True; St.append(dict(tok=np.zeros_like(ex["tok"]), act=np.zeros_like(ctx_act), v=0.0))
            write_frag(sid, St, A, LP, RW, D, DZ, batcher.policy.ver, True)
            save_death(sid, clip, info["score"]); break
        if stop.is_set(): break
    S["alive"] = False
    rec = dict(t=time.time(), sid=sid, proxy=S["proxy"], decisions=n, seconds=round(time.time() - t0, 1), start=start,
               final=info["score"], best=best, ret=ret, died=died, eps=0.0 if greedy else 1.0, greedy=greedy,
               pupd=pupd0, ver=batcher.policy.ver)
    with open(f"{OUT}/dash/episodes.jsonl", "a") as fh: fh.write(json.dumps(rec) + "\n")
    S["episodes"] += 1


async def actor(env, sid, batcher, stop, S, greedy):
    while not stop.is_set():
        try: await episode(env, sid, batcher, stop, S, greedy); S["fails_row"] = 0
        except BaseException as e:
            if isinstance(e, asyncio.CancelledError): raise                  # the watchdog is replacing this session
            S["fails"] += 1; S["fails_row"] += 1; S["alive"] = False; S["err"] = f"{type(e).__name__}: {str(e)[:80]}"
            print(f"s{sid:03d}: {S['err']}; reloading", flush=True)
            await asyncio.sleep(min(30, 3 * S["fails_row"]))
            try: await asyncio.wait_for(env.close(), 30); await asyncio.wait_for(env.open(), 180)
            except Exception: await asyncio.sleep(10)


def ensure_web():
    import subprocess
    if subprocess.run(["pgrep", "-f", "[c]loudflared tunnel"], capture_output=True).returncode != 0 and os.path.exists(f"{HERE}/web.sh"):
        subprocess.Popen(["bash", f"{HERE}/web.sh"], stdout=open(f"{OUT}/logs/websh.log", "a"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)


async def monitor(states, batcher, stop, rank):
    from PIL import Image, ImageDraw
    t0, n0, t_web = time.time(), 0, 0.0
    while not stop.is_set():
        await asyncio.sleep(5)
        if rank == 0 and time.time() - t_web > 120:
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
            dr.text((x + 6, y + 4), f"s{S['sid']}{' eval' if S['greedy'] else ''}  len {S['score']}", fill=(255, 255, 255))
        p = f"{OUT}/dash/live_{rank}.jpg"; im.save(p + ".tmp.jpg", quality=80); os.replace(p + ".tmp.jpg", p)
        R.jdump(dict(t=time.time(), rank=rank, dps=dps, policy_ms=1000 * batcher.t / max(1, batcher.n), ver=batcher.policy.ver,
                     pupd=batcher.policy.pupd, sessions=[{k: v for k, v in S.items() if k != "frame"} for S in states]),
                f"{OUT}/dash/actor_{rank}.json")


async def watchdog(pw, a, px, sids, envs, browsers, tasks, states, batcher, stop):
    """Self-healing: a session with no successful step/reset for --stale_s (crashed browser, wedged page, dead proxy
    connection) is torn down completely and rebuilt with a fresh Chromium, so the run stays near 80 live games."""
    while not stop.is_set():
        await asyncio.sleep(10)
        for k, sid in enumerate(sids):
            S = states[k]
            if time.time() - S.get("last_ok", time.time()) < a.stale_s: continue
            print(f"watchdog: s{sid} stuck {time.time() - S['last_ok']:.0f}s -> rebuilding browser", flush=True)
            S["last_ok"] = time.time(); S["rebuilds"] = S.get("rebuilds", 0) + 1; S["alive"] = False
            tasks[k].cancel()
            try: await asyncio.wait_for(asyncio.gather(tasks[k], return_exceptions=True), 15)
            except Exception: pass
            try: await asyncio.wait_for(browsers[k].close(), 20)
            except Exception: pass
            try:
                browsers[k] = await asyncio.wait_for(launch_browser(pw, proxy=px[k // a.per_proxy]), 60)
                envs[k] = SlitherEnv(browsers[k], nick=a.nick, render_fps=20, fpd=1, actions=R.ACTIONS, crop=a.crop)
                await asyncio.wait_for(envs[k].open(), 120)
            except Exception as ex: print(f"watchdog: s{sid} rebuild open failed ({type(ex).__name__}); actor will retry", flush=True)
            tasks[k] = asyncio.create_task(actor(envs[k], sid, batcher, stop, S, S["greedy"]))


async def main(a):
    from playwright.async_api import async_playwright
    px = load_proxies(a.proxies)[a.proxy_lo:a.proxy_hi]
    sids = [(a.proxy_lo + i) * a.per_proxy + j for i in range(len(px)) for j in range(a.per_proxy)]
    policy = Policy(); batcher = Batcher(policy); stop = asyncio.Event()
    R.jdump(dict(turn_centers_deg=[round(float(c), 2) for c in R.CENTERS]), f"{OUT}/dash/meta.json")
    states = [dict(sid=sid, proxy=sid // a.per_proxy, alive=False, score=0, decisions=0, episodes=0, fails=0, fails_row=0,
                   greedy=(sid % 4 == 3), err="", frame=None) for sid in sids]
    async with async_playwright() as pw:
        envs, browsers = [], []
        for k, sid in enumerate(sids):
            b = await launch_browser(pw, proxy=px[k // a.per_proxy]); browsers.append(b)
            e = SlitherEnv(b, nick=a.nick, render_fps=20, fpd=1, actions=R.ACTIONS, crop=a.crop)
            try: await e.open()
            except Exception as ex: print(f"s{sid} open failed: {ex}", flush=True)
            envs.append(e); await asyncio.sleep(a.stagger)
        print(f"rank {a.rank}: {len(sids)} sessions on proxies {a.proxy_lo}..{a.proxy_hi - 1}", flush=True)
        for S in states: S["last_ok"] = time.time()
        tasks = [asyncio.create_task(actor(e, sid, batcher, stop, states[k], states[k]["greedy"])) for k, (e, sid) in enumerate(zip(envs, sids))]
        await asyncio.gather(batcher.loop(), monitor(states, batcher, stop, a.rank),
                             watchdog(pw, a, px, sids, envs, browsers, tasks, states, batcher, stop))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--proxies", default="/tmp/slither/proxies.txt"); p.add_argument("--per_proxy", type=int, default=4)
    p.add_argument("--proxy_lo", type=int, default=0); p.add_argument("--proxy_hi", type=int, default=20)
    p.add_argument("--rank", type=int, default=0); p.add_argument("--stagger", type=float, default=1.0)
    p.add_argument("--nick", default="dyna")
    p.add_argument("--crop", type=float, default=1.0, help="zoom (calib.py: 0.6 fits the world model best, but 1.0 played no worse)")
    p.add_argument("--stale_s", type=float, default=60, help="rebuild a session with no progress for this long")
    asyncio.run(main(p.parse_args()))
