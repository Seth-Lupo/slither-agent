"""Many live sessions at once, one batched policy, every episode recorded.

Each session runs its own actor loop (the game is real time, so sessions never wait for each other; a session that is
restarting after a death doesn't hold the rest up). Decisions are batched: actors post their observation, and the
policy runs on whatever is pending every few milliseconds, in a worker thread so it can be a GPU model.

    policy(batch) -> list of action indices, batch = list of dict(frames=[k,128,128,3] uint8 (latest k frames), state=...)

Episodes are written to OUT/ep_<session>_<n>.npz (recorder.py) for the replay buffer.
"""
import asyncio, collections, time
import numpy as np
from .env import SlitherEnv, launch_browser, load_proxies, ACTIONS
from .recorder import Episode


class Batcher:
    def __init__(self, policy, max_wait=0.01):
        self.policy, self.max_wait, self.pending = policy, max_wait, []
        self.stats = collections.Counter()

    async def act(self, obs):
        fut = asyncio.get_running_loop().create_future(); self.pending.append((obs, fut)); return await fut

    async def loop(self):
        while True:
            await asyncio.sleep(self.max_wait)
            if not self.pending: continue
            batch, self.pending = self.pending, []
            t = time.time()
            acts = await asyncio.to_thread(self.policy, [o for o, _ in batch])
            self.stats["batches"] += 1; self.stats["decisions"] += len(batch); self.stats["policy_s"] += time.time() - t
            for (_, f), a in zip(batch, acts): f.set_result(int(a))


async def actor(env, sid, batcher, out, ctx, stop, log):
    while not stop.is_set():
        try: await episode(env, sid, batcher, out, ctx, stop, log)
        except Exception as e:                                  # one broken page never stops the run: reload it
            print(f"s{sid:03d}: {type(e).__name__}: {str(e)[:150]}; reloading", flush=True)
            try: await env.close(); await env.open()               # fresh page (the old one may have crashed)
            except Exception: await asyncio.sleep(10)


async def episode(env, sid, batcher, out, ctx, stop, log):
        obs = await env.reset(); ep = Episode(sid, env.episodes, obs)
        hist = collections.deque(obs["frames"], maxlen=ctx)
        while True:
            a = await batcher.act(dict(frames=np.stack(hist), state=obs["state"]))
            obs, r, done, info = await env.step(a)
            hist.extend(obs["frames"]); ep.add(a, r, done, info)
            batcher.stats["steps"] += 1; batcher.stats["late"] += max(0.0, info["late"])
            if done or stop.is_set(): break
        path = ep.save(out); log(sid, ep, path)


async def run(policy, n_sessions, out, n_browsers=None, seconds=None, ctx=8, crop=1.0, nick="rl", stagger=1.0, render_fps=0,
              proxies=None, per_proxy=4):
    """Run n_sessions live games until `seconds` pass (or forever). n_browsers Chromium processes share the pages.
    proxies: file of proxies; session i uses proxy i // per_proxy (slither.io allows ~4 live games per IP)."""
    px = load_proxies(proxies) if proxies else None
    if px: n_sessions = min(n_sessions, len(px) * per_proxy)
    from playwright.async_api import async_playwright
    n_browsers = n_browsers or n_sessions      # one Chromium per session: background tabs get throttled
    stop, batcher = asyncio.Event(), Batcher(policy)
    done_eps = []
    def log(sid, ep, path):
        done_eps.append(ep.summary()); s = ep.summary()
        print(f"[{time.strftime('%H:%M:%S')}] s{sid:03d} ep{s['episode']}: {s['decisions']} decisions, score {s['start_score']}->{s['final_score']}, "
              f"return {s['return']:+.3f}, {'died' if s['died'] else 'cut'} -> {path}", flush=True)
    async with async_playwright() as pw:
        browsers = [await launch_browser(pw, proxy=px[i // per_proxy] if px else None) for i in range(n_browsers)]
        envs = []
        for i in range(n_sessions):                                 # stagger joins so the servers don't see a burst
            e = SlitherEnv(browsers[i % n_browsers], nick=nick, ctx_frames=ctx, crop=crop, render_fps=render_fps)
            try: await e.open(); envs.append(e)
            except Exception as ex: print(f"session {i} failed to open ({type(ex).__name__}), skipped", flush=True)
            await asyncio.sleep(stagger)
        print(f"{len(envs)}/{n_sessions} sessions open", flush=True)
        tasks = [asyncio.create_task(actor(e, i, batcher, out, ctx, stop, log)) for i, e in enumerate(envs)]
        bl = asyncio.create_task(batcher.loop()); t0 = time.time()
        try:
            while seconds is None or time.time() - t0 < seconds:
                await asyncio.sleep(10)
                el = time.time() - t0; st = batcher.stats
                print(f"[{time.strftime('%H:%M:%S')}] {el:6.0f}s  {st['decisions'] / el:6.1f} decisions/s "
                      f"(target {n_sessions * 10 / 3:.0f}), episodes {len(done_eps)}, mean batch {st['decisions'] / max(1, st['batches']):.1f}, mean lateness {st['late'] / max(1, st['steps']) * 1000:.0f} ms/decision", flush=True)
        finally:
            stop.set(); await asyncio.wait(tasks, timeout=15); bl.cancel()
            for b in browsers: await b.close()
    return done_eps


def random_policy(batch):
    return np.random.randint(len(ACTIONS), size=len(batch))
