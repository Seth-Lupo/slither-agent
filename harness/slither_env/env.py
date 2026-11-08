"""One live slither.io game in a headless Chromium page, as an RL environment.

Timing matches the world model: frames are sampled in-page at 10 Hz (128x128 center-square crop, like the training
videos) and one decision lasts FRAMES_PER_DECISION = 3 frames (0.3 s). The game runs in real time, so step() waits for
those frames to happen.

Reward: change in log(length) over the decision (length = the score slither.io shows). An episode's return is therefore
log(final length / start length). Death ends the episode (done=True), detected from the game's own state.

    env = SlitherEnv(browser); await env.open()
    obs = await env.reset()                 # dict(frames=[k,128,128,3] uint8, state=...)
    obs, r, done, info = await env.step(a)  # a = index into ACTIONS
"""
import asyncio, base64, math, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_JS = open(os.path.join(HERE, "page.js")).read()
URL = "http://slither.io/"
HZ = 10
FRAMES_PER_DECISION = 3
RES = 128
TURNS_DEG = (-90, -45, -20, 0, 20, 45, 90)        # commanded heading change per decision; positive = left (dataset convention)
ACTIONS = [(t, b) for t in TURNS_DEG for b in (0, 1)]   # 14 actions: (turn degrees, boost)


RAF_CAP = """(() => { const fps = %d; if (!fps) return; const raf = window.requestAnimationFrame.bind(window); let last = 0;
  window.requestAnimationFrame = (cb) => raf(function tick(ts) { if (ts - last >= 1000 / fps - 1) { last = ts; cb(ts); } else raf(tick); }); })();"""


def decode_frame(b64):
    return np.frombuffer(base64.b64decode(b64), np.uint8).reshape(RES, RES, 3)


class SlitherEnv:
    def __init__(self, browser, nick="rl", viewport=(1280, 720), ctx_frames=8, start_timeout=45.0, crop=1.0, render_fps=0,
                 fpd=FRAMES_PER_DECISION, actions=ACTIONS):
        """fpd: frames per decision (1 = act every 0.1 s); actions: table of (turn degrees over the decision, boost)."""
        self.browser, self.nick, self.viewport, self.crop, self.render_fps = browser, nick, viewport, crop, render_fps
        self.fpd, self.actions = fpd, actions
        self.button_timeout = float(os.environ.get("SLITHER_BUTTON_TIMEOUT", "10"))    # s to wait for Play after a death
        self.debug_dir = os.environ.get("SLITHER_DEBUG_DIR")
        self.ctx_frames, self.start_timeout = ctx_frames, start_timeout
        self.context = self.page = None
        self.score = 0; self.alive = False; self.episodes = 0

    async def open(self):
        self.context = await self.browser.new_context(viewport=dict(width=self.viewport[0], height=self.viewport[1]))
        self.page = await self.context.new_page()
        await self.page.add_init_script(RAF_CAP % self.render_fps)   # cap the game's render loop (CPU: software WebGL)
        await self._load()

    async def _load(self, tries=4):
        for i in range(tries):
            try:
                await self.page.goto(URL, wait_until="commit", timeout=45000)
                await self.page.wait_for_selector("#playh .btnt", state="visible", timeout=60000)
                break
            except Exception as e:
                if i == tries - 1: raise
                print(f"page load failed ({type(e).__name__}), retry {i + 1}", flush=True); await asyncio.sleep(2 + 3 * i)
        await self.page.evaluate(PAGE_JS)
        await self.page.evaluate(f"() => window.__rl.setCrop({self.crop})")

    async def close(self):
        if self.context: await self.context.close()

    async def _pull(self):
        samples = await self.page.evaluate("() => window.__rl.pull()")
        for s in samples: s["frame"] = decode_frame(s["frame"]) if s.get("frame") else None
        return samples

    async def reset(self):
        """Start a new game and return once ctx_frames frames of it exist. Reloads the page if the menu misbehaves."""
        for attempt in range(3):
            try:
                already = await self.page.evaluate("() => window.__rl.state().alive")
                if not already:                              # a snake that is still playing is simply continued
                    await self.page.evaluate("() => { window.__rl.release(); window.__rl.stop(); }")
                    await self.page.wait_for_selector("#playh .btnt", state="visible", timeout=self.button_timeout * 1000)
                    await self.page.fill("#nick", self.nick)
                    await self.page.click("#playh .btnt")
                else: self.continued = getattr(self, "continued", 0) + 1
                t0 = time.time()
                while not (await self.page.evaluate("() => window.__rl.state().alive")):
                    if time.time() - t0 > self.start_timeout: raise TimeoutError("game did not start")
                    await asyncio.sleep(0.1)
                await self.page.evaluate(f"() => window.__rl.start({HZ})")
                frames = []
                while len(frames) < self.ctx_frames:
                    await asyncio.sleep(1 / HZ); frames += await self._pull()
                frames = frames[-self.ctx_frames:]
                st = frames[-1]; self.alive = st["alive"]; self.score = max(st["score"], 1); self.episodes += 1
                return dict(frames=np.stack([f["frame"] for f in frames]), samples=frames, state=st)
            except Exception as e:
                print(f"reset attempt {attempt + 1} failed ({str(e)[:80]}); reloading", flush=True)
                if self.debug_dir and not os.path.exists(f"{self.debug_dir}/stuck.png"):
                    try: os.makedirs(self.debug_dir, exist_ok=True); await self.page.screenshot(path=f"{self.debug_dir}/stuck.png")
                    except Exception: pass
                await self._load()
        raise RuntimeError("could not start a game")

    async def step(self, a):
        """Hold action a for one decision (fpd frames). -> obs, reward, done, info"""
        turn, boost = self.actions[a]
        await self.page.evaluate(f"() => window.__rl.act({math.radians(turn)}, {boost})")
        got, t0 = [], time.time()
        while len(got) < self.fpd:
            await asyncio.sleep(0.015 if self.fpd == 1 else 0.03); got += await self._pull()
            if got and not got[-1]["alive"]: break
            if time.time() - t0 > 3.0: break                     # page stalled
        if not got:                                               # a stall is not a death: no transition, try again
            self.stalls = getattr(self, "stalls", 0) + 1
            return dict(frames=np.zeros((0, RES, RES, 3), np.uint8), state=None), 0.0, False, dict(score=self.score, samples=[], late=0.0, stall=True)
        alive = True
        if not got[-1]["alive"]:                                  # death needs confirmation: still not alive 0.5 s later
            for _ in range(5):
                await asyncio.sleep(0.1)
                if await self.page.evaluate("() => window.__rl.state().alive"): break
            else: alive = False
            if alive: self.false_deaths = getattr(self, "false_deaths", 0) + 1
        for x in got: x["alive"] = x["alive"] or alive               # a transient flicker is not death
        live = [s["score"] for s in got if s["alive"]]
        new_score = max(live[-1] if live else self.score, 1)      # score at the end of the decision (boost can shrink it)
        r = math.log(new_score) - math.log(self.score)
        self.score, self.alive = new_score, alive
        frames = [s["frame"] for s in got if s["frame"] is not None]
        info = dict(score=new_score, samples=got, late=time.time() - t0 - self.fpd / HZ)
        if not alive: await self.page.evaluate("() => { window.__rl.release(); window.__rl.stop(); }")
        return dict(frames=np.stack(frames) if frames else np.zeros((0, RES, RES, 3), np.uint8), state=got[-1] if got else None), r, not alive, info


RENDER = {   # SLITHER_RENDER picks how the page is rasterised on GPU-less nodes (see README: cost per session)
    "swiftshader": ["--use-gl=angle", "--use-angle=swiftshader"],                    # emulated WebGL: ~1.6 cores (Mac), ~8 (Xeon)
    "nogpu": ["--disable-gpu", "--disable-gpu-compositing", "--disable-accelerated-2d-canvas"],   # plain Skia CPU raster
}


def load_proxies(path):
    """Proxy list, one per line: host:port:user:pass (Webshare) or scheme://user:pass@host:port -> Playwright proxy dicts."""
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"): continue
        if "://" in line:
            from urllib.parse import urlparse
            u = urlparse(line); out.append(dict(server=f"{u.scheme}://{u.hostname}:{u.port}", username=u.username or "", password=u.password or ""))
        else:
            h, port, user, pw = line.split(":", 3); out.append(dict(server=f"http://{h}:{port}", username=user, password=pw))
    return out


async def launch_browser(pw, headless=True, render=None, proxy=None):
    """Chromium for GPU-less or headless cluster nodes; proxy = Playwright proxy dict (all of this browser's traffic)."""
    render = render or os.environ.get("SLITHER_RENDER", "nogpu")
    return await pw.chromium.launch(headless=headless, proxy=proxy, args=RENDER[render] + [
        "--mute-audio", "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows"])
