import asyncio, os, sys, time, math, numpy as np
sys.argv = ["x"]; import live as Lv, obs as O
async def main():
    from playwright.async_api import async_playwright
    px = Lv.load_proxies("/tmp/slither/proxies.txt")
    B = Lv.Batcher(Lv.Policy()); stop = asyncio.Event()
    S = dict(sid=999, alive=False, score=0, decisions=0, episodes=0, fails=0, fails_row=0, greedy=False, err="")
    async with async_playwright() as pw:
        b = await Lv.launch_browser(pw, proxy=px[19]); e = Lv.SlitherEnv(b, nick="sp", render_fps=20, fpd=1, actions=[(0, 0)]); await e.open()
        lt = asyncio.create_task(B.loop()); t = asyncio.create_task(Lv.episode(e, 999, B, stop, S, False))
        t0 = time.time()
        while time.time() - t0 < 40 and not t.done(): await asyncio.sleep(1)
        stop.set(); 
        try: await asyncio.wait_for(t, 15)
        except Exception as ex: print("ep err", repr(ex))
        w = await Lv.world(e); print("snakes", len(w["sn"]) if w else None, "foods", len(w["fd"])//3 if w else None)
        print("decisions", S["decisions"], "rate/s", S["decisions"]/40, "policy ms", 1000*B.t/max(1,B.n))
asyncio.run(main())
