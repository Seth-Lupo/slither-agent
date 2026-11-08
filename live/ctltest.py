import asyncio, math, sys, time, numpy as np
sys.argv = ["x"]; import live as Lv, obs as O
def wrap(x): return (x + math.pi) % (2 * math.pi) - math.pi
async def main():
    from playwright.async_api import async_playwright
    px = Lv.load_proxies("/tmp/slither/proxies.txt"); pol = Lv.Policy()
    async with async_playwright() as pw:
        b = await Lv.launch_browser(pw, proxy=px[19]); e = Lv.SlitherEnv(b, nick="sp", render_fps=20, fpd=1, actions=[(0, 0)]); await e.open()
        await e.reset(); await e.page.evaluate(O.PAGE_JS)
        rec = []; last_a = 16; w = await Lv.world(e); t_prev = time.time()
        for i in range(250):
            ob, sca, me = O.build(w, e.score, last_a)
            a, lp, v = pol([dict(obs=ob, sca=sca, greedy=False)])[0]
            dturn, boost = O.turn_of(a, me["sc"]); e.actions = [(-math.degrees(dturn), boost)]
            _, r, done, info = await e.step(0)
            if done: break
            w2 = await Lv.world(e); me2 = next(o for o in w2["sn"] if o["me"]); t = time.time()
            rec.append((t - t_prev, dturn / 1.3, wrap(me2["a"] - me["a"]), boost, me2["sp"], me["sc"])); t_prev = t; w = w2; last_a = a
        R = np.array(rec); dt = R[:, 0]
        print("n", len(R), "decision interval s: median %.3f p90 %.3f" % (np.median(dt), np.percentile(dt, 90)))
        want = R[:, 1] * dt / 0.1          # policy's intended turn rate x actual interval
        big = np.abs(R[:, 1]) > 0.05
        print("intended per-0.1s turn (deg) median |x| %.1f ; actual per-decision |dang| median %.1f" % (np.degrees(np.median(np.abs(R[:, 1]))), np.degrees(np.median(np.abs(R[:, 2])))))
        print("sign agreement on big turns %.2f" % np.mean(np.sign(R[big, 1]) == np.sign(R[big, 2])))
        print("actual/intended-per-0.1s ratio (big turns) median %.2f" % np.median(np.abs(R[big, 2]) / np.abs(R[big, 1])))
        print("actual / intended over the real interval median %.2f" % np.median(np.abs(R[big, 2]) / np.abs(want[big])))
        for lag in range(4):
            c = R[:len(R) - lag, 1]; d = R[lag:, 2]; bb = R[:len(R) - lag, 3]; spd = R[lag:, 4]
            print("lag %d: turn corr %.2f  sign agree %.2f  | sp boost %.1f vs not %.1f" % (lag, np.corrcoef(c, d)[0, 1], np.mean(np.sign(c[np.abs(c) > .05]) == np.sign(d[np.abs(c) > .05])), spd[bb == 1].mean(), spd[bb == 0].mean()))
        print("boost frac %.2f ; sp when boost %.1f, when not %.1f" % (R[:, 3].mean(), R[R[:, 3] == 1, 4].mean() if (R[:, 3] == 1).any() else -1, R[R[:, 3] == 0, 4].mean()))
asyncio.run(main())
