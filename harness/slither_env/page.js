// Injected into each slither.io page. Everything the RL harness needs from the game, read from its own globals
// (window.slither = our snake, fpsls/fmlts = score tables, setAcceleration = boost, xm/ym = mouse target).
// A 10 Hz in-page sampler fills a buffer with (128x128 RGB frame, state) so Python only pulls once per decision.
(() => {
  if (window.__rl) return;
  const RES = 128;
  const off = document.createElement('canvas'); off.width = off.height = RES;
  const octx = off.getContext('2d', { willReadFrequently: true });
  const gameCanvas = () => {                      // the biggest visible canvas is the game view
    let best = null;
    for (const c of document.querySelectorAll('canvas'))
      if (c.style.display !== 'none' && (!best || c.width * c.height > best.width * best.height)) best = c;
    return best;
  };
  const score = (s) => {                          // the score slither.io shows (length)
    if (!s || !window.fpsls || !window.fmlts) return 0;
    return Math.floor(15 * (fpsls[s.sct] + s.fam / fmlts[s.sct] - 1) - 5);
  };
  // Danger of the current position, from the game's true state (used only for reward shaping, never as an observation):
  // max over other snakes' body points of exp(-gap / L) * (0.25 + 0.75 * forwardness), where gap = distance between
  // the bodies' edges and forwardness = cos of the angle between our heading and the point (0 if behind). Map edge too.
  const DANGER_L = 150;
  const danger = (s) => {
    if (!s) return { d: 0, gap: 1e9 };
    const hx = s.xx, hy = s.yy, ca = Math.cos(s.ang), sa = Math.sin(s.ang), r0 = 14.5 * (s.sc || 1);
    let d = 0, gmin = 1e9;
    const pt = (x, y, ro) => {
      const dx = x - hx, dy = y - hy, h = Math.hypot(dx, dy) || 1e-6, gap = Math.max(0, h - ro - r0);
      if (gap > 6 * DANGER_L) return;
      const fw = Math.max(0, (dx * ca + dy * sa) / h);
      if (fw <= 0 && gap > ro) return;
      const c = Math.exp(-gap / DANGER_L) * (0.25 + 0.75 * fw);
      if (c > d) d = c; if (gap < gmin) gmin = gap;
    };
    for (const o of (window.slithers || [])) {
      if (!o || o === s || o.dead) continue;
      const ro = 14.5 * (o.sc || 1); pt(o.xx, o.yy, ro);
      const P = o.pts || [];
      for (let j = P.length - 1; j >= 0; j -= 2) { const q = P[j]; if (q && !q.dying) pt(q.xx, q.yy, ro); }
    }
    const G = window.grd || 16384, cx = hx - G, cy = hy - G, rc = Math.hypot(cx, cy) || 1e-6;
    const edgeGap = Math.max(0, G * 0.98 - rc - r0), out = Math.max(0, (cx * ca + cy * sa) / rc);
    d = Math.max(d, Math.exp(-edgeGap / DANGER_L) * (0.25 + 0.75 * out));
    return { d, gap: Math.min(gmin, edgeGap) };
  };
  const state = () => {
    const s = window.slither;
    const alive = !!(s && window.playing && !s.dead && window.dead_mtm === -1);
    return { alive, playing: !!window.playing, connected: !!window.connected,
             score: s ? score(s) : 0, ang: s ? s.ang : 0, sp: s ? s.sp : 0,
             x: s ? s.xx : 0, y: s ? s.yy : 0, kills: s ? (s.kill_count || 0) : 0,
             rank: window.rank || 0, nsnakes: (window.slithers || []).length, t: performance.now(),
             danger: alive ? danger(s).d : 0 };
  };
  let crop = 1.0;                                 // fraction of the short side kept (zoom calibration vs the training videos)
  const grab = () => {                            // center square crop -> 128x128 RGB (same crop as the training videos)
    const c = gameCanvas(); if (!c) return null;
    const side = Math.min(c.width, c.height) * crop;
    octx.drawImage(c, (c.width - side) / 2, (c.height - side) / 2, side, side, 0, 0, RES, RES);
    const d = octx.getImageData(0, 0, RES, RES).data, rgb = new Uint8Array(RES * RES * 3);
    for (let i = 0, j = 0; i < d.length; i += 4, j += 3) { rgb[j] = d[i]; rgb[j + 1] = d[i + 1]; rgb[j + 2] = d[i + 2]; }
    let b = ''; for (let i = 0; i < rgb.length; i += 0x8000) b += String.fromCharCode.apply(null, rgb.subarray(i, i + 0x8000));
    return btoa(b);
  };
  let buf = [], timer = null, target = null, boost = 0;
  const steer = () => {                           // hold the commanded absolute heading
    if (target === null) return;
    window.xm = Math.cos(target) * 200; window.ym = Math.sin(target) * 200;
    if (typeof window.setAcceleration === 'function') window.setAcceleration(boost);
  };
  window.__rl = {
    state,
    setCrop(f) { crop = f; },
    start(hz) { if (timer) clearInterval(timer); buf = [];
      timer = setInterval(() => { steer(); const st = state(); st.frame = grab(); buf.push(st); if (buf.length > 600) buf.shift(); }, 1000 / hz); },
    stop() { if (timer) clearInterval(timer); timer = null; },
    pull() { const b = buf; buf = []; return b; },
    // dTurn: radians, positive = left (counter-clockwise on screen, the dataset's convention). Screen y points down,
    // so turning left means decreasing slither.ang.
    act(dTurn, b) { const s = window.slither; if (!s) return; target = s.ang - dTurn; boost = b ? 1 : 0; steer(); },
    release() { target = null; boost = 0; if (typeof window.setAcceleration === 'function') window.setAcceleration(0); },
  };
})();
