"""Live slither.io -> the exact observation the self-play policy was trained on (sp/sim.py World.observe).

The page returns a compact world (window.__rlw): every visible snake (head, angle, speed, thickness, body points) and
food near our head. We rasterise it on a 20-unit grid aligned to world coordinates and sample it the same way the
simulator does: 64x64 heading-up, zoom by size, 4 max-pooled sub-samples per cell, 8 channels + 8 scalars.
Coordinates are used raw (screen y down): the simulator's geometry is the same in any handedness, and the action
mapping below keeps the convention (positive turn = increasing angle).
"""
import math, numpy as np

CELL, V0, VB, TURN, DT = 20.0, 176.0, 425.0, math.radians(240.0), 0.1
NB, NA, OBS, NCH, NSC = 32, 64, 64, 8, 8
SP2UPS = 30.4            # game sp -> units/s (cruise sp 5.79 = 176 u/s, boost 14 = 425 u/s)
SPAN = 2700              # world units around the head the page sends (view half-diagonal at max zoom ~2260)
L = 2 * int(SPAN / CELL) + 8

PAGE_JS = """(() => { window.__rlw = (span) => {
  const s = window.slither; if (!s) return null; const hx = s.xx, hy = s.yy, S2 = span * span, sn = [], fd = [];
  for (const o of (window.slithers || [])) { if (!o || o.dead) continue; const px = [], py = [];
    for (const q of (o.pts || [])) { if (!q || q.dying) continue; const dx = q.xx - hx, dy = q.yy - hy;
      if (dx * dx + dy * dy < S2) { px.push(Math.round(q.xx)); py.push(Math.round(q.yy)); } }
    const dx = o.xx - hx, dy = o.yy - hy; if (!px.length && dx * dx + dy * dy >= S2) continue;
    sn.push({ me: o === s, x: o.xx, y: o.yy, a: o.ang, sp: o.sp, sc: o.sc || 1, px, py }); }
  for (const f of (window.foods || [])) { if (!f) continue; const dx = f.xx - hx, dy = f.yy - hy;
    if (dx * dx + dy * dy < S2) fd.push(Math.round(f.xx), Math.round(f.yy), f.sz); }
  return { grd: window.grd || 16384, sn, fd, kills: s.kill_count || 0 }; }; })();"""

_o = (np.arange(OBS) + 0.5 - OBS / 2) * CELL
_FI, _RJ = np.meshgrid(-_o, _o, indexing="ij")
OF, ORR = _FI.flatten(), _RJ.flatten()


def sc_of_score(score):  # sim: sc from length (used only for zoom/turn when the page's sc is missing)
    sct = max(2.0, (max(score, 1.0) / 4.2) ** (1 / 1.26)); return min(6.0, 1 + (sct - 2) / 106)
def scang_of(sc): return 0.13 + 0.87 * ((7 - sc) / 6) ** 2
def zoom_of(sc): return min(2.5, max(1.0, 1 + 0.3 * (sc - 1)))


def build(w, score, last_a):
    """w: page world dict, score: displayed length, last_a: previous action. -> obs [8,64,64] float16, sca [8] float32, me"""
    me = next(o for o in w["sn"] if o["me"])
    hx, hy, ang = me["x"], me["y"], me["a"]; sc = min(6.0, me["sc"])
    bx, by = math.floor(hx / CELL) - L // 2, math.floor(hy / CELL) - L // 2
    owner = np.zeros((L, L), np.int32); heads = np.zeros((L, L), np.int32); food = np.zeros((L, L), np.float32)
    n = len(w["sn"]); scs = np.zeros(n + 2, np.float32); vel = np.zeros((n + 2, 2), np.float32)
    for k, o in enumerate(w["sn"]):
        code = 1 if o["me"] else k + 2
        scs[code] = min(6.0, o["sc"]); v = o["sp"] * SP2UPS; vel[code] = (v * math.cos(o["a"]), v * math.sin(o["a"]))
        if o["px"]:
            gx = np.floor(np.asarray(o["px"]) / CELL).astype(np.int64) - bx; gy = np.floor(np.asarray(o["py"]) / CELL).astype(np.int64) - by
            m = (gx >= 0) & (gx < L) & (gy >= 0) & (gy < L); owner[gy[m], gx[m]] = code
        gx, gy = math.floor(o["x"] / CELL) - bx, math.floor(o["y"] / CELL) - by
        if 0 <= gx < L and 0 <= gy < L: owner[gy, gx] = code; heads[gy, gx] = code
    if w["fd"]:
        f = np.asarray(w["fd"], np.float32).reshape(-1, 3)
        gx = np.floor(f[:, 0] / CELL).astype(np.int64) - bx; gy = np.floor(f[:, 1] / CELL).astype(np.int64) - by
        m = (gx >= 0) & (gx < L) & (gy >= 0) & (gy < L)
        np.add.at(food, (gy[m], gx[m]), (f[m, 2] / 6.0) ** 2)      # pellet value ~ (size/6)^2: normal 0.25..2.3, remains 5+
        np.minimum(food, 50.0, out=food)
    c, s = math.cos(ang), math.sin(ang); z = zoom_of(sc)
    G, Rm = w["grd"], w["grd"] * 0.98
    ch = None
    for df, dr in ((-.25, -.25), (-.25, .25), (.25, -.25), (.25, .25)):
        of = (OF + df * CELL) * z; orr = (ORR + dr * CELL) * z
        x = hx + of * c - orr * s; y = hy + of * s + orr * c
        gx = np.floor(x / CELL).astype(np.int64) - bx; gy = np.floor(y / CELL).astype(np.int64) - by
        ok = (gx >= 0) & (gx < L) & (gy >= 0) & (gy < L); gx = gx.clip(0, L - 1); gy = gy.clip(0, L - 1)
        own = owner[gy, gx] * ok; hd = heads[gy, gx] * ok
        b0 = (own >= 2).astype(np.float32); m0 = (own == 1).astype(np.float32); z0 = scs[own] / 6 * b0
        h1 = (hd >= 2).astype(np.float32); hv = vel[hd]
        f1 = (hv[:, 0] * c + hv[:, 1] * s) / VB * h1; r1 = (-hv[:, 0] * s + hv[:, 1] * c) / VB * h1
        fo = np.log1p(food[gy, gx]) * ok / 2
        e1 = ((((x - G) ** 2 + (y - G) ** 2) > Rm ** 2) | ~ok).astype(np.float32)
        cur = [b0, h1, f1, r1, fo, e1, m0, z0]
        if ch is None: ch = cur
        else:
            take = h1 > ch[1]; ch[2] = np.where(take, f1, ch[2]); ch[3] = np.where(take, r1, ch[3])
            for i in (0, 1, 4, 5, 6, 7): ch[i] = np.maximum(ch[i], cur[i])
    obs = np.stack(ch).reshape(NCH, OBS, OBS).astype(np.float16)
    boost = me["sp"] > 10
    d = max(1e-3, math.hypot(hx - G, hy - G)); bear = math.atan2(G - hy, G - hx) - ang
    tl = last_a % NB
    sca = np.array([float(boost), (VB if boost else V0) / VB, math.log(max(score, 1)) / 10, (Rm - d) / Rm, math.sin(bear), math.cos(bear),
                    (tl - (NB - 1) / 2) / ((NB - 1) / 2), float(last_a // NB)], np.float32)
    return obs, sca, me


def turn_of(a, sc, gain=1.3):
    """action -> (heading change in radians, positive = increasing angle, boost). gain: the game lags commanded targets."""
    frac = ((a % NB) - (NB - 1) / 2) / ((NB - 1) / 2)
    return frac * TURN * scang_of(min(6.0, sc)) * DT * gain, int(a // NB)


def view_rgb(o):
    """[8,64,64] -> [64,64,3] uint8, same colours as the self-play dashboard."""
    o = o.astype(np.float32); rgb = np.zeros((OBS, OBS, 3), np.float32)
    rgb += o[4, ..., None].clip(0, 1) * np.array([90, 70, 140])
    rgb = np.where((o[5] > 0)[..., None], np.array([90, 30, 30]), rgb)
    rgb = np.where((o[6] > 0)[..., None], np.array([60, 110, 200]), rgb)
    b = 0.5 + 0.5 * o[7]
    rgb = np.where((o[0] > 0)[..., None], np.stack([b * 80, b * 220, b * 140], -1), rgb)
    rgb = np.where((o[1] > 0)[..., None], np.array([255, 255, 255]), rgb)
    rgb[OBS // 2 - 1:OBS // 2 + 1, OBS // 2 - 1:OBS // 2 + 1] = (255, 220, 0)
    return rgb.clip(0, 255).astype(np.uint8)
