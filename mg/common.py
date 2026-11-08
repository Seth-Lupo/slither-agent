"""Shared paths, video list, frame selection and decoding for the slither MaskGIT world model.

Frames: 10 fps of game time, center square crop (drops leaderboard, minimap and score), 128x128.
Actions per frame t: heading change over (t-1, t) relative to (t-2, t-1), and boost during (t-1, t).
"""
import glob, json, os, subprocess, zlib
import numpy as np

ROOT = os.environ.get("MG_ROOT", "/tmp/slither/mg")
SRC = f"{ROOT}/src"                  # HF snapshot: prepared/, raw/, actions/, reports/, metadata/, heads_v1/tracks/
DATA = f"{ROOT}/data"
DASH = f"{ROOT}/dash"
RES, FPS = 128, 10
NB = 32                              # heading-change bins; NB = "no action"
FF = os.environ.get("FFMPEG", "ffmpeg")


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f: json.dump(obj, f)
    os.replace(path + ".tmp", path)


def stage(name, **kw):
    """Pipeline progress for the dashboard."""
    p = f"{DASH}/status.json"
    try: s = json.load(open(p))
    except Exception: s = {}
    import time
    s[name] = dict(kw, t=time.time()); jdump(s, p)


def is_val(mid):
    """Train/val split by session (YouTube id): ~5% of sessions are validation."""
    return zlib.crc32(mid[:11].encode()) % 20 == 0


def videos():
    """All videos with action labels: [{mid, path, src (selected source-frame indices)}]."""
    rates = {}
    for m in glob.glob(f"{SRC}/metadata/*.json"):
        for p in json.load(open(m)).get("current_prepared", []): rates[p["media_id"]] = p["playback_rate"]
    out = []
    for a in sorted(glob.glob(f"{SRC}/actions/*.npz")):
        mid = os.path.basename(a)[:-4]
        if mid.endswith("-raw"):
            path = f"{SRC}/raw/{mid[:-4]}.mp4"
            rep = json.load(open(f"{SRC}/reports/{mid}.json"))
            n, step = rep["frames"], rep["fps"] / FPS
            src = np.round(np.arange(0, n - 1, step)).astype(np.int64)
        else:
            path, tr = f"{SRC}/prepared/{mid}.mp4", f"{SRC}/heads_v1/tracks/{mid}.npz"
            if not os.path.exists(tr): continue
            t = np.load(tr); keep = t["keep_idx"].astype(np.int64); n = int(t["src_frames"])
            game_fps = len(keep) / (n / 30.0 * rates.get(mid, 1.0))       # unique frames per second of game time
            sel = np.round(np.arange(0, len(keep), game_fps / FPS)).astype(np.int64)
            src = keep[sel[sel < len(keep)]]
        if os.path.exists(path) and len(src) > 32: out.append(dict(mid=mid, path=path, src=src))
    return out


def decode(path, src):
    """Decode only the selected source frames -> [len(src), RES, RES, 3] uint8."""
    cmd = [FF, "-v", "error", "-threads", "2", "-i", path, "-frames:v", str(int(src[-1]) + 1), "-vf", f"crop=ih:ih,scale={RES}:{RES}:flags=area",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 22)
    fs = RES * RES * 3; want = np.zeros(int(src[-1]) + 1, bool); want[src] = True
    out = np.zeros((len(src), RES, RES, 3), np.uint8); k = 0
    for i in range(len(want)):
        b = p.stdout.read(fs)
        if len(b) < fs: break
        if want[i]: out[k] = np.frombuffer(b, np.uint8).reshape(RES, RES, 3); k += 1
    p.stdout.close(); p.kill(); p.wait()
    return out[:k]


def actions(mid, src):
    """Per selected frame: turn (deg, + = counter-clockwise), boost, valid, cut (discontinuity before this frame)."""
    a = np.load(f"{SRC}/actions/{mid}.npz")
    hd, bo, am, sc = a["heading_deg"].astype(np.float64), a["boost"], a["action_mask"][:, 0], a["scene_cut"]
    n, T = len(hd), len(src)
    head = np.zeros(T); boost = np.zeros(T, bool); ok = np.zeros(T, bool); cut = np.zeros(T, bool)
    for t in range(1, T):
        s0, s1 = int(src[t - 1]), min(int(src[t]), n)
        if s1 <= s0: cut[t] = True; continue
        ang = np.radians(hd[s0:s1])
        head[t] = np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))
        boost[t] = bo[s0:s1].mean() > 0.5
        ok[t] = bool(am[s0:s1].all())
        cut[t] = bool(sc[s0:s1].any())
    turn = (head - np.roll(head, 1) + 180) % 360 - 180
    valid = ok & np.roll(ok, 1) & ~cut & ~np.roll(cut, 1); valid[:2] = False
    return dict(turn=turn.astype(np.float32), boost=boost, valid=valid, cut=cut)


def action_tokens(turn, boost, valid, edges):
    """-> [T, 2] uint8: (turn bin 0..NB-1 or NB=no-action, boost 0/1 or 2=no-action)."""
    tb = np.searchsorted(edges, turn).astype(np.uint8)
    out = np.stack([np.where(valid, tb, NB), np.where(valid, boost.astype(np.uint8), 2)], 1)
    return out.astype(np.uint8)
