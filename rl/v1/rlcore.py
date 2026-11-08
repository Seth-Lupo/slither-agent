"""Shared pieces of the Dyna-Q system: the frozen world model (s150p), its features, the RL heads, action tokens.

State features f(s) = the world model's hidden states for the latest context frame (mean over its 258 positions; the
model is block-causal so they see frames <= t and actions <= t) -- the same features the pretrained policy head used.
The world model stays frozen (stop-gradient features); only the small heads below learn:
  Q(f, .)  14 action values           R(f, .)  expected reward (10 x delta log length) per action
  D(f, .)  logit of dying during the decision, per action
"""
import json, math, os, sys
import numpy as np, torch, torch.nn as nn

MG = os.environ.get("MG_ROOT", "/tmp/slither/mg"); sys.path.insert(0, f"{MG}/code")
import common as C, dynamics as M, tokenizer as T          # noqa: E402

RL = os.environ.get("RL_ROOT", "/tmp/slither/rl")
WM_CKPT = os.environ.get("WM_CKPT", f"{MG}/runs/s150p/ckpt.pt")
TURNS = (-90, -45, -20, 0, 20, 45, 90)                     # must match harness env.ACTIONS
ACTIONS = [(t, b) for t in TURNS for b in (0, 1)]
NA = len(ACTIONS)
EDGES = np.array(json.load(open(f"{C.DATA}/meta.json"))["edges"])
REWARD_SCALE = 10.0
GAMMA = 0.98                                                 # per decision (0.3 s): horizon ~15 s


def act_token(turn_deg, boost):
    """Heading change of one frame (deg, + = left) and boost -> the world model's (turn bin, boost) action token."""
    return int(np.searchsorted(EDGES, turn_deg)), int(boost)


def load_wm(dev="cuda"):
    ck = torch.load(WM_CKPT, map_location="cpu", weights_only=False)
    m = M.Dyn(ck["cfg"]["size"], policy=ck["cfg"].get("pol_w", 0) > 0)
    m.load_state_dict(ck["model"]); return m.to(dev).eval().requires_grad_(False)


def load_tokenizer(dev="cuda"): return T.load(dev=dev)


@torch.no_grad()
def features(m, tok, act):
    """tok [B,F,256] long, act [B,F,2] long -> [B,d] float32 features of the last frame."""
    B, Fn = tok.shape[:2]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pos = torch.cat([m.aslot, (m.row + m.col).view(M.NT, m.d)])
        x = torch.cat([m.turn(act[..., :1]), m.boost(act[..., 1:]), m.tok(tok)], 2) + pos + m.time[:Fn]
        x = x.view(B, Fn * M.SLOT, m.d); mk = m.block_mask(Fn * M.SLOT, x.device)
        for b in m.blocks: x = b(x, mk)
        x = m.norm(x)
    return x.view(B, Fn, M.SLOT, m.d)[:, -1].float().mean(1)


class Heads(nn.Module):
    def __init__(s, d=1024, h=512):
        super().__init__()
        s.ln = nn.LayerNorm(d)
        s.qn = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, NA))
        s.rn = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, NA))
        s.dn = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, NA))
    def q(s, f): return s.qn(s.ln(f.float()))
    def r(s, f): return s.rn(s.ln(f.float()))
    def d(s, f): return s.dn(s.ln(f.float()))


def save_atomic(obj, path):
    torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)


def jdump(obj, path):
    with open(path + ".tmp", "w") as f: json.dump(obj, f)
    os.replace(path + ".tmp", path)
