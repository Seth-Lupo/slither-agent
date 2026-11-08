"""Shared pieces of RL v2: PPO fine-tuning of the pretrained s150p policy, on frozen s150p world-model features.

Policy  = the pretrained behaviour-cloning head (human play; 32 turn bins x 2 boost, one decision per 0.1 s frame), its
          exact weights, made trainable, plus a spatial branch whose last layer starts at zero -> at step 0 the policy
          IS the pretrained one. KL to the frozen pretrained head keeps it from drifting far.
Inputs  = g: the world model's last-frame hidden state averaged over the frame (what the pretrained head was trained on);
          feat: the same hidden states pooled to a 4x4 grid around the snake's head + global mean, through fixed random
          projections -> knows where things are.
"""
import json, math, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

MG = os.environ.get("MG_ROOT", "/tmp/slither/mg"); sys.path.insert(0, f"{MG}/code")
import common as C, dynamics as M, tokenizer as T          # noqa: E402

RL = os.environ.get("RL_ROOT", "/tmp/slither/rl2")
WM_CKPT = os.environ.get("WM_CKPT", f"{MG}/runs/s150p/ckpt.pt")
NB = C.NB                                                    # 32 turn bins
NA = NB * 2                                                  # action a = bin * 2 + boost
EDGES = np.array(json.load(open(f"{C.DATA}/meta.json"))["edges"])
CENTERS = np.concatenate([[EDGES[0] - 4.0], (EDGES[1:] + EDGES[:-1]) / 2, [EDGES[-1] + 4.0]])   # deg per frame, + = left
# Commanded heading target per bin. The snake lags the target (turnrate.py: target 26 deg -> 19.7 deg actual per 0.1 s;
# the game's max is ~24 deg at targets >= 60), so aim 1.3x further, and the hardest bin on each side aims 90 deg to the
# side = full lock, like a human pointing the mouse sideways.
def _cmd(c): return math.copysign(90.0, c) if abs(c) > 22 else 1.3 * float(c)
ACTIONS = [(_cmd(CENTERS[a // 2]), a % 2) for a in range(NA)]                                      # for SlitherEnv(fpd=1)
REWARD_SCALE = 10.0          # (v1) delta-log-length scale
# v2 reward: survival first. Per 0.1 s decision: +ALIVE while alive, GROW_W * delta log length, DEATH on dying.
ALIVE, GROW_W, DEATH = 0.1, 1.0, -10.0
# Potential-based shaping (Ng et al. 1999; leaves the optimal policy unchanged): Phi(s) = -SHAPE_K * danger(s), so each
# step adds SHAPE_K * (danger(s) - GAMMA * danger(s')). danger in [0,1] from the game's true state (page.js). No shaping
# on the terminal step. The KL leash to the pretrained policy is weighted by (1 - danger): free to swerve when in danger.
SHAPE_K = 5.0
KL_FLOOR = 0.1
TEMP = 0.4          # policy temperature: logits / TEMP everywhere (actor sampling, PPO log-probs, KL), argmax unaffected
GAMMA = 0.995                                                # per 0.1 s decision: horizon ~20 s
NSTEP = 5
TAU = float(os.environ.get("RL_TAU", "0.1"))                 # how strongly Q may override the human prior
GRID, PG, PGLOB = 4, 64, 256
FEAT = GRID * GRID * PG + PGLOB                              # 1280


def act_token(turn_deg, boost):
    """Heading change of one frame (deg, + = left) and boost -> the world model's (turn bin, boost) token."""
    return int(np.searchsorted(EDGES, turn_deg)), int(boost)


def load_wm(dev="cuda"):
    ck = torch.load(WM_CKPT, map_location="cpu", weights_only=False)
    m = M.Dyn(ck["cfg"]["size"], policy=True)
    m.load_state_dict(ck["model"]); return m.to(dev).eval().requires_grad_(False)


def load_tokenizer(dev="cuda"): return T.load(dev=dev)


_P = {}
def proj(dev):
    if dev not in _P:
        g = torch.Generator().manual_seed(1234)
        _P[dev] = (torch.randn(1024, PG, generator=g) / math.sqrt(1024)).to(dev), (torch.randn(1024, PGLOB, generator=g) / math.sqrt(1024)).to(dev)
    return _P[dev]


TRAIN_FROM = 9                                               # blocks >= this (of 12) + final norm + frame head are fine-tuned


def trunk(m, tok, act, grad_from=None):
    """World-model forward -> normed hidden states [B,F,SLOT,d]. Blocks below grad_from run without autograd."""
    B, Fn = tok.shape[:2]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pos = torch.cat([m.aslot, (m.row + m.col).view(M.NT, m.d)])
        x = torch.cat([m.turn(act[..., :1]), m.boost(act[..., 1:]), m.tok(tok)], 2) + pos + m.time[:Fn]
        x = x.view(B, Fn * M.SLOT, m.d); mk = m.block_mask(Fn * M.SLOT, x.device)
        for i, b in enumerate(m.blocks):
            if grad_from is not None and i < grad_from:
                with torch.no_grad(): x = b(x, mk)
            else: x = b(x, mk)
        x = m.norm(x)
    return x.view(B, Fn, M.SLOT, m.d)


def state_feats(h):
    """Hidden states of one frame [B,SLOT,d] -> g [B,d] (what the pretrained head saw), feat [B,FEAT] (spatial grid)."""
    B, d = h.shape[0], h.shape[-1]; h = h.float()
    g = h.mean(1)
    grid = F.avg_pool2d(h[:, 2:].reshape(B, 16, 16, d).permute(0, 3, 1, 2), 16 // GRID).flatten(2).transpose(1, 2)
    Pg, Pq = proj(h.device)
    return g, torch.cat([(grid @ Pg).flatten(1), g @ Pq], 1)


@torch.no_grad()
def encode_state(m, tok, act):
    """tok [B,8,256], act [B,8,2] long -> feat [B,FEAT], g [B,d], bc logits [B,NB+2] (the world model's own policy head)."""
    h = trunk(m, tok, act)[:, -1]
    g, feat = state_feats(h)
    bc = m.pol(g.to(m.pol[0].weight.dtype)).float()
    return feat, g, bc


def bc_logp(bc):
    """bc logits [B,NB+2] -> log pi_bc over the NA joint actions [B,NA]."""
    lt, lb = F.log_softmax(bc[:, :NB].float(), -1), F.log_softmax(bc[:, NB:].float(), -1)
    return (lt[:, :, None] + lb[:, None, :]).reshape(-1, NA)


def soft_value(q, bc):
    """V(s) = TAU * log sum_a pi_bc(a|s) exp(Q(s,a)/TAU)   (KL-regularised value toward the pretrained policy)."""
    return TAU * torch.logsumexp(bc_logp(bc) + q / TAU, -1)


def pol_net(): return nn.Sequential(nn.Linear(1024, 1024), nn.GELU(approximate="tanh"), nn.Linear(1024, NB + 2))


class Pi(nn.Module):
    """Pretrained policy head (copied) + zero-initialised spatial branch. -> logits [B, NB+2] (turn bins, then boost)."""
    def __init__(s, pol=None):
        super().__init__()
        s.pol = pol_net()
        if pol is not None: s.pol.load_state_dict({k: v.float() for k, v in pol.state_dict().items()})
        s.sp = nn.Sequential(nn.LayerNorm(FEAT), nn.Linear(FEAT, 512), nn.GELU(), nn.Linear(512, NB + 2))
        nn.init.zeros_(s.sp[-1].weight); nn.init.zeros_(s.sp[-1].bias)
    def forward(s, g, feat): return (s.pol(g.float()) + s.sp(feat.float())) / TEMP


class Vf(nn.Module):
    def __init__(s, h=512):
        super().__init__()
        s.net = nn.Sequential(nn.LayerNorm(FEAT + 1024), nn.Linear(FEAT + 1024, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
    def forward(s, g, feat): return s.net(torch.cat([g.float(), feat.float()], -1))[..., 0]


class DangerHead(nn.Module):
    """Foresight: from the current state predict danger 0.5 s ahead (regression) and death within 1 s (logit)."""
    def __init__(s, h=512):
        super().__init__()
        s.net = nn.Sequential(nn.LayerNorm(FEAT + 1024), nn.Linear(FEAT + 1024, h), nn.GELU(), nn.Linear(h, 2))
    def forward(s, g, feat): return s.net(torch.cat([g.float(), feat.float()], -1))


def backbone_trainable(m):
    """Names of the world-model parameters that are fine-tuned (top blocks, final norm, frame head)."""
    keep = tuple(f"blocks.{i}." for i in range(TRAIN_FROM, len(m.blocks))) + ("norm.", "head.")
    return [n for n, _ in m.named_parameters() if n.startswith(keep)]


def kl_ent(logits, ref):
    """Exact KL(pi || ref) and entropy of pi for the factorised (turn, boost) distribution. -> [B], [B]"""
    kl, ent = 0, 0
    for sl in (slice(0, NB), slice(NB, NB + 2)):
        lp, lq = F.log_softmax(logits[:, sl].float(), -1), F.log_softmax(ref[:, sl].float(), -1)
        p = lp.exp(); kl = kl + (p * (lp - lq)).sum(-1); ent = ent - (p * lp).sum(-1)
    return kl, ent


def turn_name(a):
    t, b = float(CENTERS[a // 2]), a % 2; return (f"{'left' if t > 0 else 'right'} {abs(t):.1f}°" if abs(t) > 0.05 else "straight") + (" + boost" if b else "")


class Heads(nn.Module):
    def __init__(s, d=FEAT, h=512):
        super().__init__()
        s.ln = nn.LayerNorm(d)
        s.qn = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, NA))
        s.rn = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, NA))
        s.dn = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, NA))
        nn.init.zeros_(s.qn[-1].weight); nn.init.zeros_(s.qn[-1].bias)    # Q = 0 -> policy = pretrained policy
    def q(s, f): return s.qn(s.ln(f.float()))
    def r(s, f): return s.rn(s.ln(f.float()))
    def d(s, f): return s.dn(s.ln(f.float()))


def save_atomic(obj, path):
    torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)


def jdump(obj, path):
    with open(path + ".tmp", "w") as f: json.dump(obj, f)
    os.replace(path + ".tmp", path)
