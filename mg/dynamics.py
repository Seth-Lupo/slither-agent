"""MaskGIT dynamics model: 8 context frames x (2 action tokens + 256 frame tokens) -> next frame's tokens.

Block-causal transformer (a frame attends to itself and earlier frames), learned time + 2D position embeddings,
RMSNorm, QK-norm. Train: mask a cosine-sampled fraction of the target frame, CE on masked tokens.
Action dropout 10% (whole window -> no-action) enables classifier-free guidance at inference.
Unexplained energy (--energy): each token's loss is weighted 1 + (what motion can't explain)^2 / its batch mean (energy()).
Death events (events.py): --ev_frac of each batch are windows whose target is 1 frame before to 4 after a worm turning
into orbs. That frame's death-mask tokens (what turns into orbs or loses the body) are always masked, so they must be
predicted from the past rather than copied from visible neighbours, and get --ev_w x loss weight. Both anneal back to
plain training over the last --ev_anneal of the schedule.
Policy head (--pol_w > 0): each context frame's hidden states (mean over its 258 positions, block-causal so it sees only
frames <= t and actions <= t) predict the next action a[t+1] (turn bin: 32-way CE, boost: 2-way CE; no-action labels ignored).
That is behaviour cloning on the human play, trained jointly with the world model: loss = dynamics CE + pol_w * (turn + boost CE).

torchrun --standalone --nproc_per_node 8 dynamics.py --name s400p --size 400 --bs 32 --init runs/s400/ckpt.pt --epochs 2 --pol_w 0.5

torchrun --standalone --nproc_per_node 4 dynamics.py --name s150 --size 150
"""
import argparse, json, math, os, subprocess, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import common as C

NCODE, MASK, NT, CTX = 1000, 1000, 256, 8
SLOT = NT + 2                                   # tokens per frame: turn, boost, 256 frame tokens
SIZES = {50: (640, 10, 10), 150: (1024, 12, 16), 400: (1280, 20, 20)}   # d, layers, heads
NULL = (C.NB, 2)
_flex = None                                    # FlexAttention needs torch 2.9 on sm_120 (2.13 segfaults in Triton); MG_FLEX=0 -> SDPA + bool mask
if os.environ.get("MG_FLEX", "1") == "1":
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _flex = torch.compile(flex_attention)


class Attn(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)
        s.qn = nn.RMSNorm(d // h); s.kn = nn.RMSNorm(d // h)
    def forward(s, x, mask):
        B, L, D = x.shape
        q, k, v = s.qkv(x).view(B, L, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        q, k = s.qn(q).to(v.dtype), s.kn(k).to(v.dtype)
        y = _flex(q, k, v, block_mask=mask) if not torch.is_tensor(mask) else F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return s.o(y.transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.n1 = nn.RMSNorm(d); s.a = Attn(d, h); s.n2 = nn.RMSNorm(d)
        s.m = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d, bias=False))
    def forward(s, x, mask): x = x + s.a(s.n1(x), mask); return x + s.m(s.n2(x))


class Dyn(nn.Module):
    def __init__(s, size=150, nf=CTX + 1, policy=False):
        super().__init__(); d, L, H = SIZES[size]; s.d = d
        s.tok = nn.Embedding(NCODE + 1, d); s.turn = nn.Embedding(C.NB + 1, d); s.boost = nn.Embedding(3, d)
        s.time = nn.Parameter(torch.randn(nf, 1, d) * 0.02); s.row = nn.Parameter(torch.randn(16, 1, d) * 0.02)
        s.col = nn.Parameter(torch.randn(1, 16, d) * 0.02); s.aslot = nn.Parameter(torch.randn(2, d) * 0.02)
        s.blocks = nn.ModuleList(Block(d, H) for _ in range(L)); s.norm = nn.RMSNorm(d); s.head = nn.Linear(d, NCODE, bias=False)
        s.pol = nn.Sequential(nn.Linear(d, d), nn.GELU(approximate="tanh"), nn.Linear(d, C.NB + 2)) if policy else None
        for m in s.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, std=0.02)
        for b in s.blocks: nn.init.zeros_(b.a.o.weight); nn.init.zeros_(b.m[2].weight)
        s._bm = {}

    @torch.compiler.disable
    def block_mask(s, n, dev):                  # block-causal: a frame sees itself and earlier frames
        if dev.type != "cuda" or _flex is None:
            if n not in s._bm: f = torch.arange(n, device=dev) // SLOT; s._bm[n] = f[:, None] >= f[None, :]
            return s._bm[n]
        if n not in s._bm: s._bm[n] = create_block_mask(lambda b, h, q, kv: q // SLOT >= kv // SLOT, None, None, n, n, device=dev)
        return s._bm[n]

    def forward(s, tok, act, pol=False):        # tok [B,F,256] long (MASK allowed), act [B,F,2] long -> logits of last frame [B,256,NCODE]
        """pol=True also returns next-action logits per frame [B,F,NB+2] (turn bins, then boost 0/1)."""
        B, Fn = tok.shape[:2]
        pos = torch.cat([s.aslot, (s.row + s.col).view(NT, s.d)])
        x = torch.cat([s.turn(act[..., :1]), s.boost(act[..., 1:]), s.tok(tok)], 2) + pos + s.time[:Fn]
        x = x.view(B, Fn * SLOT, s.d); m = s.block_mask(Fn * SLOT, x.device)
        for b in s.blocks: x = b(x, m)
        x = s.norm(x); lg = s.head(x[:, -NT:])
        if not pol: return lg
        return lg, s.pol(x.view(B, Fn, SLOT, s.d).mean(2))


@torch.no_grad()
def generate(model, ctx, cact, act, steps=8, temp=0.9, cfg=1.5, choice_temp=2.0, gen=None, trace=None):
    """MaskGIT decode of one frame. ctx [B,8,256], cact [B,8,2], act [B,2] -> [B,256]."""
    B = ctx.shape[0]; dev = ctx.device
    toks = torch.cat([ctx, torch.full((B, 1, NT), MASK, device=dev, dtype=ctx.dtype)], 1)
    acts = torch.cat([cact, act[:, None]], 1); null = torch.tensor(NULL, device=dev).expand_as(acts)
    unknown = torch.ones(B, NT, dtype=torch.bool, device=dev)
    for s in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(toks, acts).float()
            if cfg != 1.0: lu = model(toks, null).float(); lg = lu + cfg * (lg - lu)
        p = F.softmax(lg / temp, -1)
        samp = torch.multinomial(p.view(-1, NCODE), 1, generator=gen).view(B, NT)
        conf = torch.gather(p, -1, samp[..., None])[..., 0].clamp_min(1e-20).log()
        g = -torch.log(-torch.log(torch.rand(B, NT, device=dev, generator=gen).clamp(1e-20, 1)))
        conf = conf + choice_temp * (1 - (s + 1) / steps) * g
        cur = torch.where(unknown, samp, toks[:, -1])
        n_keep_masked = int(NT * math.cos(math.pi / 2 * (s + 1) / steps))
        conf = torch.where(unknown, conf, torch.full_like(conf, float("inf")))
        remask = torch.zeros_like(unknown)
        if n_keep_masked > 0:
            remask.scatter_(1, conf.argsort(1)[:, :n_keep_masked], True); remask &= unknown
        if trace is not None: trace.append((cur.clone(), remask.clone()))   # full current guess + still-uncertain positions
        toks[:, -1] = torch.where(remask, torch.full_like(cur, MASK), cur); unknown = remask
    return toks[:, -1]


@torch.no_grad()
def rollout(model, tok, act, n, **kw):
    """tok [B,8,256] context, act [B,8+n,2] -> generated [B,n,256] (sliding 8-frame context)."""
    seq, acts = tok.clone(), act
    for t in range(n):
        nxt = generate(model, seq[:, -CTX:], acts[:, t:t + CTX], acts[:, t + CTX], **kw)
        seq = torch.cat([seq, nxt[:, None]], 1)
    return seq[:, CTX:]


# ------------------------------------------------------------------ data
class Data:
    def __init__(s, data_hours=None, seed=0):
        s.tok = np.load(f"{C.DATA}/tokens.npy", mmap_mode="r"); s.act = np.load(f"{C.DATA}/acts.npy"); cut = np.load(f"{C.DATA}/cut.npy")
        index = json.load(open(f"{C.DATA}/index.json"))
        cc = np.concatenate([[0], np.cumsum(cut)])
        ok = np.zeros(len(cut), bool); N = len(cut)
        ok[:N - CTX] = (cc[CTX + 1:N + 1] - cc[1:N - CTX + 1]) == 0           # no discontinuity in frames i+1..i+8
        tr = [v for v in index if not v["val"]]; np.random.default_rng(seed).shuffle(tr)
        if data_hours:
            keep, h = [], 0.0
            for v in tr:
                if h >= data_hours: break
                keep.append(v); h += v["len"] / C.FPS / 3600
            tr = keep
        s.hours = sum(v["len"] for v in tr) / C.FPS / 3600
        sel = lambda vs: np.concatenate([np.arange(v["start"], v["start"] + v["len"]) for v in vs])
        s.train = sel(tr); s.train = s.train[ok[s.train]]
        s.val = sel([v for v in index if v["val"]]); s.val = s.val[ok[s.val]]
        s.ev, s.ev_m, s.dval = np.zeros(0, np.int64), np.zeros((0, 32), np.uint8), np.zeros(0, np.int64)
        if os.path.exists(f"{C.DATA}/deaths.npz"):       # events.py: g [K] first death frame, xy [K,2], mask [K,6,32] (targets g-1..g+4, packed bits)
            z = np.load(f"{C.DATA}/deaths.npz"); g, dm = z["g"].astype(np.int64), z["mask"]
            itr = np.zeros(N, bool); itr[s.train] = True; iva = np.zeros(N, bool); iva[s.val] = True
            w = g[:, None] + np.arange(-1, 5) - CTX                                  # [K,6] window starts
            k = (w >= 0) & (w < N); k[k] = itr[w[k]]; s.ev, s.ev_m = w[k], dm[k]
            k = (g >= CTX + 2) & (g + 4 < N); k[k] = iva[g[k] - CTX - 2] & iva[g[k] - CTX + 3]   # clean frames g-9..g+3
            s.dval, s.dval_xy, s.dval_m = g[k], z["xy"][k], dm[k]
        ec = np.load(f"{C.DATA}/evalclips.npz"); s.clip_start, s.clip_frames = ec["starts"], ec["frames"]

    def window(s, idx, n=CTX + 1):
        w = idx[:, None] + np.arange(n)
        return torch.from_numpy(s.tok[w.ravel()].reshape(len(idx), n, NT).astype(np.int64)), torch.from_numpy(s.act[w].astype(np.int64))


def mask_batch(tgt, gen=None):
    """Cosine-sampled mask ratio per sample. tgt [B,256] -> bool mask [B,256]."""
    B = tgt.shape[0]; dev = tgt.device
    r = torch.cos(math.pi / 2 * torch.rand(B, device=dev, generator=gen))
    n = (r * NT).ceil().clamp(1, NT)
    rank = torch.rand(B, NT, device=dev, generator=gen).argsort(1).argsort(1)
    return rank < n[:, None]


def unpack(m): return torch.from_numpy(np.unpackbits(m, axis=-1).astype(bool))   # packed [...,32] -> bool [...,256]


# ------------------------------------------------------------------ events: what motion can't explain (no labels)
def pmatch(a, b, r):
    """Motion-compensated residual: for each 8x8 patch of image a, the mean abs difference to the best-matching 8x8 window of
    image b shifted by up to +-r pixels (in each direction). a, b float [B,3,128,128] -> [B,256]."""
    bp = F.pad(b, (r, r, r, r), mode="replicate"); best = None
    for dy in range(0, 2 * r + 1, 2):
        for dx in range(0, 2 * r + 1, 2):
            d = F.avg_pool2d((a - bp[..., dy:dy + 128, dx:dx + 128]).abs().mean(1, keepdim=True), 8)
            best = d if best is None else torch.minimum(best, d)
    return best.flatten(1)


_RING = None
@torch.no_grad()
def energy(tokz, tok):
    """Unexplained energy of each target patch, from decoded tokens (no labels, no thresholds). tok [B,9,256] -> [B,256].
    residual = what motion can't explain: the larger of (a) the patch vs its best-matching window in the last context frame
    (something appeared, vanished or disintegrated) and (b) for spots that changed over the last 3 context frames, the patch vs
    what was there before (a worm that just vanished must stay gone). energy = residual^2, like a squared error. The outer
    ring of patches gets 0 (new content scrolls in there)."""
    global _RING
    B = tok.shape[0]
    if _RING is None or _RING.device != tok.device:
        _RING = torch.ones(16, 16, dtype=torch.bool, device=tok.device); _RING[1:-1, 1:-1] = False; _RING = _RING.view(NT)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = tokz.decode(tok[:, [-5, -2, -1]].reshape(-1, 16, 16)).float().view(B, 3, 3, 128, 128)
    old, last, y = x[:, 0], x[:, 1], x[:, 2]
    r = torch.maximum(pmatch(y, last, 6), torch.minimum(pmatch(last, old, 12), F.avg_pool2d((y - old).abs().mean(1, keepdim=True), 8).flatten(1)))
    return torch.where(_RING, torch.zeros((), device=tok.device), r.square())


def pol_loss(pl, nxt):
    """pl [B,F,NB+2] next-action logits, nxt [B,F,2] the actions that followed (NB / 2 = no action, ignored).
    -> turn CE, boost CE, turn accuracy, turn within +-1 bin, boost accuracy."""
    tl, bl = pl[..., :C.NB].reshape(-1, C.NB).float(), pl[..., C.NB:].reshape(-1, 2).float()
    ty, by = nxt[..., 0].reshape(-1), nxt[..., 1].reshape(-1); v = ty < C.NB; vb = by < 2
    tce = F.cross_entropy(tl, ty, ignore_index=C.NB) if v.any() else tl.sum() * 0
    bce = F.cross_entropy(bl, by, ignore_index=2) if vb.any() else bl.sum() * 0
    tp = tl.argmax(-1); n = v.sum().clamp_min(1)
    return tce, bce, ((tp == ty) & v).sum() / n, (((tp - ty).abs() <= 1) & v).sum() / n, ((bl.argmax(-1) == by) & vb).sum() / vb.sum().clamp_min(1)


def loss_fn(model, tok, act, gen=None, ctx_noise=0.0, force=None, weight=None, en=None, pact=None):
    """pact [B,F,2]: the true (undropped) actions; given -> the policy head's parts go to loss_fn.pol (pol_loss())."""
    """Masked CE on the target frame. force [B,256]: tokens always masked; weight [B,256]: per-token weights.
    en [B,256]: unexplained energy (energy()); each masked token is weighted 1 + en / mean(en), so half the weight is spread
    evenly and half follows what motion can't explain. Plain and energy-weighted CE are kept in loss_fn.parts."""
    m = mask_batch(tok[:, -1], gen)
    if force is not None: m = m | force
    inp = tok.clone(); inp[:, -1][m] = MASK
    if ctx_noise > 0:
        noise = torch.rand(inp[:, :-1].shape, device=tok.device, generator=gen) < ctx_noise
        inp[:, :-1][noise] = torch.randint(0, NCODE, (int(noise.sum()),), device=tok.device, generator=gen)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if pact is None: lg = model(inp, act)
        else: lg, pl = model(inp, act, pol=True); loss_fn.pol = pol_loss(pl[:, :-1], pact[:, 1:])   # frame t -> a[t+1]
    lg = lg.float()[m]; y = tok[:, -1][m]; acc = (lg.argmax(-1) == y).float().mean()
    if en is not None:
        ce = F.cross_entropy(lg, y, reduction="none"); e = en[m]; e = e / e.mean().clamp_min(1e-8)
        k = max(1, e.numel() // 10)
        loss_fn.parts = (ce.mean().detach(), ((ce * e).sum() / e.sum().clamp_min(1e-8)).detach(), (e.topk(k).values.sum() / e.sum().clamp_min(1e-8)).detach())
        return ((1 + e) * ce).sum() / (1 + e).sum(), acc
    if weight is None: return F.cross_entropy(lg, y), acc
    wm = weight[m]; return (F.cross_entropy(lg, y, reduction="none") * wm).sum() / wm.sum().clamp_min(1e-6), acc


# ------------------------------------------------------------------ evaluation
def _ent(x, k):                                 # entropy (nats) of the valid labels' marginal: the CE of a policy that ignores the screen
    c = np.bincount(x[x < k], minlength=k).astype(float); p = c[c > 0] / c.sum(); return float(-(p * np.log(p)).sum())


def psnr(a, b): return -10 * torch.log10(((a - b) ** 2).flatten(-3).mean(-1).clamp_min(1e-10))


def sharp(x): return ((x[..., 1:, :] - x[..., :-1, :]).abs().mean((-3, -2, -1)) + (x[..., 1:] - x[..., :-1]).abs().mean((-3, -2, -1)))


def tile(x):                                    # float [R,Cc,3,H,W] -> uint8 [R*H, Cc*W, 3]
    R, Cc, _, H, W = x.shape
    return (x.clamp(0, 1) * 255).byte().permute(0, 3, 1, 4, 2).reshape(R * H, Cc * W, 3).cpu().numpy()


def save_gif(frames, path, fps=5):
    from PIL import Image
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(path + ".tmp.gif", save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0); os.replace(path + ".tmp.gif", path)


def border(x, color):                           # x float [...,3,H,W]; 2px colored frame
    x = x.clone(); c = torch.tensor(color, device=x.device, dtype=x.dtype).view(3, 1, 1)
    x[..., :, :2, :] = c; x[..., :, -2:, :] = c; x[..., :, :, :2] = c; x[..., :, :, -2:] = c; return x


@torch.no_grad()
def evaluate(model, tokz, D, step, out, cfg):
    model.eval(); dev = "cuda"; g = torch.Generator(dev).manual_seed(0); ev = dict(step=step, t=time.time())
    dec = lambda t: tokz.decode(t.reshape(-1, 16, 16)).view(*t.shape[:-1], 3, 128, 128)
    # (a) teacher-forced val CE with and without actions (how much the model uses actions)
    rv = np.random.default_rng(0).choice(D.val, 512, replace=False); ce, cen, acc = [], [], []
    for i in range(0, 512, 64):
        tk, ac = D.window(rv[i:i + 64]); tk, ac = tk.to(dev), ac.to(dev)
        g1 = torch.Generator(dev).manual_seed(i); l, a = loss_fn(model, tk, ac, g1); ce.append(l.item()); acc.append(a.item())
        g1 = torch.Generator(dev).manual_seed(i); l, _ = loss_fn(model, tk, torch.tensor(NULL, device=dev).expand_as(ac), g1); cen.append(l.item())
    ev.update(val_ce=float(np.mean(ce)), val_acc=float(np.mean(acc)), val_ce_noact=float(np.mean(cen)))
    ev["action_gain"] = ev["val_ce_noact"] - ev["val_ce"]
    if getattr(model, "pol", None) is not None or getattr(getattr(model, "_orig_mod", None), "pol", None) is not None:
        ps = []                                                       # policy head on the same val windows (true action history)
        for i in range(0, 512, 64):
            tk, ac = D.window(rv[i:i + 64]); tk, ac = tk.to(dev), ac.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16): _, pl = model(tk, ac, pol=True)
            ps.append([x.item() for x in pol_loss(pl[:, :-1], ac[:, 1:])])
        ps = np.mean(ps, 0); ev.update(pol_turn_ce=ps[0], pol_boost_ce=ps[1], pol_turn_acc=ps[2], pol_turn_acc1=ps[3], pol_boost_acc=ps[4])
        a = D.act[D.train[::97]]; ev.update(pol_turn_ce_marginal=_ent(a[:, 0], C.NB), pol_boost_ce_marginal=_ent(a[:, 1], 2))
    # (b) 16-step rollouts on fixed clips with the real actions: real | tokenizer recon | predicted
    K = min(16, len(D.clip_start)); H = 16
    tk, ac = D.window(D.clip_start[:K], CTX + H); tk, ac = tk.to(dev), ac.to(dev)
    real = torch.from_numpy(D.clip_frames[:K, :CTX + H]).to(dev).permute(0, 1, 4, 2, 3).float() / 255
    gen = rollout(model, tk[:, :CTX], ac, H, cfg=cfg, gen=g)
    rec, pred = dec(tk), dec(gen)
    tgt = real[:, CTX:]; last = real[:, CTX - 1:CTX].expand_as(tgt)
    ev["psnr_pred"] = psnr(pred, tgt).mean(0).tolist(); ev["psnr_copy"] = psnr(last, tgt).mean(0).tolist()
    ev["psnr_recon"] = psnr(rec[:, CTX:], tgt).mean(0).tolist(); ev["tok_acc_h"] = (gen == tk[:, CTX:]).float().mean((0, 2)).tolist()
    ev["sharp_ratio"] = (sharp(pred) / sharp(tgt).clamp_min(1e-6)).mean(0).tolist()
    ev["beats_copy_h"] = next((h for h in range(H) if ev["psnr_pred"][h] <= ev["psnr_copy"][h]), H)
    n = 4; panel = torch.stack([real[:n], rec[:n], torch.cat([border(rec[:n, :CTX], (0.2, 0.8, 0.3)), border(pred[:n], (0.9, 0.3, 0.2))], 1)], 1)
    frames = [tile(panel[:, :, t]) for t in range(CTX + H)]          # rows: clips; cols: real | recon | pred
    save_gif(frames, f"{out}/rollout.gif"); save_gif(frames, f"{out}/rollout_{step:07d}.gif")
    # (b2) internals: the 8 MaskGIT steps of the first predicted frame; red = still masked after that step
    tr = []; generate(model, tk[:4, :CTX], ac[:4, :CTX], ac[:4, CTX], cfg=cfg, gen=g, trace=tr)
    tint = torch.tensor([0.9, 0.15, 0.1], device=dev).view(1, 3, 1, 1); steps_img = []
    for cur, rm in tr:
        im = dec(cur); m = F.interpolate(rm.view(-1, 1, 16, 16).float(), scale_factor=8) > 0
        steps_img.append(torch.where(m, im * 0.35 + tint * 0.5, im))
    dg = torch.stack([torch.stack([real[:4, CTX]] * len(tr), 1), torch.stack(steps_img, 1)], 1)   # [4, 2, steps, 3, H, W]
    save_gif([tile(dg[:, :, t]) for t in range(len(tr))] + [tile(dg[:, :, -1])] * 2, f"{out}/decode.gif", fps=2)
    # (c) coverage: 8 samples x 8 clips, 8-step rollouts; mean vs best-of-8 PSNR per horizon
    K8, S, H8 = 8, 8, 8
    tk8, ac8 = tk[:K8, :CTX].repeat_interleave(S, 0), ac[:K8, :CTX + H8].repeat_interleave(S, 0)
    g8 = rollout(model, tk8, ac8, H8, cfg=cfg, gen=g); p8 = dec(g8).view(K8, S, H8, 3, 128, 128)
    ps = psnr(p8, real[:K8, None, CTX:CTX + H8])                     # [K8,S,H8]
    ev["cov_mean"] = ps.mean((0, 1)).tolist(); ev["cov_best"] = ps.max(1).values.mean(0).tolist()
    ev["sample_div"] = (p8[:, 0] - p8[:, 1]).abs().mean((0, 2, 3, 4)).tolist()
    grid = torch.cat([real[0, CTX:CTX + H8][None], p8[0], torch.zeros_like(p8[0, :1])]).view(2, 5, H8, 3, 128, 128)   # real + 8 samples
    save_gif([tile(grid[:, :, t]) for t in range(H8)], f"{out}/samples.gif", fps=3)
    # (d) control: same context, constant commanded actions: hard left / straight / hard right / boost / no action
    K4 = 4; cmds = [("left", C.NB - 2, 0), ("straight", C.NB // 2, 0), ("right", 1, 0), ("boost", C.NB // 2, 1), ("none", C.NB, 2)]
    outs = []
    for _, tb, bb in cmds:
        a2 = ac[:K4, :CTX + H8].clone(); a2[:, CTX:, 0] = tb; a2[:, CTX:, 1] = bb
        outs.append(dec(rollout(model, tk[:K4, :CTX], a2, H8, cfg=cfg, gen=torch.Generator(dev).manual_seed(1))))
    lr = (outs[0] - outs[2]).abs().mean((0, 2, 3, 4))
    same = (p8[:K4, 0] - p8[:K4, 1]).abs().mean((0, 2, 3, 4))
    ev["control_ratio"] = (lr / same.clamp_min(1e-6)).tolist()       # left-vs-right difference / sample noise, per horizon
    ev["boost_ratio"] = ((outs[3] - outs[1]).abs().mean((0, 2, 3, 4)) / same.clamp_min(1e-6)).tolist()
    pc = torch.stack([torch.cat([border(rec[:K4, CTX - 2:CTX], (0.2, 0.8, 0.3)), o], 1) for o in outs], 1)   # [K4,5,2+H8,3,128,128]
    save_gif([tile(pc[:, :, t]) for t in range(2 + H8)], f"{out}/control.gif", fps=3)
    if len(D.dval): ev.update(death_eval(model, tokz, D, out, cfg, g))
    model.train(); return ev


@torch.no_grad()
def death_eval(model, tokz, D, out, cfg, g):
    """Worms turning into orbs (val). death_ce: CE on death-mask tokens (targets g..g+2, always masked) vs move_ce: CE on
    changed tokens of random val frames. follow_*: share of deaths where the generated frame g+3 is closer, inside its
    death mask, to the real g+3 than to the last frame before the death (g-1); cont = context already shows the first
    death frame, anti = context ends 3 frames before it."""
    dev = "cuda"; ev = {}; dec = lambda t: tokz.decode(t.reshape(-1, 16, 16)).view(*t.shape[:-1], 3, 128, 128)
    pick = np.random.default_rng(0).choice(len(D.dval), min(192, len(D.dval)), replace=False)
    ce = []
    for off in (0, 1, 2):
        for i in range(0, len(pick), 64):
            j = pick[i:i + 64]; tk, ac = D.window(D.dval[j] + off - CTX); tk, ac = tk.to(dev), ac.to(dev)
            fm = unpack(D.dval_m[j, off + 1]).to(dev)
            ce.append(loss_fn(model, tk, ac, torch.Generator(dev).manual_seed(i + off), force=fm, weight=fm.float())[0].item())
    rv = np.random.default_rng(1).choice(D.val, 192, replace=False); mv = []
    for i in range(0, 192, 64):
        tk, ac = D.window(rv[i:i + 64]); tk, ac = tk.to(dev), ac.to(dev)
        mv.append(loss_fn(model, tk, ac, torch.Generator(dev).manual_seed(i), weight=(tk[:, -1] != tk[:, -2]).float())[0].item())
    ev.update(death_ce=float(np.mean(ce)), move_ce=float(np.mean(mv)))
    K = min(32, len(D.dval)); j = pick[:K]; gd = D.dval[j]
    tk, ac = D.window(gd - CTX - 2, CTX + 6); tk, ac = tk.to(dev), ac.to(dev)      # frames g-10 .. g+3
    anti = rollout(model, tk[:, :CTX], ac, 6, cfg=cfg, gen=g)                     # generates g-2 .. g+3
    cont = rollout(model, tk[:, 3:3 + CTX], ac[:, 3:], 3, cfg=cfg, gen=g)          # context g-7 .. g, generates g+1 .. g+3
    real = dec(tk)
    disc = F.interpolate(unpack(D.dval_m[j, 4]).view(K, 1, 16, 16).float().to(dev), scale_factor=8) > 0   # death mask at g+3, pixels
    dist = lambda a, b: ((a - b).abs() * disc).sum((1, 2, 3))
    for name, gen_ in (("anti", anti), ("cont", cont)):
        p = dec(gen_[:, -1:])[:, 0]
        ev[f"follow_{name}"] = (dist(p, real[:, -1]) < dist(p, real[:, CTX + 1])).float().mean().item()
    n = 4; dd = disc[:n].float(); ring = (F.max_pool2d(dd, 5, 1, 2) - dd)[:, None]  # cyan outline of the death mask
    hi = lambda x: torch.where(ring > 0, torch.tensor([0.1, 0.9, 0.9], device=dev).view(1, 1, 3, 1, 1), x)
    red = lambda x: border(dec(x), (0.9, 0.3, 0.2))
    rows = [real[:n, 2:], torch.cat([real[:n, 2:11], red(cont[:n])], 1), torch.cat([real[:n, 2:8], red(anti[:n])], 1)]   # frames g-8 .. g+3
    panel = torch.stack([hi(x) for x in rows], 1)                                   # rows: deaths; cols: real | cont | anti
    save_gif([tile(panel[:, :, t]) for t in range(panel.shape[2])], f"{out}/deaths.gif", fps=3)
    return ev


# ------------------------------------------------------------------ train
def walltime_left_h():
    """Hours left in this Slurm job (None outside Slurm)."""
    try:
        s = subprocess.run(["squeue", "-h", "-j", os.environ["SLURM_JOB_ID"], "-o", "%L"], capture_output=True, text=True, timeout=30).stdout.strip()
        d, _, hms = s.rpartition("-"); p = [int(x) for x in hms.split(":")]
        while len(p) < 3: p.insert(0, 0)
        return (int(d or 0) * 24 + p[0] + p[1] / 60 + p[2] / 3600)
    except Exception: return None


def main():
    import torch.distributed as dist, datetime
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True); p.add_argument("--size", type=int, default=150)
    p.add_argument("--data_hours", type=float, default=None); p.add_argument("--epochs", type=float, default=None, help="default 4 (2 for a --data_hours pilot)")
    p.add_argument("--hours", type=float, default=None, help="default: until 45 min before the Slurm job ends"); p.add_argument("--bs", type=int, default=64, help="per GPU")
    p.add_argument("--lr", type=float, default=None); p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--drop", type=float, default=0.1); p.add_argument("--ctx_noise", type=float, default=0.0)
    p.add_argument("--cfg", type=float, default=1.5); p.add_argument("--eval_min", type=float, default=30)
    p.add_argument("--compile", type=int, default=1); p.add_argument("--init", default=None, help="start from this checkpoint if the run has none")
    p.add_argument("--ev_frac", type=float, default=0.0); p.add_argument("--ev_w", type=float, default=4.0); p.add_argument("--ev_anneal", type=float, default=0.15)
    p.add_argument("--cycle", type=int, default=0, help="never finish: each new Slurm job re-warms and fits a fresh cosine to its walltime (also on if runs/NAME/CYCLE exists)")
    p.add_argument("--flat", type=float, default=0.0, help="never finish, constant LR = flat * lr0, no re-warm (also read from runs/NAME/FLAT)")
    p.add_argument("--energy", type=int, default=0, help="weight each token by 1 + its unexplained energy / mean (energy())")
    p.add_argument("--pol_w", type=float, default=0.0, help="policy head: predict the next action from each frame, loss weight")
    a = p.parse_args()
    a.epochs = a.epochs or (2 if a.data_hours else 4)
    left = walltime_left_h(); a.hours = min(a.hours or 1e9, (left - 0.75) if left else 40)
    if a.hours < 0.3: print(f"only {a.hours:.2f} h of walltime left, not starting", flush=True); return
    fl = f"{C.ROOT}/runs/{a.name}/FLAT"; a.flat = a.flat or (float(open(fl).read()) if os.path.exists(fl) else 0.0)
    a.cycle = int(a.cycle or a.flat > 0 or os.path.exists(f"{C.ROOT}/runs/{a.name}/CYCLE")); job = os.environ.get("SLURM_JOB_ID", "")
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60)); rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"])); dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True; torch.manual_seed(rank)
    OUT = f"{C.ROOT}/runs/{a.name}"; DD = f"{C.DASH}/dyn/{a.name}"; os.makedirs(OUT, exist_ok=True); os.makedirs(DD, exist_ok=True)
    D = Data(a.data_hours)
    model = Dyn(a.size, policy=a.pol_w > 0).to(dev); nparam = sum(p.numel() for p in model.parameters())
    lr0 = a.lr or {50: 6e-4, 150: 4e-4, 400: 3e-4}[a.size]
    opt = torch.optim.AdamW(model.parameters(), lr=lr0, betas=(0.9, 0.95), weight_decay=0.05, fused=True)
    total = int(a.epochs * len(D.train) / (a.bs * world)); step = 0; w0 = 0
    ckp = f"{OUT}/ckpt.pt" if os.path.exists(f"{OUT}/ckpt.pt") else a.init
    if ckp:
        ck = torch.load(ckp, map_location=dev); miss = model.load_state_dict(ck["model"], strict=False); step = ck["step"]
        assert not miss.unexpected_keys and all(k.startswith("pol.") for k in miss.missing_keys), miss
        if not miss.missing_keys: opt.load_state_dict(ck["opt"])   # new policy head -> fresh optimizer (re-warms from --init anyway)
        elif rank == 0: print("new policy head on", ckp, "- fresh optimizer", flush=True)
        w0 = ck["step"] if ckp == a.init else ck.get("w0", 0)        # continuing from --init: re-warm, then a fresh cosine from there
        if a.cycle and ckp != a.init and ck.get("job") != job: w0 = ck["step"]   # cycle runs: same in each new Slurm job
    total += w0
    cfgd = dict(vars(a), params_m=nparam / 1e6, train_hours=D.hours, train_windows=len(D.train), val_windows=len(D.val), total_steps=total, world=world, lr0=lr0, w0=w0,
                ev_windows=len(D.ev), val_deaths=len(D.dval))
    if rank == 0: C.jdump(cfgd, f"{DD}/config.json"); print(json.dumps(cfgd), flush=True)
    fwd = torch.compile(model) if a.compile else model
    net = nn.parallel.DistributedDataParallel(fwd, device_ids=[int(os.environ["LOCAL_RANK"])])
    tokz = None
    if rank == 0 or a.energy:
        import tokenizer as T; tokz = T.load()
    rng = np.random.default_rng(1000 + rank + step); mlog = open(f"{DD}/metrics.jsonl", "a") if rank == 0 else None
    t0 = time.time(); tl = time.time(); last_eval = time.time(); s0 = step; s0a = s0 + (-s0) % 50   # the startup checks below run on multiples of 50
    while step < total:
        lr = lr0 * a.flat if a.flat else lr0 * min(1, (step - w0 + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - w0) / (total - w0))))
        for gp in opt.param_groups: gp["lr"] = lr
        f = min(1.0, (total - step) / max(1.0, a.ev_anneal * total)) if a.ev_anneal > 0 else 1.0
        ne = int(round(a.ev_frac * f * a.bs)) if len(D.ev) else 0
        idx = rng.choice(D.train, a.bs - ne); fm = None
        if ne:
            j = rng.integers(len(D.ev), size=ne); idx = np.concatenate([idx, D.ev[j]])
            fm = torch.zeros(a.bs, NT, dtype=torch.bool); fm[a.bs - ne:] = unpack(D.ev_m[j]); fm = fm.to(dev, non_blocking=True)
        tk, ac = D.window(idx); tk, ac = tk.to(dev, non_blocking=True), ac.to(dev, non_blocking=True)
        pact = ac.clone() if a.pol_w > 0 else None
        drop = torch.rand(a.bs, device=dev) < a.drop; ac[drop] = torch.tensor(NULL, device=dev)
        if a.energy:
            loss, acc = loss_fn(net, tk, ac, ctx_noise=a.ctx_noise, force=fm, en=energy(tokz, tk), pact=pact)
        else:
            loss, acc = loss_fn(net, tk, ac, ctx_noise=a.ctx_noise, force=fm, weight=None if fm is None else 1 + (a.ev_w - 1) * f * fm.float(), pact=pact)
        dloss = loss
        if pact is not None: loss = loss + a.pol_w * (loss_fn.pol[0] + loss_fn.pol[1])
        opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
        if rank == 0 and step % 50 == 0:
            dt = (time.time() - tl) / 50; tl = time.time()
            evl = dict(ce=loss_fn.parts[0].item(), ce_energy=loss_fn.parts[1].item(), energy_top10=loss_fn.parts[2].item()) if a.energy else {}
            if pact is not None: evl.update(dyn_loss=dloss.item(), **dict(zip(("pol_turn_ce", "pol_boost_ce", "pol_turn_acc", "pol_turn_acc1", "pol_boost_acc"), (x.item() for x in loss_fn.pol))))
            mlog.write(json.dumps(dict(step=step, t=time.time(), loss=loss.item(), acc=acc.item(), lr=lr, gn=gn.item(), sec_per_step=dt, ev_n=ne, **evl)) + "\n"); mlog.flush()
            C.stage(f"dyn_{a.name}", step=step, total=total, sec_per_step=dt, eta_h=dt * (total - step) / 3600, loss=loss.item())
        if step % 50:
            continue
        if step == s0a + 50: t50 = time.time()
        fit = total
        if step == s0a + 250:                                            # shrink the schedule to fit the time budget (LR still anneals)
            rate = (time.time() - t50) / 200; fit = min(total, int(step + 0.93 * (a.hours * 3600 - (time.time() - t0)) / rate))
        flags = torch.tensor([time.time() - last_eval > a.eval_min * 60 or step == s0a + 300, time.time() - t0 > a.hours * 3600, fit], device=dev, dtype=torch.float64)
        dist.broadcast(flags, 0)                                       # rank 0 decides, so all ranks stay in lockstep
        if flags[1].item(): break
        if int(flags[2].item()) != total:
            total = int(flags[2].item()); cfgd["total_steps"] = total
            if rank == 0: C.jdump(cfgd, f"{DD}/config.json"); print(f"schedule fitted to {a.hours:.1f} h: total_steps={total}", flush=True)
        if flags[0].item():
            last_eval = time.time()
            if rank == 0:
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, w0=w0, job=job, cfg=cfgd), f"{OUT}/ckpt.pt.tmp"); os.replace(f"{OUT}/ckpt.pt.tmp", f"{OUT}/ckpt.pt")
                te = time.time(); ev = evaluate(fwd, tokz, D, step, DD, a.cfg); ev["eval_sec"] = time.time() - te
                ev["epoch"] = step * a.bs * world / len(D.train)
                open(f"{DD}/eval.jsonl", "a").write(json.dumps(ev) + "\n"); print(json.dumps({k: v for k, v in ev.items() if not isinstance(v, list)}), flush=True)
            dist.barrier()
    if rank == 0:
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, w0=w0, job=job, cfg=cfgd), f"{OUT}/ckpt.pt.tmp"); os.replace(f"{OUT}/ckpt.pt.tmp", f"{OUT}/ckpt.pt")
        ev = evaluate(fwd, tokz, D, step, DD, a.cfg); open(f"{DD}/eval.jsonl", "a").write(json.dumps(ev) + "\n")
        if step >= total and not a.cycle: open(f"{OUT}/DONE", "w").write(str(step))
        C.stage(f"dyn_{a.name}", step=step, total=total, finished=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
