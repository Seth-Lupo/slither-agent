"""Smoke test for learner v3 on one GPU: fake fragments -> build -> one minibatch forward/backward, timing + memory."""
import time, numpy as np, torch, torch.nn.functional as F, argparse, rlcore as R, learner as L
M = R.M
def frag(T=64, term=False):
    return dict(tok=np.random.randint(0, 1000, (T + 1, 8, 256)).astype(np.uint16), act=np.random.randint(0, 2, (T + 1, 8, 2)).astype(np.uint8),
                v=np.random.randn(T + 1).astype(np.float32), a=np.random.randint(0, 64, T).astype(np.int16), logp=np.full(T, -3.0, np.float32),
                r=np.full(T, 0.1, np.float32), done=np.array([False] * (T - 1) + [term]), danger=np.random.rand(T).astype(np.float32), ver=0, terminal=term)
a = argparse.Namespace(lam=0.95)
D = L.build([frag(), frag(term=True)], a); print({k: v.shape for k, v in D.items()})
model = L.Model(R.load_wm(dev="cpu").float()).cuda(); params = [p for p in model.parameters() if p.requires_grad]
print("trainable M", sum(p.numel() for p in params) / 1e6)
opt = torch.optim.Adam(params, 1e-5); dev = "cuda"; T = {k: torch.from_numpy(v.astype(np.int16) if v.dtype == np.uint16 else v).to(dev) for k, v in D.items()}
for rep in range(3):
    torch.cuda.synchronize(); t0 = time.time()
    j = torch.arange(64, device=dev)
    tok, act, ntok = T["tok"][j].long(), T["act"][j].long(), T["ntok"][j].long()
    mask = M.mask_batch(ntok)
    inp = torch.cat([tok, torch.where(mask, torch.full_like(ntok, M.MASK), ntok)[:, None]], 1)
    h = R.trunk(model.wm, inp, torch.cat([act, T["nact"][j].long()[:, None]], 1), grad_from=R.TRAIN_FROM)
    g, feat = R.state_feats(h[:, -2])
    with torch.autocast("cuda", dtype=torch.bfloat16): lgw = model.wm.head(h[:, -1, 2:])
    loss = F.cross_entropy(lgw.float()[mask], ntok[mask]) + model.vf(g, feat).pow(2).mean() + model.pi(g, feat).logsumexp(-1).mean() + model.dh(g, feat).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step(); torch.cuda.synchronize()
    print(f"minibatch 64: {time.time() - t0:.3f}s  loss {float(loss):.3f}  mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
# policy init check: pi on features equals pretrained head / TEMP
with torch.no_grad():
    feat2, g2, bc = R.encode_state(model.wm, tok[:8], act[:8]); print("pi == pretrained/TEMP:", float((model.pi(g2, feat2) - bc / R.TEMP).abs().max()))
