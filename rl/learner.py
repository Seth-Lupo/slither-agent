"""Learner (v3): PPO + world-model + danger-foresight training through one shared, partly fine-tuned s150p backbone.
Data-parallel over GPUs 3-7 (torchrun; gradients all-reduced by hand). Per sample:

  9-frame input = the 8 context frames the actor saw + the next real frame (MaskGIT-masked)
  backbone      = s150p; blocks < TRAIN_FROM frozen, top blocks + final norm + frame head fine-tuned (lr_bb)
  frame 7 states -> policy (pretrained BC head + spatial branch, /TEMP), value, danger-foresight head
  frame 8 logits -> world-model next-frame CE on live data (keeps it a world model while RL reshapes it)

  loss = PPO clip + vf_coef * (V - GAE return)^2 + wm_coef * CE(next frame) + dz_coef * (danger 0.5 s ahead MSE + death<1 s BCE)

GAE uses the actors' value estimates (IMPALA-style; recorded per state). First --warmup_min minutes: policy loss off.
Rank 0 publishes RL/policy.pt (policy, value, fine-tuned backbone layers) for the actors every iteration.
  torchrun --nproc_per_node 5 learner.py      (CUDA_VISIBLE_DEVICES=3,4,5,6,7)
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.distributed as dist
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rlcore as R
M = R.M
OUT = R.RL


def gae(v, r, done, terminal, gamma, lam):
    T = len(r); adv = np.zeros(T, np.float32); last = 0.0; v = v.astype(np.float32).copy()
    if terminal: v[T] = 0.0
    for t in reversed(range(T)):
        nt = 0.0 if done[t] else 1.0
        delta = r[t] + gamma * v[t + 1] * nt - v[t]; last = delta + gamma * lam * nt * last; adv[t] = last
    return adv, adv + v[:T]


def build(frags, a):
    """Fragments -> flat per-transition arrays (numpy)."""
    out = {k: [] for k in ("tok", "act", "ntok", "nact", "wmv", "a", "logp", "adv", "ret", "vold", "dz", "dzf", "dzfv", "ds", "dsv")}
    for z in frags:
        T = len(z["a"]); term = bool(z["terminal"])
        adv, ret = gae(z["v"], z["r"], z["done"], term, R.GAMMA, a.lam)
        dz = z["danger"] if "danger" in z else np.zeros(T, np.float32)
        for i in range(T):
            last_term = term and i == T - 1
            out["tok"].append(z["tok"][i]); out["act"].append(z["act"][i])
            out["ntok"].append(z["tok"][i + 1][-1]); out["nact"].append(z["act"][i + 1][-1]); out["wmv"].append(not last_term)
            out["a"].append(z["a"][i]); out["logp"].append(z["logp"][i]); out["adv"].append(adv[i]); out["ret"].append(ret[i])
            out["vold"].append(z["v"][i]); out["dz"].append(dz[i])
            j = i + 5; out["dzf"].append(dz[j] if j < T else 0.0); out["dzfv"].append(j < T)
            died = term and (T - i) <= 10
            out["ds"].append(1.0 if died else 0.0); out["dsv"].append(died or i + 10 <= T)
    return {k: np.array(v) for k, v in out.items()}


class Model(nn.Module):
    def __init__(s, wm):
        super().__init__()
        s.wm, s.pi, s.vf, s.dh = wm, R.Pi(wm.pol), R.Vf(), R.DangerHead()
        s.ref = R.Pi(wm.pol).requires_grad_(False)                                       # for the KL diagnostic only
        train = set(R.backbone_trainable(wm))
        for n, p in wm.named_parameters(): p.requires_grad_(n in train)


def main(a):
    dist.init_process_group("nccl"); rank, W = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"])); torch.cuda.set_device(dev)
    torch.manual_seed(0)                                               # identical initial weights on every rank
    model = Model(R.load_wm(dev="cpu").float()).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    bb = [p for p in model.wm.parameters() if p.requires_grad]
    opt = torch.optim.Adam([{"params": bb, "lr": a.lr_bb}, {"params": list(model.pi.parameters()) + list(model.dh.parameters()), "lr": a.lr_pi},
                            {"params": list(model.vf.parameters()), "lr": a.lr_v}])
    it, ver, pupd, steps_total, stale_total = 0, 0, 0, 0, 0
    if os.path.exists(f"{OUT}/learner_state.pt"):
        ck = torch.load(f"{OUT}/learner_state.pt", map_location=dev)
        model.load_state_dict(ck["model"], strict=False); opt.load_state_dict(ck["opt"])
        for g, lr in zip(opt.param_groups, (a.lr_bb, a.lr_pi, a.lr_v)): g["lr"] = lr
        it, ver, pupd, steps_total = ck["it"], ck["ver"], ck["pupd"], ck["steps_total"]
        if rank == 0: print(f"resumed it {it}", flush=True)
    torch.manual_seed(1000 + rank)
    t_start = time.time() - (a.warmup_min * 60 if pupd > 0 else 0); t_hf = time.time()
    if rank == 0: print(f"{W} ranks; trainable backbone params {sum(p.numel() for p in bb) / 1e6:.1f}M, heads {sum(p.numel() for p in params) / 1e6 - sum(p.numel() for p in bb) / 1e6:.1f}M", flush=True)

    def publish():
        if rank != 0: return
        bbsd = {n: p.detach().to(torch.bfloat16).cpu() for n, p in model.wm.named_parameters() if p.requires_grad}
        R.save_atomic(dict(pi=model.pi.state_dict(), vf=model.vf.state_dict(), bb=bbsd, ver=ver, pupd=pupd), f"{OUT}/policy.pt")
    publish()
    while True:
        files = sorted(glob.glob(f"{OUT}/frag/f_*.npz")) if rank == 0 else None
        obj = [files]; dist.broadcast_object_list(obj, 0); files = obj[0]
        # cheap size check on rank 0's listing: ~64 transitions per fragment
        if len(files) * 60 < a.min_batch: time.sleep(2); continue
        frags = []
        for p in files[rank::W]:
            try: z = dict(np.load(p))
            except Exception: continue
            if int(z["ver"]) < ver - a.max_lag: stale_total += 1; continue
            frags.append(z)
        dist.barrier()
        if rank == 0:
            for p in files:
                try: os.remove(p)
                except OSError: pass
        n_local = torch.tensor([sum(len(z["a"]) for z in frags)], device=dev); n_all = n_local.clone()
        dist.all_reduce(n_all); dist.all_reduce(n_local, op=dist.ReduceOp.MIN)
        if int(n_local) < a.mb: continue
        t0 = time.time()
        D = build(frags, a); N = len(D["a"])
        T = {k: torch.from_numpy(v.astype(np.int16) if v.dtype == np.uint16 else v).to(dev) for k, v in D.items()}
        advn = (T["adv"] - T["adv"].mean()) / T["adv"].std().clamp_min(1e-6)
        tp = torch.tensor([float(time.time() - t_start > a.warmup_min * 60)], device=dev); dist.broadcast(tp, 0); train_pi = bool(tp.item())
        nmb = min(int(n_local) // a.mb, a.max_mb)
        names = ("pl", "vl", "wm", "dzl", "dsl", "kl", "ent", "clip", "ds_acc"); m = {k: 0.0 for k in names}; k_ = 0
        null = torch.tensor(M.NULL, device=dev)
        for ep in range(a.epochs):
            perm = torch.randperm(N, device=dev)
            for b in range(nmb):
                j = perm[b * a.mb:(b + 1) * a.mb]
                tok, act, ntok = T["tok"][j].long(), T["act"][j].long(), T["ntok"][j].long()
                wmv = T["wmv"][j]; nact = torch.where(wmv[:, None], T["nact"][j].long(), null)
                mask = M.mask_batch(ntok) & wmv[:, None]
                inp = torch.cat([tok, torch.where(mask, torch.full_like(ntok, M.MASK), ntok)[:, None]], 1)
                h = R.trunk(model.wm, inp, torch.cat([act, nact[:, None]], 1), grad_from=R.TRAIN_FROM)
                g, feat = R.state_feats(h[:, -2])
                with torch.autocast("cuda", dtype=torch.bfloat16): lgw = model.wm.head(h[:, -1, 2:])
                lgw = lgw.float()
                wm_loss = F.cross_entropy(lgw[mask], ntok[mask]) if mask.any() else lgw.sum() * 0
                lg = model.pi(g, feat); lp = R.bc_logp(lg).gather(1, T["a"][j].long()[:, None])[:, 0]
                ratio = (lp - T["logp"][j]).exp(); aj = advn[j]
                pl = -torch.min(ratio * aj, ratio.clamp(1 - a.clip, 1 + a.clip) * aj).mean()
                vl = F.mse_loss(model.vf(g, feat), T["ret"][j])
                dzo = model.dh(g, feat); fv, sv = T["dzfv"][j].float(), T["dsv"][j].float()
                dzl = ((dzo[:, 0] - T["dzf"][j]) ** 2 * fv).sum() / fv.sum().clamp_min(1)
                dsl = (F.binary_cross_entropy_with_logits(dzo[:, 1], T["ds"][j].float(), reduction="none", pos_weight=torch.tensor(5.0, device=dev)) * sv).sum() / sv.sum().clamp_min(1)
                with torch.no_grad(): kl, ent = R.kl_ent(lg.detach(), model.ref(g, feat))
                ent_live = R.kl_ent(lg, lg.detach())[1]
                loss = (pl if train_pi else pl * 0) - a.ent_coef * ent_live.mean() + a.vf_coef * vl + a.wm_coef * wm_loss + a.dz_coef * (dzl + dsl)
                opt.zero_grad(set_to_none=True); loss.backward()
                grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                flat = torch.cat([x.reshape(-1) for x in grads]); dist.all_reduce(flat); flat /= W
                o = 0
                for p in params:
                    nn_ = p.numel(); p.grad = flat[o:o + nn_].view_as(p); o += nn_
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                m["pl"] += float(pl); m["vl"] += float(vl); m["wm"] += float(wm_loss); m["dzl"] += float(dzl); m["dsl"] += float(dsl)
                m["kl"] += float(kl.mean()); m["ent"] += float(ent.mean()); m["clip"] += float(((ratio - 1).abs() > a.clip).float().mean())
                m["ds_acc"] += float((((dzo[:, 1] > 0).float() == T["ds"][j].float()).float() * sv).sum() / sv.sum().clamp_min(1)); k_ += 1
        it += 1; ver += 1; steps_total += int(n_all); pupd += int(train_pi)
        mt = torch.tensor([m[k] / max(1, k_) for k in names], device=dev); dist.all_reduce(mt); mt /= W
        publish()
        if rank == 0:
            sd = {k: v for k, v in model.state_dict().items() if not k.startswith("ref.")}
            R.save_atomic(dict(model=sd, opt=opt.state_dict(), it=it, ver=ver, pupd=pupd, steps_total=steps_total), f"{OUT}/learner_state.pt")
            ev = 1 - float((T["ret"] - T["vold"]).var() / T["ret"].var().clamp_min(1e-8))
            mm = dict(zip(names, mt.tolist()))
            rec = dict(t=time.time(), it=it, ver=ver, pupd=pupd, training_policy=train_pi, batch=int(n_all), frags=len(files), steps_total=steps_total,
                       stale_total=stale_total, pl=mm["pl"], vl=mm["vl"], wm_ce=mm["wm"], danger_mse=mm["dzl"], death_bce=mm["dsl"], death_acc=mm["ds_acc"],
                       kl_pretrained=mm["kl"], entropy=mm["ent"], clipfrac=mm["clip"], explained_var=ev, v_mean=float(T["ret"].mean()),
                       danger_mean=float(T["dz"].mean()), iter_s=time.time() - t0, samples_trained=nmb * a.epochs * a.mb * W,
                       reward="survival+danger-shaping", finetune=f"blocks>={R.TRAIN_FROM}")
            with open(f"{OUT}/dash/learner.jsonl", "a") as fh: fh.write(json.dumps(rec) + "\n")
            R.jdump(rec, f"{OUT}/dash/learner.json")
            if time.time() - t_hf > 1800:
                t_hf = time.time()
                try:
                    from huggingface_hub import HfApi
                    up = [(f"{OUT}/learner_state.pt", "rl3/learner_state.pt")] + [(p, f"rl3/dash/{os.path.basename(p)}") for p in glob.glob(f"{OUT}/dash/*.jsonl")]
                    HfApi().batch_bucket_files("mundoamundo/slither-maskgit-live", add=up); print("hf backup ok", flush=True)
                except Exception as e: print(f"hf backup failed: {e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lr_bb", type=float, default=3e-6); p.add_argument("--lr_pi", type=float, default=3e-5); p.add_argument("--lr_v", type=float, default=3e-4)
    p.add_argument("--min_batch", type=int, default=16384); p.add_argument("--mb", type=int, default=64); p.add_argument("--max_mb", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1); p.add_argument("--clip", type=float, default=0.2); p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--vf_coef", type=float, default=0.5); p.add_argument("--wm_coef", type=float, default=0.5); p.add_argument("--dz_coef", type=float, default=1.0)
    p.add_argument("--ent_coef", type=float, default=0.0); p.add_argument("--max_lag", type=int, default=3); p.add_argument("--warmup_min", type=float, default=10.0)
    main(p.parse_args())
