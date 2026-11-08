"""PPO learner (v2): fine-tunes the pretrained s150p policy head (+ zero-initialised spatial branch) on live play.

Each iteration takes the trajectory fragments the actors wrote since the last one (dropping any from policies more
than --max_lag versions old), computes GAE advantages with the value net, and runs a few PPO epochs:
  loss_pi = -min(ratio * A, clip(ratio) * A) + kl_coef * KL(pi || pretrained) - ent_coef * H(pi)
  loss_v  = (V - return)^2
The first --warmup_min minutes train only V (policy stays exactly the pretrained one = clean baseline).
Publishes RL/policy.pt every iteration; logs RL/dash/learner.jsonl; backs up to the HF bucket under rl2/.
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rlcore as R

OUT = R.RL
for d in ("frag", "dash", "logs"): os.makedirs(f"{OUT}/{d}", exist_ok=True)


def gae(v, r, done, terminal, gamma, lam):
    """numpy: v [T+1] (last = bootstrap state's value), r [T], done [T] -> adv [T], ret [T]"""
    T = len(r); adv = np.zeros(T, np.float32); last = 0.0
    v = v.copy()
    if terminal: v[T] = 0.0
    for t in reversed(range(T)):
        nonterm = 0.0 if done[t] else 1.0
        delta = r[t] + gamma * v[t + 1] * nonterm - v[t]
        last = delta + gamma * lam * nonterm * last; adv[t] = last
    return adv, adv + v[:T]


def main(a):
    torch.manual_seed(0)
    wm_pol = R.load_wm(dev="cpu").pol
    pi, ref, V = R.Pi(wm_pol).cuda(), R.Pi(wm_pol).cuda().requires_grad_(False), R.Vf().cuda()
    opt_pi = torch.optim.Adam(pi.parameters(), lr=a.lr_pi); opt_v = torch.optim.Adam(V.parameters(), lr=a.lr_v)
    it, ver, pupd, steps_total = 0, 0, 0, 0
    if os.path.exists(f"{OUT}/learner_state.pt"):
        ck = torch.load(f"{OUT}/learner_state.pt", map_location="cuda")
        pi.load_state_dict(ck["pi"]); V.load_state_dict(ck["v"]); opt_pi.load_state_dict(ck["opt_pi"]); opt_v.load_state_dict(ck["opt_v"])
        it, ver, pupd, steps_total = ck["it"], ck["ver"], ck["pupd"], ck["steps_total"]; print(f"resumed it {it}", flush=True)
        for o, lr in ((opt_pi, a.lr_pi), (opt_v, a.lr_v)):
            for g in o.param_groups: g["lr"] = lr                    # resumed optimiser state must not keep the old lr
    t_start = time.time() - (a.warmup_min * 60 if pupd > 0 else 0); t_hf = time.time()
    if os.path.exists(f"{OUT}/REWARM"):                                   # reward changed: re-fit the critic before touching pi
        t_start = time.time(); print("re-warm-up: value only for", a.warmup_min, "min", flush=True)
    R.save_atomic(dict(pi=pi.state_dict(), ver=ver, pupd=pupd), f"{OUT}/policy.pt")
    pending, stale = [], 0
    while True:
        for p in sorted(glob.glob(f"{OUT}/frag/f_*.npz")):
            try: z = dict(np.load(p))
            except Exception: continue
            os.remove(p)                                                     # on-policy scratch: consumed once
            pending.append(z)
        fresh = [z for z in pending if int(z["ver"]) >= ver - a.max_lag]; stale += len(pending) - len(fresh); pending = fresh
        if sum(len(z["a"]) for z in pending) < a.min_batch: time.sleep(1); continue
        frags, pending = pending, []
        t0 = time.time()
        # ---- advantages
        G, Fe, A, LP, ADV, RET, DZ = [], [], [], [], [], [], []
        with torch.no_grad():
            for z in frags:
                g = torch.from_numpy(z["g"]).cuda(); fe = torch.from_numpy(z["feat"]).cuda()
                v = V(g, fe).cpu().numpy()
                adv, ret = gae(v, z["r"], z["done"], bool(z["terminal"]), R.GAMMA, a.lam)
                G.append(g[:-1]); Fe.append(fe[:-1]); A.append(torch.from_numpy(z["a"].astype(np.int64)).cuda())
                DZ.append(torch.from_numpy(z["danger"] if "danger" in z else np.zeros(len(z["a"]), np.float32)).cuda())
                LP.append(torch.from_numpy(z["logp"]).cuda()); ADV.append(torch.from_numpy(adv).cuda()); RET.append(torch.from_numpy(ret).cuda())
        G, Fe, A, LP, ADV, RET, DZ = (torch.cat(x) for x in (G, Fe, A, LP, ADV, RET, DZ))
        leash = (1 - DZ).clamp_min(R.KL_FLOOR)                            # KL to pretrained relaxed in danger
        with torch.no_grad():
            pred = torch.cat([V(G[i:i + 8192], Fe[i:i + 8192]) for i in range(0, len(G), 8192)])
            ev = 1 - float((RET - pred).var() / RET.var().clamp_min(1e-8))
        advn = (ADV - ADV.mean()) / ADV.std().clamp_min(1e-6)
        train_pi = time.time() - t_start > a.warmup_min * 60
        if train_pi and os.path.exists(f"{OUT}/REWARM"): os.remove(f"{OUT}/REWARM")
        m = dict(pl=0., vl=0., kl=0., ent=0., clip=0., k=0)
        for _ in range(a.epochs):
            perm = torch.randperm(len(A), device="cuda")
            for i in range(0, len(A), a.mb):
                j = perm[i:i + a.mb]
                vl = F.mse_loss(V(G[j], Fe[j]), RET[j]); opt_v.zero_grad(); vl.backward()
                torch.nn.utils.clip_grad_norm_(V.parameters(), 1.0); opt_v.step()
                lg = pi(G[j], Fe[j]); lp = R.bc_logp(lg).gather(1, A[j][:, None])[:, 0]
                ratio = (lp - LP[j]).exp()
                pl = -torch.min(ratio * advn[j], ratio.clamp(1 - a.clip, 1 + a.clip) * advn[j]).mean()
                with torch.no_grad(): rl = ref(G[j], Fe[j])
                kl, ent = R.kl_ent(lg, rl)
                loss = pl + a.kl_coef * (leash[j] * kl).mean() - a.ent_coef * ent.mean()
                if train_pi:
                    opt_pi.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt_pi.step()
                m["pl"] += float(pl); m["vl"] += float(vl); m["kl"] += float(kl.mean()); m["ent"] += float(ent.mean())
                m["clip"] += float(((ratio - 1).abs() > a.clip).float().mean()); m["k"] += 1
        it += 1; ver += 1; steps_total += len(A); pupd += int(train_pi)
        R.save_atomic(dict(pi=pi.state_dict(), ver=ver, pupd=pupd), f"{OUT}/policy.pt")
        R.save_atomic(dict(pi=pi.state_dict(), v=V.state_dict(), opt_pi=opt_pi.state_dict(), opt_v=opt_v.state_dict(), it=it, ver=ver,
                           pupd=pupd, steps_total=steps_total), f"{OUT}/learner_state.pt")
        k = max(1, m["k"])
        rec = dict(t=time.time(), it=it, ver=ver, pupd=pupd, training_policy=train_pi, batch=len(A), frags=len(frags),
                   steps_total=steps_total, stale_total=stale, pl=m["pl"] / k, vl=m["vl"] / k, kl_pretrained=m["kl"] / k, entropy=m["ent"] / k,
                   clipfrac=m["clip"] / k, explained_var=ev, v_mean=float(RET.mean()), adv_std=float(ADV.std()),
                   r_mean=float(np.mean([z["r"].mean() for z in frags])), iter_s=time.time() - t0, reward="survival+danger-shaping", danger_mean=float(DZ.mean()), danger_hi=float((DZ > 0.5).float().mean()))
        with open(f"{OUT}/dash/learner.jsonl", "a") as fh: fh.write(json.dumps(rec) + "\n")
        R.jdump(rec, f"{OUT}/dash/learner.json")
        if time.time() - t_hf > 1800:
            t_hf = time.time()
            try:
                from huggingface_hub import HfApi
                files = [(f"{OUT}/learner_state.pt", "rl2/learner_state.pt")] + [(p, f"rl2/dash/{os.path.basename(p)}") for p in glob.glob(f"{OUT}/dash/*.jsonl")]
                HfApi().batch_bucket_files("mundoamundo/slither-maskgit-live", add=files); print("hf backup ok", flush=True)
            except Exception as e: print(f"hf backup failed: {e}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lr_pi", type=float, default=1e-4); p.add_argument("--lr_v", type=float, default=3e-4)
    p.add_argument("--min_batch", type=int, default=16384); p.add_argument("--mb", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=4); p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--lam", type=float, default=0.95); p.add_argument("--kl_coef", type=float, default=0.0)
    p.add_argument("--ent_coef", type=float, default=0.0); p.add_argument("--max_lag", type=int, default=2)
    p.add_argument("--warmup_min", type=float, default=10.0)
    main(p.parse_args())
