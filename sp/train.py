"""Massive self-play PPO for slither on all GPUs (torchrun, one World per GPU).

Each GPU: A arenas x S snakes. 75% of slots are played by the current policy (trained), 25% by frozen past snapshots
(the league: prevents copies exploiting copies). Slot ownership is re-drawn only when its snake dies.
Every update: T ticks of every snake -> PPO (GAE) on the current-policy transitions, gradients averaged over GPUs.
Rank 0 writes the dashboard (dash/), checkpoints to ckpt.pt, and backs up to Hugging Face every 30 min.
  torchrun --standalone --nproc_per_node 8 train.py
"""
import argparse, copy, glob, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.distributed as dist
from sim import World, NA, NB, NCH, NSC, OBS, DT

OUT = os.environ.get("SP_ROOT", "/tmp/slither/sp")
HF_REPO, HF_DIR = "mundoamundo/slither-maskgit-live", "selfplay"
SERIES_KEYS = ["t", "upd", "decisions", "surv_cur", "surv_snap", "surv_cur_med", "surv_cur_p90", "peak_cur", "peak_snap",
               "entropy", "explained_var", "kl", "clipfrac", "dps", "edge_frac", "biggest"]


class Res(nn.Module):
    def __init__(self, c):
        super().__init__(); self.a = nn.Conv2d(c, c, 3, padding=1); self.b = nn.Conv2d(c, c, 3, padding=1)
    def forward(self, x): return x + self.b(F.relu(self.a(F.relu(x))))


class Net(nn.Module):
    """IMPALA ResNet CNN (channels 32/64/64) + scalars -> 512 -> policy (64 actions) and value."""
    def __init__(self, ch=(32, 64, 64), hid=512):
        super().__init__(); L = []; c0 = NCH
        for c in ch: L += [nn.Conv2d(c0, c, 3, padding=1), nn.MaxPool2d(3, 2, 1), Res(c), Res(c)]; c0 = c
        self.cnn = nn.Sequential(*L)
        self.fc = nn.Linear(c0 * (OBS // 8) ** 2 + NSC, hid)
        self.pi = nn.Linear(hid, NA); self.v = nn.Linear(hid, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01); nn.init.zeros_(self.pi.bias)
    def forward(self, x, s):
        h = F.relu(self.cnn(x)).flatten(1)
        h = F.relu(self.fc(torch.cat([h, s.to(h.dtype)], 1)))
        return self.pi(h).float(), 10 * self.v(h).squeeze(1).float()


def jdump(o, p):
    with open(p + ".tmp", "w") as f: json.dump(o, f)
    os.replace(p + ".tmp", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arenas", type=int, default=1200); ap.add_argument("--snakes", type=int, default=100)
    ap.add_argument("--T", type=int, default=8); ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--nmb", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995); ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2); ap.add_argument("--ent", type=float, default=0.005)
    ap.add_argument("--vf", type=float, default=0.5); ap.add_argument("--snap_p", type=float, default=0.25)
    ap.add_argument("--snap_every", type=int, default=10); ap.add_argument("--pool", type=int, default=12)
    ap.add_argument("--max_updates", type=int, default=10 ** 9)
    a = ap.parse_args()
    dist.init_process_group("nccl"); rank, W = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))); torch.cuda.set_device(dev)
    torch.backends.cudnn.benchmark = False   # batch shapes change every tick (league subsets): re-benchmarking kills throughput; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    os.makedirs(f"{OUT}/dash", exist_ok=True)
    world = World(a.arenas, a.snakes, dev, seed=1000 + rank); N = world.N
    net = Net().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    pool, upd, decisions, t_start0 = [], 0, 0, time.time()
    ck = f"{OUT}/ckpt.pt"
    if os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); upd, decisions = st["upd"], st["decisions"]
        t_start0 = time.time() - st.get("elapsed", 0)
        for sd in st["pool"]:
            m = Net().to(dev).to(memory_format=torch.channels_last); m.load_state_dict(sd); m.eval(); pool.append(m)
        if rank == 0: print(f"resumed update {upd}, pool {len(pool)}", flush=True)
    for p in net.parameters(): dist.broadcast(p.data, 0)
    nparam = sum(p.numel() for p in net.parameters())
    model_of = torch.zeros(N, dtype=torch.long, device=dev)          # 0 = current policy, k>0 = pool[k-1]


    def redraw(idx):
        if not len(pool) or idx.numel() == 0: model_of[idx] = 0; return
        r = torch.rand(idx.numel(), device=dev)
        model_of[idx] = torch.where(r < a.snap_p, 1 + torch.randint(len(pool), (idx.numel(),), device=dev), 0)

    T = a.T
    obs_b = torch.empty(T, N, NCH, OBS, OBS, dtype=torch.bfloat16, device=dev)
    sca_b = torch.empty(T, N, NSC, device=dev); act_b = torch.empty(T, N, dtype=torch.long, device=dev)
    lp_b = torch.empty(T, N, device=dev); v_b = torch.empty(T, N, device=dev)
    r_b = torch.empty(T, N, device=dev); d_b = torch.empty(T, N, device=dev); own_b = torch.empty(T, N, dtype=torch.bool, device=dev)
    win = []; t_hf = time.time() - 1680; t_ck = time.time(); t_img = 0.0
    hf_status = dict(t=None, ok=None)
    stats_path = f"{OUT}/dash/stats.jsonl"
    series = []
    if rank == 0 and os.path.exists(stats_path):
        for line in open(stats_path):
            try: r0 = json.loads(line); series.append({k: r0.get(k) for k in SERIES_KEYS})
            except Exception: pass
        series = [r0 for r0 in series if (r0.get("decisions") or 0) <= decisions]   # drop points past the resumed checkpoint
        while len(series) > 1200: series = series[::2]
    o, s = world.observe()
    while upd < a.max_updates:
        t0 = time.time(); deaths = []
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(T):
                obs_b[t] = o; sca_b[t] = s; own_b[t] = model_of == 0
                logits = torch.empty(N, NA, device=dev); val = torch.zeros(N, device=dev)
                for k in range(len(pool) + 1):
                    m = (model_of == k).nonzero().squeeze(1)
                    if m.numel() == 0: continue
                    for mc in m.split(16384):
                        lg, v = (net if k == 0 else pool[k - 1])(o[mc].contiguous(memory_format=torch.channels_last), s[mc])
                        logits[mc] = lg.float(); val[mc] = v.float()
                dist_ = torch.distributions.Categorical(logits=logits)
                act = dist_.sample(); act_b[t] = act; lp_b[t] = dist_.log_prob(act); v_b[t] = val
                r, done, info = world.step(act)
                r_b[t] = r; d_b[t] = done.float()
                if info is not None:
                    info["model"] = model_of[info["idx"]].clone()
                    info["kmodel"] = torch.where(info["killer"] >= 0, model_of[info["killer"].clamp(min=0)], -1)
                    deaths.append(info); redraw(info["idx"])
                o, s = world.observe()
            v_last = torch.cat([net(o[c].contiguous(memory_format=torch.channels_last), s[c])[1].float() for c in torch.arange(N, device=dev).split(16384)])
        t_roll = time.time() - t0
        # GAE (per slot; a death cuts the bootstrap, so a slot changing owner after a death is fine)
        adv = torch.zeros(T, N, device=dev); last = torch.zeros(N, device=dev)
        nv = torch.where(model_of == 0, v_last.float(), 0.0)
        for t in reversed(range(T)):
            nvt = nv if t == T - 1 else v_b[t + 1]
            delta = r_b[t] + a.gamma * nvt * (1 - d_b[t]) - v_b[t]
            last = delta + a.gamma * a.lam * (1 - d_b[t]) * last
            adv[t] = last
        ret = adv + v_b
        sel = own_b.flatten().nonzero().squeeze(1)
        Ob = obs_b.view(T * N, NCH, OBS, OBS); Sb = sca_b.view(T * N, NSC); Ab = act_b.flatten(); LPb = lp_b.flatten()
        ADb = adv.flatten(); RETb = ret.flatten(); Vb = v_b.flatten()
        net.train(); mm = dict(pl=0., vl=0., ent=0., kl=0., clip=0.); n_mb = 0
        t1 = time.time()
        for ep in range(a.epochs):
            perm = sel[torch.randperm(sel.numel(), device=dev)]
            for mb in perm.chunk(a.nmb):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, v = net(Ob[mb].contiguous(memory_format=torch.channels_last), Sb[mb])
                lp_all = F.log_softmax(lg, -1); lp = lp_all.gather(1, Ab[mb][:, None]).squeeze(1)
                ratio = torch.exp(lp - LPb[mb]); ad = ADb[mb]; ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pl = -torch.min(ratio * ad, ratio.clamp(1 - a.clip, 1 + a.clip) * ad).mean()
                vl = (((v - RETb[mb]) / 10) ** 2).mean()
                ent = -(lp_all.exp() * lp_all).sum(1).mean()
                loss = pl + a.vf * vl - a.ent * ent
                opt.zero_grad(set_to_none=True); loss.backward()
                for p in net.parameters():
                    if p.grad is not None: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
                with torch.no_grad():
                    mm["pl"] += float(pl); mm["vl"] += float(vl); mm["ent"] += float(ent)
                    mm["kl"] += float(((ratio - 1) - torch.log(ratio)).mean()); mm["clip"] += float(((ratio - 1).abs() > a.clip).float().mean())
                n_mb += 1
        t_train = time.time() - t1
        upd += 1
        for k in mm: mm[k] /= max(1, n_mb)
        ev = float(1 - torch.var(RETb[sel] - Vb[sel]) / (torch.var(RETb[sel]) + 1e-8))
        if upd % a.snap_every == 0:
            m = copy.deepcopy(net).eval()
            for p in m.parameters(): p.requires_grad_(False)
            if len(pool) < a.pool: pool.append(m)
            else: pool[int(np.random.RandomState(upd).randint(0, a.pool))] = m
        # ---- episode statistics (all ranks -> sums; rank 0 keeps the distributions)
        if deaths:
            D = {k: torch.cat([d[k] for d in deaths]) for k in deaths[0]}
            cur = D["model"] == 0
            vec = torch.tensor([float(cur.sum()), float(D["age"][cur].sum()), float(D["peak"][cur].sum()), float((D["edge"] & cur).sum()),
                                float((~cur).sum()), float(D["age"][~cur].sum()), float(D["peak"][~cur].sum()),
                                float(((D["kmodel"] == 0) & ~cur).sum()), float(((D["kmodel"] > 0) & cur).sum()),
                                float(D["kills"][cur].sum())], device=dev)
        else: vec = torch.zeros(10, device=dev); D = None
        nd = torch.tensor([float(N * T)], device=dev)
        dist.all_reduce(vec); dist.all_reduce(nd)
        big = torch.tensor([float(world.score.max())], device=dev); dist.all_reduce(big, op=dist.ReduceOp.MAX)
        decisions += int(nd)
        if rank == 0:
            if D is not None:
                cur = (D["model"] == 0).cpu().numpy(); age = D["age"].cpu().numpy() * DT; pk = D["peak"].cpu().numpy()
                win.append((age[cur], pk[cur], age[~cur], pk[~cur]))
            win = win[-20:]
            ca = np.concatenate([w[0] for w in win]) if win else np.zeros(0); cp = np.concatenate([w[1] for w in win]) if win else np.zeros(0)
            sa = np.concatenate([w[2] for w in win]) if win else np.zeros(0)
            el = time.time() - t0
            rec = dict(t=time.time(), upd=upd, decisions=decisions, elapsed=time.time() - t_start0, dps=float(nd) / el,
                       iter_s=el, roll_s=t_roll, train_s=t_train, gpus=W, snakes=N * W, arenas=a.arenas * W, pool=len(pool),
                       ep_cur=int(vec[0]), ep_snap=int(vec[4]),
                       surv_cur=float(vec[1] / max(1, vec[0])) * DT, surv_snap=float(vec[5] / vec[4]) * DT if vec[4] > 0 else None,
                       surv_cur_med=float(np.median(ca)) if len(ca) else None, surv_snap_med=float(np.median(sa)) if len(sa) else None,
                       surv_cur_p90=float(np.percentile(ca, 90)) if len(ca) else None,
                       peak_cur=float(vec[2] / max(1, vec[0])), peak_snap=float(vec[6] / vec[4]) if vec[4] > 0 else None,
                       peak_cur_med=float(np.median(cp)) if len(cp) else None,
                       edge_frac=float(vec[3] / max(1, vec[0])), kills_by_cur_of_snap=int(vec[7]), kills_by_snap_of_cur=int(vec[8]),
                       biggest=float(big), pl=mm["pl"], vl=mm["vl"], entropy=mm["ent"], kl=mm["kl"], clipfrac=mm["clip"],
                       explained_var=ev, n_train=int(sel.numel()) * W, params=nparam, hf=hf_status)
            with open(stats_path, "a") as f: f.write(json.dumps(rec) + "\n")
            series.append({k: rec[k] for k in SERIES_KEYS})
            if len(series) > 1200: series = series[::2]
            jdump(series, f"{OUT}/dash/series.json")
            jdump(rec, f"{OUT}/dash/now.json")
            print(f"upd {upd} dps {rec['dps']:.0f} surv {rec['surv_cur']:.1f}s vs snap {rec['surv_snap'] or 0:.1f}s peak {rec['peak_cur']:.0f} "
                  f"ent {mm['ent']:.2f} ev {ev:.2f} kl {mm['kl']:.4f} roll {t_roll:.1f}s train {t_train:.1f}s", flush=True)
            if time.time() - t_img > 4:
                t_img = time.time()
                try: write_images(world, model_of, o)
                except Exception as e: print("img:", e, flush=True)
            if time.time() - t_ck > 600 or upd == 1 or os.path.exists(f"{OUT}/SAVE"):   # touch SAVE to force a checkpoint (before a restart)
                t_ck = time.time()
                st = dict(net=net.state_dict(), opt=opt.state_dict(), upd=upd, decisions=decisions, elapsed=time.time() - t_start0,
                          pool=[m.state_dict() for m in pool], args=vars(a))
                torch.save(st, ck + ".tmp"); os.replace(ck + ".tmp", ck)
                torch.save(net.state_dict(), f"{OUT}/policy.pt.tmp"); os.replace(f"{OUT}/policy.pt.tmp", f"{OUT}/policy.pt")
                if os.path.exists(f"{OUT}/SAVE"): os.remove(f"{OUT}/SAVE"); print(f"saved checkpoint at update {upd}", flush=True)
            if time.time() - t_hf > 1800:
                t_hf = time.time(); hf_status["t"] = time.time()
                try:
                    from huggingface_hub import HfApi
                    up = [(ck, f"{HF_DIR}/ckpt.pt"), (f"{OUT}/policy.pt", f"{HF_DIR}/policy.pt"), (stats_path, f"{HF_DIR}/stats.jsonl")]
                    up += [(p, f"{HF_DIR}/code/{os.path.relpath(p, OUT + '/code')}") for p in glob.glob(f"{OUT}/code/**", recursive=True) if os.path.isfile(p)]
                    HfApi().batch_bucket_files(HF_REPO, add=up); hf_status["ok"] = True; print("hf backup ok", flush=True)
                except Exception as e: hf_status["ok"] = False; print(f"hf backup failed: {e}", flush=True)
    dist.destroy_process_group()


def obs_rgb(x, dev):
    """[n, 8, 64, 64] observations -> [n, 64, 64, 3] uint8 pictures (same colours as the dashboard legend)."""
    x = x.float(); n = x.shape[0]
    rgb = torch.zeros(n, OBS, OBS, 3, device=dev)
    rgb += x[:, 4, ..., None].clamp(0, 1) * torch.tensor([90., 70., 140.], device=dev)
    rgb = torch.where((x[:, 5] > 0)[..., None], torch.tensor([90., 30., 30.], device=dev), rgb)
    rgb = torch.where((x[:, 6] > 0)[..., None], torch.tensor([60., 110., 200.], device=dev), rgb)      # own body: blue
    b = 0.5 + 0.5 * x[:, 7]
    rgb = torch.where((x[:, 0] > 0)[..., None], torch.stack([b * 80, b * 220, b * 140], -1), rgb)   # others: green, brighter = bigger
    rgb = torch.where((x[:, 1] > 0)[..., None], torch.tensor([255., 255., 255.], device=dev), rgb)
    rgb[:, OBS // 2 - 1:OBS // 2 + 1, OBS // 2 - 1:OBS // 2 + 1] = torch.tensor([255., 220., 0.], device=dev)
    return rgb.clamp(0, 255).byte().cpu().numpy()


def write_images(world, model_of, o):
    """Arena 0 for the dashboard: map picture, every snake's observation (sprite sheet, 10 per row), snake list."""
    from PIL import Image
    img = world.render(0, model_of)
    Image.fromarray(img).save(f"{OUT}/dash/arena.jpg.tmp.jpg", quality=85); os.replace(f"{OUT}/dash/arena.jpg.tmp.jpg", f"{OUT}/dash/arena.jpg")
    S = world.S; tiles = obs_rgb(o[:S], world.dev)
    rows = (S + 9) // 10; sheet = np.zeros((rows * OBS, 10 * OBS, 3), np.uint8)
    for i in range(S): sheet[(i // 10) * OBS:(i // 10 + 1) * OBS, (i % 10) * OBS:(i % 10 + 1) * OBS] = tiles[i]
    Image.fromarray(sheet).save(f"{OUT}/dash/eyes.png.tmp.png"); os.replace(f"{OUT}/dash/eyes.png.tmp.png", f"{OUT}/dash/eyes.png")
    h = world.G * 20.0 / 2
    pos = ((world.pos[:S] + h) / 20.0).cpu().numpy(); ang = world.ang[:S].cpu().numpy()
    sc = world.score[:S].cpu().numpy(); age = world.age[:S].cpu().numpy(); k = world.kills[:S].cpu().numpy()
    mo = model_of[:S].cpu().numpy(); bo = world.boost[:S].cpu().numpy()
    jdump(dict(t=time.time(), G=world.G, snakes=[dict(i=i, x=float(pos[i, 0]), y=float(pos[i, 1]), ang=float(ang[i]), score=float(sc[i]),
               age_s=float(age[i]) * DT, kills=int(k[i]), model=int(mo[i]), boost=bool(bo[i])) for i in range(S)]), f"{OUT}/dash/arena.json")


if __name__ == "__main__":
    main()
