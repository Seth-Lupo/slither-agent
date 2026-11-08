"""Live PPO learner: consumes OUT/frag/*.npz from live.py, updates the self-play policy, publishes OUT/policy.pt.

Starts from OUT/init_policy.pt (the self-play policy at the moment live play began). One update per >= MIN_STEPS fresh
transitions; stats -> OUT/dash/learner.jsonl; backups to Hugging Face (selfplay_live/) every 30 min.
"""
import glob, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.environ.get("SP_CODE", "/tmp/slither/sp/code"))
from train import Net                     # noqa: E402

OUT = os.environ.get("LIVE_ROOT", "/tmp/slither/live")
MIN_STEPS, EPOCHS, MB, LR, CLIP, ENT, VF, GAMMA, LAM = 8192, 2, 1024, 1e-4, 0.2, 0.005, 0.5, 0.995, 0.95
HF_REPO = "mundoamundo/slither-maskgit-live"


def jdump(o, p):
    with open(p + ".tmp", "w") as f: json.dump(o, f)
    os.replace(p + ".tmp", p)


def main():
    dev = torch.device("cuda")
    net = Net().to(dev); opt = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5); ver = 0
    if os.path.exists(f"{OUT}/learner.pt"):
        st = torch.load(f"{OUT}/learner.pt", map_location=dev, weights_only=False)
        net.load_state_dict(st["sd"]); opt.load_state_dict(st["opt"]); ver = st["ver"]; print(f"resumed ver {ver}", flush=True)
    else:
        net.load_state_dict(torch.load(f"{OUT}/init_policy.pt", map_location=dev, weights_only=False))
    t_hf = time.time() - 1500
    while True:
        files = sorted(glob.glob(f"{OUT}/frag/f_*.npz"))
        data = []
        for f in files:
            try: data.append(dict(np.load(f)))
            except Exception: continue
            finally:
                try: os.remove(f)
                except Exception: pass
        if sum(len(d["a"]) for d in data) < MIN_STEPS:
            pend = getattr(main, "pend", []); pend += data; main.pend = pend
            if sum(len(d["a"]) for d in pend) < MIN_STEPS: time.sleep(2); continue
            data, main.pend = pend, []
        else: data = getattr(main, "pend", []) + data; main.pend = []
        t0 = time.time()
        Ob, Sc, A, LP, ADV, RET, V = [], [], [], [], [], [], []
        for d in data:      # GAE per fragment, bootstrapped from the stored value of the state after it
            r, dn, v = d["r"], d["done"].astype(np.float32), d["v"]; n = len(r)
            adv = np.zeros(n, np.float32); last = 0.0
            for t in reversed(range(n)):
                nv = float(d["last_v"]) if t == n - 1 else v[t + 1]
                delta = r[t] + GAMMA * nv * (1 - dn[t]) - v[t]; last = delta + GAMMA * LAM * (1 - dn[t]) * last; adv[t] = last
            Ob.append(d["obs"]); Sc.append(d["sca"]); A.append(d["a"]); LP.append(d["logp"]); ADV.append(adv); RET.append(adv + v); V.append(v)
        Ob = torch.from_numpy(np.concatenate(Ob)).to(dev); Sc = torch.from_numpy(np.concatenate(Sc)).to(dev)
        A = torch.from_numpy(np.concatenate(A).astype(np.int64)).to(dev); LP = torch.from_numpy(np.concatenate(LP)).to(dev)
        ADV = torch.from_numpy(np.concatenate(ADV)).to(dev); RET = torch.from_numpy(np.concatenate(RET)).to(dev); V = torch.from_numpy(np.concatenate(V)).to(dev)
        N = len(A); net.train(); m = dict(pl=0., vl=0., ent=0., kl=0., clip=0.); k = 0
        for _ in range(EPOCHS):
            for mb in torch.randperm(N, device=dev).split(MB):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, v = net(Ob[mb].float(), Sc[mb])
                lpa = F.log_softmax(lg, -1); lp = lpa.gather(1, A[mb][:, None]).squeeze(1)
                ratio = torch.exp(lp - LP[mb]); ad = ADV[mb]; ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pl = -torch.min(ratio * ad, ratio.clamp(1 - CLIP, 1 + CLIP) * ad).mean()
                vl = (((v - RET[mb]) / 10) ** 2).mean(); ent = -(lpa.exp() * lpa).sum(1).mean()
                loss = pl + VF * vl - ENT * ent
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
                with torch.no_grad():
                    m["pl"] += float(pl); m["vl"] += float(vl); m["ent"] += float(ent)
                    m["kl"] += float(((ratio - 1) - torch.log(ratio)).mean()); m["clip"] += float(((ratio - 1).abs() > CLIP).float().mean())
                k += 1
        ver += 1
        for x in m: m[x] /= max(1, k)
        ev = float(1 - torch.var(RET - V) / (torch.var(RET) + 1e-8))
        sd = {x: y.detach().clone() for x, y in net.state_dict().items()}
        torch.save(dict(ver=ver, sd=sd), f"{OUT}/policy.pt.tmp"); os.replace(f"{OUT}/policy.pt.tmp", f"{OUT}/policy.pt")
        torch.save(dict(ver=ver, sd=sd, opt=opt.state_dict()), f"{OUT}/learner.pt.tmp"); os.replace(f"{OUT}/learner.pt.tmp", f"{OUT}/learner.pt")
        rec = dict(t=time.time(), ver=ver, steps=N, frags=len(data), lag=float(ver - 1 - np.mean([int(d["ver"]) for d in data])),
                   rew_mean=float(np.concatenate([d["r"] for d in data]).mean()), explained_var=ev, train_s=time.time() - t0, **m)
        with open(f"{OUT}/dash/learner.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
        jdump(rec, f"{OUT}/dash/learner.json")
        print(f"ver {ver} steps {N} ent {m['ent']:.2f} kl {m['kl']:.4f} ev {ev:.2f}", flush=True)
        if time.time() - t_hf > 1800:
            t_hf = time.time()
            try:
                from huggingface_hub import HfApi
                up = [(f"{OUT}/{f}", f"selfplay_live/{f}") for f in ("policy.pt", "init_policy.pt", "dash/episodes.jsonl", "dash/learner.jsonl") if os.path.isfile(f"{OUT}/{f}")]
                HfApi().batch_bucket_files(HF_REPO, add=up); print("hf backup ok", flush=True)
            except Exception as e: print(f"hf backup failed: {e}", flush=True)


if __name__ == "__main__":
    main()
