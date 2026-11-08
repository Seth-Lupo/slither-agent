"""Death events (a worm turning into orbs) and per-token death masks.

1. Candidates: every head track (heads_v1) of >= 10 detections that ends inside the crop, away from the player's head.
2. Orb features on the decoded frames g-3 .. g+4 (g = first frame without the head): scale-normalised determinant of
   Hessian of luminance, bright blobs only. Round orbs and white masses respond, while elongated worm bodies (ridges) do not.
   Energy within 64 px of the head: small scales (orbs, sparkles) and large scales (white masses); after (max g..g+4)
   minus before (mean g-3, g-2).
3. Score = mean population percentile of (small-orb increase, large-mass increase, track length). Keep the top 10%.
   No weights are fitted. Hand-label check (labels/): blind precision 0.54 (25/46), estimated recall 0.35, against
   0.25 / 0.46 for the earlier rule (long track, ends mid-crop, no head reappearing).
4. Mask per kept event and target frame t = g-1 .. g+4: tokens within 80 px whose orb energy rose by > MASK_T over the
   pre-death mean, dilated by one token (covers orb edges and the body that vanished under them).

Output DATA/deaths.npz: g [K], xy [K,2] (128-px frame coords), mask [K,6,32] (packed bits over 256 tokens), score [K].
Intermediate candidates/features are cached in DATA/events_cache.npz.  python events.py
"""
import glob, json, math, os
import numpy as np, torch, torch.nn.functional as F
import common as C

X0, S = 140, 360 / 128                     # crop=ih:ih on 640x360 keeps x in [140, 500]; source px per frame px
HF = f"{C.ROOT}/tracks"; CACHE = f"{C.DATA}/events_cache.npz"
KEEP_Q, MASK_T, SIG = 0.90, 0.015, [1.0, 1.6, 2.5, 4.0, 6.5]


def candidates():
    from huggingface_hub import snapshot_download
    index = json.load(open(f"{C.DATA}/index.json")); vs = [v for v in index if not v["mid"].endswith("-raw")]
    snapshot_download("mundoamundo/slither-wam-video-actions", repo_type="dataset", local_dir=HF, max_workers=8,
                      allow_patterns=["metadata/*"] + [f"heads_v1/tracks/{v['mid']}.npz" for v in vs])
    rates = {}
    for m in glob.glob(f"{HF}/metadata/*.json"):
        for p in json.load(open(m)).get("current_prepared", []): rates[p["media_id"]] = p["playback_rate"]
    cut = np.load(f"{C.DATA}/cut.npy"); rows = []
    for v in vs:
        p = f"{HF}/heads_v1/tracks/{v['mid']}.npz"
        if not os.path.exists(p): continue
        t = np.load(p); keep = t["keep_idx"].astype(np.int64); n = int(t["src_frames"])
        sel = np.round(np.arange(0, len(keep), len(keep) / (n / 30.0 * rates.get(v["mid"], 1.0)) / C.FPS)).astype(np.int64)   # as in C.videos()
        sel = sel[sel < len(keep)][:v["len"]]; fr, xy, tr = t["det_frame"], t["det_xy_score"], t["det_track"]
        for k in np.unique(tr):
            i = np.nonzero(tr == k)[0]
            if len(i) < 10: continue
            e = i[-1]; x, y = xy[e, 0], xy[e, 1]
            if not (X0 + 30 < x < X0 + 330 and 30 < y < 330) or (abs(x - 320) < 40 and abs(y - 180) < 40): continue
            g = v["start"] + int(np.searchsorted(sel, fr[e])) + 1
            if g - 4 < v["start"] or g + 5 >= v["start"] + v["len"] or cut[g - 3:g + 6].any(): continue
            rows.append((g, (x - X0) / S, y / S, len(i)))
    return np.array(rows, np.float64)


def blob_energy(lum):
    """lum [N,1,128,128] -> (small, large) bright-blob DoH energy maps [N,1,128,128]."""
    out = []
    for s in SIG:
        r = int(math.ceil(3 * s)); x = torch.arange(-r, r + 1, device=lum.device, dtype=torch.float32)
        g = torch.exp(-x ** 2 / (2 * s * s)); g /= g.sum(); d2 = (x ** 2 / s ** 4 - 1 / s ** 2) * g; d1 = -x / s ** 2 * g
        sep = lambda im, kx, ky: F.conv2d(F.pad(F.conv2d(F.pad(im, (r, r, 0, 0), mode="reflect"), kx.view(1, 1, 1, -1)), (0, 0, r, r), mode="reflect"), ky.view(1, 1, -1, 1))
        Lxx, Lyy, Lxy = sep(lum, d2, g), sep(lum, g, d2), sep(lum, d1, d1)
        det = s ** 4 * (Lxx * Lyy - Lxy ** 2)
        out.append(torch.where((Lxx < 0) & (Lyy < 0), det, torch.zeros_like(det)).clamp_min(0))
    return out[0] + out[1], out[3] + out[4]


def frames(tok, tokz, gs, offs):
    w = (gs[:, None] + offs).ravel()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = tokz.decode(torch.from_numpy(tok[w].astype(np.int64)).cuda().view(-1, 16, 16)).float().clamp(0, 1)
    return x.mean(1, keepdim=True).view(len(gs), len(offs), 1, 128, 128)


def disc(xy, r, n=128, cell=1):
    c = (torch.arange(n, device="cuda") + 0.5) * cell
    return torch.hypot(c[None, None, :] - xy[:, 0, None, None], c[None, :, None] - xy[:, 1, None, None]) <= r


def features(R, tok, tokz):
    offs = np.arange(-3, 5); out = []                                         # g-3 .. g+4
    for b in range(0, len(R), 256):
        r = R[b:b + 256]; B = len(r)
        with torch.no_grad():
            lum = frames(tok, tokz, r[:, 0].astype(np.int64), offs)
            sm, lg = blob_energy(lum.flatten(0, 1))
            d = disc(torch.from_numpy(r[:, 1:3]).float().cuda(), 64).float()[:, None, None]
            m = lambda t: (t.view(B, len(offs), 1, 128, 128) * d).sum((-1, -2, -3)) / d.sum((-1, -2, -3))
            inc = lambda v: v[:, 3:].max(1).values - v[:, :2].mean(1)
            out.append(torch.stack([inc(m(sm)), inc(m(lg))], 1).cpu().numpy())
        if b % 25600 == 0: print("features", b, len(R), flush=True)
    return np.concatenate(out)


def masks(E, tok, tokz):
    offs = np.arange(-3, 5); out = []                                         # g-3, g-2 = before; g-1 .. g+4 = targets
    for b in range(0, len(E), 256):
        e = E[b:b + 256]; B = len(e)
        with torch.no_grad():
            lum = frames(tok, tokz, e[:, 0].astype(np.int64), offs)
            sm, lg = blob_energy(lum.flatten(0, 1)); en = F.avg_pool2d(sm + lg, 8).view(B, len(offs), 16, 16)   # per-token energy
            rise = en[:, 2:] - en[:, :2].mean(1, keepdim=True)                                                 # [B,6,16,16]
            near = disc(torch.from_numpy(e[:, 1:3]).float().cuda(), 80, 16, 8)[:, None]
            mk = F.max_pool2d(((rise > MASK_T) & near).float(), 3, 1, 1) > 0
            out.append(np.packbits(mk.view(B, 6, 256).cpu().numpy(), axis=-1))
    return np.concatenate(out)


def main():
    import tokenizer as T
    tok = np.load(f"{C.DATA}/tokens.npy", mmap_mode="r"); tokz = T.load()
    if os.path.exists(CACHE): z = np.load(CACHE); R, Fe = z["R"], z["F"]
    else: R = candidates(); Fe = features(R, tok, tokz); np.savez(CACHE, R=R, F=Fe)
    pct = lambda v: (np.argsort(np.argsort(v)) + 0.5) / len(v)
    score = (pct(Fe[:, 0]) + pct(Fe[:, 1]) + pct(R[:, 3])) / 3
    E = R[score >= np.quantile(score, KEEP_Q)]; sc = score[score >= np.quantile(score, KEEP_Q)]
    M = masks(E, tok, tokz); nt = np.unpackbits(M, axis=-1).sum(-1)               # [K,6] mask sizes
    np.savez(f"{C.DATA}/deaths.npz", g=E[:, 0].astype(np.int64), xy=E[:, 1:3].astype(np.float32), mask=M, score=sc.astype(np.float32))
    C.stage("events", candidates=len(R), deaths=len(E), mask_tokens_mean=float(nt.mean()), empty_masks=float((nt.sum(1) == 0).mean()), finished=True)
    print(f"{len(R)} candidates -> {len(E)} deaths; mean mask tokens per frame {nt.mean():.1f}; empty {100 * (nt.sum(1) == 0).mean():.0f}%")


if __name__ == "__main__":
    main()
