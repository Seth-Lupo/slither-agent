"""v2 sanity check on the node: Pi == pretrained head at init, shapes, KL = 0, GAE/learner step on fake fragments."""
import numpy as np, torch, rlcore as R, learner as Lr
wm = R.load_wm(); tok = torch.randint(0, 1000, (8, 8, 256)).cuda(); act = torch.randint(0, 32, (8, 8, 2)).cuda(); act[..., 1] %= 2
feat, g, bc = R.encode_state(wm, tok, act); pi = R.Pi(wm.pol).cuda()
lg = pi(g, feat); print("feat", tuple(feat.shape), "g", tuple(g.shape), "max |pi - pretrained|", float((lg - bc).abs().max()))
kl, ent = R.kl_ent(lg, bc); print("KL at init", float(kl.max()), "entropy", float(ent.mean()))
lp = R.bc_logp(lg); print("logp sums to 1:", float(lp.exp().sum(-1).mean()), "actions", R.NA, R.turn_name(0), "|", R.turn_name(33), "|", R.turn_name(63))
adv, ret = Lr.gae(np.array([1, 1, 1, 1], np.float32), np.array([.1, .1, .1], np.float32), np.array([0, 0, 1]), True, 0.995, 0.95)
print("gae", adv.round(3), ret.round(3))
print("first turn bins (deg/frame):", [round(R.ACTIONS[a][0], 2) for a in range(0, 64, 8)])
