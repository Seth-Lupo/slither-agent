"""Batched slither.io-like world on one GPU: A arenas x S snakes, all state in tensors.

Physics constants are measured from live slither.io (2026-11-08, scratchpad/probe/phys.py):
  cruise 176 units/s (sp 5.79), boost 425 units/s (sp 14), max turn 240 deg/s at sc=1, scaled by scang,
  sc = min(6, 1 + (sct-2)/106), scang = 0.13 + 0.87*((7-sc)/6)^2, body ~ sct points ~20 units apart,
  score ~ 4.2 * sct^1.26. No self-collision (as in slither.io); head into any other body or past the edge = death.

Grids (per arena, CELL units per cell): owner (snake id + 1 of a body point), heads (snake id + 1), food (score value).
Observation per snake: 64x64 heading-up egocentric grid (20 units/cell, x zoom_of(size) up to 2.5), 6 channels:
  0 other snakes' bodies (binary)  1 other heads  2/3 head velocity forward/right (/ boost speed)
  4 food value (log)  5 outside the map edge  6 own body (binary)  7 other bodies' size (sc / 6)
plus 8 scalars: speed, boosting, log score, edge distance, bearing to centre (sin, cos), last turn, last boost.
Action a in [0, 64): turn bin t = a % 32 (fraction of max turn rate, -1..1), boost = a // 32.
"""
import math, torch

CELL = 20.0          # grid cell and body point spacing, game units
V0, VB = 176.0, 425.0
TURN = math.radians(240.0)
DT, SUB = 0.1, 3     # one decision = 0.1 s of game time, physics in 3 sub-steps
P = 2048             # max body points per snake (length cap ~63k: real giants reach 50k)
NB, NA = 32, 64
OBS, NCH, NSC = 64, 8, 8
R_ARENA = 4500.0     # 100 snakes per arena (20% denser than 50 in 3500): encounters constant
# AGGRESSIVE: kills are the main income (+KILL*log(victim score)), grazing pays little, death is cheap,
# length decays (sitting still shrinks you; the way to stay big is to keep killing), ambient food scarce
ALIVE, GROW, DEATH, KILL = 0.0, 1.0, -6.0, 3.0
DECAY = 0.0005       # fraction of length lost per 0.1 s (~3%/min)
FOOD_FRAC = 0.01     # ambient food target (was 0.05)    # no pay for merely existing (-> coiling); KILL: to the snake whose body another head ran into     # per 0.1 s alive, x d log(length), on death (pure survival -> everyone coils in place)


def sct_of(score): return ((score.clamp(min=1.0) / 4.2) ** (1 / 1.26)).clamp(2, P - 8)
def sc_of(sct): return torch.clamp(1 + (sct - 2) / 106, max=6.0)
def scang_of(sc): return 0.13 + 0.87 * ((7 - sc) / 6) ** 2
def rad_of(sc): return 14.0 * sc
def zoom_of(sc): return (1 + 0.3 * (sc - 1)).clamp(1.0, 2.5)    # view widens as the snake grows, like the game camera


def _win(r):
    o = torch.arange(-r, r + 1)
    oy, ox = torch.meshgrid(o, o, indexing="ij")
    return ox.flatten(), oy.flatten()


class World:
    def __init__(self, A, S, dev, R=R_ARENA, seed=0):
        self.A, self.S, self.N, self.R, self.dev = A, S, A * S, R, dev
        self.G = int(2 * R / CELL) + 4; G2 = self.G * self.G
        self.gen = torch.Generator(device=dev); self.gen.manual_seed(seed)
        N = self.N; f = dict(device=dev)
        self.arena = torch.arange(N, **f) // S
        self.pos = torch.zeros(N, 2, **f); self.ang = torch.zeros(N, **f); self.vel = torch.zeros(N, 2, **f)
        self.score = torch.ones(N, **f); self.boost = torch.zeros(N, dtype=torch.bool, **f)
        self.body = torch.zeros(N, P, 2, **f); self.hp = torch.zeros(N, dtype=torch.long, **f)
        self.nb = torch.zeros(N, dtype=torch.long, **f); self.acc = torch.zeros(N, **f)
        self.age = torch.zeros(N, dtype=torch.long, **f); self.kills = torch.zeros(N, dtype=torch.long, **f)
        self.peak = torch.ones(N, **f); self.last_a = torch.full((N,), NB // 2, dtype=torch.long, **f)
        self.owner = torch.zeros(self.A * G2, dtype=torch.int32, **f)
        self.heads = torch.zeros(self.A * G2, dtype=torch.int32, **f)
        self.food = torch.zeros(self.A * G2, **f)
        c = (torch.arange(self.G, **f).float() + 0.5) * CELL - self.G * CELL / 2
        cy, cx = torch.meshgrid(c, c, indexing="ij")
        self.inside = ((cx ** 2 + cy ** 2) < (R - 30) ** 2).flatten()          # [G2]
        self.n_inside = int(self.inside.sum())
        self.inside_idx = self.inside.nonzero().squeeze(1)
        self.cwx, self.cwy = _win(9); self.cwx, self.cwy = self.cwx.to(dev), self.cwy.to(dev)      # collision window
        self.ewx, self.ewy = _win(6); self.ewx, self.ewy = self.ewx.to(dev), self.ewy.to(dev)      # eating window
        o = (torch.arange(OBS, **f).float() + 0.5 - OBS / 2) * CELL
        fi, rj = torch.meshgrid(-o, o, indexing="ij")                  # row 0 = straight ahead, column -> right
        self.of, self.orr = fi.flatten(), rj.flatten()
        self.food_target = FOOD_FRAC * self.n_inside
        self.rebuild(); self.spawn(torch.arange(N, **f))
        for _ in range(30): self._spawn_food()

    # ---------------------------------------------------------------- helpers
    def rnd(self, *shape): return torch.rand(*shape, device=self.dev, generator=self.gen)

    def cell(self, x, y):
        h = self.G * CELL / 2
        gx = ((x + h) / CELL).floor().long(); gy = ((y + h) / CELL).floor().long()
        ok = (gx >= 0) & (gx < self.G) & (gy >= 0) & (gy < self.G)
        return gx.clamp(0, self.G - 1), gy.clamp(0, self.G - 1), ok

    def flat(self, arena, gx, gy): return arena * (self.G * self.G) + gy * self.G + gx

    def body_points(self, idx=None):
        """-> (snake index [M], point xy [M,2]) of every live body point (all snakes or those in idx)."""
        ids = torch.arange(self.N, device=self.dev) if idx is None else idx
        k = torch.arange(P, device=self.dev)
        m = k[None] < self.nb[ids][:, None]
        j = (self.hp[ids][:, None] - k[None]) % P
        sid = ids[:, None].expand(-1, P)[m]
        return sid, self.body[sid, j[m]]

    def rebuild(self):
        self.owner.zero_(); self.heads.zero_()
        sid, xy = self.body_points()
        gx, gy, ok = self.cell(xy[:, 0], xy[:, 1])
        self.owner[self.flat(self.arena[sid], gx, gy)[ok]] = (sid[ok] + 1).int()
        gx, gy, ok = self.cell(self.pos[:, 0], self.pos[:, 1])
        ids = torch.arange(self.N, device=self.dev)
        self.heads[self.flat(self.arena, gx, gy)[ok]] = (ids[ok] + 1).int()
        self.owner[self.flat(self.arena, gx, gy)[ok]] = (ids[ok] + 1).int()

    def _spawn_food(self):
        tot = self.food.view(self.A, -1).sum(1)
        need = tot < self.food_target
        K, M = 4, 16                                   # food appears in patches (4 per arena per tick, 16 pellets each)
        ctr = self.inside_idx[(self.rnd(self.A, K) * self.n_inside).long().clamp(max=self.n_inside - 1)]
        cx = (ctr % self.G)[..., None].float() + torch.randn(self.A, K, M, device=self.dev, generator=self.gen) * 4
        cy = (ctr // self.G)[..., None].float() + torch.randn(self.A, K, M, device=self.dev, generator=self.gen) * 4
        gx = cx.round().long().clamp(0, self.G - 1); gy = cy.round().long().clamp(0, self.G - 1)
        cells = gy * self.G + gx
        val = (0.5 + 1.5 * self.rnd(self.A, K, M)) * need[:, None, None] * self.inside[cells]
        fl = torch.arange(self.A, device=self.dev)[:, None, None] * (self.G * self.G) + cells
        self.food.index_add_(0, fl.flatten(), val.flatten())

    # ---------------------------------------------------------------- spawning
    def spawn(self, idx):
        n = idx.numel()
        if n == 0: return
        big = self.rnd(n) < 0.4
        # heavy tail like real servers: Pareto(alpha 0.5) on [30, 50000] -> ~1 giant (>10k) per arena
        q = 1 - self.rnd(n) * (1 - (50000 / 30) ** -0.5)
        score = torch.where(big, 30 * q ** -2.0, 10 + 20 * self.rnd(n))
        score = torch.where(idx % self.S < 5, torch.full_like(score, 10.0), score)   # slots 0-4 of every arena always respawn at the minimum length
        # head position: best of 8 random candidates (fewest body cells nearby)
        cand = torch.sqrt(self.rnd(n, 8)) * (0.75 * self.R); th = self.rnd(n, 8) * 2 * math.pi
        cx, cy = cand * torch.cos(th), cand * torch.sin(th)
        ar = self.arena[idx]
        o = torch.arange(-8, 9, 2, device=self.dev)      # +-160 units: clear of even a giant's girth
        dy, dx = torch.meshgrid(o, o, indexing="ij"); dx, dy = dx.flatten(), dy.flatten()
        gx, gy, _ = self.cell(cx, cy)
        gx2 = (gx[..., None] + dx).clamp(0, self.G - 1); gy2 = (gy[..., None] + dy).clamp(0, self.G - 1)
        occ = (self.owner[self.flat(ar[:, None, None], gx2, gy2)] > 0).sum(-1)
        best = occ.argmin(1)
        hx, hy = cx.gather(1, best[:, None]).squeeze(1), cy.gather(1, best[:, None]).squeeze(1)
        ang = self.rnd(n) * 2 * math.pi
        npt = (sct_of(score) + 3).long().clamp(max=P)
        # body: an arc trailing behind the head (curls so long snakes stay compact)
        k = torch.arange(P, device=self.dev).float()
        # body: a smooth random wiggle trailing behind the head (like a snake that has been wandering)
        turn = torch.randn(n, P, device=self.dev, generator=self.gen) * 0.12
        turn = torch.cumsum(turn, 1); turn = turn - 0.97 * torch.cat([torch.zeros_like(turn[:, :1]), turn[:, :-1]], 1)   # smooth, mean-reverting curvature
        d = (ang + math.pi)[:, None] + torch.cumsum(turn, 1)
        path = torch.stack([hx[:, None] + CELL * torch.cumsum(torch.cos(d), 1) - CELL * torch.cos(d[:, :1]),
                            hy[:, None] + CELL * torch.cumsum(torch.sin(d), 1) - CELL * torch.sin(d[:, :1])], -1)
        j = (npt[:, None] - 1 - k.long()[None]).clamp(min=0)          # body[j] = path[npt-1-j]: hp = npt-1 is the head end
        self.body[idx] = path.gather(1, j[..., None].expand(-1, -1, 2))
        self.hp[idx] = npt - 1; self.nb[idx] = npt; self.acc[idx] = 0
        self.pos[idx] = torch.stack([hx, hy], 1); self.ang[idx] = ang
        self.vel[idx] = V0 * torch.stack([torch.cos(ang), torch.sin(ang)], 1)
        self.score[idx] = score; self.peak[idx] = score; self.boost[idx] = False
        self.age[idx] = 0; self.kills[idx] = 0; self.last_a[idx] = NB // 2

    # ---------------------------------------------------------------- one decision (0.1 s)
    def step(self, a):
        """a: [N] actions. -> reward [N], done [N] bool, info dict of tensors for the snakes that died."""
        N, dev = self.N, self.dev
        t = a % NB; frac = (t.float() - (NB - 1) / 2) / ((NB - 1) / 2)
        sc = sc_of(sct_of(self.score)); r_me = rad_of(sc)
        dang = frac * TURN * scang_of(sc) * DT
        self.boost = (a // NB == 1) & (self.score > 15)
        v = torch.where(self.boost, VB, V0)
        log0 = torch.log(self.score)
        dead = torch.zeros(N, dtype=torch.bool, device=dev); killer = torch.full((N,), -1, dtype=torch.long, device=dev)
        ids = torch.arange(N, device=dev)
        rad_all = rad_of(sc)
        for _ in range(SUB):
            self.ang = self.ang + dang / SUB
            dv = (v * DT / SUB)[:, None] * torch.stack([torch.cos(self.ang), torch.sin(self.ang)], 1)
            alive = ~dead
            self.pos = torch.where(alive[:, None], self.pos + dv, self.pos)
            self.acc = self.acc + torch.where(alive, v * DT / SUB, 0.0)
            m = (self.acc >= CELL) & alive
            self.hp = torch.where(m, (self.hp + 1) % P, self.hp)
            self.body[ids[m], self.hp[m]] = self.pos[m]
            self.nb = torch.where(m, self.nb + 1, self.nb); self.acc = torch.where(m, self.acc - CELL, self.acc)
            # collisions: head vs other snakes' body cells
            gx, gy, _ = self.cell(self.pos[:, 0], self.pos[:, 1])
            wx = gx[:, None] + self.cwx[None]; wy = gy[:, None] + self.cwy[None]
            okw = (wx >= 0) & (wx < self.G) & (wy >= 0) & (wy < self.G)
            own = self.owner[self.flat(self.arena[:, None], wx.clamp(0, self.G - 1), wy.clamp(0, self.G - 1))].long() - 1
            h = self.G * CELL / 2
            ccx = (wx.float() + 0.5) * CELL - h; ccy = (wy.float() + 0.5) * CELL - h
            dist = torch.hypot(ccx - self.pos[:, :1], ccy - self.pos[:, 1:2])
            thr = r_me[:, None] + rad_all[own.clamp(min=0)] - 6.0
            hit = okw & (own >= 0) & (own != ids[:, None]) & (dist < thr) & (self.age[own.clamp(min=0)] >= 10)   # a just-spawned body is harmless for 1 s
            hb = hit.any(1) & ~dead
            first = torch.where(hit, dist, torch.full_like(dist, 1e9)).argmin(1)
            killer = torch.where(hb, own.gather(1, first[:, None]).squeeze(1), killer)
            edge = (self.pos.norm(dim=1) > self.R) & ~dead
            dead = dead | hb | edge
        self.vel = v[:, None] * torch.stack([torch.cos(self.ang), torch.sin(self.ang)], 1)
        tgt = (sct_of(self.score) + 3).long().clamp(max=P)
        self.nb = torch.minimum(self.nb, tgt)
        # eating
        gx, gy, _ = self.cell(self.pos[:, 0], self.pos[:, 1])
        wx = (gx[:, None] + self.ewx[None]).clamp(0, self.G - 1); wy = (gy[:, None] + self.ewy[None]).clamp(0, self.G - 1)
        h = self.G * CELL / 2
        dd = torch.hypot((wx.float() + 0.5) * CELL - h - self.pos[:, :1], (wy.float() + 0.5) * CELL - h - self.pos[:, 1:2])
        em = (dd < (r_me + 35.0)[:, None]) & ~dead[:, None]
        fi = self.flat(self.arena[:, None], wx, wy)
        eaten = (self.food[fi] * em).sum(1)
        self.food[fi[em]] = 0.0
        self.score = self.score + eaten
        # boosting costs mass, dropped as food behind the snake
        loss = torch.where(self.boost & ~dead, (2.0 + 0.03 * self.score) * DT, 0.0)
        self.score = (self.score - loss).clamp(min=1.0)
        self.score = torch.where(self.score > 10, (self.score * (1 - DECAY)).clamp(min=10.0), self.score)
        tail = self.body[ids, (self.hp - self.nb + 1) % P]
        tx, ty, ok = self.cell(tail[:, 0], tail[:, 1])
        self.food.index_add_(0, self.flat(self.arena, tx, ty), 0.6 * loss * ok)
        self.peak = torch.maximum(self.peak, self.score)
        self.age += 1
        r = torch.where(dead, torch.full_like(log0, DEATH), ALIVE + GROW * (torch.log(self.score) - log0))
        # deaths: the body becomes food, the killer gets the credit, a new snake spawns in the slot
        info = None
        di = dead.nonzero().squeeze(1)
        if di.numel():
            sid, xy = self.body_points(di)
            px, py, ok = self.cell(xy[:, 0], xy[:, 1])
            share = (0.5 * self.score[sid] / self.nb[sid].float()) * ok
            self.food.index_add_(0, self.flat(self.arena[sid], px, py), share)
            k = killer[di]; kv = k >= 0
            self.kills.index_add_(0, k[kv], torch.ones_like(k[kv]))
            r.index_add_(0, k[kv], KILL * torch.log(self.score[di][kv].clamp(min=10.0)))   # bigger prey, bigger prize
            info = dict(idx=di, age=self.age[di].clone(), score=self.score[di].clone(), peak=self.peak[di].clone(),
                        edge=(killer[di] < 0), killer=k, kills=self.kills[di].clone())
            self.spawn(di)
        self.last_a = a.clone()
        self._spawn_food()
        self.food.clamp_(max=50.0)
        self.rebuild()
        return r, dead, info

    # ---------------------------------------------------------------- observation
    @torch.no_grad()
    def observe(self, chunk=4096):
        N, dev = self.N, self.dev
        out = torch.empty(N, NCH, OBS * OBS, dtype=torch.bfloat16, device=dev)
        h = self.G * CELL / 2
        sc_all = sc_of(sct_of(self.score))
        zoom = zoom_of(sc_all)
        for s in range(0, N, chunk):
            i = torch.arange(s, min(N, s + chunk), device=dev)
            c, sn = torch.cos(self.ang[i])[:, None], torch.sin(self.ang[i])[:, None]
            z = zoom[i][:, None]
            ch0 = oh = vf = vr = fd = None; edge = None
            for (df, dr) in ((-.25, -.25), (-.25, .25), (.25, -.25), (.25, .25)):     # 4 sub-samples per cell, max-pooled
                of = (self.of[None] + df * CELL) * z; orr = (self.orr[None] + dr * CELL) * z
                x = self.pos[i, :1] + of * c - orr * sn
                y = self.pos[i, 1:] + of * sn + orr * c
                gx, gy, ok = self.cell(x, y)
                fl = self.flat(self.arena[i][:, None], gx, gy)
                own = self.owner[fl].long() - 1
                oth = (own >= 0) & (own != i[:, None]) & ok
                b0 = oth.float(); m0 = ((own == i[:, None]) & ok).float(); z0 = sc_all[own.clamp(min=0)] / 6 * oth
                hd = self.heads[fl].long() - 1
                h1 = ((hd >= 0) & (hd != i[:, None])).float()
                hv = self.vel[hd.clamp(min=0)]
                f1 = (hv[..., 0] * c + hv[..., 1] * sn) / VB * h1; r1 = (-hv[..., 0] * sn + hv[..., 1] * c) / VB * h1
                fo = torch.log1p(self.food[fl]) * ok / 2
                e1 = (((x ** 2 + y ** 2) > self.R ** 2) | ~ok).float()
                if ch0 is None: ch0, oh, vf, vr, fd, edge, mine, sz = b0, h1, f1, r1, fo, e1, m0, z0
                else:
                    take = h1 > oh
                    vf = torch.where(take, f1, vf); vr = torch.where(take, r1, vr)
                    ch0 = torch.maximum(ch0, b0); mine = torch.maximum(mine, m0); sz = torch.maximum(sz, z0); oh = torch.maximum(oh, h1); fd = torch.maximum(fd, fo); edge = torch.maximum(edge, e1)
            out[i, 0] = ch0.bfloat16(); out[i, 1] = oh.bfloat16(); out[i, 2] = vf.bfloat16(); out[i, 3] = vr.bfloat16()
            out[i, 4] = fd.bfloat16(); out[i, 5] = edge.bfloat16(); out[i, 6] = mine.bfloat16(); out[i, 7] = sz.bfloat16()
        d = self.pos.norm(dim=1).clamp(min=1e-3)
        bear = torch.atan2(-self.pos[:, 1], -self.pos[:, 0]) - self.ang
        tl = (self.last_a % NB).float(); sca = torch.stack([
            self.boost.float(), torch.where(self.boost, VB, V0) / VB, torch.log(self.score) / 10, (self.R - d) / self.R,
            torch.sin(bear), torch.cos(bear), (tl - (NB - 1) / 2) / ((NB - 1) / 2), (self.last_a // NB).float()], 1)
        return out.view(N, NCH, OBS, OBS), sca

    # ---------------------------------------------------------------- picture of one arena for the dashboard
    @torch.no_grad()
    def render(self, arena, model_of=None):
        G2 = self.G * self.G; sl = slice(arena * G2, (arena + 1) * G2)
        own = self.owner[sl].long() - 1; hd = self.heads[sl].long() - 1; fd = self.food[sl]
        img = torch.zeros(G2, 3, device=self.dev)
        img[self.inside] = torch.tensor([14., 18., 26.], device=self.dev)
        img += (torch.log1p(fd) / 3).clamp(max=1)[:, None] * torch.tensor([60., 50., 90.], device=self.dev)
        if model_of is not None:
            mo = model_of[own.clamp(min=0)]
            col = torch.where((mo == 0)[:, None], torch.tensor([60., 200., 120.], device=self.dev), torch.tensor([230., 150., 60.], device=self.dev))
        else: col = torch.tensor([60., 200., 120.], device=self.dev).expand(G2, 3)
        img = torch.where((own >= 0)[:, None], col, img)
        img = torch.where((hd >= 0)[:, None], torch.tensor([255., 255., 255.], device=self.dev), img)
        return img.clamp(0, 255).byte().view(self.G, self.G, 3).cpu().numpy()
