"""One episode -> one .npz, everything the RL side needs (replay buffer, death/growth heads, world-model fine-tuning).

Per frame (10 Hz, k = all frames of the episode incl. the starting context):
  frames [k,128,128,3] uint8, t [k] ms (page clock), ang [k] rad (screen coords), score [k], alive [k] bool,
  turn_deg [k] heading change from the previous frame, positive = left (dataset convention; bin it with the world model's
  edges to get its action tokens), boost [k] 0/1 (commanded).
Per decision (3 frames each):
  action [n] index into env.ACTIONS, frame_idx [n] index of the last frame seen when deciding, reward [n] (delta log
  length), done [n] (last one True if the snake died).
"""
import math, os
import numpy as np
from .env import ACTIONS


class Episode:
    def __init__(self, sid, n, obs):
        self.sid, self.n = sid, n
        self.samples = [x for x in obs["samples"] if x.get("frame") is not None]; self.boost = [0] * len(self.samples)
        self.start_score = obs["state"]["score"]
        self.action, self.frame_idx, self.reward, self.done = [], [], [], []

    def add(self, a, r, done, info):
        self.frame_idx.append(len(self.samples) - 1); self.action.append(a); self.reward.append(r); self.done.append(done)
        new = [x for x in info["samples"] if x.get("frame") is not None]
        self.samples += new; self.boost += [ACTIONS[a][1]] * len(new)

    def summary(self):
        return dict(session=self.sid, episode=self.n, decisions=len(self.action), start_score=self.start_score,
                    final_score=([s["score"] for s in self.samples if s["alive"]] or [self.start_score])[-1],
                    max_score=max([s["score"] for s in self.samples if s["alive"]] or [self.start_score]),
                    **{"return": float(sum(self.reward))}, died=bool(self.done and self.done[-1]))

    def save(self, out):
        os.makedirs(out, exist_ok=True)
        s = self.samples
        ang = np.array([x["ang"] for x in s], np.float64)
        d = np.diff(ang, prepend=ang[:1]); d = (d + math.pi) % (2 * math.pi) - math.pi
        path = f"{out}/ep_{self.sid:03d}_{self.n:05d}.npz"
        np.savez_compressed(path + ".tmp.npz",
            frames=np.stack([x["frame"] for x in s]), t=np.array([x["t"] for x in s]), ang=ang.astype(np.float32),
            score=np.array([x["score"] for x in s], np.int32), alive=np.array([x["alive"] for x in s]),
            turn_deg=(-np.degrees(d)).astype(np.float32),
            boost=np.array(self.boost, np.uint8),
            action=np.array(self.action, np.int16), frame_idx=np.array(self.frame_idx, np.int32),
            reward=np.array(self.reward, np.float32), done=np.array(self.done), actions_table=np.array(ACTIONS, np.int16))
        os.replace(path + ".tmp.npz", path)
        return path
