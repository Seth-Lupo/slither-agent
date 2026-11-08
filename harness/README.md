# Slither.io RL harness

Live slither.io games in headless Chromium, as an RL environment for the Dyna-Q agent. One Chromium per session; any
number of sessions run concurrently, each in real time; one batched policy decides for all of them; every episode is
saved for the replay buffer.

## Interface (matches the world model)
| | |
|---|---|
| Observation | 128x128 RGB, center-square crop of the game canvas, sampled at **10 Hz** in the page |
| Decision | every **3 frames (0.3 s)**; the policy sees the last 8 frames |
| Actions | 14 = 7 heading changes (-90, -45, -20, 0, 20, 45, 90 deg; **positive = left**, the dataset convention) x boost (0/1). `env.ACTIONS` |
| Reward | change in log(length) over the decision (length = the game's score); return = log(final / start length) |
| Done | the snake died (read from the game's own state, exact) |

Steering holds an absolute target heading (current heading + turn) for the whole decision, via the game's mouse target.
Boost uses the game's own `setAcceleration`.

## Files
- `slither_env/page.js`: injected into the page; reads `window.slither` (our snake), score tables, frames; steers.
- `slither_env/env.py`: `SlitherEnv`: `reset()`, `step(a)` (async).
- `slither_env/runner.py`: `run(policy, n_sessions, out)`: one actor per session, batched policy
  (`policy(batch) -> actions`, runs in a thread so it can be a GPU model), auto-restart after death or page crash.
- `slither_env/recorder.py`: one `.npz` per episode (format in its docstring): all 10 Hz frames, score, alive, heading,
  per-frame `turn_deg` (bin with the world model's edges to get its action tokens), per-decision action/reward/done.
- `scripts/smoke.py`: one session, scripted turns: checks turn sign, boost, reward, death; writes a gif.
- `scripts/run.py`: N sessions with a random policy, recording episodes (`--sessions --seconds --render_fps --out`).
- `scripts/setup_node.sh`, `push.sh`: put it on the cluster node (everything under /tmp).

## Run
```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/playwright install chromium
.venv/bin/python scripts/smoke.py
.venv/bin/python scripts/run.py --sessions 8 --seconds 120 --render_fps 20 --out episodes/
./push.sh            # cluster: copies to the job's node, installs, checks slither.io is reachable
```

## Verified (Mac, 2026-11-07)
- Game start, steering (commanded +90 left -> +46..82 deg per 0.6 s; -90 -> -62..-72), boost (burns length), score,
  death detection, auto-restart, episode files (3.0 frames per decision, frames ~100 ms apart).
- One Chromium per session is required: several pages in one browser throttle to ~1 Hz.

## Open items
1. **CPU per session**: software WebGL renders a fixed 1500x845 canvas; ~1.6 cores/session at 4 sessions, less with
   `--render_fps 20`. Not measurable on the Mac (shared load); benchmark on the node: 8, 32, 64 sessions, watch
   "mean lateness" in the log.
2. **Zoom calibration**: the game zooms out as the snake grows; compare hex size / snake width against training frames
   of similar-size snakes and set `crop` (fraction of the short side) so the views match.
3. **Many sessions from one IP**: slither.io may limit connections; ramp up (`--stagger`) and watch for failed joins.
4. **Network from compute nodes**: `setup_node.sh` checks slither.io is reachable.
