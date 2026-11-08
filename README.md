# slither agent

Experiments toward a strong slither.io agent, on one 8×H200 node.

| dir | what |
|---|---|
| `harness/` | live slither.io as an RL environment: headless Chromium per game (Playwright), in-page sampler (`slither_env/page.js`), proxies for many concurrent games |
| `mg/` | world-action model ("WAM"): FSQ video tokenizer + MaskGIT dynamics/policy trained on gameplay video |
| `rl/` | live PPO fine-tuning of the video-pretrained policy on 80 real games (v1–v3) |
| `sp/` | massive GPU self-play: a batched slither simulator in PyTorch (`sim.py`, physics measured from the live game) + PPO league training (`train.py`) on up to ~1M concurrent snakes |
| `live/` | the self-play policy in real games: browser state → the simulator's 8-channel egocentric observation (`obs.py`), actors (`live.py`) and a PPO learner (`learn.py`) |

Each folder has `launch.sh` / `stop.sh` / `push.sh` for the cluster node (paths assume `/tmp/slither/...` on the node).
Credentials (Hugging Face token, proxy list, Cloudflare tunnel) are read at runtime from files outside this repo and are not included.
