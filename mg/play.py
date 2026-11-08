"""Live play against a trained world model.

python play.py --run s150 --port 8771      (on the node; tunnel with ssh -L 8771:localhost:8771 <node>)
Arrows or A/D turn, Space or W boosts, R restarts from a real validation context.
"""
import argparse, base64, io, json, random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import numpy as np, torch
from PIL import Image
import common as C, tokenizer as T, dynamics as M

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Slither World Model</title><style>
body{background:#0f141b;color:#e3e8ef;font:14px system-ui;display:grid;place-items:center;min-height:100vh;margin:0}
img{width:min(90vw,512px);image-rendering:pixelated;border:1px solid #253042}
.r{display:flex;gap:16px;align-items:center;margin-top:10px;flex-wrap:wrap}</style></head><body><div>
<img id="v"><div class="r"><span id="s">press a key to start</span>
<label>cfg <input id="cfg" type="range" min="1" max="4" step="0.25" value="1.5"><b id="cv">1.5</b></label>
<label>temp <input id="tp" type="range" min="0.5" max="1.2" step="0.05" value="0.9"><b id="tv">0.9</b></label></div>
<div class="r" style="color:#8a96a8">arrows / A-D turn, space / W boost, R reset</div></div><script>
const k={};let busy=false,n=0,t0=performance.now();
onkeydown=e=>{k[e.key.toLowerCase()]=1;if(e.key=='r'||e.key=='R')reset();if(e.key==' ')e.preventDefault()};
onkeyup=e=>{k[e.key.toLowerCase()]=0};
for(const [i,o] of [['cfg','cv'],['tp','tv']])document.getElementById(i).oninput=e=>document.getElementById(o).textContent=e.target.value;
async function reset(){const r=await (await fetch('/reset')).json();v.src=r.img}
async function loop(){const turn=(k.arrowleft||k.a?1:0)-(k.arrowright||k.d?1:0),boost=!!(k[' ']||k.w||k.arrowup);
 const r=await (await fetch('/step',{method:'POST',body:JSON.stringify({turn,boost,cfg:+cfg.value,temp:+tp.value})})).json();
 v.src=r.img;n++;s.textContent=`${(n*1000/(performance.now()-t0)).toFixed(1)} fps · ${r.ms} ms/frame · turn ${turn>0?'left':turn<0?'right':'straight'}${boost?' · boost':''}`;
 requestAnimationFrame(loop)}
reset().then(loop);
</script></body></html>"""


class Game:
    def __init__(s, run):
        ck = torch.load(f"{C.ROOT}/runs/{run}/ckpt.pt", map_location="cuda")
        s.model = M.Dyn(ck["cfg"]["size"], policy=ck["cfg"].get("pol_w", 0) > 0).cuda(); s.model.load_state_dict(ck["model"]); s.model = torch.compile(s.model.eval())
        s.tokz = T.load(); s.D = M.Data(); s.lock = threading.Lock(); s.reset()

    def reset(s):
        i = random.choice(list(s.D.clip_start))
        tk, ac = s.D.window(np.array([i]), M.CTX); s.tok, s.act = tk.cuda(), ac.cuda()
        return s.png(s.tok[0, -1])

    def png(s, t):
        x = s.tokz.decode(t.view(1, 16, 16))[0].permute(1, 2, 0).mul(255).byte().cpu().numpy()
        b = io.BytesIO(); Image.fromarray(x).save(b, "PNG"); return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

    def step(s, turn, boost, cfg, temp):
        tb = C.NB - 2 if turn > 0 else 1 if turn < 0 else C.NB // 2
        a = torch.tensor([[tb, int(boost)]], device="cuda")
        nxt = M.generate(s.model, s.tok, s.act, a, cfg=cfg, temp=temp)
        s.tok = torch.cat([s.tok[:, 1:], nxt[:, None]], 1); s.act = torch.cat([s.act[:, 1:], a[:, None]], 1)
        return s.png(nxt[0])


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", default="s150"); p.add_argument("--port", type=int, default=8771); a = p.parse_args()
    g = Game(a.run)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x): pass
        def send(self, body, ctype="application/json"):
            b = body.encode(); self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            if self.path == "/reset":
                with g.lock: self.send(json.dumps(dict(img=g.reset())))
            else: self.send(PAGE, "text/html")
        def do_POST(self):
            r = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            import time; t = time.time()
            with g.lock: img = g.step(r["turn"], r["boost"], r.get("cfg", 1.5), r.get("temp", 0.9))
            self.send(json.dumps(dict(img=img, ms=int((time.time() - t) * 1000))))
    print(f"play at http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
