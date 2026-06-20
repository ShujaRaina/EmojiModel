"""Interactive emoji-diffusion server: type emoji, get an emoji reply.

Serves a single-page GUI at / and a POST /reply endpoint that runs the MDLM
conditional sampler (emoji prompt -> emoji reply). Same-origin, so no CORS.

  python scripts/serve_emoji.py --checkpoint <ckpt> --port 8000
Reach it from a laptop with:  ssh -fN -L 8000:localhost:8000 primeintellect
then open http://localhost:8000
"""
import argparse, os, sys, types
from typing import Optional
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
import hydra
import eval_permutation_stability as eps
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

PAGE = """<!DOCTYPE html><html><head><meta charset=utf-8>
<title>Emoji Diffusion</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0c0e14;--panel:#151823;--ink:#e8ecf4;--mut:#9aa3b8;--acc:#7c8cff;--good:#3ddc97;--line:#262b3a}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;display:flex;align-items:center;justify-content:center}
.card{width:min(640px,92vw);background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:2.2rem}
h1{margin:0 0 .2em;font-size:1.7rem;letter-spacing:-.02em}
.sub{color:var(--mut);margin:0 0 1.4em;font-size:.95rem}
.row{display:flex;gap:.6rem}
input{flex:1;font-size:1.8rem;padding:.7rem 1rem;border-radius:12px;border:1px solid var(--line);
background:#0a0c12;color:var(--ink);outline:none}
input:focus{border-color:var(--acc)}
button{font-size:1.05rem;font-weight:600;padding:0 1.4rem;border:0;border-radius:12px;background:var(--acc);
color:#0a0c12;cursor:pointer}button:disabled{opacity:.5;cursor:wait}
.chips{margin:.9rem 0 0;display:flex;flex-wrap:wrap;gap:.4rem}
.chip{background:#0a0c12;border:1px solid var(--line);border-radius:999px;padding:.3rem .8rem;font-size:1.2rem;cursor:pointer}
.out{margin-top:1.6rem;min-height:96px;border-radius:14px;border:1px dashed var(--line);
display:flex;align-items:center;justify-content:center;font-size:3rem;letter-spacing:.1em}
.meta{margin-top:.8rem;color:var(--mut);font-size:.8rem;display:flex;justify-content:space-between;align-items:center}
.meta input[type=range]{flex:0 0 120px}
</style></head><body>
<div class=card>
  <h1>🌀 Emoji Diffusion</h1>
  <p class=sub>Type an emoji message — a masked-diffusion model replies in emoji.</p>
  <div class=row>
    <input id=p placeholder="🎤🎶🌃" autocomplete=off>
    <button id=go>Reply</button>
  </div>
  <div class=chips id=chips></div>
  <div class=out id=out>·</div>
  <div class=meta>
    <span>steps <input id=steps type=range min=8 max=128 value=32 oninput="st.textContent=this.value"></span>
    <span><b id=st>32</b> · <span id=ms></span></span>
  </div>
</div>
<script>
const ex=["🎤🎶🌃","📚💯","☕🚫🌙","🎉🥳","🥶❄️","💘🌹","🌧️☔","🏋️💪","🍕😋","😱"];
const chips=document.getElementById("chips");
ex.forEach(e=>{const c=document.createElement("span");c.className="chip";c.textContent=e;
c.onclick=()=>{p.value=e;go.click()};chips.appendChild(c)});
const p=document.getElementById("p"),go=document.getElementById("go"),out=document.getElementById("out"),
steps=document.getElementById("steps"),ms=document.getElementById("ms");
async function run(){const prompt=p.value.trim();if(!prompt)return;
go.disabled=true;out.textContent="…";const t=performance.now();
try{const r=await fetch("/reply",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({prompt,steps:+steps.value})});const j=await r.json();
out.textContent=j.reply||"∅";ms.textContent=Math.round(performance.now()-t)+" ms";}
catch(e){out.textContent="error";}go.disabled=false;}
go.onclick=run;p.addEventListener("keydown",e=>{if(e.key==="Enter")run()});
p.focus();
</script></body></html>"""

app = FastAPI()
STATE = {}


class Req(BaseModel):
    prompt: str
    steps: int = 32
    cap: Optional[int] = None


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.post("/reply")
def reply(r: Req):
    tok, model, length = STATE["tok"], STATE["model"], STATE["length"]
    emoji = "".join(eps.dataloader.extract_emoji_graphemes(r.prompt))
    if len(emoji) < 1:
        return JSONResponse({"reply": "", "note": "type at least 1 emoji"})
    prefix = eps.prefix_for_prompt(tok, emoji, length)
    with torch.no_grad():
        out = model.restore_model_and_cond_sample(
            [prefix], num_steps=r.steps,
            max_response_tokens=r.cap).cpu().tolist()[0]
    return {"reply": eps.extract_response(tok, out, len(prefix))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-cache", default="/tmp/emoji_mdlm_two_phase_hard1")
    ap.add_argument("--vocab-cache", default="/tmp/emoji_mdlm_two_phase_hard1/atomic_emoji_vocab.json")
    ap.add_argument("--reply-data-file", default="data/emoji_reply/emoji_reply.jsonl")
    ap.add_argument("--challenge-file", default="data/emoji_reply/curated_permutation.jsonl")
    ap.add_argument("--backbone", default="dit")
    ap.add_argument("--parameterization", default="subs")
    ap.add_argument("--steps", type=int, default=32)
    a = ap.parse_args()
    a.checkpoint = os.path.abspath(a.checkpoint)
    a.reply_data_file = os.path.abspath(a.reply_data_file)
    a.challenge_file = os.path.abspath(a.challenge_file)
    with hydra.initialize(version_base=None, config_path="../configs"):
        config = eps.build_config(a, a.checkpoint)
    tok = eps.dataloader.get_tokenizer(config)
    model = eps.diffusion.Diffusion.load_from_checkpoint(
        a.checkpoint, tokenizer=tok, config=config)
    model.to(a.device).eval()
    STATE.update(tok=tok, model=model, length=a.length)
    print("MODEL READY on port", a.port, flush=True)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
