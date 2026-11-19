"""Stage 2a — manual sort of DermNet folder 11 ("Melanoma Skin Cancer Nevi and Moles").

Serves a tiny local web app on http://127.0.0.1:8765 that shows one image at a time and
records a keyboard label:

    M  melanoma        N  nevus / mole (not melanoma)        U or ?  unsure (skip)
    <- previous        -> next

Labels are appended to data/dermnet_folder11_labels.csv as they are pressed, so the
session can be closed and resumed at any time. The annotator is a layperson; the CSV is
marked as such and every label is later cross-checked against Model B (see plan §3).

Run:  python scripts/02a_sort_folder11.py      (opens the browser automatically)
Stop: Ctrl+C in the terminal.
"""
from __future__ import annotations

import csv
import json
import mimetypes
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
FOLDER = "Melanoma Skin Cancer Nevi and Moles"
IMG_DIRS = [ROOT / "data" / "raw" / "dermnet" / s / FOLDER for s in ("train", "test")]
OUT_CSV = ROOT / "data" / "dermnet_folder11_labels.csv"
HOST, PORT = "127.0.0.1", 8765

LABELS = {"M": "melanoma", "N": "nevus", "U": "unsure"}

items: list[Path] = sorted(p for d in IMG_DIRS for p in d.glob("*.jpg"))
labels: dict[str, str] = {}
lock = threading.Lock()


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def load_existing() -> None:
    if not OUT_CSV.exists():
        return
    with OUT_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            labels[r["path"]] = r["label"]  # last write wins


def append_label(path: str, label: str) -> None:
    new = not OUT_CSV.exists()
    with lock:
        labels[path] = label
        with OUT_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["path", "label", "annotator", "ts"])
            w.writerow([path, label, "lay-user", time.strftime("%Y-%m-%dT%H:%M:%S")])


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Folder 11 sort</title>
<style>
 body{margin:0;background:#111;color:#eee;font:15px/1.4 system-ui,sans-serif;display:flex;height:100vh}
 #left{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:16px;gap:10px}
 #img{max-width:100%;max-height:78vh;border-radius:6px;background:#000}
 #right{width:330px;padding:18px;background:#1b1b1b;overflow:auto}
 h2{margin:0 0 8px;font-size:17px} h3{margin:16px 0 6px;font-size:14px;color:#9cf}
 .key{display:inline-block;min-width:26px;padding:2px 7px;border:1px solid #666;border-radius:4px;background:#222;font-weight:600;text-align:center}
 .row{margin:6px 0} .mel{color:#f66} .nev{color:#6d6} .uns{color:#fc6}
 #bar{height:8px;background:#333;border-radius:4px;overflow:hidden;margin:8px 0}
 #fill{height:100%;background:#4a8;width:0}
 #status{font-size:13px;color:#aaa} #cur{font-weight:700;font-size:18px}
 ul{padding-left:18px;margin:4px 0} li{margin:3px 0}
 small{color:#999}
</style></head><body>
<div id="left">
  <img id="img" alt="">
  <div id="cur"></div>
  <div id="status"></div>
</div>
<div id="right">
  <h2>DermNet folder 11 — hate sort</h2>
  <div id="bar"><div id="fill"></div></div>
  <div id="counts"></div>
  <h3>Keys</h3>
  <div class="row"><span class="key">M</span> <span class="mel">melanoma</span> — sure hole i</div>
  <div class="row"><span class="key">N</span> <span class="nev">nevus / til / mole</span> — cancer na</div>
  <div class="row"><span class="key">U</span> or <span class="key">?</span> <span class="uns">unsure</span> — sure na, skip</div>
  <div class="row"><span class="key">&larr;</span> agher chobi &nbsp; <span class="key">&rarr;</span> porer chobi</div>
  <h3>Rule 1</h3>
  <div><b>Sure na hole U.</b> Bhul label deya theke skip onek valo. Unsure chobi Model B te jabe na.</div>
  <h3>Melanoma sign — ABCDE</h3>
  <ul>
   <li><b>A</b>symmetry — ek pash arek pash er moto na</li>
   <li><b>B</b>order — kinara vanga, jhapsa, notched</li>
   <li><b>C</b>olor — ek chobir moddhe onek rong: kalo, badami, lal, sada, nil</li>
   <li><b>D</b>iameter — boro (&gt;6 mm, pencil eraser er beshi)</li>
   <li><b>E</b>volving / ugly duckling — pasher til gular theke alada dekhte</li>
  </ul>
  <h3>Shadharon til (nevus)</h3>
  <ul>
   <li>Gol/oval, symmetric</li>
   <li>Ek rong, even badami</li>
   <li>Smooth kinara</li>
   <li>Choto</li>
  </ul>
  <small>Label auto-save hoy <code>data/dermnet_folder11_labels.csv</code> e. Bondho kore abar chalale jekhane chile sekhane ashbe. Pore Model B er shathe cross-check hobe.</small>
</div>
<script>
let items=[], idx=0;
const img=document.getElementById('img');
function firstUnlabeled(){const i=items.findIndex(x=>!x.label);return i<0?0:i;}
function render(){
  const it=items[idx]; if(!it) return;
  img.src='/img/'+idx;
  const done=items.filter(x=>x.label).length;
  const m=items.filter(x=>x.label==='melanoma').length, n=items.filter(x=>x.label==='nevus').length, u=items.filter(x=>x.label==='unsure').length;
  document.getElementById('cur').innerHTML=(idx+1)+' / '+items.length+(it.label?' &nbsp;<span class="'+(it.label==='melanoma'?'mel':it.label==='nevus'?'nev':'uns')+'">['+it.label+']</span>':'');
  document.getElementById('status').textContent=it.path;
  document.getElementById('fill').style.width=(100*done/items.length)+'%';
  document.getElementById('counts').innerHTML=done+' / '+items.length+' done &nbsp;·&nbsp; <span class="mel">M '+m+'</span> &nbsp;<span class="nev">N '+n+'</span> &nbsp;<span class="uns">U '+u+'</span>';
}
async function label(l){
  const it=items[idx];
  await fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({idx:idx,label:l})});
  it.label=l;
  // advance to next unlabeled after current, else next
  let j=items.findIndex((x,i)=>i>idx && !x.label);
  idx = j<0 ? Math.min(idx+1, items.length-1) : j;
  render();
}
document.addEventListener('keydown',e=>{
  const k=e.key;
  if(k==='m'||k==='M') label('melanoma');
  else if(k==='n'||k==='N') label('nevus');
  else if(k==='u'||k==='U'||k==='?') label('unsure');
  else if(k==='ArrowLeft'){idx=Math.max(0,idx-1);render();}
  else if(k==='ArrowRight'){idx=Math.min(items.length-1,idx+1);render();}
});
fetch('/state').then(r=>r.json()).then(s=>{items=s.items; idx=firstUnlabeled(); render();
  // preload next few
  for(let k=1;k<=3;k++){ if(items[idx+k]) { const p=new Image(); p.src='/img/'+(idx+k);} }
});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/state":
            data = {"items": [{"path": rel(p), "label": labels.get(rel(p), "")} for p in items]}
            self._send(200, json.dumps(data).encode(), "application/json")
        elif path.startswith("/img/"):
            try:
                p = items[int(path[5:])]
                self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "image/jpeg")
            except (ValueError, IndexError):
                self._send(404, b"no", "text/plain")
        else:
            self._send(404, b"no", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/label":
            return self._send(404, b"no", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        try:
            p = items[int(body["idx"])]
            lab = body["label"]
            if lab not in LABELS.values():
                raise ValueError(lab)
        except (KeyError, ValueError, IndexError) as e:
            return self._send(400, str(e).encode(), "text/plain")
        append_label(rel(p), lab)
        self._send(200, b"ok", "text/plain")


def main() -> int:
    if not items:
        print(f"no images found under {IMG_DIRS}", file=sys.stderr)
        return 1
    load_existing()
    done = sum(1 for p in items if rel(p) in labels)
    print(f"{len(items)} images, {done} already labelled -> {OUT_CSV.relative_to(ROOT)}")
    url = f"http://{HOST}:{PORT}/"
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"serving {url}  (Ctrl+C to stop)", flush=True)
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
