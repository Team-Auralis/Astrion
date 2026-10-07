"""Lightweight live web dashboard for the ORION Pokemon training runs.

Serves a self-refreshing HTML page (Chart.js) plus a JSON feed. History comes
from each run's append-only `training_history.jsonl` when present, and is
otherwise accumulated in memory while this process is alive.

Usage:
    python dashboard_web.py --port 8765
Then open http://127.0.0.1:8765/
"""

import argparse
import glob
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from orion.pokemon.knowledge import map_name

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = "checkpoints/pokemon"
KEYS = [
    "avg_reward",
    "max_badges",
    "best_badges",
    "max_party",
    "max_level",
    "max_map",
    "entropy",
    "global_step",
]

_lock = threading.Lock()
# {run_name: {key: [[rollout, value], ...]}}
_HISTORY = {}


def _update_history(name, entry):
    h = _HISTORY.setdefault(name, {k: [] for k in KEYS})
    rollouts = [p[0] for p in h["avg_reward"]]
    r = entry.get("rollout", 0)
    if r in rollouts:
        # keep the latest values for this rollout
        idx = rollouts.index(r)
        for k in KEYS:
            if k in entry:
                h[k][idx] = [r, entry[k]]
        return
    for k in KEYS:
        h[k].append([r, entry.get(k, 0) or 0])


def _load_history_file(name):
    path = os.path.join(ROOT, BASE, name, "training_history.jsonl")
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    _update_history(name, json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def snapshot():
    runs = {}
    logs = {
        os.path.basename(os.path.dirname(p)): p
        for p in glob.glob(os.path.join(ROOT, BASE, "*", "training_log.json"))
    }
    lives = {
        os.path.basename(os.path.dirname(p))
        for p in glob.glob(os.path.join(ROOT, BASE, "*", "live.png"))
    }
    for name in sorted(set(logs) | set(lives)):
        latest = None
        if name in logs:
            try:
                with open(logs[name]) as f:
                    latest = json.load(f)
            except (json.JSONDecodeError, OSError):
                latest = None
        with _lock:
            if name not in _HISTORY:
                _load_history_file(name)
            if latest:
                _update_history(name, latest)
            series = {k: list(v) for k, v in _HISTORY[name].items()}
        live_age = -1
        try:
            live_age = int(
                time.time()
                - os.path.getmtime(os.path.join(ROOT, BASE, name, "live.png"))
            )
        except OSError:
            pass
        runs[name] = {
            "latest": latest or {},
            "series": series,
            "live_age": live_age,
            "map_name": map_name(latest.get("max_map", 0)) if latest else None,
        }
    return runs


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>ORION Pokemon Training</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 body{background:#11141a;color:#e6e9ef;font-family:system-ui,Segoe UI,Arial;margin:0;padding:16px}
 h1{font-size:18px;margin:0 0 12px}
 .grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
 .card{background:#1a1f29;border:1px solid #262d3a;border-radius:10px;padding:12px}
 .card h2{font-size:13px;margin:0 0 8px;color:#9aa7bd;font-weight:600}
 table{width:100%;border-collapse:collapse;font-size:12px}
 th,td{padding:6px 8px;text-align:left;border-bottom:1px solid #262d3a}
 th{color:#9aa7bd;font-weight:600}
 .muted{color:#7c8aa3}
 canvas{max-height:250px}
</style></head><body>
<h1>ORION Pokemon Training <span class="muted" id="ts"></span></h1>
<div class="grid">
  <div class="card"><h2>Average Reward</h2><canvas id="reward"></canvas></div>
  <div class="card"><h2>Policy Entropy (health)</h2><canvas id="entropy"></canvas></div>
  <div class="card"><h2>Max Badges</h2><canvas id="badges"></canvas></div>
  <div class="card"><h2>Progress (party / lvl / map) — map is the map ID, not level</h2><canvas id="prog"></canvas></div>
  <div class="card" style="grid-column:span 2"><h2>Status</h2><div id="table"></div></div>
</div>
<script>
const COLORS=["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c"];
const charts={};
function mk(id){
  if(charts[id])return charts[id];
  charts[id]=new Chart(document.getElementById(id),{type:"line",data:{datasets:[]},
    options:{animation:false,responsive:true,parsing:false,
      plugins:{legend:{labels:{color:"#cbd5e1",font:{size:10}}}},
      scales:{x:{type:"linear",ticks:{color:"#7c8aa3",maxTicksLimit:8},grid:{color:"#222a36"}},
              y:{ticks:{color:"#7c8aa3"},grid:{color:"#222a36"}}},
      elements:{point:{radius:0}}}});
  return charts[id];
}
function draw(id,names,d,keys){
  const c=mk(id);
  const ds=[];
  names.forEach((n,i)=>{
    keys.forEach((k,ki)=>{
      const pts=(d[n].series[k]||[]).map(p=>({x:p[0],y:p[1]}));
      if(pts.length) ds.push({label:n+(keys.length>1?" ·"+k:""),data:pts,
        borderColor:COLORS[i%COLORS.length],borderDash:ki?[5,4]:[],
        borderWidth:ki?1.5:2,pointRadius:0});
    });
  });
  c.data.datasets=ds;c.update();
}
async function tick(){
  const d=await (await fetch("/data.json")).json();
  document.getElementById("ts").textContent=new Date().toLocaleTimeString();
  const names=Object.keys(d).sort();
  draw("reward",names,d,["avg_reward"]);
  draw("entropy",names,d,["entropy"]);
  draw("badges",names,d,["max_badges"]);
  draw("prog",names,d,["max_party","max_level","max_map"]);
  let rows="<table><tr><th></th><th>Run</th><th>Status</th><th>Rollout</th><th>Steps</th><th>Reward</th><th>Badges</th><th>Party</th><th>Level</th><th>Map</th><th>Entropy</th><th>Live</th></tr>";
  for(const n of names){const L=d[n].latest||{};
    const st=(d[n].live_age>=0&&d[n].live_age<150)?"LIVE":"stopped";
    const stc=st==="LIVE"?"color:#2ecc71;font-weight:700":"color:#e74c3c";
    const mn=(d[n].map_name||(L.max_map??""));
    rows+=`<tr><td><b>${n}</b></td><td>${L.rollout??"-"}</td>
      <td style="${stc}">${st}</td><td>${(L.global_step||0).toLocaleString()}</td>
      <td>${(L.avg_reward??0).toFixed(1)}</td><td>${L.max_badges??0}</td><td>${L.max_party??0}</td>
      <td style="font-weight:700">${L.max_level??0}</td>
      <td>${(L.max_map??0)!=null&&L.max_map!=null?`${mn} (${L.max_map})`:"-"}</td><td>${(L.entropy??0).toFixed(2)}</td>
      <td><img src="/live/${encodeURIComponent(n)}?t=${Date.now()}" style="width:120px;height:108px;background:#000;border-radius:6px;image-rendering:pixelated" alt=""></td></tr>`;}
  rows+="</table>";document.getElementById("table").innerHTML=rows;
}
tick();setInterval(tick,3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def do_GET(self):
        if self.path.startswith("/live/"):
            name = os.path.basename(self.path.split("/")[2].split("?")[0])
            p = os.path.join(ROOT, BASE, name, "live.png")
            try:
                with open(p, "rb") as f:
                    self._send(200, f.read(), "image/png")
            except OSError:
                self._send(404, "no live frame yet", "text/plain")
        elif self.path.startswith("/data.json"):
            self._send(200, json.dumps(snapshot()), "application/json")
        else:
            self._send(200, PAGE, "text/html; charset=utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dashboard: http://127.0.0.1:{args.port}/")
    srv.serve_forever()


if __name__ == "__main__":
    main()
