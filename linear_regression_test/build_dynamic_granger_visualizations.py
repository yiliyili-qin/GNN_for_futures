"""Build the offline timeline significance board for dynamic Granger results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_granger_economic_graph import FUTURES_NODES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(__file__).resolve().parent / "results" / "dynamic_pairwise_granger" / "dynamic_pairwise_granger_results.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "graph_data"
ECHARTS_SOURCE = "echarts.min.js"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--suffix", default="", help="Optional output filename suffix, such as _smoke.")
    return parser.parse_args()


def node_metadata() -> dict[str, dict[str, str]]:
    records = {}
    for line in FUTURES_NODES.splitlines():
        code, name, industry, exchange = line.split("|")
        records[code] = {"code": code, "name": name, "industry": industry, "exchange": exchange}
    return records


def significance_level(value: object) -> int:
    return {"*": 1, "**": 2, "***": 3}.get(value, 0)


def build_payload(results: pd.DataFrame) -> dict[str, object]:
    metadata = node_metadata()
    symbols = [symbol for symbol in metadata if symbol in set(results["explanatory_futures"])]
    dates = sorted(results["evaluation_date"].astype(str).unique())
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    daily_matrix: list[list[list[int]]] = []
    daily_counts = []
    for date in dates:
        daily = results.loc[results["evaluation_date"].astype(str).eq(date)]
        levels = []
        counts = [0, 0, 0, 0]
        for row in daily.itertuples(index=False):
            level = significance_level(row.significance_stars)
            counts[level] += 1
            levels.append([symbol_index[row.explanatory_futures], symbol_index[row.dependent_futures], level])
        daily_matrix.append(levels)
        daily_counts.append(counts)
    return {
        "symbols": symbols, "dates": dates, "matrix": daily_matrix, "counts": daily_counts,
    }


COMMON_STYLE = """
:root{--ink:#18201c;--muted:#65716a;--paper:#eef2ed;--panel:rgba(255,255,255,.94);--line:#cbd4cc;--accent:#255f4b}*{box-sizing:border-box}html,body{width:100%;height:100%;margin:0;overflow:hidden;font-family:"Microsoft YaHei","Noto Sans SC",sans-serif;color:var(--ink)}body{background:linear-gradient(rgba(30,53,41,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(30,53,41,.04) 1px,transparent 1px),var(--paper);background-size:24px 24px}.header{position:fixed;z-index:10;top:14px;left:18px;pointer-events:none}.header h1{margin:0;font-family:"Microsoft JhengHei","Noto Serif SC",serif;font-size:22px;letter-spacing:0}.subtitle{margin-top:5px;color:var(--muted);font-size:12px}.timeline{position:fixed;z-index:20;left:50%;bottom:14px;display:grid;grid-template-columns:36px minmax(260px,52vw) 100px 58px;align-items:center;gap:9px;transform:translateX(-50%);padding:9px 11px;border:1px solid var(--line);border-radius:6px;background:var(--panel);box-shadow:0 6px 22px rgba(24,32,28,.12)}button{height:30px;border:1px solid #bfcac2;border-radius:4px;background:#fff;color:var(--ink);cursor:pointer}button:hover{background:#edf3ee}.date{font-variant-numeric:tabular-nums;font-size:12px;font-weight:700}.index{color:var(--muted);font-size:11px;text-align:right}input[type=range]{width:100%;accent-color:var(--accent)}@media(max-width:720px){.timeline{grid-template-columns:34px minmax(120px,1fr) 82px;left:10px;right:10px;transform:none}.index{display:none}.header h1{font-size:17px}}
"""


def timeline_markup() -> str:
    return """<div class="timeline"><button id="play" type="button" title="播放">▶</button><input id="range" type="range" min="0" value="0" step="1"><span id="date" class="date"></span><span id="index" class="index"></span></div>"""


def heatmap_html(payload: dict[str, object]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>动态 Granger 显著性棋盘图</title><style>{COMMON_STYLE}#chart{{width:100%;height:calc(100% - 62px);margin-top:58px}}.legend{{position:fixed;z-index:12;right:16px;top:14px;padding:8px 10px;border:1px solid var(--line);border-radius:5px;background:var(--panel);font-size:11px}}.swatch{{display:inline-block;width:11px;height:11px;margin:0 4px 0 10px;vertical-align:-2px}}</style></head><body><div class="header"><h1>动态 Granger 显著性棋盘图</h1><div id="subtitle" class="subtitle">列 → 行：解释品种 → 被解释品种</div></div><div class="legend">不显著<span class="swatch" style="background:#fff;border:1px solid #ccd3ce"></span><span class="swatch" style="background:#efb3a8"></span>*<span class="swatch" style="background:#cf6656"></span>**<span class="swatch" style="background:#762f2a"></span>***</div><div id="chart"></div>{timeline_markup()}<script src="{ECHARTS_SOURCE}"></script><script>const payload={data};const chart=echarts.init(document.getElementById('chart'));const range=document.getElementById('range'),dateLabel=document.getElementById('date'),indexLabel=document.getElementById('index'),play=document.getElementById('play');range.max=payload.dates.length-1;let timer=null;function render(i){{const counts=payload.counts[i];dateLabel.textContent=payload.dates[i];indexLabel.textContent=`${{i+1}} / ${{payload.dates.length}}`;document.getElementById('subtitle').textContent=`列 → 行：解释品种 → 被解释品种 · 显著边 ${{counts[1]+counts[2]+counts[3]}} / ${{payload.symbols.length*(payload.symbols.length-1)}}`;chart.setOption({{animation:false,grid:{{left:66,right:30,top:20,bottom:64}},tooltip:{{position:'top',formatter:p=>{{const [x,y,level]=p.data;return `<strong>${{payload.symbols[x]}} → ${{payload.symbols[y]}}</strong><br>${{['不显著','*','**','***'][level]}}`}}}},xAxis:{{type:'category',data:payload.symbols,splitArea:{{show:true}},axisLabel:{{fontSize:9,interval:0,rotate:90}}}},yAxis:{{type:'category',data:payload.symbols,splitArea:{{show:true}},axisLabel:{{fontSize:9,interval:0}}}},visualMap:{{show:false,min:0,max:3,inRange:{{color:['#ffffff','#efb3a8','#cf6656','#762f2a']}}}},series:[{{type:'heatmap',data:payload.matrix[i],itemStyle:{{borderColor:'#d8ddd9',borderWidth:.45}},emphasis:{{itemStyle:{{borderColor:'#18201c',borderWidth:1.5}}}}}}]}} ,true)}}range.oninput=()=>render(Number(range.value));play.onclick=()=>{{if(timer){{clearInterval(timer);timer=null;play.textContent='▶';return}}play.textContent='Ⅱ';timer=setInterval(()=>{{range.value=(Number(range.value)+1)%payload.dates.length;render(Number(range.value))}},550)}};window.addEventListener('resize',()=>chart.resize());render(0);</script></body></html>"""


def graph_html(payload: dict[str, object]) -> str:
    compact = {key: payload[key] for key in ("dates", "links", "counts", "nodes", "categories")}
    data = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>动态期货 Granger 因果图</title><style>{COMMON_STYLE}#chart{{width:100%;height:100%}}.panel{{position:fixed;z-index:12;right:16px;top:14px;width:230px;padding:9px 10px;border:1px solid var(--line);border-radius:5px;background:var(--panel);font-size:11px;line-height:1.55}}.panel strong{{font-size:12px}}#reset{{width:100%;margin-top:7px}}@media(max-width:720px){{.panel{{top:66px;width:180px}}}}</style></head><body><div class="header"><h1>动态期货 Granger 因果图</h1><div id="subtitle" class="subtitle"></div></div><div id="chart"></div><aside class="panel"><strong id="mode">全网络</strong><br>黑线粗细对应一至三星；箭头表示 Granger 方向。双击节点只看该品种的动态关系。<button id="reset" type="button">恢复全网络</button></aside>{timeline_markup()}<script src="{ECHARTS_SOURCE}"></script><script>const payload={data};const chart=echarts.init(document.getElementById('chart'));const range=document.getElementById('range'),dateLabel=document.getElementById('date'),indexLabel=document.getElementById('index'),play=document.getElementById('play');range.max=payload.dates.length-1;let timer=null,focus=null;const styles=[null,{{width:.8,type:'dashed',opacity:.42}},{{width:1.25,type:'solid',opacity:.52}},{{width:2.1,type:'solid',opacity:.68}}];function nodes(){{const w=chart.getWidth(),h=chart.getHeight();return payload.nodes.map(n=>({{...n,x:w*(.04+((n.x-10)/86)*.92),y:h*(.1+((n.y-8)/94)*.8),draggable:true,opacity:focus&&(n.code!==focus)?0.38:1}}))}}function render(i){{let raw=payload.links[i];if(focus)raw=raw.filter(e=>e[0]===focus||e[1]===focus);const links=raw.map(e=>({{source:e[0],target:e[1],level:e[2],pValue:e[3],lag:e[4],qValue:e[5],symbol:['none','arrow'],symbolSize:[0,6],lineStyle:{{color:'#17201c',...styles[e[2]],curveness:.06}}}}));const counts=payload.counts[i];dateLabel.textContent=payload.dates[i];indexLabel.textContent=`${{i+1}} / ${{payload.dates.length}}`;document.getElementById('subtitle').textContent=`20 交易日滚动窗口 · 当日显著 ${{counts[1]+counts[2]+counts[3]}} 条 · 当前显示 ${{links.length}} 条`;chart.setOption({{animationDuration:250,tooltip:{{confine:true,formatter:p=>p.dataType==='node'?`<strong>${{p.data.name}}（${{p.data.code}}）</strong><br>${{p.data.industry}}`:`<strong>${{p.data.source}} → ${{p.data.target}}</strong><br>${{'*'.repeat(p.data.level)}} · p=${{Number(p.data.pValue).toPrecision(4)}} · q=${{Number(p.data.qValue).toPrecision(4)}}<br>BIC 阶数 ${{p.data.lag}}`}},series:[{{type:'graph',layout:'none',data:nodes(),links,categories:payload.categories,roam:true,label:{{show:true,formatter:p=>p.data.displayName,position:'inside',fontSize:8,lineHeight:10,fontWeight:600,color:'#fff'}},itemStyle:{{borderColor:'#fff',borderWidth:2}},emphasis:{{focus:'adjacency',scale:1.15,lineStyle:{{opacity:1,width:3}}}}}}]}},true)}}range.oninput=()=>render(Number(range.value));play.onclick=()=>{{if(timer){{clearInterval(timer);timer=null;play.textContent='▶';return}}play.textContent='Ⅱ';timer=setInterval(()=>{{range.value=(Number(range.value)+1)%payload.dates.length;render(Number(range.value))}},550)}};chart.on('dblclick',p=>{{if(p.dataType!=='node')return;focus=p.data.code;document.getElementById('mode').textContent=`聚焦：${{p.data.name}}（${{focus}}）`;render(Number(range.value))}});document.getElementById('reset').onclick=()=>{{focus=null;document.getElementById('mode').textContent='全网络';render(Number(range.value))}};window.addEventListener('resize',()=>render(Number(range.value)));render(0);</script></body></html>"""


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.input, keep_default_na=False)
    required = {"evaluation_date", "explanatory_futures", "dependent_futures", "significance_stars", "p_value", "bic_selected_lag", "fdr_bh_q_value"}
    if missing := required.difference(results.columns):
        raise ValueError(f"Missing required result columns: {sorted(missing)}")
    payload = build_payload(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = args.output_dir / f"dynamic_granger_significance_board{args.suffix}.html"
    heatmap_path.write_text(heatmap_html(payload), encoding="utf-8")
    print(f"Wrote {heatmap_path}")
    print(f"Embedded {len(payload['dates'])} dates for {len(payload['symbols'])} symbols")


if __name__ == "__main__":
    main()