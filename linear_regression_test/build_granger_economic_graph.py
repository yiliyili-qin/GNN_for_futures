"""Build an offline ECharts graph from pairwise Granger-causality results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "linear_regression_test" / "results" / "no_smm_pairwise_granger" / "pairwise_granger_results.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "graph_data" / "futures_granger_causality_graph.html"

FUTURES_NODES = """
I|铁矿石|黑色金属|大连商品交易所
JM|焦煤|黑色金属|大连商品交易所
J|焦炭|黑色金属|大连商品交易所
RB|螺纹钢|黑色金属|上海期货交易所
HC|热轧卷板|黑色金属|上海期货交易所
SF|硅铁|黑色金属/铁合金|郑州商品交易所
SM|锰硅|黑色金属/铁合金|郑州商品交易所
SC|原油|能源化工|上海国际能源交易中心
FU|燃料油|能源化工|上海期货交易所
LU|低硫燃料油|能源化工|上海国际能源交易中心
BU|石油沥青|能源化工|上海期货交易所
PG|液化石油气|能源化工|大连商品交易所
PX|对二甲苯|能源化工|郑州商品交易所
TA|精对苯二甲酸（PTA）|能源化工|郑州商品交易所
EG|乙二醇|能源化工|大连商品交易所
PF|涤纶短纤|能源化工|郑州商品交易所
EB|苯乙烯|能源化工|大连商品交易所
L|聚乙烯|能源化工|大连商品交易所
PP|聚丙烯|能源化工|大连商品交易所
V|聚氯乙烯|能源化工|大连商品交易所
MA|甲醇|能源化工|郑州商品交易所
SH|烧碱|能源化工|郑州商品交易所
UR|尿素|能源化工|郑州商品交易所
BR|合成橡胶|能源化工|上海期货交易所
RU|天然橡胶|能源化工|上海期货交易所
NR|20号胶|能源化工|上海国际能源交易中心
AL|铝|有色金属|上海期货交易所
AO|氧化铝|有色金属|广州期货交易所
SI|工业硅|有色金属|广州期货交易所
AU|黄金|贵金属|上海期货交易所
AG|白银|贵金属|上海期货交易所
A|黄大豆1号|农产品|大连商品交易所
B|黄大豆2号|农产品|大连商品交易所
M|豆粕|农产品|大连商品交易所
Y|豆油|农产品|大连商品交易所
P|棕榈油|农产品|大连商品交易所
OI|菜籽油|农产品|郑州商品交易所
RM|菜籽粕|农产品|郑州商品交易所
C|玉米|农产品|大连商品交易所
CS|玉米淀粉|农产品|大连商品交易所
LH|生猪|农产品|大连商品交易所
CF|棉花|农产品/纺织|郑州商品交易所
CY|棉纱|农产品/纺织|郑州商品交易所
PK|花生|农产品|郑州商品交易所
""".strip()

# These coordinates keep the industry groups in the same visual neighborhoods as
# the existing futures economic graph, while making the Granger edges fixed.
NODE_POSITIONS = {
    "I": (10, 18), "JM": (10, 35), "J": (20, 30), "RB": (30, 22), "HC": (30, 38), "SF": (22, 12), "SM": (22, 48),
    "SC": (47, 16), "FU": (58, 8), "LU": (58, 18), "BU": (58, 28), "PG": (58, 39), "PX": (68, 12), "TA": (78, 12),
    "EG": (68, 26), "PF": (86, 18), "EB": (68, 40), "L": (78, 32), "PP": (78, 43), "V": (88, 38), "MA": (58, 52),
    "SH": (48, 58), "UR": (58, 65), "BR": (78, 57), "RU": (88, 57), "NR": (96, 57),
    "AL": (29, 68), "AO": (20, 68), "SI": (11, 65), "AU": (11, 82), "AG": (20, 82),
    "A": (40, 82), "B": (49, 82), "M": (58, 82), "Y": (67, 82), "P": (76, 82), "OI": (85, 82), "RM": (58, 92),
    "C": (40, 92), "CS": (32, 92), "LH": (40, 102), "CF": (86, 94), "CY": (96, 94), "PK": (49, 102),
}

PALETTE = ["#315c49", "#b94e3e", "#247787", "#c58c20", "#745d9c", "#a9537d", "#3e70a8", "#69782c", "#8e5d40", "#536863"]
STAR_LEVELS = {"*": 1, "**": 2, "***": 3}


def wrap_name(name: str) -> str:
    """Insert a compact label break for long commodity names."""
    if len(name) <= 5:
        return name
    if "（" in name:
        return name.replace("（", "\n（")
    midpoint = (len(name) + 1) // 2
    return f"{name[:midpoint]}\n{name[midpoint:]}"


def build_nodes() -> tuple[list[dict[str, object]], dict[str, dict[str, str]]]:
    """Build ECharts nodes and a symbol-to-metadata lookup."""
    records = []
    for line in FUTURES_NODES.splitlines():
        code, name, industry, exchange = line.split("|")
        records.append({"code": code, "name": name, "industry": industry, "exchange": exchange})
    industries = list(dict.fromkeys(record["industry"] for record in records))
    nodes = []
    for record in records:
        x, y = NODE_POSITIONS[record["code"]]
        nodes.append({
            "id": record["code"], "name": record["name"], "code": record["code"],
            "displayName": wrap_name(record["name"]), "industry": record["industry"], "exchange": record["exchange"],
            "category": industries.index(record["industry"]), "x": x, "y": y, "symbol": "circle",
            "symbolSize": 58 if len(record["name"]) > 8 else 52 if len(record["name"]) > 5 else 44,
            "draggable": True,
        })
    return nodes, {record["code"]: record for record in records}


def build_links(results: pd.DataFrame) -> list[dict[str, object]]:
    """Merge significant directional tests into single or bidirectional graph links."""
    significant = results.loc[results["significance_stars"].isin(STAR_LEVELS)].copy()
    significant["pair_key"] = significant.apply(
        lambda row: "__".join(sorted((row["explanatory_futures"], row["dependent_futures"]))), axis=1
    )
    links = []
    for _, pair in significant.groupby("pair_key", sort=True):
        directions = pair.to_dict("records")
        level = max(STAR_LEVELS[row["significance_stars"]] for row in directions)
        first = directions[0]
        bidirectional = len(directions) == 2
        if bidirectional:
            source, target = sorted((first["explanatory_futures"], first["dependent_futures"]))
        else:
            source, target = first["explanatory_futures"], first["dependent_futures"]
        links.append({
            "source": source,
            "target": target,
            "bidirectional": bidirectional,
            "level": level,
            "directions": [{
                "source": row["explanatory_futures"], "target": row["dependent_futures"],
                "stars": row["significance_stars"], "pValue": row["p_value"], "lag": int(row["bic_selected_lag"]),
            } for row in directions],
        })
    return links


def write_html(nodes: list[dict[str, object]], links: list[dict[str, object]], metadata: dict[str, dict[str, str]]) -> None:
    """Write the self-contained interactive Granger graph HTML."""
    industries = list(dict.fromkeys(item["industry"] for item in metadata.values()))
    categories = [{"name": industry, "itemStyle": {"color": PALETTE[index % len(PALETTE)]}} for index, industry in enumerate(industries)]
    template = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>期货 Granger 因果图</title>
<style>
:root{color-scheme:light;--ink:#18201c;--muted:#607068;--paper:#edf2ed;--panel:rgba(255,255,255,.95);--line:#d0d9d2}*{box-sizing:border-box}html,body,#chart{width:100%;height:100%;margin:0;overflow:hidden;font-family:"Microsoft YaHei","Noto Sans SC",sans-serif;color:var(--ink)}body{background:linear-gradient(rgba(30,53,41,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(30,53,41,.045) 1px,transparent 1px),var(--paper);background-size:24px 24px}.title-block{position:fixed;z-index:10;top:18px;left:22px;pointer-events:none}h1{margin:0;font-family:"Microsoft JhengHei","Noto Serif SC",serif;font-size:24px;letter-spacing:0}.subtitle{margin-top:6px;color:var(--muted);font-size:12px}.control-panel{position:fixed;z-index:10;top:16px;right:16px;width:250px;padding:11px;border:1px solid var(--line);border-radius:6px;background:var(--panel);box-shadow:0 7px 22px rgba(24,33,29,.1);backdrop-filter:blur(10px)}.panel-title{display:flex;justify-content:space-between;align-items:center;margin-bottom:9px;font-size:13px;font-weight:700}.panel-count{color:var(--muted);font-size:11px;font-weight:400}.legend{display:grid;gap:6px}.legend-row{display:grid;grid-template-columns:30px 1fr;align-items:center;gap:8px;font-size:12px}.edge-sample{height:0;border-top:1.4px solid #1d2522}.edge-sample.strong{border-top-width:2.6px}.edge-sample.dashed{border-top-style:dashed}.legend-note{margin-top:9px;padding-top:8px;border-top:1px solid var(--line);color:var(--muted);font-size:11px;line-height:1.5}.panel-actions{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:9px}.panel-actions button,.legend-row button{padding:6px;border:1px solid #c7d1c9;border-radius:4px;color:#34413a;background:#fff;cursor:pointer}.legend-row button{display:grid;grid-template-columns:30px 1fr auto;align-items:center;gap:8px;width:100%;padding:4px 5px;text-align:left;font-size:12px}.legend-row button.off{opacity:.35}.legend-row button:hover,.panel-actions button:hover{background:#f0f4f0}.legend-count{color:var(--muted);font-size:11px}.hint{position:fixed;z-index:10;left:22px;bottom:18px;color:var(--muted);font-size:12px;pointer-events:none}@media(max-width:760px){.title-block{top:12px;left:14px;width:calc(100vw - 190px)}h1{font-size:17px;line-height:1.35}.control-panel{top:10px;right:10px;width:174px;padding:8px}.hint{display:none}}
</style></head><body>
<div class="title-block"><h1>期货 Granger 因果图</h1><div class="subtitle">15 分钟收益率 · BIC 选滞后阶数 · 仅显示 $EDGE_COUNT 条显著关系</div></div><div id="chart" role="img" aria-label="期货品种 Granger 因果关系图"></div>
<aside class="control-panel"><div class="panel-title">显著性与方向 <span class="panel-count">44 节点 · $EDGE_COUNT 边</span></div><div id="legend" class="legend"></div><div class="legend-note">单箭头 A → B：A 是 B 的 Granger 原因。双箭头 A ↔ B：两个方向均显著。双向边按更高星级显示。</div><div class="panel-actions"><button id="show-all" type="button">显示全部</button><button id="three-star" type="button">仅看三星</button></div></aside><div class="hint">滚轮缩放 · 拖动画布或节点 · 点击节点聚焦邻接关系 · 悬停查看检验详情</div>
<script src="echarts.min.js"></script><script>
const nodes=$NODES;const links=$LINKS;const categories=$CATEGORIES;const styles={1:{width:.9,type:'dashed',color:'#1d2522'},2:{width:1.4,type:'solid',color:'#1d2522'},3:{width:2.6,type:'solid',color:'#1d2522'}};const activeLevels=new Set([1,2,3]);const chart=echarts.init(document.getElementById('chart'));
function positionedNodes(){const width=chart.getWidth(),height=chart.getHeight();return nodes.map(node=>({...node,x:width*(.04+((node.x-10)/86)*.92),y:height*(.1+((node.y-8)/94)*.8)}))}function makeLink(link){const style=styles[link.level];return {...link,symbol:link.bidirectional?['arrow','arrow']:['none','arrow'],symbolSize:link.bidirectional?[6,6]:[0,6],lineStyle:{...style,opacity:link.level===3?.64:.48,curveness:link.bidirectional?.11:.06}}}function tooltip(params){if(params.dataType==='node'){const node=params.data;return `<strong>${node.name}</strong>（${node.code}）<br>${node.industry}<br>${node.exchange}`}const edge=params.data;const direction=edge.bidirectional?'↔':'→';const details=edge.directions.map(item=>`${item.source} → ${item.target}：${item.stars}，p=${Number(item.pValue).toPrecision(4)}，BIC 阶数 ${item.lag}`).join('<br>');return `<strong>${edge.source} ${direction} ${edge.target}</strong><br>${details}`}function render(){chart.setOption({animationDuration:700,animationEasingUpdate:'quinticInOut',tooltip:{confine:true,backgroundColor:'rgba(255,255,255,.98)',borderColor:'#c7d1c9',textStyle:{color:'#18201c',fontFamily:'Microsoft YaHei',fontSize:12},formatter:tooltip},series:[{type:'graph',layout:'none',data:positionedNodes(),links:links.filter(link=>activeLevels.has(link.level)).map(makeLink),categories,roam:true,zoom:1,center:['50%','50%'],label:{show:true,formatter:params=>params.data.displayName,position:'inside',fontSize:8,fontWeight:600,lineHeight:11,color:'#fff'},itemStyle:{borderColor:'rgba(255,255,255,.96)',borderWidth:2,shadowColor:'rgba(24,33,29,.14)',shadowBlur:7},emphasis:{focus:'adjacency',scale:1.18,label:{fontSize:10},lineStyle:{width:3.2,opacity:1}},lineStyle:{opacity:.56}}]},{notMerge:true})}const legend=document.getElementById('legend');[3,2,1].forEach(level=>{const count=links.filter(link=>link.level===level).length;const button=document.createElement('button');button.type='button';button.dataset.level=level;button.innerHTML=`<span class="edge-sample ${level===3?'strong':''} ${level===1?'dashed':''}"></span><span>${'*'.repeat(level)} ${level===3?'三星':level===2?'二星':'一星'}</span><span class="legend-count">${count}</span>`;button.addEventListener('click',()=>{activeLevels.has(level)?activeLevels.delete(level):activeLevels.add(level);button.classList.toggle('off',!activeLevels.has(level));render()});legend.appendChild(button)});document.getElementById('show-all').addEventListener('click',()=>{[1,2,3].forEach(level=>activeLevels.add(level));document.querySelectorAll('[data-level]').forEach(button=>button.classList.remove('off'));render()});document.getElementById('three-star').addEventListener('click',()=>{activeLevels.clear();activeLevels.add(3);document.querySelectorAll('[data-level]').forEach(button=>button.classList.toggle('off',Number(button.dataset.level)!==3));render()});render();window.addEventListener('resize',()=>{chart.resize();render()});
</script></body></html>"""
    html = (template.replace("$NODES", json.dumps(nodes, ensure_ascii=False))
            .replace("$LINKS", json.dumps(links, ensure_ascii=False))
            .replace("$CATEGORIES", json.dumps(categories, ensure_ascii=False))
            .replace("$EDGE_COUNT", str(len(links))))
    OUTPUT_PATH.write_text(html, encoding="utf-8")


def main() -> None:
    """Build the graph from the synchronized no-SMM Granger experiment output."""
    results = pd.read_csv(RESULTS_PATH)
    nodes, metadata = build_nodes()
    links = build_links(results)
    write_html(nodes, links, metadata)
    directional_count = int(results["significance_stars"].isin(STAR_LEVELS).sum())
    bidirectional_count = sum(link["bidirectional"] for link in links)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Significant directional tests: {directional_count}")
    print(f"Displayed links: {len(links)} ({bidirectional_count} bidirectional)")


if __name__ == "__main__":
    main()