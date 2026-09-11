# 加入 SMM 节点后的经济图谱完整边清单

## 1. 用途与建图口径

本文件是下一版经济图谱的建图依据，覆盖当前期货图中的全部逻辑边，以及加入 SMM 实体节点后新增的边。它取代旧版“边变更审计”，不再只是列出争议边。

- 每条逻辑边只写一次。`A -> B` 为有向边，`A <-> B` 为无向边。
- 保留直接产业链、间接产业链、真实能源投入、同级商品、品级/产地和市场联动关系，但必须使用不同的 `relation_type`。
- 当前“期货原始文字关系”已经支持的关系属于既有边，按要求不再逐条重复来源。
- 至少一端为新增 SMM 节点的边属于新增边，必须列出能够证明生产路径或指标归属的来源。
- “跨级产业边”表示产业路径真实存在，但至少一个中间实体没有可用数据，因而直接连接已收集数据的上游和下游；不得和一跳直接投入使用同一种关系类型。
- 若已收集到全部中间节点的可用数据，并能形成完整的 `A -> B -> ... -> C` 路径，则只保留逐级边，不建立 `A -> C` 跨级边。
- “有节点”不等于“有可用数据”：节点特征语义错误、为空或尚未通过质量检查时，仍视为中间数据缺失，可以暂时保留跨级边。
- SMM 指标是节点特征，不因某个指标同时提到两个商品就自动产生商品边。

### 关系类型

| `relation_type` | 方向 | 含义 |
| --- | --- | --- |
| `direct_input` | 有向 | 上游商品直接作为下游生产投入，或直接加工得到下游商品 |
| `indirect_chain` | 有向 | 产业路径成立，但至少一个中间节点没有可用数据 |
| `energy_input` | 有向 | 电力、燃料等生产能源投入 |
| `peer_product` | 无向 | 具有相似商品生态位的同级关系，包括条件替代、联合产出和同源产品；具体机制保留在边的说明字段中 |
| `grade_or_origin` | 无向 | 同一原料体系、品级重叠或不同交割标准 |
| `market_linkage` | 无向 | 宏观、板块、价差或用途路径造成的市场联动，不代表实物流 |

## 2. 重建后保留的既有期货边

本节列出重建后实际保留的既有期货逻辑边。来源均为“期货原始文字关系（免列来源）”。原图中已被完整中间路径替代的 7 条跨级边不在有效边表内，统一记录在第 4 节。表中编号沿用原审计编号，因此有意保留空号，便于追溯原关系。

### 2.1 直接产业链边

| 序号 | 边 | 中文关系 | `relation_type` |
| ---: | --- | --- | --- |
| 1 | `I -> RB` | 铁矿石 -> 螺纹钢 | `direct_input` |
| 2 | `I -> HC` | 铁矿石 -> 热轧卷板 | `direct_input` |
| 3 | `JM -> J` | 焦煤 -> 焦炭 | `direct_input` |
| 4 | `J -> RB` | 焦炭 -> 螺纹钢 | `direct_input` |
| 5 | `J -> HC` | 焦炭 -> 热轧卷板 | `direct_input` |
| 6 | `J -> SM` | 焦炭 -> 锰硅 | `direct_input` |
| 7 | `SF -> RB` | 硅铁 -> 螺纹钢 | `direct_input` |
| 8 | `SF -> HC` | 硅铁 -> 热轧卷板 | `direct_input` |
| 9 | `SM -> RB` | 锰硅 -> 螺纹钢 | `direct_input` |
| 11 | `SC -> FU` | 原油 -> 燃料油 | `direct_input` |
| 12 | `SC -> LU` | 原油 -> 低硫燃料油 | `direct_input` |
| 13 | `SC -> BU` | 原油 -> 石油沥青 | `direct_input` |
| 14 | `SC -> PG` | 原油 -> 液化石油气 | `direct_input` |
| 15 | `PX -> TA` | 对二甲苯 -> PTA | `direct_input` |
| 16 | `TA -> PF` | PTA -> 涤纶短纤 | `direct_input` |
| 17 | `EG -> PF` | 乙二醇 -> 涤纶短纤 | `direct_input` |
| 18 | `SH -> AO` | 烧碱 -> 氧化铝 | `direct_input` |
| 19 | `AO -> AL` | 氧化铝 -> 铝 | `direct_input` |
| 20 | `UR -> C` | 尿素 -> 玉米 | `direct_input` |
| 23 | `B -> M` | 黄大豆2号 -> 豆粕 | `direct_input` |
| 24 | `B -> Y` | 黄大豆2号 -> 豆油 | `direct_input` |
| 25 | `M -> LH` | 豆粕 -> 生猪 | `direct_input` |
| 26 | `C -> CS` | 玉米 -> 玉米淀粉 | `direct_input` |
| 27 | `C -> LH` | 玉米 -> 生猪 | `direct_input` |
| 28 | `CF -> CY` | 棉花 -> 棉纱 | `direct_input` |

说明：`UR -> C` 表示尿素作为玉米种植投入，不表示尿素成为玉米的物质组成；若模型区分农业投入，可进一步改名为 `agricultural_input`。

### 2.2 真实但跨级的产业链边

这些边来自原始期货关系。只有在至少一个中间节点没有可用数据、无法形成完整逐级路径时才保留，并标记为 `indirect_chain`。

| 序号 | 边 | 中文关系 | `relation_type` | 当前缺少的中间环节 |
| ---: | --- | --- | --- | --- |
| 33 | `PG -> EG` | 液化石油气 -> 乙二醇 | `indirect_chain` | 乙烯、环氧乙烷 |
| 34 | `SC -> EB` | 原油 -> 苯乙烯 | `indirect_chain` | 纯苯、乙烯、乙苯 |
| 36 | `SC -> V` | 原油 -> 聚氯乙烯 | `indirect_chain` | 乙烯、氯乙烯单体 |
| 37 | `J -> V` | 焦炭 -> 聚氯乙烯 | `indirect_chain` | 电石、乙炔、氯乙烯单体 |

`PG -> EG`、`SC -> EB`、`SC -> V` 和 `J -> V` 因至少一个中间节点仍无可用数据，暂时保留为 `indirect_chain`。后续一旦补齐中间节点数据并通过质量检查，就删除相应跨级边。

### 2.3 同级商品、品级和市场联动边

| 序号 | 边 | 中文关系 | `relation_type` | 语义范围 |
| ---: | --- | --- | --- | --- |
| 41 | `SF <-> SM` | 硅铁 <-> 锰硅 | `peer_product` | 条件替代：炼钢配料的有限技术替代 |
| 42 | `CF <-> PF` | 棉花 <-> 涤纶短纤 | `peer_product` | 条件替代：纺纱、纺织原料替代及混纺 |
| 43 | `BR <-> RU` | 合成橡胶 <-> 天然橡胶 | `peer_product` | 条件替代：轮胎等橡胶用途的配方替代 |
| 44 | `BR <-> NR` | 合成橡胶 <-> 20号胶 | `peer_product` | 条件替代：轮胎等橡胶用途的配方替代 |
| 45 | `RU <-> NR` | 天然橡胶 <-> 20号胶 | `grade_or_origin` | 天然橡胶体系与 TSR20 品级/交割范围重叠 |
| 46 | `A <-> B` | 黄大豆1号 <-> 黄大豆2号 | `grade_or_origin` | 同源品种、不同质量用途和交割标准 |
| 47 | `A <-> PK` | 黄大豆1号 <-> 花生 | `peer_product` | 条件替代：食品用途及种植收益替代；不表示豆一进入压榨链 |
| 48 | `B <-> PK` | 黄大豆2号 <-> 花生 | `peer_product` | 条件替代：压榨油料替代 |
| 49 | `M <-> RM` | 豆粕 <-> 菜籽粕 | `peer_product` | 条件替代：饲料蛋白原料替代 |
| 50 | `Y <-> P` | 豆油 <-> 棕榈油 | `peer_product` | 条件替代：食用油和部分工业用途替代 |
| 51 | `Y <-> OI` | 豆油 <-> 菜籽油 | `peer_product` | 条件替代：食用油用途替代 |
| 52 | `P <-> OI` | 棕榈油 <-> 菜籽油 | `peer_product` | 条件替代：食用油用途替代 |
| 53 | `M <-> Y` | 豆粕 <-> 豆油 | `peer_product` | 联合产出：黄大豆2号压榨的两类主要产品 |
| 54 | `RB <-> HC` | 螺纹钢 <-> 热轧卷板 | `peer_product` | 同源产品：同属钢材生产体系及卷螺价差 |
| 55 | `SC <-> P` | 原油 <-> 棕榈油 | `market_linkage` | 生物柴油用途路径和阶段性价格联动 |
| 56 | `SC <-> AU` | 原油 <-> 黄金 | `market_linkage` | 通胀、美元和风险偏好联动 |
| 57 | `PX <-> EB` | 对二甲苯 <-> 苯乙烯 | `market_linkage` | 芳烃化工板块和共同成本驱动 |
| 58 | `AU <-> AG` | 黄金 <-> 白银 | `market_linkage` | 贵金属共同宏观因子和部分用途联系 |

`market_linkage` 可以保留，但应与产业链消息传递分开；更稳妥的做法是让其边权由仅使用历史窗口的统计量动态决定。

第 21、22 条旧边 `A -> M` 和 `A -> Y` 已删除。黄大豆1号以食用消费为主，进入压榨并生产豆粕、豆油的比例很低；黄大豆2号才是国内压榨链的主要期货映射，因此仅保留 `B -> M` 和 `B -> Y`。

## 3. 加入 SMM 实体节点后的新增边

### 3.1 黑色金属和铁合金

| 序号 | 边 | 中文关系 | `relation_type` | 严格依据 | 证据等级 |
| ---: | --- | --- | --- | --- | --- |
| 59 | `MANGANESE_ORE -> SM` | 锰矿 -> 锰硅 | `direct_input` | 锰硅以锰矿、焦炭等碳质还原剂和电力为主要投入。[郑商所锰硅品种资料](https://www.czce.com.cn/cn/sspz/mg/H770202index_1.htm)；[国家发展改革委高耗能行业能效文件](https://www.gov.cn/zhengce/zhengceku/2021-11/15/content_5651017.htm) | A |
| 61 | `SILICA_STONE -> SF` | 硅石 -> 硅铁 | `direct_input` | 硅铁生产原料包括硅石和碳质还原剂。[郑商所硅铁品种资料](https://www.czce.com.cn/cn/sspz/gt/H770201index_1.htm)；[国家发展改革委高耗能行业能效文件](https://www.gov.cn/zhengce/zhengceku/2021-11/15/content_5651017.htm) | A |
| 62 | `CHARCOAL -> SF` | 兰炭 -> 硅铁 | `direct_input` | 郑商所硅铁产业资料将兰炭列为主要还原剂原料。来源同上。 | A |

### 3.2 铝和工业硅

| 序号 | 边 | 中文关系 | `relation_type` | 严格依据 | 证据等级 |
| ---: | --- | --- | --- | --- | --- |
| 64 | `BAUXITE -> AO` | 铝土矿 -> 氧化铝 | `direct_input` | 氧化铝由铝土矿经拜耳法等工艺生产。[美国地质调查局：Bauxite and Alumina Statistics and Information](https://www.usgs.gov/centers/national-minerals-information-center/bauxite-and-alumina-statistics-and-information)；[上期所氧化铝期货](https://www.shfe.com.cn/products/futures/metal/ao_f/) | A |
| 67 | `SILICA_STONE -> SI` | 硅石 -> 工业硅 | `direct_input` | 工业硅由硅石和碳质还原剂在矿热炉中冶炼。[广期所工业硅品种资料](https://www.gfex.com.cn/gfex/sspz/gygspz.shtml) | A |
| 68 | `PETCOKE -> SI` | 石油焦 -> 工业硅 | `direct_input` | 石油焦属于工业硅生产使用的碳质还原剂。[广期所工业硅品种资料](https://www.gfex.com.cn/gfex/sspz/gygspz.shtml) | A |

### 3.3 石油化工和聚酯链

| 序号 | 边 | 中文关系 | `relation_type` | 严格依据 | 证据等级 |
| ---: | --- | --- | --- | --- | --- |
| 70 | `SC -> NAPHTHA` | 原油 -> 石脑油 | `direct_input` | 炼油将原油转化为石脑油等石油产品。[EIA：Refining crude oil](https://www.eia.gov/energyexplained/oil-and-petroleum-products/refining-crude-oil.php)；[郑商所 PX 品种资料](https://www.czce.com.cn/cn/sspz/px/H770308index_1.htm) | A |
| 71 | `NAPHTHA -> MX` | 石脑油 -> 混合二甲苯 | `direct_input` | 郑商所给出的芳烃路径为“原油 -> 石脑油 -> 混合二甲苯 -> PX”。[郑商所 PX 品种资料](https://www.czce.com.cn/cn/sspz/px/H770308index_1.htm) | A |
| 72 | `MX -> PX` | 混合二甲苯 -> 对二甲苯 | `direct_input` | 混合二甲苯经分离等工艺得到 PX。来源同上。 | A |
| 73 | `NAPHTHA -> ETHYLENE` | 石脑油 -> 乙烯 | `direct_input` | 石脑油裂解是乙烯生产路线，交易所聚乙烯资料给出“原油 -> 石脑油 -> 乙烯 -> 聚乙烯”。[大商所聚乙烯品种资料](https://www.dce.com.cn/dalianshangpin/sspz/nyhg/llDPE/index.html) | A |
| 74 | `ETHYLENE -> L` | 乙烯 -> 聚乙烯 | `direct_input` | 聚乙烯由乙烯聚合生产。来源同上。 | A |
| 75 | `NAPHTHA -> PROPYLENE` | 石脑油 -> 丙烯 | `direct_input` | 石脑油裂解/炼化路线产生丙烯，丙烯再聚合为聚丙烯。[大商所聚丙烯品种资料](https://www.dce.com.cn/dalianshangpin/sspz/nyhg/jbx/index.html) | A |
| 76 | `MA -> PROPYLENE` | 甲醇 -> 丙烯 | `direct_input` | 甲醇制烯烃/甲醇制丙烯路线得到丙烯。[郑商所甲醇品种资料](https://www.czce.com.cn/cn/sspz/ma/H770204index_1.htm) | A |
| 77 | `PROPYLENE -> PP` | 丙烯 -> 聚丙烯 | `direct_input` | 丙烯是聚丙烯的聚合单体。[大商所聚丙烯品种资料](https://www.dce.com.cn/dalianshangpin/sspz/nyhg/jbx/index.html) | A |
| 78 | `MA -> ETHYLENE` | 甲醇 -> 乙烯 | `direct_input` | 甲醇制烯烃路线可得到乙烯，再用于聚乙烯生产。[郑商所甲醇品种资料](https://www.czce.com.cn/cn/sspz/ma/H770204index_1.htm)；[大商所聚乙烯品种资料](https://www.dce.com.cn/dalianshangpin/sspz/nyhg/llDPE/index.html) | A |
| 79 | `NAPHTHA -> BUTADIENE` | 石脑油 -> 丁二烯 | `indirect_chain` | 丁二烯主要由石脑油裂解副产 C4 馏分抽提获得；因图中缺“裂解 C4”节点，暂记跨一级路径。[国际橡胶研究组织：Synthetic Rubber](https://www.rubberstudy.org/synthetic-rubber)；[上期所合成橡胶期货](https://www.shfe.com.cn/products/futures/energyandchemical/br_f/) | B |
| 80 | `BUTADIENE -> BR` | 丁二烯 -> 合成橡胶 | `direct_input` | 丁二烯是丁二烯橡胶的直接原料。[上期所合成橡胶期货](https://www.shfe.com.cn/products/futures/energyandchemical/br_f/)；[国际橡胶研究组织：Synthetic Rubber](https://www.rubberstudy.org/synthetic-rubber) | A |
| 81 | `ETHYLENE -> EO` | 乙烯 -> 环氧乙烷 | `direct_input` | 环氧乙烷由乙烯氧化生产，是油制乙二醇的直接中间体。[大商所乙二醇品种资料](https://www.dce.com.cn/dalianshangpin/sspz/nyhg/yy/index.html) | A |
| 82 | `EO -> EG` | 环氧乙烷 -> 乙二醇 | `direct_input` | 环氧乙烷水合生产乙二醇。来源同上。 | A |
| 83 | `CALCIUM_CARBIDE -> V` | 电石 -> 聚氯乙烯 | `indirect_chain` | 电石法路径还经过乙炔和氯乙烯单体；现有图缺乙炔节点。[大商所 PVC 品种资料](https://www.dce.com.cn/dalianshangpin/sspz/nyhg/jlyx/index.html)；[国家发展改革委高耗能行业能效文件](https://www.gov.cn/zhengce/zhengceku/2021-11/15/content_5651017.htm) | A |

### 3.4 煤化工和农业化肥

| 序号 | 边 | 中文关系 | `relation_type` | 严格依据 | 证据等级 |
| ---: | --- | --- | --- | --- | --- |
| 84 | `THERMAL_COAL -> MA` | 动力煤 -> 甲醇 | `indirect_chain` | 中国甲醇存在煤气化制合成气、再合成甲醇的路线；“动力煤”节点口径宽于实际化工用煤，因此只记路线依赖，不记一跳直接原料。[郑商所甲醇品种资料](https://www.czce.com.cn/cn/sspz/ma/H770204index_1.htm) | B |
| 85 | `NATURAL_GAS -> MA` | 天然气 -> 甲醇 | `indirect_chain` | 天然气经合成气路线生产甲醇。[郑商所甲醇品种资料](https://www.czce.com.cn/cn/sspz/ma/H770204index_1.htm)；[国际甲醇生产商和消费者协会：Methanol](https://www.methanol.org/about-methanol/) | A |
| 86 | `THERMAL_COAL -> UR` | 动力煤 -> 尿素 | `indirect_chain` | 中国尿素上游以煤制合成氨路线为主，但动力煤与实际原料煤并非完全同义。[郑商所尿素品种资料](https://www.czce.com.cn/cn/sspz/ur/H770301index_1.htm) | B |
| 87 | `NATURAL_GAS -> UR` | 天然气 -> 尿素 | `indirect_chain` | 天然气可经合成气、合成氨路线生产尿素。[郑商所尿素品种资料](https://www.czce.com.cn/cn/sspz/ur/H770301index_1.htm) | A |

## 4. 重建时应删除的旧跨级捷径

第 3 节已经用具有可用数据的中间节点补全下列路径，因此这些旧边不属于最终有效边，重建 CSV 时不得写入：

| 删除旧边 | 使用的新路径 |
| --- | --- |
| `SC -> PX` | `SC -> NAPHTHA -> MX -> PX` |
| `SC -> L` | `SC -> NAPHTHA -> ETHYLENE -> L` |
| `SC -> EG` | `SC -> NAPHTHA -> ETHYLENE -> EO -> EG` |
| `SC -> PP` | `SC -> NAPHTHA -> PROPYLENE -> PP` |
| `SC -> BR` | `SC -> NAPHTHA -> BUTADIENE -> BR` |
| `MA -> L` | `MA -> ETHYLENE -> L` |
| `MA -> PP` | `MA -> PROPYLENE -> PP` |

其余旧跨级边暂时保留为 `indirect_chain`，直到补齐中间节点的可用数据。每次更新节点数据后，都应重新执行一次路径检查：若从上游到下游已经存在完整逐级路径，则删除对应的跨级边。

## 5. 暂不接入或必须先修正的新增节点

### 5.1 `POWER`、`THERMAL_COAL` 和 `NATURAL_GAS`

当前 `POWER` 和 `NATURAL_GAS` 两个目录都包含同一个 SMM 指标 `a12750852`：“预焙阳极成本模型：天然气及电力等能耗”。该指标是混合能耗成本，既不能独立代表电价，也不能独立代表天然气价格。

- `POWER -> SM/SF/AL/SI/CALCIUM_CARBIDE` 是成立的真实结构关系，但当前不得作为有效特征边写入图；取得独立且通过时序质量检查的电价后，才能新增这些 `energy_input` 边。
- `POWER -> AO` 暂不保留。氧化铝属于高耗能行业并不等于电力是其主要成本，蒸汽、烧碱、铝土矿等投入同样重要；仅凭能效限额不足以建立“主要电力成本”边。
- 原第 60、63、65、66、69 条 `THERMAL_COAL -> 产品` 煤电代理边已删除。动力煤价格只覆盖部分燃料成本，无法代表上网电价、市场化交易电价、地区电源结构及自备电厂长协成本；在没有独立真实电价数据时，不建立这些电力传导边。
- 但在修正特征映射前，不得把 `a12750852` 同时作为两个节点的独立特征。
- 如果无法取得独立电力和天然气数据，应保留节点并将节点特征设为缺失及有效性掩码，而不是复制同一序列。

### 5.2 `VCM`

当前 `VCM` 目录下载到的是 PVC 期货/现货数据和 PVC 阻燃母粒数据，不是氯乙烯单体数据。因此：

- 暂不新增 `VCM -> V`，否则节点名称与数据语义不一致。
- 取得真实 VCM 价格、产量或库存后，再新增 `VCM -> V`，关系类型为 `direct_input`。
- 电石法完整路径应最终表示为 `CALCIUM_CARBIDE -> 乙炔 -> VCM -> V`。

### 5.3 原 `AL_SUPPLY_DEMAND` 目录

它不是经济实体，而是下载阶段按 SMM 目录组织的一组指标。当前内容混入“镁铝比”“镁铝价差”和金属镁供需中的铝产品需求，并不是纯粹的铝产业供需集合，因此不应建成图节点：

- 真实铝产量、库存、消费、进出口等指标经逐项复核后直接并入 `AL` 的特征。
- 镁铝比、镁铝价差属于跨品种市场特征，不应伪装成铝供需。
- 删除 `AL_SUPPLY_DEMAND` 节点及原候选第 88 条 `indicator_scope` 边。

### 5.4 原 `MN_SUPPLY_DEMAND` 目录

它同样是指标目录而非经济实体。当前内容主要是锰矿价格，与 `MANGANESE_ORE` 语义重复；将复核通过的指标并入 `MANGANESE_ORE` 特征，并删除 `MN_SUPPLY_DEMAND` 节点及原候选第 89 条 `indicator_scope` 边。

### 5.5 删除孤立节点

本版只保留至少参与一条有效边的节点，不为连通性强造边。删除下列零度节点：

- 期货节点：`CU`、`BC`、`PB`、`SN`、`LC`、`JD`、`AP`、`CJ`、`SR`、`FB`。
- SMM 节点：`POWER`、`VCM`。
- 指标目录伪节点：`AL_SUPPLY_DEMAND`、`MN_SUPPLY_DEMAND`（在孤立检查前已按第 5.3、5.4 节删除）。
- 与主网络隔离的二节点分量：`NI`、`SS` 及第 10 条 `NI -> SS`；`SA`、`FG` 及第 29 条 `SA -> FG`。这些生产关系本身成立，但当前无法向图中其他节点传递信息，因此本版不注册。

将来新增有依据的边或修复真实特征后，可以连同相关边重新注册对应实体节点。

## 6. 最终数量与实施规则

按逻辑边计数：

- 原图既有期货关系共 58 条；重建后保留 47 条，其中直接产业链 25 条、跨级产业链 4 条、横向关系 18 条。除 7 条跨级捷径外，另删除第 10、29 条孤立分量边和第 21、22 条豆一弱生产边。
- 新增有效边 24 条，均为生产或路线边；原第 60、63、65、66、69 条煤电代理边及候选第 88、89 条指标归属边均不进入有效图。
- 7 条旧跨级边已被具有可用数据的中间节点路径替代，仅在第 4 节留作审计记录，最终建图时必须删除，不能重复保留。
- 最终有效逻辑边为 71 条，节点为 58 个，其中期货节点 44 个、SMM 实体节点 14 个。删除煤电代理边后，`SI` 与主网络形成由 `SILICA_STONE`、`PETCOKE` 连接的独立分量。

建议规范化边表字段：

```text
source,target,relation_type,directed,route,confidence,source_url,is_active,note
```

实施约束：

1. 无向边只存一条，加载图时再生成两个方向的消息传递边。
2. `direct_input`、`indirect_chain`、`energy_input`、`peer_product`、`grade_or_origin` 和 `market_linkage` 使用不同的关系编码或参数。`peer_product` 内部的条件替代、联合产出、同源产品机制保留在说明字段中。
3. 不要仅凭 SMM 指标相关性新增结构边；动态统计边只能用预测时点以前的数据估计。
4. 新增边来源失效或页面迁移时，只能替换为同一交易所、政府部门、国际行业组织或同等级权威来源。
5. 节点有结构边但暂时没有可靠特征时，使用缺失掩码；不得用语义不匹配的数据填充。
6. 对每条候选 `indirect_chain` 边 `A -> C`，在仅由逐级产业边构成的图上检查 `A` 到 `C` 的可达性；若存在完整路径且路径中所有节点都有可用数据，则禁用 `A -> C`。
7. 电力传导边必须以具有独立语义、通过时序质量检查的 `POWER` 数据为前提；不得使用 `THERMAL_COAL` 价格替代电价。
