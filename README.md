# GNN for Futures — 模型与冻结权重发布版

本分支发布正式模型实现、图消融入口与精选 checkpoint，不分发市场数据、测试代码或失败实验入口。
原研究工作区及封存实验均未被此次打包修改；所有权重和 scaler 保持原始字节。
历史提交中已有的研究脚本/图资料保持不变，本次提交不新增这些数据或测试。

## 1. 模型与输入输出

- **Full / History / FH**：15 分钟因果历史 → 节点编码 → 图关系聚合 → 有无关系概率 q、条件方向概率 d。
  Full 使用经济先验 E、因果滚动回归 R、注意力 A；History 使用历史事件关系。
  FH 使用 Full 的 q 与 History 的 d，不是另一套独立训练权重。
- **GM baseline**：已实现历史事件、时间编码与结构信息 → GraphMixer → Follow-only q。
  发布的二分类权重使用 39 节点身份特征、50 事件槽、55 结构事件窗口。
- **Full+Dir119Linear-v1**：冻结 Full 的 100 维 pair 表征，拼接 19 维因果收益/动量，
  用 train-only scaler 标准化后接 Linear(119,1)+sigmoid，共 120 个方向参数。
  只对 q>0.5 的可行动、非对角关系调用方向头。七年和十年分别封存，主种子均为 20261002。
- **十年 FF/FH Follow-only 图消融**：27 节点，8 个 E/R/A 子集 × 3 种子，共 24 个 checkpoint。
  FF/FH 去掉方向后共享 Full q，是 alias，不重复训练；none 保留节点编码与 Follow head，所有图输入为空。

原三分类规则：q<=0.5 为 N；否则 d>0.5 为 S，其余 O。二分类映射 N→0、O/S→1，严格 q>0.5。
不得把不同节点数、双头与 Follow-only checkpoint 混用。

## 2. 运行环境与正式入口

Python >=3.11；安装已有项目依赖（numpy、pandas、torch、scipy、scikit-learn、statsmodels）：

```bash
python -m pip install -e .
python -I -B tools/run_ten_year_graph_ablation.py --help
python -I -B tools/run_follow_binary_experiment.py --help
python -I -B tools/run_full_dir119_baseline.py --help
```

正式入口：

| 功能 | 入口 |
| --- | --- |
| 十年 Follow-only 全图消融 | [tools/run_ten_year_graph_ablation.py](tools/run_ten_year_graph_ablation.py) |
| 七年 FH 图消融 | [tools/run_alpha39_fh_ablation.py](tools/run_alpha39_fh_ablation.py) |
| 七年 Full / HH / GM 二分类比较 | [tools/run_follow_binary_experiment.py](tools/run_follow_binary_experiment.py) |
| GM 数据适配与训练 | [tools/train_gm_baseline.py](tools/train_gm_baseline.py) |
| 七年/十年 119 维冻结基准 | [tools/full_dir119_baseline.py](tools/full_dir119_baseline.py) |
| 十年 119 维从头训练入口 | [tools/run_full_dir119_baseline.py](tools/run_full_dir119_baseline.py) |

安装包提供 src 下模型；tools 与 graph_model 是仓库级运行模块，调用它们时请在仓库根目录执行。
启动脚本只使用已有 Python 环境，不安装依赖、不部署远端、不自动运行完整实验。
部分文件沿用 research/grid 命名，但发布内容仅保留正式运行所需的特征、指标、分配与合同 helper；
旧残差、MOE、MLP/树候选搜索入口不在此次发布中。

## 3. Checkpoint 位置与加载

| 发布内容 | 位置 | 数量 |
| --- | --- | ---: |
| 十年图消融 | models_performance/ten_year_graph_ablation/20261109_full24/training/{variant}_{seed}/best.pt | 24 |
| 七年 Full/HH/GM Follow-only | models_performance/follow_binary/20261031_primary/training/{family}_{seed}/best.pt | 9 |
| 七年/十年119维基准合同 | models_performance/full_dir119linear_v1/20261103_primary/{seven_year,ten_year}_frozen.json | 2 |

三种子均为 20261002、20261003、20261004，不按 test 结果筛选。
原七年 Full / History 权重保留合同引用位置；History 虽位于 alpha39_residual 旧目录，
这里只发布其基准骨干 best.pt，不含残差修正模型。
scaler.npz 是方向模型必要参数，不是原始市场数据，共发布两个。

在仓库根目录加载两个119维基准（无需市场数据；预测仍需要符合合同的因果输入）：

```python
from pathlib import Path
from tools.full_dir119_baseline import load

root = Path("models_performance/full_dir119linear_v1/20261103_primary")
full, direction, contract = load(root / "seven_year_frozen.json")
full10, direction10, contract10 = load(root / "ten_year_frozen.json")
```

七年二分类模型按 checkpoint_manifest.json 的 constructor/arguments 构建：

```python
import torch
from tools.run_follow_binary_experiment import make_model

model = make_model("gm", 39)  # family: full / HH / gm
path = "models_performance/follow_binary/20261031_primary/training/gm_20261002/best.pt"
model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
model.eval()
```

十年消融按 tools.run_ten_year_graph_ablation.make_model(variant, 27) 构建，再严格加载 state_dict；
variant 为 full / no_economic / no_regression / no_attention / economic_only /
regression_only / attention_only / none。none 必须用该入口的专用构造器。

## 4. 分发边界与重现限制

- 不含原始行情、面板、日图、预测数组、逐笔账本、完整结果归档、测试代码或凭据。
- 仅保留紧凑合同、选轮记录、分类/辅助回测汇总；历史 completed.json 和 prediction hashes
  描述原完整实验，**不表示本发布包含全部证据或可重做全量产物哈希审计**。
- 原训练 admission gate 未关闭。完整重训必须具备原私有数据、audit/verification、原封存源码与合同。
  七年父合同绑定未分发的论文及历史源码；十年合同发布副本脱敏了远端地址，原 parent hash 保留。
  因此不能只 clone 本分支并补一份 NPZ 就声称通过原始实验准入。
- 119维推理合同的 implementation hashes 对裁剪后的发布 helper 重新绑定；十年 protocol 引用
  对脱敏副本重新绑定，并记录原 SHA。权重、scaler、种子、选轮、特征公式不变。
- .gitattributes 禁用平台换行转换，避免 checkout 后源码哈希失配。
- [release-20261114.json](release-20261114.json) 记录来源/发布 SHA-256 与精确白名单；它不含自身递归哈希。

## 5. 验证与结果解释

发布副本隔离测试：62 项通过、1 项真实私有面板集成测试因未分发数据跳过；
另核查运行时本地 import 符号完整、七年/十年119维权重可加载。测试脚本未发布。
未为此次打包重跑正式训练，也未修改成本、标签、切分、阈值或选种子规则。

十年图消融 Full test macro-F1 约 0.6449，none 约 0.6313（三种子均值）；
Full 的 AP 并非最优，且费用压力表现差，不能据此宣称全面排序优势或可交易盈利。
119维线性头七年方向指标有小幅改善，但十年没有复现方向优势，冻结不代表最佳模型。
所有研究仍是 strict_pit=false、fresh_oos=false、previously_inspected_test=true；
三种子标准差不是置信区间，不新增显著性或 fresh-OOS 声明。
