from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RESULT_DIR = Path(__file__).resolve().parent / "results" / "graph_edge_regressions_full"
OUTPUT_PATH = RESULT_DIR / "initial_analysis_report.md"
SIGNIFICANCE_PATH = RESULT_DIR / "edge_significance_summary.csv"


def markdown_table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    display = frame.copy()
    for column, pattern in (formats or {}).items():
        if column in display:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else pattern.format(value)
            )
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    header = "|" + "|".join(headers) + "|"
    divider = "|" + "|".join("---" for _ in headers) + "|"
    body = ["|" + "|".join(row) + "|" for row in rows]
    return "\n".join([header, divider, *body])


def direction_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, dropna=False)
    result = grouped.agg(
        窗口数=("forecast_date", "size"),
        命中数=("forecast_direction_hits", "sum"),
        可比较数=("forecast_direction_comparisons", "sum"),
        中位窗口命中率=("forecast_direction_hit_rate", "median"),
        中位样本外R2=("oos_r_squared", "median"),
        样本外R2为正比例=("oos_r_squared", lambda values: float((values > 0).mean())),
        非零阶窗口比例=("bic_selected_lag", lambda values: float((values > 0).mean())),
        原始p小于0点1窗口数=("edge_wald_p_value", lambda values: int((values < 0.1).sum())),
        可检验窗口数=("edge_wald_p_value", "count"),
    ).reset_index()
    result["汇总命中率"] = result["命中数"] / result["可比较数"]
    result["原始p小于0点1占全部窗口"] = result["原始p小于0点1窗口数"] / result["窗口数"]
    result["原始p小于0点1占可检验窗口"] = (
        result["原始p小于0点1窗口数"] / result["可检验窗口数"].replace(0, np.nan)
    )
    return result


def build_report() -> tuple[str, pd.DataFrame]:
    static = pd.read_csv(RESULT_DIR / "static_model_summary.csv")
    dynamic = pd.read_csv(RESULT_DIR / "dynamic_model_summary.csv")
    ablation = pd.read_csv(RESULT_DIR / "smm_ablation_summary.csv")
    registry = pd.read_csv(RESULT_DIR / "directed_edge_registry.csv")

    baseline = dynamic.loc[dynamic["model_variant"].eq("baseline")].copy()
    baseline["forecast_date"] = pd.to_datetime(baseline["forecast_date"])
    baseline["年份"] = baseline["forecast_date"].dt.year
    static_baseline = static.loc[static["model_variant"].eq("baseline_full")].copy()

    hits = int(baseline["forecast_direction_hits"].sum())
    comparisons = int(baseline["forecast_direction_comparisons"].sum())
    zero_returns = int(baseline["forecast_zero_return_count"].sum())
    expected_rows = baseline["forecast_date"].nunique() * baseline["directed_edge"].nunique()
    duplicate_keys = int(baseline.duplicated(["directed_edge", "forecast_date"]).sum())

    lag_counts = (
        baseline["bic_selected_lag"]
        .value_counts()
        .sort_index()
        .rename_axis("BIC阶数")
        .reset_index(name="窗口数")
    )
    lag_counts["占比"] = lag_counts["窗口数"] / len(baseline)

    relation = direction_summary(baseline, ["relation_type"]).rename(
        columns={"relation_type": "关系类型"}
    )
    relation = relation.sort_values("汇总命中率", ascending=False)

    yearly = direction_summary(baseline, ["年份"])

    edge = direction_summary(
        baseline,
        ["graph_edge_number", "directed_edge", "relation_type"],
    ).rename(
        columns={
            "graph_edge_number": "边号",
            "directed_edge": "有向边",
            "relation_type": "关系类型",
        }
    )
    best_edges = edge.nlargest(10, "汇总命中率")
    worst_edges = edge.nsmallest(10, "汇总命中率")
    most_nonzero = edge.nlargest(10, ["非零阶窗口比例", "汇总命中率"])

    static_lag_counts = (
        static_baseline["bic_selected_lag"]
        .value_counts()
        .sort_index()
        .rename_axis("BIC阶数")
        .reset_index(name="方向数")
    )
    static_fitted = static_baseline["bic_selected_lag"].gt(0)
    static_significant = static_baseline["edge_wald_p_value"].lt(0.05).fillna(False)
    static_significant_10 = static_baseline["edge_wald_p_value"].lt(0.1).fillna(False)
    dynamic_significance = direction_summary(
        baseline, ["graph_edge_number", "directed_edge"]
    )[
        [
            "graph_edge_number",
            "directed_edge",
            "窗口数",
            "可检验窗口数",
            "原始p小于0点1窗口数",
            "原始p小于0点1占全部窗口",
            "原始p小于0点1占可检验窗口",
        ]
    ]
    static_significance_fields = static_baseline[
        ["graph_edge_number", "directed_edge", "edge_wald_p_value"]
    ].copy()
    static_significance_fields["static_raw_p_lt_0_10"] = static_significant_10
    significance_summary = registry.merge(
        static_significance_fields,
        on=["graph_edge_number", "directed_edge"],
        how="left",
    ).merge(
        dynamic_significance,
        on=["graph_edge_number", "directed_edge"],
        how="left",
    ).rename(
        columns={
            "edge_wald_p_value": "static_edge_wald_p_value",
            "窗口数": "dynamic_windows",
            "可检验窗口数": "dynamic_testable_windows",
            "原始p小于0点1窗口数": "dynamic_raw_p_lt_0_10_windows",
            "原始p小于0点1占全部窗口": "dynamic_raw_p_lt_0_10_share_all_windows",
            "原始p小于0点1占可检验窗口": "dynamic_raw_p_lt_0_10_share_testable_windows",
        }
    )
    significance_display = significance_summary[
        [
            "graph_edge_number",
            "directed_edge",
            "relation_type",
            "static_edge_wald_p_value",
            "static_raw_p_lt_0_10",
            "dynamic_raw_p_lt_0_10_windows",
            "dynamic_windows",
            "dynamic_raw_p_lt_0_10_share_all_windows",
            "dynamic_testable_windows",
            "dynamic_raw_p_lt_0_10_share_testable_windows",
        ]
    ].rename(
        columns={
            "graph_edge_number": "边号",
            "directed_edge": "有向边",
            "relation_type": "关系类型",
            "static_edge_wald_p_value": "静态原始p",
            "static_raw_p_lt_0_10": "静态p<0.1",
            "dynamic_raw_p_lt_0_10_windows": "动态p<0.1窗口",
            "dynamic_windows": "动态全部窗口",
            "dynamic_raw_p_lt_0_10_share_all_windows": "动态显著率/全部",
            "dynamic_testable_windows": "动态可检验窗口",
            "dynamic_raw_p_lt_0_10_share_testable_windows": "动态显著率/可检验",
        }
    ).sort_values(
        ["动态显著率/全部", "静态p<0.1", "静态原始p"],
        ascending=[False, False, True],
    )
    static_top = (
        static_baseline.loc[static_significant]
        .sort_values(["edge_wald_p_value", "edge_wald_f_statistic"])
        .head(12)[
            [
                "directed_edge",
                "relation_type",
                "bic_selected_lag",
                "edge_wald_f_statistic",
                "edge_wald_p_value",
                "r_squared",
            ]
        ]
        .rename(
            columns={
                "directed_edge": "有向边",
                "relation_type": "关系类型",
                "bic_selected_lag": "BIC阶数",
                "edge_wald_f_statistic": "边Wald F",
                "edge_wald_p_value": "边Wald p值",
                "r_squared": "样本内R2",
            }
        )
    )

    paired = ablation.loc[ablation["ablation_status"].eq("paired")].copy()
    paired_display = paired[
        [
            "directed_edge",
            "forecast_date",
            "smm_nodes",
            "bic_selected_lag",
            "delta_oos_r_squared",
            "delta_oos_rmse",
            "delta_oos_mae",
            "delta_direction_hit_rate",
        ]
    ].rename(
        columns={
            "directed_edge": "有向边",
            "forecast_date": "预测日",
            "smm_nodes": "SMM节点",
            "bic_selected_lag": "BIC阶数",
            "delta_oos_r_squared": "Δ样本外R2",
            "delta_oos_rmse": "ΔRMSE",
            "delta_oos_mae": "ΔMAE",
            "delta_direction_hit_rate": "Δ方向命中率",
        }
    )

    oos_r2 = baseline["oos_r_squared"].dropna()
    median_direction_rate = baseline["forecast_direction_hit_rate"].median()
    mean_direction_rate = baseline["forecast_direction_hit_rate"].mean()
    zero_lag_share = baseline["bic_selected_lag"].eq(0).mean()
    positive_oos_share = baseline["oos_r_squared"].gt(0).mean()
    static_significance_rate = static_significant.sum() / len(static_baseline)
    static_significance_10_rate = static_significant_10.sum() / len(static_baseline)
    dynamic_significant_10_count = int(baseline["edge_wald_p_value"].lt(0.1).sum())
    dynamic_testable_count = int(baseline["edge_wald_p_value"].notna().sum())

    formats = {
        "占比": "{:.2%}",
        "汇总命中率": "{:.3%}",
        "中位窗口命中率": "{:.3%}",
        "中位样本外R2": "{:.5f}",
        "样本外R2为正比例": "{:.2%}",
        "非零阶窗口比例": "{:.2%}",
        "原始p小于0点1占全部窗口": "{:.2%}",
        "原始p小于0点1占可检验窗口": "{:.2%}",
    }
    edge_formats = {
        "汇总命中率": "{:.3%}",
        "中位窗口命中率": "{:.3%}",
        "中位样本外R2": "{:.5f}",
        "样本外R2为正比例": "{:.2%}",
        "非零阶窗口比例": "{:.2%}",
        "原始p小于0点1占全部窗口": "{:.2%}",
        "原始p小于0点1占可检验窗口": "{:.2%}",
    }

    lines = [
        "# 经济先验图回归结果：初步汇总与分析",
        "",
        "> 生成日期：2026-09-10  ",
        "> 范围：57 条经济图边、114 个回归方向、全样本静态回归，以及 20+1 交易日滚动预测。  ",
        "> 当前阶段只评价回归结果，不定义 GNN 标量边权。",
        "",
        "## 1. 先看结论",
        "",
        f"1. 动态实验完整覆盖 {baseline['forecast_date'].nunique():,} 个预测交易日和 {baseline['directed_edge'].nunique():,} 个方向，共 {len(baseline):,} 个基线方向-日期窗口；预期也是 {expected_rows:,} 行，重复主键为 {duplicate_keys}。",
        f"2. 动态方向预测共命中 {hits:,} / {comparisons:,}，汇总命中率为 **{hits / comparisons:.3%}**。窗口命中率均值为 {mean_direction_rate:.3%}，中位数为 {median_direction_rate:.3%}，整体只略高于 50%。",
        f"3. BIC 在 **{zero_lag_share:.2%}** 的动态窗口选择 0 阶；这意味着绝大多数 20 日窗口中，模型退化为仅截距基线，跨期货滞后信息并未稳定进入预测方程。",
        f"4. 只有 **{positive_oos_share:.2%}** 的方向-日期窗口取得正样本外 $R^2$；样本外 $R^2$ 中位数为 {oos_r2.median():.5f}。方向略高于随机，并不等于收益率数值预测已经有效。",
        f"5. 静态全样本中，{int(static_fitted.sum())} / {len(static_baseline)} 个方向选择非零阶；原始 $p<0.05$ 有 {int(static_significant.sum())} 个（{static_significance_rate:.2%}），原始 $p<0.1$ 有 {int(static_significant_10.sum())} 个（{static_significance_10_rate:.2%}）。",
        f"6. 动态原始 $p<0.1$ 的方向-日期窗口为 {dynamic_significant_10_count:,} / {len(baseline):,}（{dynamic_significant_10_count / len(baseline):.2%}）；在 {dynamic_testable_count:,} 个非零阶可检验窗口中占 {dynamic_significant_10_count / dynamic_testable_count:.2%}。",
        f"7. SMM 候选窗口共 {len(ablation):,} 个，但仅 {len(paired)} 个形成真实增强配对；其余 {int(ablation['ablation_status'].eq('zero_lag_no_smm_terms').sum()):,} 个因 BIC 为 0 阶而未加入 SMM。现有 SMM 证据太少且结果混合。",
        "",
        "## 2. 实验覆盖与口径",
        "",
        "- 图注册表：57 条无向经济边，展开为 114 个有向回归方向。",
        "- 静态区间：各品种可用重叠历史；不同品种上市时间不同，因此起点不完全一致。",
        "- 动态区间：最近两年内 484 个预测交易日；每次使用此前 20 个交易日训练，并冻结参数预测下一交易日。",
        "- 阶数：二元 VAR 在 0 至 40 阶中按共同样本 BIC 选择，两方向共享阶数。",
        "- 方向命中：预测与真实收益率均非零且同号记为命中；任一为零时单列，不进入命中率分母。",
        f"- 动态零收益相关记录：{zero_returns:,}；程序验证了训练期早于预测日、方向字段完整、SMM 发布时间不晚于预测信息时点。",
        "",
        "## 3. 动态选阶分布",
        "",
        markdown_table(lag_counts, {"占比": "{:.2%}"}),
        "",
        "0 阶占比接近 96%，是本次结果最重要的结构特征。它说明在短至 20 个交易日的训练窗里，BIC 对增加滞后参数非常谨慎。少数 8 阶和 15 阶窗口值得逐一检查，但不应仅因阶数高就认定关系更强。",
        "",
        "## 4. 按经济关系类型汇总",
        "",
        markdown_table(
            relation[
                [
                    "关系类型",
                    "窗口数",
                    "可比较数",
                    "汇总命中率",
                    "中位窗口命中率",
                    "中位样本外R2",
                    "样本外R2为正比例",
                    "非零阶窗口比例",
                    "原始p小于0点1窗口数",
                    "原始p小于0点1占全部窗口",
                    "原始p小于0点1占可检验窗口",
                ]
            ],
            formats,
        ),
        "",
        "关系类型之间的方向命中率差异应视为描述性结果。各组包含的品种、上市年限和波动尺度不同，当前没有做显著性检验，也没有控制横截面依赖。",
        "",
        "## 5. 按预测年份汇总",
        "",
        markdown_table(
            yearly[
                [
                    "年份",
                    "窗口数",
                    "可比较数",
                    "汇总命中率",
                    "中位窗口命中率",
                    "中位样本外R2",
                    "样本外R2为正比例",
                    "非零阶窗口比例",
                    "原始p小于0点1窗口数",
                    "原始p小于0点1占全部窗口",
                    "原始p小于0点1占可检验窗口",
                ]
            ],
            formats,
        ),
        "",
        "年度表用于观察总体漂移，但 2024 和 2026 都是不完整年度，不能直接与完整的 2025 年做等权比较。",
        "",
        "## 6. 有向边方向命中率排名",
        "",
        "### 6.1 前十",
        "",
        markdown_table(
            best_edges[
                [
                    "边号",
                    "有向边",
                    "关系类型",
                    "可比较数",
                    "汇总命中率",
                    "中位样本外R2",
                    "样本外R2为正比例",
                    "非零阶窗口比例",
                ]
            ],
            edge_formats,
        ),
        "",
        "### 6.2 后十",
        "",
        markdown_table(
            worst_edges[
                [
                    "边号",
                    "有向边",
                    "关系类型",
                    "可比较数",
                    "汇总命中率",
                    "中位样本外R2",
                    "样本外R2为正比例",
                    "非零阶窗口比例",
                ]
            ],
            edge_formats,
        ),
        "",
        "这些排名是两年内所有可比较 bar 的汇总，不代表结构因果排序。前后差距若只有一两个百分点，也可能受到共同市场方向、零收益比例和波动状态影响。",
        "",
        "## 7. 非零阶最常见的方向",
        "",
        markdown_table(
            most_nonzero[
                [
                    "边号",
                    "有向边",
                    "关系类型",
                    "窗口数",
                    "非零阶窗口比例",
                    "汇总命中率",
                    "中位样本外R2",
                ]
            ],
            edge_formats,
        ),
        "",
        "这一表比单纯命中率排名更接近“动态滞后关系是否经常被 BIC 选中”的问题，但仍需进一步结合系数符号稳定性和跨窗口显著性。",
        "",
        "## 8. 原始 p<0.1 的静态与动态统计",
        "",
        "这里使用跨期货滞后项的联合 Wald 检验，而不是挑选某一个滞后系数的最小 p 值。动态 0 阶窗口没有跨期货项，记为不可检验，不记为显著。",
        "",
        markdown_table(
            significance_display,
            {
                "静态原始p": "{:.3e}",
                "动态显著率/全部": "{:.2%}",
                "动态显著率/可检验": "{:.2%}",
            },
        ),
        "",
        "`动态显著率/全部` 可直接在 114 个方向间比较，并自然惩罚经常选到 0 阶的边；`动态显著率/可检验` 回答的是模型一旦选入滞后项后有多少窗口达到原始 p<0.1。二者含义不同，不建议只保留后者。完整字段同时写入 `edge_significance_summary.csv`。",
        "",
        "## 9. 静态全样本结果",
        "",
        "### 8.1 静态 BIC 阶数",
        "",
        markdown_table(static_lag_counts),
        "",
        f"静态样本更长，非零阶比例为 {static_fitted.mean():.2%}，明显高于动态 20 日窗口的 {1 - zero_lag_share:.2%}。这提示关系可能在长样本上可检测，但在短窗预测中不够稳定，或 BIC 在短窗下的参数惩罚占主导。",
        "",
        "### 8.2 原始 p 值最小的跨期货联合检验",
        "",
        markdown_table(
            static_top,
            {
                "边Wald F": "{:.3f}",
                "边Wald p值": "{:.3e}",
                "样本内R2": "{:.5f}",
            },
        ),
        "",
        "这里的 $p$ 值没有经过 114 个方向的多重检验校正，且超长高频样本可能使很小的经济效应获得统计显著性。因此应同时看效应大小、预测增益和时间稳定性，不能只按 $p$ 值选边。",
        "",
        "## 10. SMM 消融",
        "",
        markdown_table(
            paired_display,
            {
                "Δ样本外R2": "{:+.6f}",
                "ΔRMSE": "{:+.6e}",
                "ΔMAE": "{:+.6e}",
                "Δ方向命中率": "{:+.3%}",
            },
        ),
        "",
        "- 正的 $\\Delta R^2_{OOS}$ 表示加入 SMM 后改善；负的 $\\Delta RMSE$、$\\Delta MAE$ 表示误差下降。",
        "- 四个配对全部来自 `SC↔L` 的两个预测日，不能推广到其他 SMM 路径。",
        "- 两个窗口数值误差略改善、两个恶化；三个窗口方向命中不变，一个下降。当前应判断为“证据不足且没有稳定增益”。",
        "- 96% 左右的零阶选择实际上阻断了绝大多数 SMM 消融。后续若要专门评价 SMM，宜把“是否允许 SMM 在期货阶数为 0 时仍以一阶进入”作为独立实验，不应悄悄修改本轮批准口径。",
        "",
        "## 11. 初步判断",
        "",
        "### 10.1 可以较有把握地说",
        "",
        "1. 输出覆盖和无未来函数检查通过，本轮结果可作为后续研究的完整基线。",
        "2. 经济图关系在全历史静态样本中能找到一批统计显著的滞后关联，但这种关联很少在 20 日滚动窗口中被 BIC 选中。",
        "3. 两年总体方向命中率略高于 50%，但样本外数值拟合普遍较弱；目前不能把命中率直接解释成可交易预测力。",
        "4. 本轮 SMM 消融有效样本太少，不支持“SMM 有效”或“SMM 无效”的强结论。",
        "",
        "### 10.2 暂时不能说",
        "",
        "1. 不能把静态显著方向解释为结构性因果。",
        "2. 不能因某条边方向命中率排名靠前就直接定义较大 GNN 权重。",
        "3. 不能根据四个 SMM 配对窗口评价三个 A 级 SMM 节点的总体价值。",
        "4. 不能把 50% 以上的表观命中率直接转换为策略收益；尚未计入手续费、滑点、信号阈值和仓位规则。",
        "",
        "## 12. 构建边权前还必须明确什么",
        "",
        "不需要把经济先验图再做成一套复杂的统计筛选系统，也不建议用 p<0.1 删除边。图的存在由经济先验决定，回归结果更适合决定边的方向、符号、强弱或置信度。真正必须先明确的只有四项：",
        "",
        "1. **边权时间口径**：训练和推理时只能使用当时已经完成的滚动窗口参数，不能把全样本静态系数放进历史时点；静态结果只适合初始化或研究对照。",
        "2. **多滞后如何压缩**：一个方向可能有多个 `cross_futures` 系数，必须固定聚合规则，例如求和 $b_{ij,t}=\\sum_{l=1}^{p_t}\\beta_{ij,l,t}$，不能事后挑选表现最好的滞后。",
        "3. **符号和尺度**：明确 GNN 是否允许负边权；不同窗口的极端系数应裁剪或稳健缩放，避免少数病态回归主导消息传递。",
        "4. **0 阶和缺失处理**：0 阶时回归增量权重应为 0，但经济先验边可以保留一个基础通道；这样不会把“短窗未检出”误写成“经济关系不存在”。",
        "",
        "满足这四项后，就可以开始设计并回测边权函数；block bootstrap、FDR 和复杂稳定性模型是增强项，不是现阶段开工的硬门槛。",
        "",
        "## 13. 建议的简单边权函数",
        "",
        "建议先把经济先验和回归证据分开，而不是用显著性做硬筛选。对方向 $i\\to j$、时点 $t$：",
        "",
        "$$",
        "b_{ij,t}=\\sum_{\\ell=1}^{p_t}\\beta_{ij,\\ell,t},",
        "\\qquad",
        "c_{ij,t}=\\max(0,1-p^{Wald}_{ij,t}),",
        "$$",
        "",
        "再对 $b_{ij,t}$ 做只基于训练期横截面的稳健缩放：",
        "",
        "$$",
        "z_{ij,t}=\\operatorname{clip}\\left(\\frac{b_{ij,t}}{Q_{0.75}(|b_t|)+\\epsilon},-z_{max},z_{max}\\right).",
        "$$",
        "",
        "第一版推荐保留两个通道：",
        "",
        "$$",
        "w^{prior}_{ij}=1,",
        "\\qquad",
        "w^{reg}_{ij,t}=\\mathbf{1}[p_t>0]\\,z_{ij,t}\\,c_{ij,t}.",
        "$$",
        "",
        "GNN 可以分别学习经济先验消息和回归消息的组合比例。这比预先手工确定 $\\lambda$ 更稳妥，也保留负系数信息。若模型接口只能接收一个标量，可先用：",
        "",
        "$$",
        "w_{ij,t}=\\operatorname{clip}\\left(1+\\lambda w^{reg}_{ij,t},0,w_{max}\\right),",
        "$$",
        "",
        "其中 $\\lambda$ 只在训练集上调参。原始 $p<0.1$ 可另存为布尔置信字段，或用分段置信度 $c=1,0.5,0$ 对应 $p<0.05$、$0.05\\le p<0.1$、其余；不建议直接令不显著边权为 0。",
        "",
        "## 14. 建议的下一轮分析",
        "",
        "1. 对方向命中率做按日聚类或 block bootstrap 置信区间，避免把同日、同品种 bar 当成完全独立样本。",
        "2. 报告每条边的滚动系数符号稳定率、非零阶持续长度和显著窗口占比，而不是只看全期汇总命中率。",
        "3. 加入简单基准：零预测、历史均值、仅自身滞后模型。当前 baseline 同时含自身与对端滞后，还不能把全部表现归因于图边。",
        "4. 做多重检验控制，例如 Benjamini-Hochberg FDR，并区分统计显著与经济显著。",
        "5. 对 SMM 单独开展预注册实验：保持当前结果不变，另行比较“期货 0 阶时仍允许 SMM 一阶进入”与更长训练窗。",
        "6. 在讨论 GNN 边权前，先形成边级特征面板，例如静态显著性、动态非零阶比例、方向稳定性和样本外增益；不要过早压缩成单一标量。",
        "",
        "## 15. 结果文件索引",
        "",
        "- `static_model_summary.csv`：114 个静态方向的模型统计。",
        "- `static_coefficients.csv`：静态系数、标准误和置信区间。",
        "- `dynamic_model_summary.csv`：逐方向、逐预测日模型与样本外指标。",
        "- `dynamic_coefficients.csv`：逐窗口回归系数。",
        "- `dynamic_forecasts.csv`：逐 15 分钟 bar 的真实值、预测值与方向分类。",
        "- `smm_ablation_summary.csv`：匹配样本上的 SMM 增强差异。",
        "- `directed_edge_registry.csv`：114 个方向及经济关系分类。",
        "- `validation_report.json`：时序和输出验证结果。",
        "- `edge_significance_summary.csv`：114 个方向的静态原始 p<0.1 标记和动态显著窗口统计。",
        "",
        "---",
        "",
        f"注册表核对：{len(registry)} 个有向边；报告由 `build_graph_edge_regression_report.py` 自动生成。",
    ]
    return "\n".join(lines) + "\n", significance_summary


def main() -> None:
    report, significance_summary = build_report()
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    significance_summary.to_csv(SIGNIFICANCE_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {SIGNIFICANCE_PATH}")


if __name__ == "__main__":
    main()