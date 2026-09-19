# GRPO 自适应 RAG 评测设计

## 目标

评估 GRPO 训练得到的**单一自适应 RAG 策略**。它应针对每个问题自主决定是否检索、是否继续检索及何时作答；不应把训练后的模型人为拆成 B0、B1、B2 三类。

当前 pilot 配置的 `GRPO_MAX_TURNS=2`，故策略可选择 0、1 或 2 轮检索。若后续提高该上限，评测按新的上限统计实际轮数。

## 固定评测集

- 正式数据集：NQ、HotpotQA。
- 规模：每个数据集固定 200 条，共 400 条（资源允许时扩展至每集 500 条）。
- 在首次生成时保存数据集版本、每题 `id`、抽样 seed 和顺序到 manifest；之后所有条件必须使用该 manifest，不得重新随机抽样。
- 每题保存：问题、gold answers、预测、Strict EM、token F1、检索轮数、每轮 query、Top-k 文档/分数、延迟、失败信息。

## 对照条件

所有条件共享模型版本、固定评测集、E5-base-v2、Wiki-18 Flat FAISS 索引、Top-k、提示词、评分器与解码设置；仅改变下列明确指定的策略。

1. **B0（固定无检索）**：训练前原模型，0 轮检索。
2. **B1（固定单轮）**：训练前原模型，强制 1 轮检索。
3. **B2（固定多轮）**：训练前原模型，最多 2 轮检索。
4. **GRPO step-0 adaptive（推荐且必要）**：未更新的初始模型，使用与 GRPO 完全相同的自适应 rollout 协议和最大轮数。
5. **GRPO trained adaptive**：训练后 checkpoint，使用相同的自适应 rollout 协议。

其中第 5 项是训练后模型的唯一正式结果；它不是三组 B0/B1/B2 结果。第 4 项与第 5 项构成评估 GRPO 更新效果的主配对比较。B0/B1/B2 用于衡量固定检索预算的参考上限/下限及自适应策略的相对收益。

正式结论以 NQ/HotpotQA 各 200 条为准；B0/B1/B2 与 GRPO 必须使用相同的 400 条冻结题目。不能用旧的各 50 条 smoke 结果替代此设计。已生成的六数据集结果仅保留作诊断，不计入主结论。

## 必报指标

### 答案评分口径

唯一的主评分是 **Strict EM**：在仓库统一的 QA 归一化后，完整预测必须精确等于至少一个完整 gold alias。多别名按“任一别名匹配”处理；NumPy 风格的序列化数组（如 `['alias one' 'alias two']`）必须解析为两个别名，不能拼接为一个字符串。

不再报告或用于决策 Extracted EM／答案包含式匹配。模型回答应遵守短答案格式；完整句即使包含正确实体，也不计 Strict EM 正确。GRPO 仅对 `<answer>` 字段评分，B0/B1/B2 对其完整预测评分。

### 质量

- 每数据集及总体的 Strict EM、token F1。
- 逐题配对的正确数变化：improved / regressed / unchanged。
- 对 step-0 adaptive 与 trained adaptive 做配对显著性检验（如 McNemar 或 paired bootstrap），并报告置信区间。

### 检索策略与成本

- 实际检索轮数分布：0、1、2（或配置允许的更高轮数）轮的比例，以及平均轮数。
- 每轮检索命中率/证据命中率、重复 query 率、无效 action 率。
- 平均与 p95 端到端延迟、检索延迟、生成延迟。
- 每题检索文档数与失败率。

## 运行顺序与产物

1. 生成并冻结 NQ 与 HotpotQA 各 200 条的 ID manifest。
2. 运行 B0/B1/B2 和 GRPO step-0 adaptive，保存逐题 JSONL 与汇总报告。
3. 训练 GRPO，并保存可恢复 checkpoint。
4. 使用完全相同的 manifest 运行 GRPO trained adaptive。
5. 输出按数据集的汇总表、step-0 vs trained 的逐题配对报告、检索轮数/延迟报告及失败案例清单。

## 本次 50-step pilot 的限制

本次 pilot 的验证集为框架以固定 `random_state=42` 抽取的 50 条混合样本，并非“每数据集前 50 条”；各数据集样本数从 4 到 14 不等。它仅验证训练链路、资源配置和奖励流程可运行，不能作为上述正式评测或统计结论的替代品。该 run 还设置了 `save_freq=-1`，没有训练后 checkpoint，因而无法补做逐题的训练后配对评测。
