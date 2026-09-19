# 自适应 RAG 检索策略优化项目：六数据集综合总结

## 项目简介

搭建基于 2018 Wikipedia（Wiki-18）的本地大规模 RAG 检索服务，并使用 GRPO 训练 Qwen2.5-3B-Instruct 自主决定是否检索、检索次数和最终回答时机。检索器采用 `intfloat/e5-base-v2 + FAISS IndexFlatIP`，明确不使用 IVF-PQ96。

评测使用冻结 manifest，覆盖 NQ、HotpotQA、MuSiQue、PopQA、2WikiMultiHopQA、TriviaQA 六个数据集，每个数据集 200 题，共 1,200 题。所有方法使用相同的 Wiki-18 Flat 索引、Top-k=3、提示词、解码与 Strict EM 评分器。

## 主要工作

- 构建 E5 向量编码、FAISS `IndexFlatIP` 检索、Wiki-18 JSONL 语料懒加载和 HTTP 检索服务；将约 60 GiB 索引放在 CPU 内存，GPU 仅负责 E5 编码与生成。
- 将 Search-R1/verl 改造成可执行 `<think>/<search>/<information>/<answer>` 的多轮 RAG 环境，策略最多自主检索两轮。
- 实现 B0（0 轮）、B1（固定 1 轮）、B2（最多 2 轮）基线，以及单一自适应 GRPO 策略的统一评测。
- 冻结六数据集 1,200 题 manifest，保存逐题预测、检索文档、查询、检索轮数、延迟、有效动作和错误信息，支持逐题配对比较。
- 修复 Strict EM 的多别名解析：NumPy 风格 `['alias1' 'alias2']` 被解析为独立别名，完整归一化预测必须精确匹配任一 alias。
- 针对纯 outcome reward 的奖励稀疏与协议退化，增加格式规范奖励，形成“结果奖励 + 格式奖励”消融实验。

## 技术配置

- 生成模型：Qwen2.5-3B-Instruct。
- 检索模型：`intfloat/e5-base-v2`。
- 知识库：2018 Wikipedia（Wiki-18）。
- 索引：FAISS `IndexFlatIP`，CPU 常驻；Top-k=3。
- 强化学习：GRPO，`n_agent=4`，训练 batch size=1，学习率 `2e-7`，最多两轮检索。
- 结果+格式实验：600-step 预算，在 step 500 保存并评测 checkpoint；实际训练使用原始 train parquet 前 1,000 条（由于文件按来源排序，该训练切片为 NQ）。
- 主指标：Strict EM；辅助指标：Token F1、检索轮数、延迟、无效动作与空答案率。

## 六数据集 Strict EM 结果

| 数据集（各 200 题） | B0 | B1 | B2 | GRPO 纯结果 step-500 | GRPO 结果+格式 step-500 |
|---|---:|---:|---:|---:|---:|
| NQ | 3.0%（6） | 10.0%（20） | 7.0%（14） | 15.0%（30） | **31.0%（62）** |
| HotpotQA | 11.0%（22） | 12.5%（25） | 13.5%（27） | 22.5%（45） | **23.0%（46）** |
| MuSiQue | 1.0%（2） | 2.0%（4） | 2.5%（5） | **3.0%（6）** | 2.5%（5） |
| PopQA | 11.5%（23） | 23.0%（46） | 23.5%（47） | 11.0%（22） | **32.5%（65）** |
| 2WikiMultiHopQA | **17.5%（35）** | 11.0%（22） | 13.5%（27） | 8.5%（17） | 15.0%（30） |
| TriviaQA | 23.0%（46） | 36.0%（72） | 36.0%（72） | 1.5%（3） | **53.5%（107）** |
| **总体（1,200）** | **11.17%（134）** | **15.75%（189）** | **16.00%（192）** | **10.25%（123）** | **26.25%（315）** |

结果+格式奖励相对 B2：

- 总体 Strict EM：`16.00% → 26.25%`，提升 **10.25 个百分点**，多答对 123 题。
- NQ：+24.0 个百分点；HotpotQA：+9.5 个百分点。
- PopQA：+9.0 个百分点；TriviaQA：+17.5 个百分点。
- 2WikiMultiHopQA：+1.5 个百分点；MuSiQue 持平。

结果+格式奖励相对纯结果奖励 GRPO：总体提升 **16.0 个百分点**（10.25% → 26.25%），其中 TriviaQA 提升 52.0 个百分点、PopQA 提升 21.5 个百分点；MuSiQue 下降 0.5 个百分点。

## 辅助指标

结果+格式 checkpoint 在 1,200 题上的平均 Token F1 为 **32.01%**。实际检索轮数为：0 轮 153 题、1 轮 842 题、2 轮 205 题，平均 **1.043 轮**。这说明模型已经能够选择不同检索预算，但整体仍偏向一次检索。

## 结论与局限

格式规范奖励明显缓解了纯 outcome reward 的稀疏反馈问题，使模型在总体六数据集测试中超过固定检索基线，尤其改善了 NQ、PopQA 和 TriviaQA。MuSiQue 仍然困难，2WikiMultiHopQA 仅略高于 B2。

需要诚实说明：训练切片前 1,000 条实际全部来自 NQ，因此六数据集结果主要反映 NQ 训练后的跨数据集迁移，而不是六数据集混合训练。下一步应使用分层 NQ/HotpotQA 或六数据集混合训练，并增加证据命中、有效动作和空答案的低权重 shaping，同时控制 KL 与无效动作率。

## 简历可直接使用

**自适应 RAG 检索策略优化｜GRPO、FAISS、E5、Qwen2.5**

搭建基于 Wiki-18、E5-base-v2 与 FAISS IndexFlatIP 的本地大规模 RAG 服务，将 Qwen2.5-3B-Instruct 改造成可自主选择 0–2 次检索的 GRPO 策略；设计六数据集 1,200 题冻结评测与 Strict EM 评分，加入格式规范奖励缓解稀疏 outcome reward，使总体 Strict EM 从固定多轮检索 B2 的 16.00% 提升至 26.25%（+10.25pp），并在 NQ、PopQA、TriviaQA 上取得显著提升。

## 可复现产物

- 六数据集基线：`outputs/b0_eval200x6_20260818_151628.jsonl`、`b1_...jsonl`、`b2_...jsonl`。
- 纯结果奖励 GRPO：`outputs/grpo500_adaptive_eval200x6_20260819_162320.jsonl`。
- 结果+格式奖励 GRPO：`outputs/grpo500_format_adaptive_nq_hotpotqa_20260821_134000.jsonl` 与 `outputs/grpo500_format_adaptive_remaining4_20260821_142000.jsonl`。
- checkpoint：`checkpoints/grpo_flat_rag_outcome_em_format_nagent4_batch1_600step_20260821_083813/actor/global_step_500`。
