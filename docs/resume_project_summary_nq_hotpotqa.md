# 自适应 RAG 检索策略优化项目

## 项目概述

围绕“语言模型应自主决定是否检索、检索几次以及何时回答”的问题，搭建了一个本地可复现的 RAG 与 GRPO 强化学习训练/评测闭环。系统使用 Qwen2.5-3B-Instruct 作为策略模型，基于 `intfloat/e5-base-v2`、2018 Wikipedia（Wiki-18）和 FAISS `IndexFlatIP` 构建知识检索服务；明确不使用 IVF-PQ96。训练后模型通过 `<search>` 与 `<answer>` 动作协议，在 0–2 次检索之间自主选择。

正式结论仅使用 NQ 与 HotpotQA 的固定测试集：每集 200 题、共 400 题，所有方法共享同一题目、检索器、Top-k=3、提示词和贪心解码设置。

## 个人工作

- 构建本地 Wiki-18 RAG 服务：E5 向量编码、FAISS `IndexFlatIP` 检索、CPU 常驻大规模索引与 GPU 推理分工；实现 HTTP 检索接口、Top-k 文档返回和检索日志记录。
- 基于 Search-R1/verl 改造并跑通单卡 GRPO：实现 `<think>/<search>/<information>/<answer>` 多轮环境，最大两轮检索，记录轨迹、检索轮数、延迟、有效动作和 GPU 内存。
- 设计三类基线：B0 无检索、B1 固定一次检索、B2 最多两次检索；冻结 NQ/HotpotQA 各 200 题 manifest，保证所有实验逐题可配对比较。
- 统一实现 Strict EM 评分和多别名解析：归一化后的完整预测必须匹配任一 gold alias；修复 NumPy 风格多别名序列化被错误拼接的问题，避免指标低估。
- 定位纯结果奖励的稀疏反馈与协议退化问题，设计“结果奖励 + 格式规范奖励”消融：合法完整轨迹但答案错误仍获得 0.2，小幅奖励可提取但格式不合法的答案 0.1；正确且合法的 Strict EM 奖励保持 1.0。
- 增强训练工程可靠性：修复 checkpoint 保存边界，支持仅保存指定周期 checkpoint；在单张 A800 80GB、约 20GB 可用磁盘条件下完成训练、checkpoint 完整性验证与离线评测。

## 技术方案

```text
Question
  └─ Qwen2.5-3B policy
       ├─ <answer>：直接输出短答案
       └─ <search>：E5-base-v2 → FAISS IndexFlatIP / Wiki-18 → Top-3 evidence
                                                   └─ policy decides next action (max 2 searches)
```

- 策略模型：Qwen2.5-3B-Instruct。
- 检索：`intfloat/e5-base-v2`，Wiki-18，FAISS `IndexFlatIP`；索引位于 CPU 内存，编码器位于 GPU。
- RL：GRPO，`n_agent=4`，训练 batch size=1，学习率 `2e-7`，最大两轮检索。
- 结果+格式实验：600 步训练，在 step 500 保存并评测 checkpoint；训练使用原始 NQ/HotpotQA train parquet 中采样的 1,000 条数据。
- 评测：NQ/HotpotQA 各 200 题，主指标 Strict EM；同时记录 token F1、检索轮数、延迟、无效动作和空答案。

## 正式结果（Strict EM）

| 方法 | NQ（200） | HotpotQA（200） | 合计（400） |
|---|---:|---:|---:|
| B0：无检索 | 3.0%（6） | 11.0%（22） | 7.00%（28） |
| B1：固定 1 次检索 | 10.0%（20） | 12.5%（25） | 11.25%（45） |
| B2：最多 2 次检索 | 7.0%（14） | 13.5%（27） | 10.25%（41） |
| GRPO：纯结果奖励，step-500 | 15.0%（30） | 22.5%（45） | 18.75%（75） |
| **GRPO：结果+格式奖励，step-500** | **31.0%（62）** | **23.0%（46）** | **27.00%（108）** |

结果+格式奖励模型相对最强固定基线 B1，在 400 题上提升 **15.75 个百分点**（108 vs. 45，增加 63 道严格匹配正确题）；相对纯结果奖励 GRPO 提升 **8.25 个百分点**。其中 NQ 从 15.0% 提升至 31.0%，证明非稀疏格式反馈显著缓解了纯 outcome reward 的训练信号不足。

## 策略与效率

结果+格式奖励 checkpoint 在 400 题上的 token F1 为 35.22%，平均检索轮数为 0.985：NQ 为 0.915，HotpotQA 为 1.055。实际轮数分布为 0/1/2 轮 = 55/296/49，说明策略可以作出不同检索预算的选择，而非被固定为单一基线行为。

该模型在 NQ/HotpotQA 的平均端到端延迟分别为 9.174 秒和 10.819 秒。后续优化方向包括降低无效最终动作率、将检索证据质量纳入低权重奖励，以及在不损害 Strict EM 的前提下缩短轨迹和推理延迟。

## 简历表述（可直接使用）

**自适应 RAG 检索策略优化｜GRPO、FAISS、E5、Qwen2.5**

搭建基于 Wiki-18、E5-base-v2 与 FAISS IndexFlatIP 的本地 RAG 服务，并以 GRPO 训练 Qwen2.5-3B 自主选择 0–2 次检索；设计严格配对的 NQ/HotpotQA 400 题评测与 Strict EM 评分，加入格式规范奖励缓解稀疏 outcome reward，使 Strict EM 从固定单检索基线的 11.25% 提升至 27.00%（+15.75pp），NQ 达到 31.0%。

## 可复现产物

- 结果+格式 checkpoint：`checkpoints/grpo_flat_rag_outcome_em_format_nagent4_batch1_600step_20260821_083813/actor/global_step_500`
- NQ/HotpotQA 逐题评测：`outputs/grpo500_format_adaptive_nq_hotpotqa_20260821_134000.jsonl`
- 基线与纯结果奖励对照：`reports/grpo500_nq_hotpotqa_strict_em_evaluation_20260819.md`
- 实验配置：`configs/grpo_flat_rag_outcome_em_format_nagent4_batch1_600step.env`
