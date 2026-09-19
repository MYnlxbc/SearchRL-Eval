# GRPO：结果奖励 + 格式规范奖励（1,000-step 对照实验）

本实验与 `grpo_flat_rag_outcome_em_nagent4_batch1_1000step` 保持完全相同的训练设置：原始 `nq_hotpotqa_train/train.parquet`、前端配置的 1,000 条训练数据、50 条验证数据、1,000 steps、`n_agent=4`、batch=1、学习率 `2e-7`、最多 2 轮检索、seed `1000` 和 checkpoint 间隔 500。

唯一改动是奖励：

| 轨迹状态 | 旧 outcome-only | 本实验 |
|---|---:|---:|
| 合法完整格式 + Strict EM 正确 | 1.0 | 1.0 |
| Strict EM 正确但格式不合法 | 1.0 | 0.8 |
| 合法完整格式但答案错误或无可提取答案 | 0.0 | 0.2 |
| 格式不合法但有错误最终答案 | 0.0 | 0.1 |
| 其他无效轨迹 | 0.0 | 0.0 |

`retrieval_score=0`，因此没有引入证据命中等额外奖励变量。数据路径及其实际前 1,000 条仍全为 NQ；这是与上一轮的刻意对齐，不是 NQ/HotpotQA 混合训练。

磁盘只能同时容纳一个约 13 GiB 的 actor checkpoint：在 `global_step_500` 生成后需先下载、验证、删除，再继续保留最终 `global_step_1000`。
