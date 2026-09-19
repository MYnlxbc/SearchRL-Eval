# GRPO outcome-only：`n_agent=4`、batch=1 的 1,000-step 正式首轮

## 固定训练设置

- 配置：`configs/grpo_flat_rag_outcome_em_nagent4_batch1_1000step.env`。
- 模型与检索：Qwen2.5-3B-Instruct；`intfloat/e5-base-v2 + Wiki-18 + FAISS IndexFlatIP`。
- 训练：1,000 个实际 update；每个 update 为一个问题的四条 rollout，合计 4,000 条 rollout。
- 学习率：`2e-7`；纯 outcome extracted-EM reward；最多两轮检索。
- 验证：step 500 与最终 step 1,000（最终会额外运行一次 final validation）。

## checkpoint 与磁盘操作

- 修复后的训练器按实际 update 计数，保存发生在对应验证之前。
- 预期 checkpoint：`actor/global_step_500`、`actor/global_step_1000`。
- 实测单个 actor checkpoint 约 13 GiB；当前磁盘只能同时安全容纳一个。
- step 500 完整落盘后，用户下载该目录并从服务器删除它；训练无需暂停，会继续运行。
- 最终仅保留并验证 `actor/global_step_1000`，再进行固定 1,200 条 adaptive 评测。
