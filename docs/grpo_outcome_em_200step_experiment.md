# Outcome-only GRPO：200-step 首轮实验

## 目的

在已验证可运行的单卡 Flat-FAISS GRPO 链路上，隔离检验纯结果奖励是否能带来稳定的策略更新。该实验不是最终性能训练，也不应据此直接扩展至数千 step。

## 冻结配置

- 配置文件：`configs/grpo_flat_rag_outcome_em_1gpu_200step.env`。
- 模型：`Qwen2.5-3B-Instruct`；检索器：`intfloat/e5-base-v2 + FAISS IndexFlatIP + Wiki-18`。
- 训练集：训练 Parquet 的确定性 400 条随机子集（训练器 `random_state=42`）；batch=2，不 shuffle，共 200 更新。
- 训练策略：GRPO，2 个 rollout agent，最多 2 轮检索。
- 学习率：`5e-7`。
- 奖励：正确抽取答案的 EM 为 1，其他均为 0；`structure_format_score`、`final_format_score`、`retrieval_score` 均为 0。
- 验证：确定性 50 条随机子集，step 100 和训练结束时运行。
- 保存：仅 step 200 保存 actor checkpoint；预留 6--8 GiB 磁盘。

## 判定与后续

训练结束后，不以训练期 50 条验证集作正式结论。应加载 step-200 actor checkpoint，在冻结的六数据集各 200 条 manifest 上运行 GRPO adaptive 评测，并与 step-0 adaptive、B0、B1、B2 对照。先检查非零奖励比例、验证趋势、检索轮数和逐题配对变化，再决定是否增加训练步数。
