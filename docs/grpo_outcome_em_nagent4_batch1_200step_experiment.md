# GRPO outcome-only：`n_agent=4`、batch=1 的 200-step 对照

## 目的

这是相对 `grpo_flat_rag_outcome_em_1gpu_200step_20260818_200646` 的独立后续实验，检验每个 GRPO 组使用四条同题 rollout，而不是两个问题各两条 rollout，是否提供更稳定的组内相对奖励信号。

它不是对上一次失败 run 的续训，且不得使用其不存在的 checkpoint。

## 冻结配置

- 配置文件：`configs/grpo_flat_rag_outcome_em_nagent4_batch1_200step.env`。
- 模型与检索：Qwen2.5-3B-Instruct；`intfloat/e5-base-v2 + Wiki-18 + FAISS IndexFlatIP`。
- `GRPO_TRAIN_BATCH_SIZE=1`：每个 optimizer update 只放入一个问题。
- `GRPO_N_AGENTS=4`：对该问题采样四条独立 rollout，构成一个 GRPO 比较组。
- `GRPO_TOTAL_STEPS=200`、`GRPO_TRAIN_DATA_NUM=200`、不 shuffle：一轮恰好 200 updates。
- 总 rollout 预算为 `200 × 1 × 4 = 800`，与前一轮 `200 × 2 × 2 = 800` 相同；改变的是每组的比较结构，不是总采样量。
- 仍使用 outcome-only EM 奖励与最多两轮检索；学习率由上轮的 `5e-7` 降至 `2e-7`，以抑制已观察到的 KL 上升和协议退化。因而这是“更大 rollout 组 + 保守学习率”的稳定性实验，不是严格的单因素 group-size 消融。
- step 100 验证，step 200 保存。训练器以实际 update 编号计数，并在最后一次 update 后、final validation 前保证保存一次 actor；保存是否成功仍须通过日志及 actor 目录实际存在双重确认。

## 成功与停止条件

- 成功只表示训练完成且存在可加载的 `actor/global_step_200`；这不等同于效果提升。
- 训练中监控 reward 非零率、`actor/kl_loss`、无效动作率、格式可解析率与显存。
- 若再次出现持续的混合 token/无效动作或验证全零，停止在该 run 边界，保留日志，不将其用于正式 1,200 条 adaptive 评测。
- 只有保存成功且基本协议稳定时，才在冻结的六数据集各 200 条 manifest 上比较 B0/B1/B2、step-0 adaptive 与 trained adaptive。
