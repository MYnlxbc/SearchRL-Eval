# GRPO：结果奖励 + 格式规范奖励（200-step）

## 目的

在 outcome-only 训练出现稀疏奖励和协议退化后，隔离检验格式奖励是否能提高轨迹合法性与训练稳定性。唯一相对上一轮的算法改动是格式奖励；检索证据奖励仍为 0。

## 数据与训练

- 训练：NQ 100 + HotpotQA 100，分层、固定 seed `20260821`，共 200 条。
- 验证：NQ 50 + HotpotQA 50，分层、同一 seed 派生。
- 200 updates；每 update 1 个问题 × 4 条 rollout，合计 800 条 rollout。
- `n_agent=4`、`train_batch_size=1`、学习率 `2e-7`、最多 2 轮检索。
- 只保存最后 `global_step_200` checkpoint。

## 奖励

实现：`verl/utils/reward_score/qa_em_format.py`。

| 轨迹状态 | 总奖励 |
|---|---:|
| 合法完整格式 + Strict EM 正确 | 1.0 |
| Strict EM 正确但格式不合法 | 0.8 |
| 合法完整格式但答案错误/未提取到答案 | 0.2 |
| 格式不合法但提取到最终答案且错误 | 0.1 |
| 其他无效轨迹 | 0.0 |

## 成功判据

1. 无 NaN/OOM，最终 checkpoint 可加载；
2. `reward mean` 不长期退化为 0，`actor/kl_loss` 不持续上升；
3. 无效动作率低于 outcome-only 后段；
4. 在冻结 NQ/HotpotQA 各 200 条测试集上，以 Strict EM 对比 B0/B1/B2 与 outcome-only checkpoint 的历史结果。
