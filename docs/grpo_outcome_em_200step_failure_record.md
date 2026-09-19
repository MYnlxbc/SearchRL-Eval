# GRPO pure outcome-only 200-step：失败/不稳定性记录

## 实验身份

- Run：`grpo_flat_rag_outcome_em_1gpu_200step_20260818_200646`
- 配置：`configs/grpo_flat_rag_outcome_em_1gpu_200step.env`
- 模型：Qwen2.5-3B-Instruct。
- 检索：intfloat/e5-base-v2 + Wiki-18 + FAISS `IndexFlatIP`。
- 训练：400 条确定性训练子集，batch=2，`n_agent=2`，200 updates，最多两轮检索。
- 奖励：correct extracted-answer EM=`1.0`，其余=`0.0`；`structure_format_score=0`、`final_format_score=0`、`retrieval_score=0`。

## 已确认的现象

训练末段主日志中出现大量 assistant rollout 的无意义混合 token、多语言碎片词和不完整标记。例如模型在 assistant 位置生成随机词串，而非 `<think>`、`<search>` 或 `<answer>` 协议。

环境随后打印：

```text
My previous action is invalid. If I want to search, I should put the query
between <search> and </search>. If I want to give the final answer, I should
put the answer between <answer> and </answer>.
```

这表示环境检测到了模型动作无效并要求重试；它不是检索服务错误、语料损坏或终端编码造成的输入乱码。

日志中的 `\u001b[36m ... \u001b[0m` 是 Ray 的 ANSI 颜色控制符，显示异常但无运行语义；`<|im_start|>` / `<|im_end|>` 是 Qwen chat template token，也属正常。

## 训练指标背景

在中段已观察到：

- 纯 outcome reward 并非全零，但高度稀疏且 batch 方差大。
- 有效动作/正常结束率曾上升。
- `actor/kl_loss` 从前 20 step 平均约 `0.0071` 升至后续 20 step 平均约 `0.2576`。
- 当时 `kl_loss_coef=0.001`，因此其直接损失项仍较小；但上升趋势说明策略正显著偏离 reference model。

最终 checkpoint 的正式固定评测尚未完成前，不能只凭训练日志宣称整体质量下降；但无效动作与词串生成已经构成明确的协议稳定性风险。

## 最终状态与可复现证据

- 主训练进程已退出；训练日志与 launcher 日志均在 final validation 后停止。
- 最后可见输出是六个验证集均为 `0.0`：NQ、MuSiQue、HotpotQA、PopQA、2WikiMultiHopQA、TriviaQA。
- 日志中没有 `Saving actor checkpoint` 或保存完成记录；预期目录
  `checkpoints/grpo_flat_rag_outcome_em_1gpu_200step_20260818_200646/actor/global_step_200`
  不存在。因此本次 run **没有可用于正式 adaptive 评测的最终 actor checkpoint**。
- 日志末尾没有 Python traceback、CUDA OOM 或 `Killed` 的明确文本。后续代码审计已确认漏存的直接原因：训练循环在首个 batch 前把 step 预增为 1，最后实际更新到 step 199 后即退出；`save_freq=200` 因而从未命中。这不是磁盘、显存或验证结果本身导致的保存失败。
- 原始日志保留了发生过程；其中末段的 invalid-action 提示和混合 token 可直接复核。不要删除或覆盖这些日志。

## 根因判断

最可能是纯 0/1 EM 奖励与当前多轮工具协议不匹配：

1. “答错但格式正确”和“答错且动作无效”都得到 0，缺乏维持协议格式的相对学习信号。
2. EM 正例稀疏，GRPO 很多 group 内没有有效的奖励排序。
3. 学习率 `5e-7` 加上弱 KL 约束，可能加速了对初始策略的漂移。
4. 训练时 temperature=1.0 的多 agent rollout 放大了高方差探索。

这是一项因果假设，而非已经由消融实验严格证明的单一根因。

## 处置与解释

- 本次没有完成最终 checkpoint 保存，故不能作为可评测模型，也不能补跑“最后保存”来伪装成同一个受控 run。
- 将本次保留为 outcome-only 的失败/不稳定性记录；可复现资产是配置、离线 W&B 记录和原始日志，而非 actor 权重。
- 后续实验必须使用新的 run ID；不要在这个已退出 run 的中途状态上修改奖励、学习率或 rollout 数。
- 训练器已修复为按实际 update 编号计数，并在启用保存时于最后一次 update、任何验证之前保证一次 actor 保存；历史 run 不能据此补出不存在的 checkpoint。

## 下一轮建议

下一轮不再使用完全无 shaping 的 0/1 reward。保持 EM 的最高奖励为 1，并增加受限的辅助信号：

```text
EM exact correct:                  1.00
valid, parseable protocol:         +0.05  (仅错误轨迹的低权重辅助项)
retrieved evidence contains gold:  +0.15  (仅错误轨迹的低权重辅助项)
Token F1 of <answer> field:        +0.30 * F1, capped below EM reward
```

实施前需要：

1. 修复多个 gold alias 被序列化为单字符串的问题。
2. 只对 `<answer>` 字段计算 Token F1，避免用长文本堆词获得奖励。
3. 降低学习率到 `2e-7`，并设定 KL、无效动作率和 reward 非零率的早停监控。
4. 将 `n_agent=4, batch_size=1` 作为独立后续消融，而非与奖励重设计同时改变。

## 相关产物

- 主日志：`logs/grpo_flat_rag_outcome_em_1gpu_200step_20260818_200646.log`
- launcher 日志：`logs/grpo_flat_rag_outcome_em_1gpu_200step_launcher_20260818_200646.log`
- 训练配置：`configs/grpo_flat_rag_outcome_em_1gpu_200step.env`
- 实验设计：`docs/grpo_outcome_em_200step_experiment.md`
- 离线 W&B：`../Search-R1/wandb/offline-run-20260818_200704-ypl3eerv/`
