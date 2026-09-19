# SearchRL-Eval README 草稿

## 项目目标

在固定模型、数据切片和生成参数下，复现并比较无检索、单轮 RAG、多轮检索及后续 GRPO 奖励变体。项目强调可复现输出、异常保留和配对评测，不覆盖官方 `Search-R1` 源码。

## 环境与资源

- 远端目录：`/root/autodl-tmp/searchrl_eval/SearchRL-Eval`
- GPU：1×A800 80GB
- `searchr1`：Python 3.9、Torch 2.4、vLLM 0.6.3、Qwen2.5-3B-Instruct
- `retriever`：Python 3.10、E5-base-v2、FAISS CPU 索引
- Wiki-18：JSONL offsets 懒加载；禁止生成大型 Arrow 缓存

## 目录约定

```text
configs/    实验配置
scripts/    启动、评测、验证和指标脚本
src/        SearchRL-Eval 实现
outputs/    每条样本一行的 JSONL 与 summary.json
reports/    CSV/JSON/Markdown 汇总和人工审查
logs/       服务与运行日志
```

官方 `Search-R1` 仓库保持只读，不在其中修改实验代码。

## 快速开始

```bash
cd /root/autodl-tmp/searchrl_eval/SearchRL-Eval

# 1. 检查资源与环境
bash scripts/00_check_assets.sh

# 2. 启动检索器（单独终端）
bash scripts/02_start_retriever.sh

# 3. 运行单条最小推理
bash scripts/03_smoke_inference.sh

# 4. 运行评测；输入数据固定为同一 Parquet 与同一批 ID
bash scripts/05_evaluate.sh \
  --config configs/b2_1_deduplicated.yaml \
  --input /root/autodl-tmp/searchrl_eval/datasets/raw/nq_hotpotqa_train/test.parquet \
  --dataset nq --limit 50 \
  --output outputs/example.jsonl

# 5. 验证输出行数、重复 ID、错误文档和异常字段
find outputs -maxdepth 1 -type f -name 'example.jsonl' -exec \
  env PYTHONPATH=src conda run --no-capture-output -n searchr1 \
  python scripts/06_validate_b2_1.py {} --expected-count 50 \;
```

## 输出约定

每条 JSONL 记录至少包含：`question_id`、`dataset`、`question`、`gold_answers`、`prediction`、`retrieved_documents`、`retrieval_rounds`、`retrieval_stop_reason`、`latency_seconds`、`status`。summary 文件记录成功、失败、超时、空预测、停止原因和总耗时。

## 已完成结果

第一版 B0/B1/B2 表见 `SearchRL-Eval第一版结果表_2026-07-21.md`。B2.1 已在 NQ/HotpotQA 各 50 条上通过结构验证，但质量指标未超过 B2，因此不直接扩大到 500 条。

## 下一阶段：GRPO 奖励消融

当前检索基线用于提供稳定评测入口；主研究线切换为 GRPO 奖励设计：

1. 正确性奖励基线；
2. 正确性 + 格式奖励；
3. 正确性 + 证据/grounding 奖励；
4. 联合奖励及去掉单项奖励的消融。

每个变体固定模型、数据、rollout、训练步数、seed 和评测 ID，并同时记录 EM、Token F1、格式合法率、证据命中、奖励分解、长度、耗时和显存。

## 可复现要求

- 不删除失败输出和日志；
- 不覆盖已有结果文件；
- 记录配置路径、模型 revision、数据版本、随机种子和 SHA-256；
- 任何结论必须能由 `outputs/`、`reports/` 和对应命令复核。
