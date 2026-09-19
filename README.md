# SearchRL-Eval

基于 Wiki-18、FAISS `IndexFlatIP`、`intfloat/e5-base-v2` 和 Qwen2.5-3B-Instruct 的 RAG 检索与 GRPO 自适应检索评测项目。

项目包含：

- B0（无检索）、B1（单轮检索）、B2（多轮检索）基线；
- 自适应 0–2 次检索的 GRPO 训练与评测脚本；
- NQ、HotpotQA 及其他数据集的评测输出；
- Strict EM / Extracted EM 报告、训练曲线和实验配置；
- `vendor/search_r1/` 中用于训练的 Search-R1/veRL 源码快照。

## 重要资产

由于 GitHub 不适合存放几十 GB 的二进制文件，以下资产不会进入 Git：

- `retrieval/e5_Flat.index`
- `retrieval/wiki-18.jsonl` 及 offsets
- `models/` 中的 Qwen 和 E5 权重
- `datasets/` 中的原始数据集
- `checkpoints/` 中的 GRPO checkpoint
- `cache/`、`logs/`、`wandb/` 和临时运行状态

这些路径由 `.gitignore` 排除。请先准备本地资产，再复制配置模板：

```bash
cp configs/paths.env.example configs/paths.env
# 按实际服务器路径编辑 configs/paths.env
source configs/paths.env
```

Wiki-18 和模型的下载/校验辅助脚本位于 `scripts/asset_setup/`。完整运行前应先完成资产校验和小规模 smoke test。

## 基本用法

```bash
pip install -e .
bash scripts/00_check_assets.sh
bash scripts/02_start_retriever.sh
bash scripts/05_evaluate.sh
bash scripts/04_stop_retriever.sh
```

具体实验参数保存在 `configs/`，评测结果在 `outputs/`，汇总报告在 `reports/`，项目设计与实验记录在 `docs/`。

本项目使用 FAISS Flat 索引，不使用 IVF-PQ96。
