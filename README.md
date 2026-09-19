# SearchRL-Eval

> 面向检索增强生成（RAG）与检索策略学习的可复现实验仓库。

SearchRL-Eval 用统一的评测协议比较 **B0 无检索、B1 固定单轮检索、B2 最多两轮检索**，并评估基于 GRPO 训练的自适应检索策略。项目围绕 `Qwen2.5-3B-Instruct`、`intfloat/e5-base-v2`、Wiki-18 语料和 FAISS `IndexFlatIP` 构建，支持 NQ、HotpotQA、MuSiQue、PopQA、2WikiMultiHopQA、TriviaQA 等数据集。

仓库同时保存了实验配置、逐题 JSONL 输出、汇总报告、训练曲线和实验记录；大模型、检索索引、原始数据集及 checkpoint 需要在本地或服务器上准备，不会提交到 Git。

## 目录

- [项目结构](#项目结构)
- [核心流程](#核心流程)
- [环境与硬件](#环境与硬件)
- [准备本地资产](#准备本地资产)
- [快速开始](#快速开始)
- [运行基线评测](#运行基线评测)
- [生成固定评测集](#生成固定评测集)
- [运行 GRPO smoke 训练](#运行-grpo-smoke-训练)
- [评测自适应 GRPO](#评测自适应-grpo)
- [输出与指标](#输出与指标)
- [测试与故障排查](#测试与故障排查)
- [实验结论](#实验结论)
- [限制与复现注意事项](#限制与复现注意事项)
- [许可证与第三方代码](#许可证与第三方代码)

## 项目结构

```text
configs/
  b0_no_retrieval.yaml                 # B0：直接生成
  b1_single_rag.yaml                   # B1：原问题检索一次
  b2_multi_retrieval.yaml              # B2：最多两轮检索
  paths.env.example                    # 本地路径与服务配置模板
  grpo_flat_rag_*.env                  # 单卡 GRPO 配置

environment/
  searchr1.yml                         # Search-R1 / veRL 训练环境
  retriever.yml                        # FAISS / FastAPI 检索环境

src/searchrl_eval/
  evaluate.py                          # B0/B1/B2 统一评测入口
  eval_utils.py                        # 答案归一化、去重、相似度、超时控制
  schema.py                            # EvalExample / EvalRecord / JSONL schema
  smoke_inference.py                   # Qwen 最小 B0 GPU 推理
  lazy_retrieval_server.py             # 懒加载 Wiki JSONL 的检索服务

tools/
  create_fixed_eval_manifest.py         # 按数据集固定抽样并生成 manifest
  build_ivfpq_index.py                 # 索引构建辅助工具

scripts/
  00_check_assets.sh                   # 资产、GPU、空间和环境预检
  01_create_envs.sh                    # 创建或修复两个 Conda 环境
  02_start_retriever.sh                # 启动本地 /retrieve 服务
  03_smoke_inference.sh                # B0 最小推理 smoke test
  04_stop_retriever.sh                 # 停止检索服务
  05_evaluate.sh                       # 运行 B0/B1/B2 评测
  12_train_grpo_flat_rag_1gpu_smoke.sh # 单卡 GRPO smoke 训练
  14/16_plot_grpo_*.py                 # 生成训练 reward / loss 曲线
  15_evaluate_grpo_adaptive.py          # 评测训练后的自适应检索策略
  asset_setup/                         # 下载、校验模型/数据/Wiki-18 的脚本

vendor/search_r1/                      # Search-R1 / veRL 源码快照及许可证说明
configs/                                # 实验配置
manifests/                              # 固定评测题目及摘要
outputs/                               # 逐题运行输出（JSON/JSONL）
reports/                               # EM、F1、延迟、配对分析和曲线
 docs/                                  # 设计文档、实验记录和复现说明
tests/                                 # 轻量级单元测试和 run_example 测试
```

`vendor/search_r1/` 是训练所依赖的源码快照，不是本项目的核心评测实现。运行训练前，`configs/paths.env` 中的 `SEARCHR1_CODE_DIR` 必须指向该源码目录（或兼容的 Search-R1 checkout）。

## 核心流程

### 固定基线

1. `src/searchrl_eval.evaluate` 从 Parquet、JSON 或 JSONL 读取题目，并统一为 `EvalExample`。
2. `LocalGenerator` 在 GPU 上以 bfloat16 加载 Qwen，并使用 deterministic decoding（`do_sample=false`、`num_beams=1`）。
3. B0 直接回答；B1 先用原问题检索一次；B2 先检索一次，再由模型生成 follow-up query，必要时进行第二轮检索。
4. `RetrieverClient` 调用本地 HTTP `/retrieve` 服务；B2 可依据 Jaccard query similarity、`NONE`、空 query 或无新文档提前停止。
5. 每题通过 `EvalRecord` 立即追加到 JSONL，因此中途失败时仍保留已完成结果；同时记录延迟、token 数、GPU 峰值显存、检索文档和失败 traceback。

### 检索服务

`lazy_retrieval_server.py` 复用 Search-R1 的 `retrieval_server.py`，但将 Wiki JSONL 包装为 `LazyJsonlCorpus`：首次启动时创建紧凑的 `.offsets.u64` 偏移文件，之后通过 mmap 随机读取文档，避免一次性将整份语料加载进 Python 内存。默认设计是 **FAISS Flat 索引留在 CPU，E5 编码器使用 GPU**；单卡配置禁止将约 60 GiB 的 Flat 索引放入 GPU。

### GRPO 自适应策略

训练脚本 `scripts/12_train_grpo_flat_rag_1gpu_smoke.sh` 在 `searchr1` 环境中调用 `verl.trainer.main_ppo_format`，通过 `<search>...</search>` 与 `<answer>...</answer>` 动作进行最多两轮检索。训练后的 checkpoint 使用 `scripts/15_evaluate_grpo_adaptive.py` 评测；该脚本保存动作轨迹、检索轮数、检索文档、Strict EM、token F1 和汇总统计。

## 环境与硬件

项目按 Linux + Conda + NVIDIA GPU 设计，默认目标是 **1×A800 80GB**。仓库定义两个环境：

- `searchr1`：Python 3.9，训练、vLLM、veRL、Search-R1、flash-attn。
- `retriever`：Python 3.10，PyTorch 2.4 / CUDA 12.1、FAISS GPU、Transformers、Datasets、FastAPI、Uvicorn。

主要版本约束见 [`environment/searchr1.yml`](environment/searchr1.yml)、[`environment/retriever.yml`](environment/retriever.yml) 和 `scripts/01_create_envs.sh`。完整 Wiki-18 Flat 索引约需 64.56 GB，解压后的语料约需 14.39 GB；建议数据盘至少预留 180 GB，且主机内存充足。

## 准备本地资产

先复制路径模板并按服务器实际路径修改。不要把 Token、密码或服务器私密路径提交到 Git。

```bash
cp configs/paths.env.example configs/paths.env
$EDITOR configs/paths.env
source configs/paths.env
```

默认路径变量包括：

| 变量 | 用途 |
|---|---|
| `SEARCHRL_ROOT` | 模型、数据、索引、checkpoint 和缓存根目录 |
| `SEARCHR1_CODE_DIR` | Search-R1 / veRL 源码目录 |
| `MODEL_PATH` | Qwen2.5-3B-Instruct 目录 |
| `RETRIEVER_MODEL_PATH` | E5-base-v2 目录 |
| `QA_DATA_DIR` | NQ/HotpotQA 等 Parquet 数据目录 |
| `INDEX_PATH` | `retrieval/e5_Flat.index` |
| `CORPUS_PATH` | `retrieval/wiki-18.jsonl` |
| `RETRIEVER_URL` | 默认 `http://127.0.0.1:8000/retrieve` |
| `FAISS_GPU` | 单卡默认必须为 `0` |

可使用 `scripts/asset_setup/` 下载和校验核心资产。该目录的说明针对远程 Linux/A800 环境；下载全量 Wiki-18 前应先运行预检。

```bash
bash scripts/asset_setup/00_preflight.sh
bash scripts/asset_setup/01_download_core.sh
# 磁盘空间 >= 180 GB 时再执行
bash scripts/asset_setup/02_download_wiki18.sh
bash scripts/asset_setup/03_verify_assets.sh
```

## 快速开始

以下流程假设已经准备好 GPU、Conda、模型、数据、Wiki-18 语料和 Flat 索引：

```bash
cp configs/paths.env.example configs/paths.env
$EDITOR configs/paths.env

# 创建两个环境；脚本会先执行资产预检
bash scripts/01_create_envs.sh --yes

# 检查模型、数据、索引、GPU 和磁盘
bash scripts/00_check_assets.sh

# 先验证 Qwen 可正常生成答案
bash scripts/03_smoke_inference.sh

# B1/B2 需要检索服务；B0 不需要
bash scripts/02_start_retriever.sh

# 用真实输入运行评测（示例见下一节）
bash scripts/05_evaluate.sh \
  --config configs/b1_single_rag.yaml \
  --input /path/to/test.parquet \
  --limit 10

# 完成后停止服务
bash scripts/04_stop_retriever.sh
```

`03_smoke_inference.sh` 默认提问 “Who wrote the novel Pride and Prejudice?”，也可以自定义：

```bash
QUESTION='Who wrote the novel Pride and Prejudice?' \
EXPECTED_ANSWER='Jane Austen' \
bash scripts/03_smoke_inference.sh
```

## 运行基线评测

统一入口接受 `.parquet`、`.json` 和 `.jsonl`，并支持 `--dataset`、`--start-index`、`--limit`。B0 不需要运行检索服务；B1/B2 必须先启动服务并确保 `RETRIEVER_URL` 可访问。

```bash
# B0：无检索
bash scripts/05_evaluate.sh \
  --config configs/b0_no_retrieval.yaml \
  --input /path/to/test.parquet \
  --output outputs/b0_test.jsonl \
  --limit 200

# B1：原问题检索一次
bash scripts/05_evaluate.sh \
  --config configs/b1_single_rag.yaml \
  --input /path/to/test.parquet \
  --output outputs/b1_test.jsonl \
  --dataset nq \
  --limit 200

# B2：最多两轮检索，启用 follow-up query
bash scripts/05_evaluate.sh \
  --config configs/b2_multi_retrieval.yaml \
  --input /path/to/test.parquet \
  --output outputs/b2_test.jsonl \
  --limit 200
```

默认配置使用 Qwen 本地模型、bfloat16、SDPA、固定 seed=42，并将每题超时设为 B0/B1 300 秒、B2 600 秒。`scripts/05_evaluate.sh` 会在 `searchr1` 环境中运行，并强制 Hugging Face 离线模式。

## 生成固定评测集

正式比较必须让 B0/B1/B2/GRPO 使用完全相同的题目、顺序和 gold answers。`tools/create_fixed_eval_manifest.py` 默认从六个数据集中各抽取 200 题，seed 为 `20260818`，同时生成 JSONL manifest 和 `.summary.json`：

```bash
python tools/create_fixed_eval_manifest.py \
  --input /path/to/nq_hotpotqa_train/test.parquet \
  --output manifests/eval200x6.jsonl \
  --per-dataset 200 \
  --seed 20260818
```

也可以只选择 NQ 与 HotpotQA：

```bash
python tools/create_fixed_eval_manifest.py \
  --input /path/to/test.parquet \
  --output manifests/eval200_nq_hotpotqa.jsonl \
  --datasets nq hotpotqa \
  --per-dataset 200
```

要在不同方法之间重新随机抽样，也不要用旧的 smoke50 输出替代固定 manifest 的正式结果。

## 运行 GRPO smoke 训练

训练前必须完成检索服务预检、训练/验证 Parquet 检查和 B0 smoke test。默认 smoke 配置只使用 8 条训练数据、4 条验证数据和 2 个 update，用于验证链路，不代表正式训练结果。

```bash
# 查看配置
bash scripts/12_train_grpo_flat_rag_1gpu_smoke.sh --print-config

# 只做训练前检查
bash scripts/12_train_grpo_flat_rag_1gpu_smoke.sh --check

# 执行最小训练
bash scripts/12_train_grpo_flat_rag_1gpu_smoke.sh
```

可通过 `GRPO_CONFIG` 指定其他配置：

```bash
GRPO_CONFIG=configs/grpo_flat_rag_outcome_em_nagent4_batch1_200step.env \
bash scripts/12_train_grpo_flat_rag_1gpu_smoke.sh
```

训练输出默认位于 `${SEARCHRL_ROOT}/checkpoints/`，日志位于 `${LOG_DIR}/`。单卡运行时不要同时启动高显存 vLLM、GPU Flat FAISS 和训练；当前策略是 Flat FAISS 使用 CPU 内存、检索编码器使用 GPU。

## 评测自适应 GRPO

训练后 checkpoint 使用与训练协议一致的动作格式进行评测，而不是把训练后的模型人为拆成 B0/B1/B2：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

conda run --no-capture-output -n searchr1 \
  python scripts/15_evaluate_grpo_adaptive.py \
  --checkpoint /path/to/checkpoints/actor/global_step_500 \
  --manifest manifests/eval200x6.jsonl \
  --output outputs/grpo500_adaptive.jsonl \
  --retriever-url "$RETRIEVER_URL" \
  --topk 3 \
  --limit 1200
```

常用选项：

- `--datasets nq,hotpotqa`：只评测指定 `data_source`。
- `--resume`：从已有输出中跳过已完成的 `question_id`。
- `--limit N`：限制本次处理数量。

该评测最多允许两次搜索，最多再进行一次最终答题尝试；每题会记录 `raw_trajectory`、`valid_actions`、`valid_searches`、`retrieval_stop_reason`、Strict EM 和 token F1。

## 输出与指标

### 逐题输出

`outputs/*.jsonl` 中每行是一道题的 `EvalRecord`，主要字段包括：

- 题目与答案：`dataset`、`question_id`、`question`、`gold_answers`、`prediction`。
- 评测状态：`status`、`error_type`、`error_message`、`timed_out`、`timeout_stage`。
- 检索轨迹：`retrieved_documents`、`retrieval_rounds`、`followup_query`、`retrieval_stop_reason`、`duplicate_documents_filtered`。
- 成本信息：`latency_seconds`、`retrieval_latency_seconds`、`generation_latency_seconds`、`prompt_tokens`、`generated_tokens`、`peak_gpu_memory_gib`。

每个运行还会写出同名 `.summary.json`，包括成功/失败/超时计数、停止原因分布和总耗时。

### 评分口径

项目统一使用 `normalize_answer_for_metrics()` 做 QA 归一化：大小写折叠、去标点、去除英文冠词并规范空白。正式主指标是 **Strict EM**：归一化后的完整预测必须精确等于任一完整 gold alias。不要使用“答案包含正确实体”的 Extracted EM 代替 Strict EM 做正式结论。

GRPO 额外报告 token F1、有效动作/搜索次数、检索轮数分布和延迟。比较实验应至少报告：

- 每个数据集及总体 Strict EM、token F1。
- 逐题 improved / regressed / unchanged，必要时做 McNemar 或 paired bootstrap。
- 实际检索轮数 0/1/2 的分布、平均轮数、重复 query 和无效 action 比例。
- 平均及 p95 的端到端、检索和生成延迟。
- 失败、超时和空答案案例。

## 测试与故障排查

轻量测试不需要真实模型或 GPU；测试文件会为重量级依赖安装 import stub。推荐：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests -v
```

常见问题：

1. **缺少 `configs/paths.env`**：复制 `configs/paths.env.example`，并确认所有路径已存在。
2. **资产检查失败**：先看 `bash scripts/00_check_assets.sh` 的 `[FAIL]` 项；完整 Flat 索引和语料会做最小文件大小检查。
3. **检索服务未就绪**：检查 `${LOG_DIR}/retriever_*.log`，确认端口 8000、E5 `config.json`、索引和语料存在。
4. **CPU 内存不足**：不要把 Flat FAISS 索引切换到 GPU；先使用更小索引或降低实验规模。
5. **模型下载错误**：评测脚本强制离线模式，需先在 `MODEL_PATH` 和 `RETRIEVER_MODEL_PATH` 准备完整本地权重。
6. **训练显存/内存不足**：先运行 smoke 配置，降低 `GRPO_VLLM_GPU_MEMORY_UTILIZATION`，并确认没有并行运行其他 GPU 服务。
7. **中途停止**：评测 JSONL 是逐题 flush；可使用已有输出和 GRPO 的 `--resume` 继续。
8. **超时**：`ExampleDeadline` 会在检索和生成阶段记录超时 stage，但不会强制中断已经运行的 CUDA kernel；应结合日志定位。

## 实验结论

仓库中的正式报告说明了当前方法的边界，不能把 smoke 结果当成最终结论。例如 `reports/grpo500_nq_hotpotqa_strict_em_evaluation_20260819.md` 报告了 NQ/HotpotQA 各 200 题的结果：GRPO step-500 的总体 Strict EM 为 18.75%，高于 B1 的 11.25% 和 B2 的 10.25%，但平均检索轮数仍约为 0.983，行为接近固定单轮检索。

在六数据集、共 1,200 题的报告中，GRPO step-500 总体 Strict EM 为 10.25%，低于 B1 的 15.75% 和 B2 的 16.00%；TriviaQA、PopQA、2WikiMultiHopQA 上出现明显退化。因此，GRPO 是否优于基线必须结合固定 manifest、数据集分项结果和配对分析判断，不能只引用 NQ/HotpotQA 子集。

详细设计与实验记录：

- [`docs/grpo_adaptive_evaluation_design.md`](docs/grpo_adaptive_evaluation_design.md)
- [`docs/grpo_outcome_em_nagent4_batch1_1000step_experiment.md`](docs/grpo_outcome_em_nagent4_batch1_1000step_experiment.md)
- [`reports/grpo500_nq_hotpotqa_strict_em_evaluation_20260819.md`](reports/grpo500_nq_hotpotqa_strict_em_evaluation_20260819.md)
- [`reports/grpo500_final_strict_em_evaluation_20260819.md`](reports/grpo500_final_strict_em_evaluation_20260819.md)

## 限制与复现注意事项

- `pyproject.toml` 当前为空，项目运行依赖由两个 Conda 环境文件和 `scripts/01_create_envs.sh` 管理；不要假定 `pip install -e .` 能完成全部安装。
- 仓库没有提交大模型权重、Wiki-18、FAISS 索引、原始数据、训练 checkpoint、cache、logs 或 wandb 运行目录。
- 评测依赖 CUDA、NVIDIA GPU、Transformers、Datasets、Requests、PyYAML；检索服务额外依赖 FAISS、FastAPI 和 Uvicorn。
- `configs/` 中的 `.env` 文件是实验参数，不应写入访问 Token、密码等敏感信息。
- 完整复现需要固定模型版本、索引版本、数据版本、manifest、seed、配置、硬件和软件环境。
- `vendor/search_r1/` 含第三方源码，请同时遵守其 `LICENSE` 和 `Notice.txt`。

## 许可证与第三方代码


