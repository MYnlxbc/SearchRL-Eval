# SearchRL-Eval：单张 A800 远端下载指南

适用配置：Linux 服务器、1×A800 80GB。  
默认根目录优先使用 `/root/autodl-tmp/searchrl_eval`；可通过 `SEARCHRL_ROOT` 修改。

## 1. 上传脚本包

在本地 PowerShell 中执行，替换服务器地址：

```powershell
scp -r "F:\Workspace\finding-intern\project\searchrl_remote_1xa800" user@server:/tmp/
```

不要在聊天中发送 SSH 密码或 Hugging Face Token。

## 2. 进入服务器并设置目录

```bash
ssh user@server
cd /tmp/searchrl_remote_1xa800
chmod +x *.sh

export SEARCHRL_ROOT=/root/autodl-tmp/searchrl_eval
```

如果服务器没有 `/root/autodl-tmp`，改成实际数据盘，例如：

```bash
export SEARCHRL_ROOT=/data/searchrl_eval
```

## 3. 先运行服务器检查

```bash
bash 00_preflight.sh
```

判定规则：

- 可用空间不足 40GB：停止，不下载。
- 40～179GB：只允许下载核心模型和问答数据。
- 180GB 以上：允许下载 Wiki-18；250GB 以上更稳妥。
- 内存低于 96GB：后续不要默认使用 CPU Flat FAISS，应考虑 ANN、小索引或分时使用 GPU。

## 4. 下载核心资产

```bash
bash 01_download_core.sh
```

下载内容：

- `Qwen/Qwen2.5-3B-Instruct`
- `intfloat/e5-base-v2`
- `PeterJinGo/nq_hotpotqa_train`

日志保存在：

```text
$SEARCHRL_ROOT/logs/download_core_*.log
```

## 5. 下载全量 Wiki-18 与 E5 索引

仅在预检显示可用空间不少于 180GB 时执行：

```bash
bash 02_download_wiki18.sh
```

该脚本会：

1. 下载 `part_aa` 与 `part_ab`。
2. 下载 `wiki-18.jsonl.gz`。
3. 合并为 `retrieval/e5_Flat.index`。
4. 解压为 `retrieval/wiki-18.jsonl`。

脚本默认保留原始分片和 `.gz` 文件，不自动删除，防止合并失败后无法恢复。

## 6. 验证下载结果

```bash
bash 03_verify_assets.sh
```

报告输出：

```text
$SEARCHRL_ROOT/manifests/verification_report.json
```

## 7. 单卡 A800 的运行约束

- 下载期间无需占用 GPU。
- 后续基线评测优先采用“Qwen/vLLM 占 GPU + FAISS 占 CPU”的结构。
- 如果 CPU 内存不足，先用小规模检索索引跑通 B0/B1/B2，再处理全量索引。
- 不在同一张卡上同时启动高显存 vLLM、全量 GPU Flat FAISS 和 GRPO 训练。
- 完整 GRPO 训练开始前必须先完成 10～50 steps smoke test。

## 8. 发生错误时

- 不删除 Hugging Face 缓存，重复运行原脚本即可断点续传。
- 不自动重试训练或修改训练代码。
- 将对应日志最后 100 行保存后再定位：

```bash
tail -n 100 "$SEARCHRL_ROOT/logs/对应日志文件.log"
```

