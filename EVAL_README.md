# DLLM-Searcher 评估运行指南

## 一、快速评估流程（Dataroller + 任意 OpenAI 兼容 API）

适用于：已有 SGLang / vLLM / OpenAI 等推理服务时，跑通评估 pipeline。

### 1. 环境与配置

```bash
conda activate dllmsearch
```

确保 `config.json` 已配置：
- `google_search_key`：Serper API Key（必填）
- `openai_api_base`：推理服务地址，如 `http://localhost:30000/v1`（SGLang 默认）
- `openai_api_key`：API Key，本地 SGLang 可填 `arbitrary`

### 2. 启动推理服务（二选一）

**方式 A：SGLang 本地部署**

```bash
# 新终端，激活含 SGLang 的环境后：
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 30000 \
  --context-length 32768 \
  --allow-auto-truncate
```

> **重要**：ReAct 多轮对话会累积大量 token，需提高最大长度：
> - `--context-length 32768`：显式支持 32k 上下文，与 agent 的 MAX_LENGTH (~31k) 匹配
> - `--allow-auto-truncate`：超长时自动截断，避免 400 报错
> - 若仍 OOM，可加 `--mem-fraction-static 0.85` 降低 KV cache 占用

**或使用项目自带脚本**（需在项目根目录执行）：

```bash
cd /research/cbim/vast/mz751/Projects/DLLM-Searcher
bash scripts/start_sglang.sh
# 或指定 16k：bash scripts/start_sglang.sh meta-llama/Llama-3.1-8B-Instruct 30000 16384
```

**方式 B：使用云端 API**

在 `config.json` 中设置 `openai_api_base` 和 `openai_api_key` 指向你的 API。

### 3. 运行 Rollout（生成预测）

```bash
cd Dataroller

# 使用示例数据（3 条）
bash run.sh "Llama-3.1-8B" example base

# 输出目录：base/Llama-3.1-8B_sglang/example/iter1.jsonl
```

- 第 1 个参数：模型名（与推理服务中一致）
- 第 2 个参数：数据集名（对应 `data/{dataset}.jsonl`）
- 第 3 个参数：输出根目录

### 4. 评估准确率

```bash
cd my_eval
# 在 cal_acc.py 中设置 API_KEY、API_BASE_URL、MODEL_NAME（用于 LLM Judge）
python cal_acc.py
```

需在 `cal_acc.py` 中修改：
- `data_path`：指向 rollout 输出，如 `../Dataroller/base/Llama-3.1-8B_sglang/example/iter1.jsonl`
- `API_KEY`、`API_BASE_URL`、`MODEL_NAME`：LLM Judge 所用的 API

---

## 二、评估数据格式

**输入**（`Dataroller/data/example.jsonl`）：

```
{"question": "问题文本", "answer": "标准答案"}
```

**输出**（rollout 生成的 jsonl）：

每行包含 `question`、`answer`、`prediction`、`messages` 等字段。

---

## 三、P-ReAct 评估（需训练模型）

若使用训练好的 P-ReAct 模型（SFT + VRPO）：

```bash
cd dLLM_trainer/VRPO
# 修改 run_test.sh 中的 MODEL_PATH、BASE_DATA_DIR、BASE_OUTPUT_DIR
bash ../my_eval/run_test.sh
```

需提前配置 `my_train/my_test.py` 中的 `GOOGLE_SEARCH_KEY`、`SEARCH_API_URL` 等。
