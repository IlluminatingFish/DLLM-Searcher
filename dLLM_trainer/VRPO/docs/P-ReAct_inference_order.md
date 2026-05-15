# P-ReAct 推理顺序说明

基于 `run_test.sh` 调用的 `my_test.py` + jetengine 实现，说明模型推理时 tool_call 与 thinking 的解码顺序。

---

## 一、训练配置（与推理顺序无关）

模型训练配置见 `recipes/dpo.yaml`：`num_train_epochs: 5`。训练阶段不涉及 toolcall 优先解码，该策略仅在推理时由 `toolcall_pre_rl` 实现。

---

## 二、推理时：第几步出现 tool_call？

**第 0 步（第一个 denoising step）** 就出现 tool_call 的框架。

- `denoising_steps = 128`（`my_test.py`）
- `block_length = 128`
- 在 `current_denoising_step == 0` 时，代码直接写入位置 63 和 126 的 token，不做采样
- 因此：在第一个去噪步，`<tool_call>` 和 `</tool_call>` 就已确定；thinking 区（0–62）仍全是 [M]

---

## 三、是否「先有 tool_call，才有 thinking」？

**是，从显式文本来看是这样。**

| 时机 | thinking 区 (0–62) | tool_call 区 (63–127) |
|------|-------------------|------------------------|
| Step 0 后 | 仍为 [M] | 位置 63、126 已是 `<tool_call>`、`</tool_call>` |
| Step 1–127 | 逐步 unmask（且 confidence 较低） | 逐步 unmask（confidence 被 +0.5 抬高） |

**为什么？** 代码在 `scheduler.py` 中显式实现：`current_denoising_step == 0` 时强制写入，`step >= 1` 时对 `confidence[63:]` 加 0.5，使 tool_call 区在选 unmask 时总是被优先选中。这是 P-ReAct 的设计选择，目的是让 tool_call 尽早可执行，降低端到端延迟。

---

## 四、实现机制：pre-filling + confidence biasing

代码位置：`jetengine/engine/scheduler.py`，策略 `toolcall_pre_rl`。

### 4.1 Block 结构

- `block_length = 128`
- 每个 block 内大致：
  - **位置 0–62**：thinking 区（前半段）
  - **位置 63–127**：tool_call 区（后半段）

### 4.2 Step 0：强制 pre-fill

在第一个 denoising step，直接写入：

- 位置 63 → `<tool_call>`
- 位置 126 → `</tool_call>`

不做采样，直接赋值，这两个 token 在第一步就确定。

### 4.3 Step ≥ 1：confidence biasing

```python
confidence[63:] = confidence[63:] + 0.5
```

后半段（63–127）的 confidence 被抬高，因此 `topk` 选 unmask 时，tool_call 区更易先被解出，thinking 区更晚完成。

---

## 五、两个层面的区分

| 层面 | 顺序 | 说明 |
|------|------|------|
| **文本** | 先 tool_call，后 thinking | 由 pre-fill + confidence biasing 强制实现 |
| **计算/表征** | 推理信息可能更早存在 | dLLM block 内双向 attention，latent reasoning 可在 thinking 显式输出前形成 |

论文主张：即便 thinking 文字尚未输出，dLLM 在 block 内双向注意力 + 去噪过程中已形成 latent reasoning。因此 tool_call 并非“完全没推理就写出来”（"know the answer before decoding it"）。

---

## 六、对 tool_call 质量的解释

若担心“先写 tool_call 再补 thinking 会导致 tool_call 质量差”，论文/实现的回答是：

dLLM 的生成不是严格 left-to-right，tool_call 可以利用同一 block 内尚未显式解码的推理轨迹，因此不必然变差。

---

## 七、工具调用的触发判定

「当 tool_call 内容足够完整且可解析时」的判定逻辑在 `my_test.py` 的 `rollout_batch` 中：

### 7.1 触发时机

**不是流式、逐 token 检查**，而是在**每个 turn 的 generation 完全结束后**，对 `output['text']`（本轮生成的整段文本）做一次性检查。

### 7.2 完整性条件

```python
if self.TOOL_CALL_START in new_text and self.TOOL_CALL_END in new_text:
    tool_info = self.parse_tool_call(new_text)
    if tool_info is not None:
        tool_calls_to_execute.append(...)
```

1. **双 tag 必须同在**：文本中必须同时出现 `<tool_call>` 和 `</tool_call>`
2. **内容必须可解析**：`parse_tool_call` 会：
   - 用 `split(TOOL_CALL_START)[1].split(TOOL_CALL_END)[0]` 取出两标签之间的字符串
   - 用 `json.loads(tool_call_str.strip())` 解析
   - 若 JSON 非法则 `except` 返回 `None`，不触发执行
   - 解析成功则取出 `name` 和 `arguments`，加入 `tool_calls_to_execute`

### 7.3 与 stop_words 的关系

`stop_words=[151645, 151658]`（`<|im_end|>` 和 `</tool_call>`）。在 `sequence.commit_block` 中，遇到 stop 会截断并结束该 sequence。因此一旦生成了 `</tool_call>`，后续 token 通常不会继续生成，工具调用被视作「这一段已完成」。

### 7.4 总结

| 项目 | 实现 |
|------|------|
| 检查粒度 | 按 turn 批处理，非逐 token |
| 完整性 | 有 `<tool_call>`、`</tool_call>`，且中间内容为合法 JSON |
| 失败时 | `parse_tool_call` 返回 None → 不执行，回写错误信息给模型，继续下一轮 |

---

## 八、多次 tool call 的实现：按 turn 循环

「多次 tool call」依赖 **ReAct 的 turn 循环**，不是在同一 block 内解析多个 tool_call。

### 8.1 机制

| 层级 | 机制 |
|------|------|
| **单 turn** | 一次 `llm.generate_streaming` 生成一个或多个 block，遇到 `</tool_call>` 即 stop，因此一次生成中最多出现一个完整 tool_call |
| **parse_tool_call** | 使用 `split(TOOL_CALL_START)[1].split(TOOL_CALL_END)[0]`，只解析**第一个** `<tool_call>...</tool_call>` |
| **多次 tool call** | 由 **turn 循环**实现：每轮生成 → 解析并执行当前 tool_call → 将 `<tool_response>` 拼回 context → 下一轮在包含新 context 的 prompt 下继续生成 |

### 8.2 流程

```
Turn 0: 生成 block → 得到 tool_call_1 → 执行 → context += tool_response_1
Turn 1: 生成 block（含 tool_response_1）→ 得到 tool_call_2 → 执行 → context += tool_response_2
...
Turn 4: 最多 5 轮（MAX_TURNS=5）
```

### 8.3 代码位置

- `my_test.py` 第 374 行：`for turn in range(MAX_TURNS)` 实现 turn 循环
- 第 436–446 行：每轮检测并解析 tool_call，加入 `tool_calls_to_execute`
- 第 449–469 行：执行 tool call，把 tool_response 拼回 context，并将 sample 放入 `still_active` 进入下一轮

---

## 九、toolcall_only_nocompute 准确率差距说明

若 toolcall_first 准确率远低于 baseline（如 1% vs 更高），主要原因：

1. **answer_rl 被覆盖**：若每个 turn 都用 toolcall 策略，turn ≥ 3 时本应切到 `answer_rl` 生成 `<|box_start|>答案<|box_end|>`，但被强制用 toolcall，模型无法输出答案格式，导致 `prediction=null` 比例很高。

2. **建议**：turn 0、1、2 用 `toolcall_only_nocompute`，turn ≥ 3 必须用 `answer_rl`。当前 `my_test.py` 已恢复该逻辑。

3. **其他可能因素**：64-token tool_call 区可能截断长 JSON；thinking 与 tool_call 分两阶段生成，可能影响连贯性。

---

## 十、toolcall_only_nocompute：先 thinking 再 tool_call

策略 `toolcall_only_nocompute` 实现「先 thinking 再 tool_call」，符合 ReAct 顺序：

1. **Phase 1**：只对 0–62（63 token）做 forward，生成 thinking
2. **Phase 2**：thinking 完成后，再对 63–127（64 token）做 forward，生成 tool_call
3. **Commit**：按顺序合并 `[thinking, tool_call]` 后提交

运行：`MAX_SAMPLES=100 DATASETS=hotpot GPU_NUM=4 bash ../../my_eval/run_test_toolcall_first.sh`

输出目录：`output/preact_eval_toolcall_first/`

---

## 十一、answer 区格式问题：assistant、\`<think>\` 等

### 11.1 什么是 `assistant\n`？

`assistant\n` 即字面含义：字符串 `"assistant"` 加上一个换行符 `\n`（`\n` 是换行的转义写法）。

### 11.2 问题表现

在 turn ≥ 3 切到 `answer_rl` 时，模型应在 `<|box_start|>` 与 `<|box_end|>` 之间输出**纯答案**。但 toolcall_first 下，模型常输出 chat 结构，例如：

```
assistant
<think>
（一段推理内容）
</think>
The search results confirm... Both
```

对应原始字符串（示意）：

```
"assistant\n<think>\n...</think>\nThe search results confirm that Scott Derrickson is an American filmmaker. Ed Wood was an American filmmaker. Both\n</think>\nBoth"
```

### 11.3 理想格式 vs 错误格式

| 理想 | 错误 |
|------|------|
| `<\|box_start\|>yes<\|box_end\|>` | `<\|box_start\|>assistant\n<think>...</think> Both<\|box_end\|>` |
| `<\|box_start\|>Scott Derrickson and Ed Wood were both American.<\|box_end\|>` | 同上 |

### 11.4 为何出现

toolcall_only_nocompute 的两阶段生成（先 63 token thinking，再 64 token tool_call）改变了模型的生成节奏。切到 answer_rl 时，模型仍延续「对话式」结构，把 chat 模板中的 `assistant`、`<think>` 等带入答案区。

### 11.5 缓解措施

1. **answer_turn_hint**：在 turn ≥ 3 的 context 末尾加提示：`[Now provide only your final answer between <|box_start|> and <|box_end|>, without assistant or think tags.]`
2. **extract_answer 后处理**：对 toolcall_first 单独实现 `extract_answer`，尝试去掉 `assistant\n`、`<think>...</think>` 后再提取答案。

---

## 十二、相关代码位置

| 文件 | 内容 |
|------|------|
| `my_train/my_test.py` | 调用 `remasking_strategy = "toolcall_pre_rl"`（turn < 3）；工具调用触发逻辑（436–446 行）、`parse_tool_call`（280–292 行） |
| `jetengine/engine/scheduler.py` | `toolcall_pre_rl` 的具体逻辑（步骤 0 强制 + 步骤 ≥1 confidence 偏置） |
| `jetengine/engine/sequence.py` | `commit_block` 中对 stop_words 的截断逻辑；`toolcall_only_nocompute` 的 `toolcall_phase` 与 `get_active_block_length` |
| `jetengine/engine/model_runner.py` | `prepare_denoise` 中按 phase 只传 64/63 token |
