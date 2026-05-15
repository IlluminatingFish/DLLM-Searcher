# Case: "When was the first iPhone released?"

**来源**: `rollout_results_rank2_20260308_201336.jsonl`

---

## 基本信息

| 项目 | 值 |
|------|-----|
| question | When was the first iPhone released? |
| answer (GT) | June 29, 2007 |
| prediction | `""`（空，提取失败） |
| num_turns | 4 |
| termination_reason | answer |

---

## 发生了什么

### Turn 1（assistant）
```
<think>
I need to find the first iPhone release year. Let's search for "first iPhone release year" to get the release details.<|im_end|>
```
- 只有 thinking，没有 `<tool_call>`
- 以 `<|im_end|>` 结束，可能被 stop_words 截断，没有输出 `</think>`
- 未执行工具

### Turn 2（assistant）
```
<tool_call>
{"name": "search", "arguments": {"query": ["first iPhone release year"]}}
<|im_end|>
```
- 有 `<tool_call>` 和合法 JSON
- **缺少 `</tool_call>`**：输出在 `<|im_end|>` 处被 stop_words 截断
- 条件 `TOOL_CALL_START in text and TOOL_CALL_END in text` 不满足 → **工具未执行**
- 没有 tool_response 注入，context 中也没有搜索结果

### Turn 3（assistant）
```
<|im_end|>
```
- 几乎为空，仅 `<|im_end|>`
- 上一轮未拿到 tool_response，本轮只能基于不完整 context 继续生成

### Turn 4（assistant，answer_rl）
```
<|box_start|>











































<|box_end|><|im_end|>
```
- 已切换到 answer_rl
- `<|box_start|>` 与 `<|box_end|>` 之间只有换行，无答案文本
- `extract_answer` 得到空字符串 → `prediction: ""`

---

## 根因归纳

1. **Stop 截断**：Turn 2 在生成 `</tool_call>` 前就遇到 `<|im_end|>` 而停止，导致 tool_call 不完整，工具未执行。
2. **无工具结果**：未注入 search 结果，模型无法基于真实检索内容作答。
3. **空答案区**：Turn 4 的答案框内被大量换行填满，无实质文本，可能与 alignment / mode collapse 相关。

---

## 可能对策

- 将 `</tool_call>` 从 stop_words 中移出，或调整 stop 逻辑，保证 tool_call 可完整生成。
- 对不完整 tool_call 做恢复或重试（如补全 `</tool_call>` 后再 parse）。
- 排查并缓解训练与推理的 alignment 问题，减少 answer 区只出换行的情况。
