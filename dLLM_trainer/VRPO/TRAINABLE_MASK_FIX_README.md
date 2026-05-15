# Trainable Mask 修复说明 (VRPO DPO)

## 一、问题背景

训练时出现 `rewards/chosen`, `rewards/rejected`, `logps/chosen`, `logps/rejected` 为 NaN，而 `loss` 有值。根因是 **trainable mask 与真实 token 序列长度不一致**，导致 block diffusion 的 `valid_positions`、`blk_num_masks` 计算出错。

## 二、原因分析

### 旧逻辑（有 bug）

1. **`get_trainable_mask`**：对 `full_text = prompt + chosen` 做**分段 tokenize**
   - 按 `<tool_response>...</tool_response>` 切分文本
   - 每段单独 `tokenizer.encode(seg_text)`，再拼成 mask
   - 问题：BPE 分段 tokenize 与整体 tokenize 的边界不一致

2. **`tokenize_row`**：对 `prompt` 和 `chosen` **分别** tokenize：
   - `chosen_input_ids = tokenizer(chosen) + [eos]`，再 truncate
   - 与 full_text 的分段 tokenize 结果**长度不同**

3. **后果**：`chosen_trainable_mask = full_mask[prompt_len : prompt_len + len(chosen_input_ids)]` 时，mask 长度与 `chosen_input_ids` 不对齐，导致后续 `valid_positions` 等出错，产生 NaN。

## 三、修复思路

> 按 `tokenize_row` 的 tokenization 构建 mask，保证 mask 与真实 token 序列一一对应。

### 1. 重写 `get_trainable_mask`（`my_dpo_trainer.py`）

**新接口**（不再需要 prompt）：

```python
def get_trainable_mask(
    completion_text: str,           # 仅 completion（chosen/rejected）
    tokenizer: PreTrainedTokenizerBase,
    tool_resp_left: str,
    tool_resp_right: str,
) -> List[bool]:
```

**核心改动（token 空间实现）**：

- 与 `tokenize_row` 完全一致：使用 `tokenizer(completion_text, add_special_tokens=False)["input_ids"]` 得到 token 序列
- 将 `tool_resp_left`、`tool_resp_right` 也 tokenize，得到 `left_ids`、`right_ids`
- 在 token 序列中搜索 `left_ids` 和 `right_ids`，找到所有 `<tool_response>...</tool_response>` 的 token 区间 [start, end)
- 对每个 token 位置 j，若在任一区间内 → 非 trainable（False）
- 返回的 mask 长度 = `len(tokenizer(completion_text))`，与 token 序列**严格一一对应**
- **不依赖** offset_mapping 或 decode+find，兼容 slow/fast tokenizer

### 2. 更新 `tokenize_row` 中的 mask 构建（`my_dpo_trainer.py`）

**旧逻辑**：

```python
chosen_full_mask = get_trainable_mask(chosen_full_text, tokenizer, prompt_text, ...)
chosen_trainable_mask = chosen_full_mask[prompt_len : prompt_len + len(chosen_input_ids)]
# 再 pad/truncate mask
```

**新逻辑**：

```python
chosen_trainable_mask = get_trainable_mask(features["chosen"], tokenizer, ...)
chosen_trainable_mask = chosen_trainable_mask + [True]   # EOS token
chosen_trainable_mask = chosen_trainable_mask[:len(chosen_input_ids)]
```

对 `rejected` 同理。mask 与 `chosen_input_ids` / `rejected_input_ids` 严格对齐。

## 四、涉及文件

| 文件 | 修改内容 |
|------|----------|
| `my_train/my_dpo_trainer.py` | 重写 `get_trainable_mask`；更新 `tokenize_row` 中的 mask 构建 |
| `check_mask_align.py` | 使用新 `get_trainable_mask` 做对齐检查（可选） |

## 五、如何回滚

若修复引入新问题，可按以下步骤回滚：

1. **恢复 `get_trainable_mask`**：改回旧签名和分段 tokenize 实现：

```python
def get_trainable_mask(text, tokenizer, prompt, tool_left, tool_right):
    segments = []
    segments.append((prompt, False))
    remaining = text[len(prompt):]
    while remaining:
        seg_start = remaining.find(tool_left)
        if seg_start == -1:
            if remaining:
                segments.append((remaining, True))
            break
        if seg_start > 0:
            segments.append((remaining[:seg_start], True))
        left_end = seg_start + len(tool_left)
        right_start = remaining.find(tool_right, left_end)
        if right_start == -1:
            segments.append((remaining[seg_start:], False))
            break
        seg_end = right_start + len(tool_right)
        segments.append((remaining[seg_start:seg_end], False))
        remaining = remaining[seg_end:]
    trainable = []
    for seg_text, is_trainable in segments:
        if seg_text:
            seg_ids = tokenizer.encode(seg_text, add_special_tokens=False)
            trainable.extend([is_trainable] * len(seg_ids))
    return trainable
```

2. **恢复 `tokenize_row` 中的 mask 构建**：

```python
prompt_text = features["prompt"]
chosen_full_text = prompt_text + features["chosen"]
rejected_full_text = prompt_text + features["rejected"]

chosen_full_mask = get_trainable_mask(
    chosen_full_text, tokenizer, prompt_text,
    self.tool_resp_left, self.tool_resp_right
)
rejected_full_mask = get_trainable_mask(
    rejected_full_text, tokenizer, prompt_text,
    self.tool_resp_left, self.tool_resp_right
)

prompt_len = len(prompt_input_ids)
chosen_trainable_mask = chosen_full_mask[prompt_len:prompt_len + len(chosen_input_ids)]
rejected_trainable_mask = rejected_full_mask[prompt_len:prompt_len + len(rejected_input_ids)]

if len(chosen_trainable_mask) < len(chosen_input_ids):
    chosen_trainable_mask.extend([True] * (len(chosen_input_ids) - len(chosen_trainable_mask)))
elif len(chosen_trainable_mask) > len(chosen_input_ids):
    chosen_trainable_mask = chosen_trainable_mask[:len(chosen_input_ids)]

if len(rejected_trainable_mask) < len(rejected_input_ids):
    rejected_trainable_mask.extend([True] * (len(rejected_input_ids) - len(rejected_trainable_mask)))
elif len(rejected_trainable_mask) > len(rejected_input_ids):
    rejected_trainable_mask = rejected_trainable_mask[:len(rejected_input_ids)]
```

## 六、汇报用摘要

- **问题**：VRPO DPO 训练中 rewards/logps 为 NaN
- **根因**：trainable mask 用分段 tokenize 构建，与 `tokenize_row` 中的 token 序列长度不一致
- **修复**：用 `tokenizer(completion, return_offsets_mapping=True)` 按字符偏移构建 mask，保证 mask 与 `tokenizer(completion)` 一一对应
- **改动**：重写 `get_trainable_mask`，并调整 `tokenize_row` 中的 mask 构建逻辑
