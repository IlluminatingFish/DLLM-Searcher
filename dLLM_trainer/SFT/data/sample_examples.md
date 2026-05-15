# SFT data.json 样本示例

数据来源：`dLLM_trainer/SFT/data/data.json`  
总样本数：3977，其中 **2812 条** 含多于一次 tool_call。

---

## 样本 1：2 次 tool_call

**问题**：Junk Mail 这本书的作者及其国籍

### prompt（前 600 字符）

```
<|im_start|>system
You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions. No matter how complex the query, you will not give up until you find the corresponding information.

As you proceed, adhere to the following principles:

1. **Persistent Actions for Answers**: You will engage in many interactions, delving deeply into the topic to explore all possible aspects until a satisfactory answer is found.

2. **Repeated Verification**: Before presenting a Final Answer, you will **cross-check** and **validate the information** you've gathered to confirm its accuracy and reliability.

3. **Attention to Detail**: You will carefully analyze each information source to ensure that all data is current, relevant...
[含 tools 说明、格式示例、User: 问题]
```

### response 完整结构

```
<tool_call>
{"name": "wikisearch", "arguments": {"query": ["Junk Mail book author", "Junk Mail (book) writer", "Junk Mail author nationality"]}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Wikipedia search for 'Junk Mail book author' found 3 results:
1. Junk Mail (book) - Junk Mail is a 1995 book by Will Self published by Bloomsbury...
2. Junk (novel) - by Melvin Burgess...
3. Julia Bell (author)...
...
</tool_response><|im_end|>
<|im_start|>assistant
<think>
From the tool response, the book "Junk Mail" is by Will Self. To confirm his nationality, I need to search for Will Self's nationality.
</think>

<tool_call>
{"name": "wikisearch", "arguments": {"query": ["Will Self nationality"]}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Wikipedia search for 'Will Self nationality' found 3 results:
1. Will Self - William Woodard Self (born 26 September 1961) is an English author...
...
</tool_response><|im_end|>
<|im_start|>assistant
<think>

</think>

<|box_start|>
The book *Junk Mail* was written by Will Self, who is of English nationality.
<|box_end|><|im_end|>
```

**流程**：第 1 次搜索 → 得到作者 Will Self → 第 2 次搜索确认国籍 → 给出答案

---

## 样本 2：3 次 tool_call

**问题**：由 Mian Abdul Khaliq and Company 建造的板球场，2017 年在该场地举行的板球赛冠军是谁？

### response 结构概览

```
<think>
To determine which road is longer, I need to find the lengths of Springfield Boulevard and New York State Route 25A...
</think>

<tool_call>
{"name": "wikisearch", "arguments": {"query": ["Springfield Boulevard length", "New York State Route 25A length"]}}
</tool_call>
<tool_response>...</tool_response>

<think>
[根据第一次结果继续推理]...
</think>

<tool_call>
{"name": "wikisearch", "arguments": {"query": ["Gaddafi Stadium constructed by Mian Abdul Khaliq and Company", "2017 cricket match Gaddafi Stadium winner"]}}
</tool_call>
<tool_response>...</tool_response>

<think>
[根据第二次结果，确认 Gaddafi Stadium 由 Mian Abdul Khaliq 建造，需再查 2017 PSL 决赛冠军]
</think>

<tool_call>
{"name": "wikisearch", "arguments": {"query": ["2017 Pakistan Super League final winner"]}}
</tool_call>
<tool_response>
Peshawar Zalmi won by 58 runs...
</tool_response>

<think>

</think>

<|box_start|>The team that won the 2017 cricket match at the specified ground (Gaddafi Stadium in Lahore) is Peshawar Zalmi. This match was the final of the 2017 Pakistan Super League, played on March 5, 2017, where Peshawar Zalmi defeated Quetta Gladiators by 58 runs. Gaddafi Stadium was designed by architect and engineer Nasreddin Murat-Khan and constructed by Mian Abdul Khaliq and Company in 1959.<|box_end|>
```

**流程**：第 1 次搜索（路长）→ 第 2 次搜索（球场建造商 + 2017 比赛）→ 第 3 次搜索（PSL 决赛冠军）→ 综合给出答案

---

## 样本 3：2 次 tool_call（En Garde! 游戏设计师）

### response 结构

```
<think>
I need to identify the designer of the game "En Garde!" who is both an award-winning game designer and a New York Times best-selling author. First, I should search for information about the creator of "En Garde!"...
</think>

<tool_call>
{"name": "wikisearch", "arguments": {"query": ["En Garde! game designer", "En Garde! board game creator", "En Garde! designer award-winning author"]}}
</tool_call>
<tool_response>
En Garde! was designed by Frank Chadwick... [Wikipedia 结果]
</tool_response>

<think>
[根据结果继续推理，可能需要第二次搜索验证作者身份]...
</think>

<tool_call>
{"name": "wikisearch", "arguments": {"query": ["Frank Chadwick New York Times bestseller", ...]}}
</tool_call>
<tool_response>...</tool_response>

<|box_start|>最终答案<|box_end|>
```

---

## 数据格式总结

| 字段 | 说明 |
|------|------|
| `prompt` | system + user，含工具说明、格式示例、问题 |
| `response` | 完整 assistant 轨迹，含 `<think>`、`<tool_call>`、`<tool_response>`、`<|box_start|>答案<|box_end|>` |

**注意**：SFT 数据使用 `wikisearch` 工具和 `<|box_start|>` 答案格式，与 Dataroller 的 `search` 工具和 `<answer>` 格式略有不同，但结构一致。

**多轮调用模式**：每轮为 `<think>` → `<tool_call>` → `<tool_response>`，可重复多次，最后以 `<|box_start|>答案<|box_end|>` 收束。
