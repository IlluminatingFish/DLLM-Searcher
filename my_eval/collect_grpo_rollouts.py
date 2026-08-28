#!/usr/bin/env python3
"""
GRPO rollout 收集：对每道题跑 G 次，用 CEM-1 自动判断对错并计算 reward。
不依赖 LLM judge，支持全量数据集。

用法（8 GPU 并行，每卡跑一份）：
  for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i python collect_grpo_rollouts.py \
      --rank $i --world 8 --n_rolls 8 &
  done
"""

import argparse, json, os, re, requests, string, torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── 超参 ──────────────────────────────────────────────────────────────────────
BLOCK_SIZE  = 64
NUM_STEPS   = 64
MAX_BLOCKS  = 16
MAX_TURNS   = 5
TEMPERATURE = 0.9   # 略高于 DPO rollout，增加 rollout 多样性

DEFAULT_MODEL = (
    "/common/users/mz751/Projects/dLLM_trainer/checkpoints/RL/plan_a/model/checkpoint-3"
)
DEFAULT_INPUT = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/VRPO/data/sft_train_as_eval.jsonl"
)
DEFAULT_OUTPUT = (
    "/research/cbim/vast/mz751/Projects/DLLM-Searcher"
    "/dLLM_trainer/VRPO/output/rl/grpo/rollouts"
)

# ── Search ────────────────────────────────────────────────────────────────────
def _load_config():
    for p in ["/research/cbim/vast/mz751/Projects/DLLM-Searcher/config.json"]:
        if os.path.exists(p):
            return json.load(open(p))
    return {}

_cfg = _load_config()
GOOGLE_KEY = os.getenv("GOOGLE_SEARCH_KEY") or _cfg.get("google_search_key", "")
SEARCH_URL = "https://google.serper.dev/search"

def google_search(q):
    try:
        r = requests.post(
            SEARCH_URL,
            headers={"X-API-KEY": GOOGLE_KEY, "Content-Type": "application/json"},
            json={"q": q, "num": 5}, timeout=10
        )
        if r.status_code != 200:
            return f"Search error {r.status_code}"
        data = r.json()
        if "organic" not in data:
            return f"No results for '{q}'."
        return f"Results for '{q}':\n" + "\n".join(
            f"{i}. [{p['title']}] {p.get('snippet','')}"
            for i, p in enumerate(data["organic"][:5], 1)
        )
    except Exception as e:
        return f"Search failed: {e}"

def run_search(queries):
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(google_search, queries))
    return "\n\n---\n\n".join(results)

# ── Reward (自动，不用 LLM judge) ─────────────────────────────────────────────
def _normalize(s):
    s = s.lower().replace("_", " ")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch if ch not in string.punctuation + "''´`" else " " for ch in s)
    return " ".join(s.split())

def _bool_map(s):
    return {"True": "yes", "False": "no"}.get(s, s)

def _first_sentence(s):
    """取第一句话（第一个句号/换行前的内容）作为关键答案。"""
    s = s.strip()
    for sep in [". ", ".\n", "\n\n", "\n"]:
        idx = s.find(sep)
        if idx > 10:    # 至少 10 个字符
            return s[:idx + 1]
    return s[:200]      # 若无句号，取前 200 字符

def _yes_no(s):
    """从文本开头提取 yes/no 倾向，返回 'yes'/'no'/None。"""
    s = s.strip().lower()
    if re.match(r'^(yes[,\s]|yes$)', s):  return 'yes'
    if re.match(r'^(no[,\s]|no$)', s):    return 'no'
    # 间接 yes/no：通过关键词推断
    if re.search(r'\bnot the same\b|\bdifferent\b|\bdid not\b|\bis not\b', s): return 'no'
    if re.search(r'\bthe same\b|\bboth\b|\byes\b', s): return 'yes'
    return None

def f1_score(prediction, ground_truth):
    """token-level F1（GT 取第一句关键词 vs prediction）。"""
    if not prediction or not str(prediction).strip():
        return 0.0
    gt_key    = _normalize(_bool_map(_first_sentence(str(ground_truth))))
    pred_norm = _normalize(_bool_map(str(prediction)))
    gt_tok    = set(gt_key.split())
    pred_tok  = set(pred_norm.split())
    if not gt_tok:
        return 0.0
    common = gt_tok & pred_tok
    if not common:
        return 0.0
    precision = len(common) / max(len(pred_tok), 1)
    recall    = len(common) / len(gt_tok)
    return 2 * precision * recall / (precision + recall + 1e-9)

def is_correct(prediction, ground_truth):
    """综合判断：yes/no 问题用极性匹配，其余用 F1 ≥ 0.35。"""
    if not prediction:
        return False
    # 尝试 yes/no 匹配
    gt_yn   = _yes_no(str(ground_truth))
    pred_yn = _yes_no(str(prediction))
    if gt_yn is not None and pred_yn is not None:
        return gt_yn == pred_yn
    # 普通问题：F1 ≥ 0.35
    return f1_score(prediction, ground_truth) >= 0.35

def count_searches(messages):
    return sum(
        1 for m in messages
        if m.get("role") == "user" and "<tool_response>" in m.get("content", "")
    )

def compute_reward(prediction, ground_truth, messages):
    """连续 reward：F1 分数 × 效率系数（Round 3+）。

    改进原因：原始 binary reward (F1 ≥ 0.35 → 0.5-1.0, 否则 0) 太松，
    导致 80% rollout 判"正确"但 eval LLM judge 只有 54%。
    连续 F1 reward 提供更细粒度的梯度信号，区分"部分正确"和"完全正确"。

    yes/no 问题保持 binary（正确=1.0×效率，错误=0）。
    """
    n_searches = count_searches(messages)
    efficiency = max(0.5, 1.0 - 0.1 * max(0, n_searches - 2))

    # yes/no 问题：保持 binary
    gt_yn   = _yes_no(str(ground_truth)) if prediction else None
    pred_yn = _yes_no(str(prediction)) if prediction else None
    if gt_yn is not None and pred_yn is not None:
        return round(efficiency, 3) if gt_yn == pred_yn else 0.0

    # 普通问题：连续 F1 × 效率
    f1 = f1_score(prediction, ground_truth) if prediction else 0.0
    if f1 <= 0.0:
        return 0.0
    return round(f1 * efficiency, 3)

# ── Parse ─────────────────────────────────────────────────────────────────────
def parse_tool_call(text):
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip()).get("arguments", {}).get("query", [])
    except Exception:
        m2 = re.search(r'"query"\s*:\s*(\[.*?\])', m.group(1), re.DOTALL)
        if m2:
            try: return json.loads(m2.group(1))
            except: pass
    return None

def extract_answer(text):
    m = re.search(r"<\|box_start\|>(.*?)<\|box_end\|>", text, re.DOTALL)
    return m.group(1).strip() if m else None

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a Web Information Seeking Master. Your task is to thoroughly seek "
    "the internet for information and provide accurate answers to questions. "
    "No matter how complex the query, you will not give up until you find the "
    "corresponding information.\n\nAs you proceed, adhere to the following principles:\n\n"
    "1. **Persistent Actions for Answers**: You will engage in many interactions, "
    "delving deeply into the topic to explore all possible aspects until a satisfactory "
    "answer is found.\n\n"
    "2. **Repeated Verification**: Before presenting a Final Answer, you will "
    "**cross-check** and **validate the information** you've gathered to confirm its "
    "accuracy and reliability.\n\n"
    "3. **Attention to Detail**: You will carefully analyze each information source "
    "to ensure that all data is current, relevant, and from credible origins."
)

USER_PROMPT_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the assistant solves it by calling one or more of the following tools.
<tools>
{
  "name": "search",
  "description": "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Array of query strings. Include 3 or less search queries in a single call."
      }
    },
    "required": ["query"]
  }
}
</tools>

The assistant starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends with answer of the question.

Example response:
<think>
thinking process here
</think>
<tool_call>
{"name": "search", "arguments": {"query": ["query string 1", "query string 2"]}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
<think>
thinking process here
</think>
<|box_start|>
answer here
<|box_end|>
The assistant must strictly abide by the above format.
User: """

# ── Block diffusion generation ────────────────────────────────────────────────
@torch.no_grad()
def denoise_block_sample(model, input_ids, block_start, block_end,
                         mask_id, num_steps, temperature=1.0):
    for _ in range(num_steps):
        logits = model(input_ids=input_ids).logits[0, block_start:block_end]
        still_masked = (input_ids[0, block_start:block_end] == mask_id)
        if still_masked.sum() == 0:
            break
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits.float(), dim=-1)
        pred_ids   = torch.multinomial(probs, 1).squeeze(-1)
        confidence = probs.max(dim=-1).values
        n_reveal = max(1, round((block_end - block_start) / num_steps))
        conf_m   = confidence.clone()
        conf_m[~still_masked] = -1.0
        n_reveal = min(n_reveal, still_masked.sum().item())
        top_idx  = conf_m.topk(n_reveal).indices
        for idx in top_idx:
            input_ids[0, block_start + idx] = pred_ids[idx]
    return input_ids

@torch.no_grad()
def llada_generate_sample(model, tokenizer, prompt_ids, mask_id,
                          block_size=BLOCK_SIZE, num_steps=NUM_STEPS,
                          max_blocks=MAX_BLOCKS, stop_strings=None,
                          temperature=TEMPERATURE):
    device       = next(model.parameters()).device
    eos_id       = tokenizer.eos_token_id or 0
    stop_strings = stop_strings or []
    input_ids    = prompt_ids.to(device).clone()
    generated_ids = []

    for _ in range(max_blocks):
        new_block   = torch.full((1, block_size), mask_id, dtype=torch.long, device=device)
        input_ids   = torch.cat([input_ids, new_block], dim=1)
        block_start = input_ids.shape[1] - block_size
        block_end   = input_ids.shape[1]

        input_ids = denoise_block_sample(
            model, input_ids, block_start, block_end,
            mask_id, num_steps, temperature
        )

        block_tokens  = input_ids[0, block_start:block_end].tolist()
        generated_ids.extend(block_tokens)
        current_text  = tokenizer.decode(generated_ids, skip_special_tokens=False)

        for stop in stop_strings:
            if stop in current_text:
                return current_text[:current_text.index(stop) + len(stop)]

        if all(t == eos_id for t in block_tokens[-8:]):
            return tokenizer.decode(generated_ids, skip_special_tokens=True)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# ── Single rollout ─────────────────────────────────────────────────────────────
def run_one_rollout(model, tokenizer, mask_id, question, temperature=TEMPERATURE):
    stop_strings = ["</tool_call>", "<|box_end|>"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT_TEMPLATE + question},
    ]
    full_context = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prediction = None

    for turn in range(MAX_TURNS):
        prompt_ids = tokenizer(
            full_context, return_tensors="pt", add_special_tokens=False
        ).input_ids
        new_text = llada_generate_sample(
            model, tokenizer, prompt_ids, mask_id,
            stop_strings=stop_strings, temperature=temperature
        )
        messages.append({"role": "assistant", "content": new_text.strip()})

        answer = extract_answer(new_text)
        if answer:
            prediction = answer
            break

        queries = parse_tool_call(new_text)
        if queries:
            search_result = run_search(queries)
            messages.append({
                "role": "user",
                "content": f"<tool_response>\n{search_result}\n</tool_response>"
            })
            tool_resp = (
                f"<|im_start|>user\n<tool_response>\n{search_result}\n</tool_response>"
                f"<|im_end|>\n<|im_start|>assistant\n"
            )
            full_context += new_text + tool_resp
        else:
            full_context += new_text

    return {
        "prediction":  prediction,
        "messages":    messages,
        "termination": "answer" if prediction else "max_turns",
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--input",       default=DEFAULT_INPUT)
    parser.add_argument("--output",      default=DEFAULT_OUTPUT)
    parser.add_argument("--n_rolls",     type=int,   default=8)
    parser.add_argument("--n_questions", type=int,   default=0,   help="0=全部")
    parser.add_argument("--rank",        type=int,   default=0)
    parser.add_argument("--world",       type=int,   default=8)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = parser.parse_args()

    samples = [json.loads(l) for l in open(args.input) if l.strip()]
    if args.n_questions > 0:
        samples = samples[:args.n_questions]
    samples = samples[args.rank::args.world]
    print(f"Rank {args.rank}/{args.world}: {len(samples)} 题 × {args.n_rolls} rolls")

    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"rank{args.rank}.jsonl"

    done = set()
    if out_file.exists():
        for l in open(out_file):
            try:
                d = json.loads(l)
                done.add((d["question"], d["roll_idx"]))
            except: pass
        print(f"已完成 {len(done)} 条，继续")

    print(f"\n加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    model.config.use_cache = False
    mask_id = model.config.mask_token_id
    print(f"mask_id={mask_id}  temperature={args.temperature}\n")

    with open(out_file, "a") as f:
        for sample in tqdm(samples, desc=f"rank{args.rank}"):
            q = sample["question"]
            a = sample["answer"]
            for roll_idx in range(args.n_rolls):
                if (q, roll_idx) in done:
                    continue
                try:
                    result = run_one_rollout(
                        model, tokenizer, mask_id, q, args.temperature
                    )
                except Exception as e:
                    tqdm.write(f"[ERROR] roll={roll_idx} {q[:40]}: {e}")
                    result = {"prediction": None, "messages": [], "termination": "error"}

                reward = compute_reward(result["prediction"], a, result["messages"])
                record = {
                    "question":    q,
                    "answer":      a,
                    "roll_idx":    roll_idx,
                    "prediction":  result["prediction"],
                    "messages":    result["messages"],
                    "termination": result["termination"],
                    "reward":      reward,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    print(f"\n完成！输出: {out_file}")

if __name__ == "__main__":
    main()
