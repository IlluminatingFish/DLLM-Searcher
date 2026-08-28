#!/usr/bin/env python3
"""
Phase 2: Round 0 Rollout Collector

使用 SFT ckpt_6 对 rl_pool_256.jsonl 的 256 个问题各生成 8 条 rollout，
共 2048 条 raw trajectories，保存结构化格式。

每条 trajectory 包含：
- round_id, question_id, group_id, rollout_id
- policy 元信息（checkpoint, hash, block_length, seed）
- turns（think, tool_call, tool_response, parse_success）
- final_answer, terminated_by, num_searches, valid_format
- n_policy_tokens, n_env_tokens, total_tokens
- gold（short_answers, answer_type）

Usage（8 GPU 并行，每卡处理 32 题 × 8 rollouts = 256 trajectories）:
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i python collect_round0_rollouts.py --rank $i --world 8 &
    done
    wait
"""

import argparse, hashlib, json, os, re, requests, string, time, torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.modeling_utils as _mu
from transformers import PreTrainedModel

# ── transformers 5.x 兼容补丁 ─────────────────────────────────────────────────
if not isinstance(PreTrainedModel.__dict__.get("all_tied_weights_keys"), property):
    def _atk_getter(self): return getattr(self, "_all_tied_weights_keys_storage", {})
    def _atk_setter(self, val): object.__setattr__(self, "_all_tied_weights_keys_storage", val)
    PreTrainedModel.all_tied_weights_keys = property(_atk_getter, _atk_setter)

def _patched_get_tied(module):
    keys = []
    for name, sub in module.named_modules():
        tied = getattr(sub, "_tied_weights_keys", None) or {}
        keys.extend([f"{name}.{k}" if name else k for k in (tied.keys() if isinstance(tied, dict) else tied)])
    return keys
_mu._get_tied_weight_keys = _patched_get_tied

if "_finalize_model_loading" in PreTrainedModel.__dict__:
    _orig = PreTrainedModel.__dict__["_finalize_model_loading"].__func__
    def _patched_finalize(model, load_config, loading_info):
        _orig_tw = type(model).tie_weights
        def _safe_tw(self, **kw):
            try: return _orig_tw(self, **kw)
            except TypeError: return _orig_tw(self)
        type(model).tie_weights = _safe_tw
        try: return _orig(model, load_config, loading_info)
        finally: type(model).tie_weights = _orig_tw
    PreTrainedModel._finalize_model_loading = staticmethod(_patched_finalize)

# ── 超参 ──────────────────────────────────────────────────────────────────────
BLOCK_SIZE        = 64
NUM_STEPS         = 64
MAX_BLOCKS        = 16   # max 1024 generated tokens per assistant turn
MAX_TURNS         = 5    # max search cycles
FORCE_ANSWER_TURN = 3    # 第 3 轮起强制注入 <|box_start|>，与 run_llada_eval.py 一致
TEMPERATURE       = 0.9  # rollout 用采样以获多样性（eval 用 greedy）
N_ROLLS           = 8    # rollouts per question

ROOT = Path("/research/cbim/vast/mz751/Projects/DLLM-Searcher")
DEFAULT_MODEL = str(ROOT / "dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized")
DEFAULT_INPUT = str(ROOT / "dLLM_trainer/VRPO/data/rl_pool/rl_pool_256.jsonl")
DEFAULT_OUTPUT= str(ROOT / "dLLM_trainer/Agentic/output/round0_rollouts")
def _load_cfg():
    for path in [ROOT / "config.json",
                 Path(os.path.dirname(__file__)) / ".." / "config.json"]:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return {}

_cfg = _load_cfg()
GOOGLE_KEY  = os.getenv("GOOGLE_SEARCH_KEY") or _cfg.get("google_search_key", "")
SEARCH_URL  = "https://google.serper.dev/search"

# ── 搜索（与 run_llada_eval.py 保持完全一致）───────────────────────────────────
def _google_search_one(query: str) -> str:
    """单条查询；返回格式化字符串。"""
    headers = {"X-API-KEY": GOOGLE_KEY, "Content-Type": "application/json"}
    try:
        resp = requests.post(SEARCH_URL, headers=headers,
                             json={"q": query, "num": 5}, timeout=10)
        if resp.status_code != 200:
            return f"Search error {resp.status_code}"
        data = resp.json()
        if "organic" not in data:
            return f"No results for '{query}'."
        lines = [
            f"{i}. [{p['title']}] {p.get('snippet', '')}"
            for i, p in enumerate(data["organic"][:5], 1)
        ]
        return f"Results for '{query}':\n" + "\n".join(lines)
    except Exception as e:
        return f"Search failed: {e}"


def run_search(queries: list) -> str:
    """并发执行多条查询，合并结果，与 run_llada_eval.py 的 run_search 完全对齐。"""
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(_google_search_one, queries))
    return "\n\n---\n\n".join(results)

# ── 答案归一化 ─────────────────────────────────────────────────────────────────
def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch if ch not in set(string.punctuation) else " " for ch in s)
    return " ".join(s.split())

# ── 生成 ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def denoise_block(model, input_ids, block_start, block_end, mask_id, num_steps,
                  temperature=0.0):
    """Block-by-block 去噪。temperature=0 → greedy（eval 模式）；>0 → 采样（rollout 模式）。
    confidence 始终基于 greedy 概率（与 run_llada_eval.py 对齐），
    采样只影响 revealed token 的具体值。
    """
    block_len = block_end - block_start
    for _ in range(num_steps):
        logits = model(input_ids=input_ids).logits[0, block_start:block_end]
        probs_greedy = torch.softmax(logits.float(), dim=-1)
        confidence   = probs_greedy.max(dim=-1).values          # 始终用 greedy 概率决定 reveal 顺序

        if temperature > 0:
            pred_ids = torch.multinomial(
                torch.softmax(logits.float() / temperature, dim=-1), 1
            ).squeeze(-1)
        else:
            pred_ids = probs_greedy.argmax(dim=-1)

        still_masked = (input_ids[0, block_start:block_end] == mask_id)
        n_still = still_masked.sum().item()
        if n_still == 0:
            break

        n_reveal = max(1, round(block_len / num_steps))
        conf_m = confidence.clone()
        conf_m[~still_masked] = -1.0
        top_idx = conf_m.topk(min(n_reveal, n_still)).indices

        for idx in top_idx:
            input_ids[0, block_start + idx] = pred_ids[idx]

    return input_ids


@torch.no_grad()
def llada_generate_blocks(model, tokenizer, prompt_ids, mask_id,
                          block_size=64, num_steps=64, max_blocks=16,
                          stop_strings=None, temperature=0.0):
    """Block-by-block 生成，每块后检查 stop_strings 并立刻截断返回。
    与 run_llada_eval.py 的 llada_generate_blocks 完全对齐，新增 temperature 参数。
    """
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id or 0
    stop_strings = stop_strings or []

    input_ids    = prompt_ids.to(device).clone()
    generated_ids = []

    for _ in range(max_blocks):
        new_block = torch.full((1, block_size), mask_id, dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, new_block], dim=1)
        block_start = input_ids.shape[1] - block_size
        block_end   = input_ids.shape[1]

        input_ids = denoise_block(model, input_ids, block_start, block_end,
                                  mask_id, num_steps, temperature)

        block_tokens = input_ids[0, block_start:block_end].tolist()
        generated_ids.extend(block_tokens)

        current_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # ← 关键：发现 stop_string 立即截断，与 run_llada_eval.py 完全一致
        for stop_str in stop_strings:
            if stop_str in current_text:
                idx = current_text.index(stop_str) + len(stop_str)
                return current_text[:idx], input_ids   # 同时返回 input_ids 供后续拼接

        # EOS 检测
        if all(t == eos_id for t in block_tokens[-8:]):
            return tokenizer.decode(generated_ids, skip_special_tokens=True), input_ids

    return tokenizer.decode(generated_ids, skip_special_tokens=True), input_ids

# ── Prompts（与 run_llada_eval.py 完全一致，含 example response）────────────────
SYSTEM_PROMPT = """You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions. No matter how complex the query, you will not give up until you find the corresponding information.

As you proceed, adhere to the following principles:

1. **Persistent Actions for Answers**: You will engage in many interactions, delving deeply into the topic to explore all possible aspects until a satisfactory answer is found.

2. **Repeated Verification**: Before presenting a Final Answer, you will **cross-check** and **validate the information** you've gathered to confirm its accuracy and reliability.

3. **Attention to Detail**: You will carefully analyze each information source to ensure that all data is current, relevant, and from credible origins."""

USER_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the assistant solves it by calling one or more of the following tools.
<tools>
{{
  "name": "search",
  "description": "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
  "parameters": {{
    "type": "object",
    "properties": {{
      "query": {{
        "type": "array",
        "items": {{
          "type": "string"
        }},
        "description": "Array of query strings. Include 3 or less search queries in a single call."
      }}
    }},
    "required": [
      "query"
    ]
    }}
}}
</tools>

The assistant starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends with answer of the question.

Example response:
<think>
thinking process here
</think>
<tool_call>
{{"name": "search", "arguments": {{"query": ["query string 1", "query string 2"]}}}}
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
User: {question}"""

TOOL_LEFT  = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<tool_response>"
TOOL_RIGHT = "</tool_response><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


# ── Parse helpers（与 run_llada_eval.py 完全对齐）────────────────────────────────
def _parse_tool_call(text: str):
    """从 <tool_call>...</tool_call> 提取 queries list；失败返回 None。"""
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1).strip())
        return obj.get("arguments", {}).get("query", [])
    except Exception:
        m2 = re.search(r'"query"\s*:\s*(\[.*?\])', m.group(1), re.DOTALL)
        if m2:
            try:
                return json.loads(m2.group(1))
            except Exception:
                pass
    return None


def _extract_answer(text: str):
    """从 <|box_start|>...</|box_end|> 提取答案；失败返回 None。"""
    m = re.search(r"<\|box_start\|>(.*?)<\|box_end\|>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def run_one_rollout(model, tokenizer, mask_id, question: str, seed: int,
                    device=None) -> dict:
    """Run a single agentic rollout；返回结构化 trajectory dict。

    生成逻辑与 run_llada_eval.py 的 run_one() 完全对齐：
      - stop_strings=["</tool_call>", "<|box_end|>"]，每 block 后检查并立刻截断
      - turn >= FORCE_ANSWER_TURN 时注入 <|box_start|> 前缀（与 eval 一致）
      - temperature=TEMPERATURE 采样（rollout 需要多样性，eval 用 greedy）
    """
    if device is None:
        device = next(model.parameters()).device
    torch.manual_seed(seed)

    # ── 构建初始 prompt ─────────────────────────────────────────────────────────
    # apply_chat_template 产生 Llama3 格式（<|start_header_id|> 等真实 special token）。
    # <|im_start|>/<|im_end|> 不在该 tokenizer 词表，会拆成普通字符，
    # 而 Llama3 special token 是模型已学过的强结构信号，实测 eval 性能更好（+5pp）。
    user_content = USER_TEMPLATE.format(question=question)
    messages_for_template = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    full_context = tokenizer.apply_chat_template(
        messages_for_template, tokenize=False, add_generation_prompt=True
    )

    stop_strings = ["</tool_call>", "<|box_end|>"]   # 与 run_llada_eval.py 完全一致

    turns          = []
    terminated_by  = "max_turns"
    final_answer   = None
    n_policy_tokens = 0
    n_env_tokens   = 0
    t_start        = time.time()

    for turn_id in range(MAX_TURNS):
        force_answer = (turn_id >= FORCE_ANSWER_TURN)

        if force_answer:
            # 与 run_llada_eval.py 一致：注入 <|box_start|>，强制模型回答
            gen_context = full_context + "<|box_start|>"
            cur_stops   = ["<|box_end|>"]
        else:
            gen_context = full_context
            cur_stops   = stop_strings

        prompt_ids = tokenizer(
            gen_context, return_tensors="pt", add_special_tokens=False
        ).input_ids

        new_text, _ = llada_generate_blocks(
            model, tokenizer, prompt_ids,
            mask_id     = mask_id,
            block_size  = BLOCK_SIZE,
            num_steps   = NUM_STEPS,
            max_blocks  = MAX_BLOCKS,
            stop_strings= cur_stops,
            temperature = TEMPERATURE,
        )
        n_policy_tokens += len(tokenizer(new_text, add_special_tokens=False).input_ids)

        think_m    = re.search(r"<think>(.*?)</think>", new_text, re.DOTALL)
        think_text = think_m.group(1).strip() if think_m else ""
        turn_data  = {"turn_id": turn_id, "think": think_text}

        if force_answer:
            # 截到 <|box_end|> 前（与 run_llada_eval.py 一致）
            ans_text = new_text.split("<|box_end|>")[0].strip() \
                       if "<|box_end|>" in new_text else new_text.strip()
            if ans_text:
                final_answer  = ans_text
                terminated_by = "forced_answer"
            turn_data["final_answer"] = final_answer
            # raw_text：force_answer 时 prompt 已注入 <|box_start|>，完整 policy 输出是两者拼接
            turn_data["raw_text"] = "<|box_start|>" + new_text
            turns.append(turn_data)
            full_context += "<|box_start|>" + new_text
            break

        # ── 正常 turn：先检查 answer，再检查 tool_call（与 run_llada_eval.py 顺序一致）
        ans = _extract_answer(new_text)
        if ans is not None:
            final_answer  = ans
            terminated_by = "final_answer"
            turn_data["final_answer"] = final_answer
            # raw_text：正常 answer turn，new_text 本身包含完整的 <|box_start|>...<|box_end|>
            turn_data["raw_text"] = new_text
            turns.append(turn_data)
            full_context += new_text
            break

        queries = _parse_tool_call(new_text)
        if queries:
            search_result = run_search(queries)
            turn_data["tool_call"]     = {"name": "search", "query": queries}
            turn_data["tool_response"] = search_result
            turn_data["parse_success"] = True
            # raw_text：tool_call turn，new_text 包含 <tool_call>...</tool_call>
            # tool_response 是 env token，单独存在 tool_response 字段，不放进 raw_text
            turn_data["raw_text"] = new_text

            # tool_response 注入方式与 run_llada_eval.py 完全一致
            tool_response_text = (
                f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"<tool_response>\n{search_result}\n</tool_response>"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            full_context += new_text + tool_response_text
            n_env_tokens += len(tokenizer(tool_response_text,
                                          add_special_tokens=False).input_ids)
        else:
            # <tool_call> 解析失败 或 生成了其他内容
            turn_data["parse_success"] = False
            turn_data["raw_text"]      = new_text   # 保存原始生成，便于调试
            terminated_by = "no_action"
            turns.append(turn_data)
            full_context += new_text
            break

        turns.append(turn_data)

        # 上下文长度检查
        ctx_len = len(tokenizer(full_context, add_special_tokens=False).input_ids)
        if ctx_len > 3900:
            terminated_by = "context_overflow"
            break

    latency      = time.time() - t_start
    num_searches = sum(1 for t in turns if t.get("tool_call") and t.get("parse_success"))
    all_queries  = []
    for t in turns:
        q = t.get("tool_call", {}).get("query")
        if isinstance(q, list):
            all_queries.extend([str(qi).lower() for qi in q])
    dup_queries = len(all_queries) - len(set(all_queries))

    return {
        "turns":             turns,
        "final_answer":      final_answer,
        "terminated_by":     terminated_by,
        "num_searches":      num_searches,
        "duplicate_queries": dup_queries,
        "valid_format":      final_answer is not None,
        "n_policy_tokens":   n_policy_tokens,
        "n_env_tokens":      n_env_tokens,
        "total_tokens":      n_policy_tokens + n_env_tokens,
        "rollout_latency_s": round(latency, 2),
    }

# ── 模型 hash ─────────────────────────────────────────────────────────────────
def get_checkpoint_hash(model_path: str) -> str:
    cfg = Path(model_path) / "config.json"
    if cfg.exists():
        return hashlib.md5(cfg.read_bytes()).hexdigest()[:8]
    return "unknown"

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    global BLOCK_SIZE, NUM_STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank",          type=int, default=0)
    parser.add_argument("--world",         type=int, default=8)
    parser.add_argument("--model",         default=DEFAULT_MODEL)
    parser.add_argument("--input",         default=DEFAULT_INPUT)
    parser.add_argument("--output",        default=DEFAULT_OUTPUT)
    parser.add_argument("--n_rolls",       type=int, default=N_ROLLS)
    parser.add_argument("--round_id",      type=int, default=0)
    parser.add_argument("--max_questions", type=int, default=None,
                        help="每 rank 最多处理 N 道题（pilot 模式；None = 全量）")
    parser.add_argument("--block_size",    type=int, default=BLOCK_SIZE,
                        help="Block size for diffusion rollout (must match SFT training)")
    parser.add_argument("--num_steps",     type=int, default=NUM_STEPS,
                        help="每个 block 的 denoising steps")
    args = parser.parse_args()
    BLOCK_SIZE = args.block_size
    NUM_STEPS  = args.num_steps

    # 加载问题池
    with open(args.input) as f:
        all_questions = [json.loads(l) for l in f if l.strip()]

    # 按 rank 分片
    shard = all_questions[args.rank::args.world]
    if args.max_questions is not None:
        shard = shard[:args.max_questions]
        print(f"[rank {args.rank}/{args.world}] [PILOT] 处理前 {len(shard)} 题 × {args.n_rolls} rolls "
              f"= {len(shard)*args.n_rolls} trajectories")
    else:
        print(f"[rank {args.rank}/{args.world}] 处理 {len(shard)} 题 × {args.n_rolls} rolls "
              f"= {len(shard)*args.n_rolls} trajectories")

    # 输出路径
    out_dir = Path(args.output) / f"round{args.round_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"rollouts_rank{args.rank}.jsonl"

    # 断点续跑
    done_qids = set()
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                try: done_qids.add(json.loads(line)["question_id"] + "_" + str(json.loads(line)["rollout_id"]))
                except: pass
        print(f"  已完成 {len(done_qids)//args.n_rolls} 题（断点续跑）")

    # 加载模型（对齐 collect_agentic_rollouts.py）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    model.config.use_cache = False

    ckpt_hash = get_checkpoint_hash(args.model)
    mask_id   = model.config.mask_token_id or 126336

    print(f"模型加载完成. checkpoint_hash={ckpt_hash}  mask_id={mask_id}")

    # 每题生成 n_rolls 条
    seeds_base = [13842, 27391, 45921, 61823, 78254, 92714, 103847, 117659]

    with open(out_file, "a") as fout:
        for q_item in tqdm(shard, desc=f"rank{args.rank}"):
            qid   = q_item["question_id"]
            group_key = f"r{args.round_id}_{qid}"

            # 跳过已完成的 rollout
            done_rolls = sum(1 for r in range(args.n_rolls)
                             if f"{qid}_{r}" in done_qids)
            if done_rolls >= args.n_rolls:
                continue

            group_records = []
            for rollout_id in range(args.n_rolls):
                if f"{qid}_{rollout_id}" in done_qids:
                    continue
                seed = seeds_base[rollout_id % len(seeds_base)] + args.rank * 100000

                traj = run_one_rollout(
                    model, tokenizer, mask_id,
                    q_item["question"], seed
                )

                record = {
                    "round_id":    args.round_id,
                    "question_id": qid,
                    "group_id":    group_key,
                    "rollout_id":  rollout_id,
                    "policy": {
                        "checkpoint":      args.model,
                        "checkpoint_hash": ckpt_hash,
                        "block_length":    BLOCK_SIZE,
                        "denoising_steps": NUM_STEPS,
                        "seed":            seed,
                    },
                    **traj,
                    "question": q_item["question"],
                    "gold": {
                        "short_answers": q_item["short_answers"],
                        "aliases":       q_item.get("aliases", q_item["short_answers"]),
                        "answer_type":   q_item.get("answer_type", "entity"),
                    },
                    "dataset": q_item["dataset"],
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

            print(f"  {qid}: {args.n_rolls} rolls 完成  "
                  f"(ans={'✓' if any(r.get('valid_format') for r in group_records) else '✗'})")

    # 验证完整性
    with open(out_file) as f:
        records = [json.loads(l) for l in f if l.strip()]
    from collections import Counter
    qid_counts = Counter(r["question_id"] for r in records)
    complete = sum(1 for c in qid_counts.values() if c == args.n_rolls)
    print(f"\n[rank {args.rank}] 完成: {len(qid_counts)} 题  "
          f"严格{args.n_rolls}条: {complete}/{len(qid_counts)}  "
          f"总records: {len(records)}")

if __name__ == "__main__":
    main()
