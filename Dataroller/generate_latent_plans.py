#!/usr/bin/env python3
"""
根据已有的成功推理路径，为每道训练题生成压缩的短搜索计划（1-3句话），
作为 latent think 的监督信号。

输入：包含完整 messages 轨迹的 rollout JSONL（raw_rollouts_shard*.jsonl）
输出格式（JSONL）：
  {"question": "...", "answer": "...", "plan": "I need to find X and Y, then compare..."}

用法：
  python generate_latent_plans.py \
    --input sft_output_v2/raw_rollouts_shard00_of08_*.jsonl \
    --output data/sft_train_plans.jsonl
"""

import argparse
import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


# ── 加载配置 ──────────────────────────────────────────────────────────────────
def _load_config():
    for p in [
        "/research/cbim/vast/mz751/Projects/DLLM-Searcher/config.json",
        os.path.join(os.path.dirname(__file__), "..", "config.json"),
    ]:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}

_cfg = _load_config()
API_KEY  = os.getenv("JUDGE_API_KEY")  or _cfg.get("judge_api_key",  "")
API_BASE = os.getenv("JUDGE_API_BASE") or _cfg.get("judge_api_base", "https://api.openai.com/v1")
MODEL    = os.getenv("JUDGE_MODEL")    or _cfg.get("judge_model",    "gpt-4o")


SYSTEM_PROMPT = """You are an expert at compressing successful reasoning trajectories into concise search plans.

You will be given a question and its ACTUAL successful reasoning trajectory (think steps, search queries, search results, and final answer).

Your task: compress this trajectory into a SHORT search plan of 1-3 sentences that captures:
- What key facts needed to be found
- What search strategy was used (in what order)
- How the results were combined to reach the answer

Important:
- Base your plan STRICTLY on the actual trajectory provided, not on hypothetical approaches
- Be concise and specific
- Do NOT include the answer itself
- Output only the plan, nothing else"""

USER_TEMPLATE = """Question: {question}

Actual successful trajectory:
{trajectory}

Compressed search plan (1-3 sentences):"""


def extract_trajectory(messages: list) -> str:
    """从 messages 里提取 think + tool_call + tool_response 的关键部分（截断过长内容）。"""
    parts = []
    for m in messages:
        role    = m["role"]
        content = m["content"]
        if role == "system":
            continue
        if role == "user":
            # 只保留 tool_response 部分（去掉长 system prompt）
            if "<tool_response>" in content:
                # 截取 tool_response 内容，最多200字
                tr = re.search(r"<tool_response>(.*?)</tool_response>", content, re.DOTALL)
                if tr:
                    snippet = tr.group(1).strip()[:300]
                    parts.append(f"[Search results]: {snippet}...")
        elif role == "assistant":
            # 提取 think 内容
            for think in re.findall(r"<think>(.*?)</think>", content, re.DOTALL):
                parts.append(f"[Think]: {think.strip()[:200]}")
            # 提取 tool_call 的 query
            tc = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
            if tc:
                try:
                    obj = json.loads(tc.group(1).strip())
                    queries = obj.get("arguments", {}).get("query", [])
                    parts.append(f"[Search queries]: {queries}")
                except Exception:
                    pass
            # 提取最终答案
            box = re.search(r"<\|box_start\|>(.*?)<\|box_end\|>", content, re.DOTALL)
            if box:
                parts.append(f"[Final answer]: {box.group(1).strip()[:100]}")
    return "\n".join(parts)


def generate_plan(client: OpenAI, question: str, trajectory: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": USER_TEMPLATE.format(
                        question=question, trajectory=trajectory
                    )},
                ],
                temperature=0.3,
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [ERROR] {e}")
                return ""
            time.sleep(2 ** attempt)
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default="sft_output_v2/raw_rollouts_shard*.jsonl",
                        help="支持 glob 通配符")
    parser.add_argument("--output",  default="data/sft_train_plans.jsonl")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--termination", default="answer",
                        help="只处理指定 termination 的样本（默认 answer）")
    args = parser.parse_args()

    # 加载所有 rollout 数据
    input_files = sorted(glob.glob(args.input))
    if not input_files:
        print(f"找不到文件：{args.input}")
        return

    samples = []
    for fpath in input_files:
        with open(fpath) as f:
            for line in f:
                if line.strip():
                    s = json.loads(line)
                    if s.get("termination") == args.termination:
                        samples.append(s)
    print(f"共 {len(samples)} 条成功 rollout，模型：{MODEL}")

    # 跳过已完成的
    out_path = Path(args.output)
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["question"])
                except Exception:
                    pass
        print(f"已完成 {len(done)} 条，继续剩余")

    todo = [s for s in samples if s["question"] not in done]
    if not todo:
        print("全部完成！")
        return

    client = OpenAI(api_key=API_KEY, base_url=API_BASE)

    def process(sample):
        trajectory = extract_trajectory(sample["messages"])
        plan = generate_plan(client, sample["question"], trajectory)
        return {
            "question": sample["question"],
            "answer":   sample["answer"],
            "plan":     plan,
        }

    with open(out_path, "a") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process, s): s for s in todo}
            for fut in tqdm(as_completed(futures), total=len(todo), desc="生成搜索计划"):
                result = fut.result()
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()

    print(f"\n完成！结果保存至 {out_path}")

    # 展示几个例子
    print("\n=== 示例 ===")
    with open(out_path) as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            d = json.loads(line)
            print(f"Q: {d['question'][:80]}")
            print(f"P: {d['plan']}")
            print()


if __name__ == "__main__":
    main()
