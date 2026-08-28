#!/usr/bin/env python3
"""
把 collect_agentic_rollouts.py 产生的 rollout 文件转成训练数据，支持 Plan A/B/C。

Plan A / Plan B:
    与原 make_grpo_data.py 完全相同：组内 z-score 归一化 → 统一 advantage。
    输出格式: {prompt, completion, advantage, reward}

Plan C (SALT-lite):
    对每个 rollout 生成多条训练记录——每条对应一个 search turn 的 prefix completion，
    加上最终 full completion，各自携带按 turn-level credit 调整的 advantage。

    Per-turn advantage:
        credit = 1.0  if gt_keywords_found (search found relevant content)
               = 0.5  if searched but did NOT find GT keywords
        final_turn credit = 0.8  (no search utility to measure)

    advantage_t = base_adv × credit_t
    (base_adv 来自组内归一化，保持组间可比性；credit 在组内调整 turn 级别权重)

    Prefix completion up to search turn t includes:
        - All assistant turns and tool_responses from turn 0 through turn t
        - Ends right after the t-th tool_response (i.e., including the separator
          "<|im_start|>assistant\\n" that primes the next generation)

    Final completion = full messages_to_completion() as in Plan A/B.

    This provides denser training signal (N+1 samples per rollout vs. 1) and
    focuses gradient on search turns that retrieved relevant information.

用法:
    # Plan A / B
    python make_agentic_data.py \\
        --plan A \\
        --rollout_dir /path/to/rollouts/plan_a \\
        --output      /path/to/train_plan_a.jsonl

    # Plan C (SALT-lite)
    python make_agentic_data.py \\
        --plan C \\
        --rollout_dir /path/to/rollouts/plan_c \\
        --output      /path/to/train_plan_c.jsonl
"""

import argparse
import json
import re
import statistics
import string
from collections import defaultdict
from pathlib import Path


# ── Pure-F1 reward helpers（用于 --reward_fn pure_f1 重新计算 reward）──────────
def _normalize_text(s: str) -> str:
    s = s.lower().replace("_", " ")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch if ch not in string.punctuation + "''´`" else " " for ch in s)
    return " ".join(s.split())

def _bool_map(s: str) -> str:
    return {"True": "yes", "False": "no"}.get(s, s)

def _first_sentence(s: str) -> str:
    s = s.strip()
    for sep in [". ", ".\n", "\n\n", "\n"]:
        idx = s.find(sep)
        if idx > 10:
            return s[: idx + 1]
    return s[:200]

def _yes_no(s: str):
    s = s.strip().lower()
    if re.match(r"^(yes[,\s]|yes$)", s):
        return "yes"
    if re.match(r"^(no[,\s]|no$)", s):
        return "no"
    if re.search(r"\bnot the same\b|\bdifferent\b|\bdid not\b|\bis not\b", s):
        return "no"
    if re.search(r"\bthe same\b|\bboth\b|\byes\b", s):
        return "yes"
    return None

def _token_f1(prediction: str, ground_truth: str) -> float:
    if not prediction or not str(prediction).strip():
        return 0.0
    gt_key  = _normalize_text(_bool_map(_first_sentence(str(ground_truth))))
    pred_n  = _normalize_text(_bool_map(str(prediction)))
    gt_tok  = set(gt_key.split())
    pred_tok = set(pred_n.split())
    if not gt_tok:
        return 0.0
    common = gt_tok & pred_tok
    if not common:
        return 0.0
    precision = len(common) / max(len(pred_tok), 1)
    recall    = len(common) / len(gt_tok)
    return 2 * precision * recall / (precision + recall + 1e-9)

def compute_reward_pure_f1(prediction, ground_truth) -> float:
    """Type-aware F1 only: binary exact-match for yes/no, token-F1 for entities.

    No efficiency multiplier, no format bonus — clean correctness signal.
    """
    if not prediction:
        return 0.0
    gt_yn   = _yes_no(str(ground_truth))
    pred_yn = _yes_no(str(prediction))
    if gt_yn is not None and pred_yn is not None:
        return 1.0 if gt_yn == pred_yn else 0.0
    return round(_token_f1(prediction, ground_truth), 4)

# ── Prompt builder（与 collect_agentic_rollouts.py 保持完全一致）──────────────
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


# ── Prompt / Completion builders ──────────────────────────────────────────────
def build_prompt(question: str) -> str:
    """Build the full prompt string (system + user header), Llama3 canonical format."""
    return (
        f"<|startoftext|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{USER_PROMPT_TEMPLATE + question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def messages_to_completion(messages: list) -> str:
    """Flatten messages[2:] into a single completion string.

    Follows the same convention as the original make_grpo_data.py:
    - assistant turns: raw content
    - user (tool_response) turns: wrapped with Llama3 eot_id/header + assistant primer
    """
    parts = []
    for m in messages[2:]:
        if m["role"] == "assistant":
            parts.append(m["content"].strip())
        elif m["role"] == "user":
            content = m["content"].strip()
            parts.append(
                f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
    return "".join(parts)


def messages_to_completion_up_to_search(messages: list, n_search_turns: int) -> str:
    """Build a prefix completion that includes exactly the first `n_search_turns` searches.

    Used for Plan C SALT-lite prefix samples. Iterates messages[2:] in order,
    counting tool_response turns. Stops immediately after the n_search_turns-th
    tool_response (which includes the "<|start_header_id|>assistant<|end_header_id|>\\n\\n" primer that
    conditions the next generation step).

    Args:
        messages: Full message list from the rollout.
        n_search_turns: Stop after this many tool_response turns (1-indexed).

    Returns:
        Completion string ending after the n_search_turns-th tool_response.
        Returns empty string if n_search_turns == 0.
    """
    if n_search_turns == 0:
        return ""

    parts = []
    searches_seen = 0

    for m in messages[2:]:
        if m["role"] == "assistant":
            parts.append(m["content"].strip())
        elif m["role"] == "user":
            content = m["content"].strip()
            if "<tool_response>" in content:
                # This is a search result turn
                parts.append(
                    f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                )
                searches_seen += 1
                if searches_seen >= n_search_turns:
                    break  # Stop after the requested number of search turns
            else:
                # Non-tool-response user turn (edge case, shouldn't occur normally)
                parts.append(
                    f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                )

    return "".join(parts)


# ── Advantage computation ─────────────────────────────────────────────────────
def group_normalize(rewards: list) -> list:
    """Group-level z-score normalization → advantages.

    Returns [0.0, ...] when fewer than 2 rollouts or zero variance (no gradient).
    """
    if len(rewards) <= 1:
        return [0.0] * len(rewards)
    mean = statistics.mean(rewards)
    std  = statistics.stdev(rewards)
    if std < 1e-8:
        return [0.0] * len(rewards)
    return [(r - mean) / std for r in rewards]


def get_turn_credit(gt_found: bool, did_search: bool) -> float:
    """Credit multiplier for a single turn in Plan C.

    - 1.0: searched AND found GT-relevant content → reinforce this search behaviour
    - 0.5: searched but found nothing useful → still train on it, but down-weighted
    - 0.8: final answer turn (no search utility signal available)
    """
    if did_search:
        return 1.0 if gt_found else 0.5
    return 0.8  # final answer turn or no-search turn


# ── Plan A / B processing (identical logic to make_grpo_data.py) ──────────────
def process_plan_ab(rollouts_by_question: dict, min_group: int,
                    max_correct_frac: float, out_f,
                    reward_fn: str = "original") -> dict:
    """Write Plan A/B training records. One record per rollout. Returns stats.

    Args:
        reward_fn: "original" = use stored reward field as-is;
                   "pure_f1"  = recompute reward as type-aware F1 (no efficiency/format bonus).
    """
    stats = {
        "skipped_small": 0,
        "skipped_no_var": 0,
        "skipped_easy": 0,
        "total_samples": 0,
    }

    for question, rollouts in rollouts_by_question.items():
        if len(rollouts) < min_group:
            stats["skipped_small"] += 1
            continue

        # Recompute rewards if requested
        if reward_fn == "pure_f1":
            rewards = [
                compute_reward_pure_f1(r.get("prediction"), r.get("answer", ""))
                for r in rollouts
            ]
            # Back-fill into rollouts so the record carries the new reward value
            for r, rw in zip(rollouts, rewards):
                r["_reward_recomputed"] = rw
        else:
            rewards = [r.get("reward", 0.0) for r in rollouts]

        # Skip groups with no reward variance: advantage = 0 everywhere, no gradient
        if len(set(rewards)) == 1:
            stats["skipped_no_var"] += 1
            continue

        # Skip "easy" questions where the model almost always succeeds
        if max_correct_frac < 1.0:
            correct_frac = sum(1 for r in rewards if r > 0) / len(rewards)
            if correct_frac > max_correct_frac:
                stats["skipped_easy"] += 1
                continue

        advantages = group_normalize(rewards)
        prompt     = build_prompt(question)

        for rollout, adv, rw in zip(rollouts, advantages, rewards):
            completion = messages_to_completion(rollout["messages"])
            if not completion.strip():
                continue

            record = {
                "prompt":     prompt,
                "completion": completion,
                "advantage":  round(adv, 6),
                "reward":     rw,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["total_samples"] += 1

    return stats


# ── Plan C (SALT-lite) processing ─────────────────────────────────────────────
def process_plan_c(rollouts_by_question: dict, min_group: int,
                   max_correct_frac: float, out_f) -> dict:
    """Write Plan C training records.

    For each rollout with N search turns, generates N+1 samples:
      - N prefix samples (one per search turn, ending after that turn's tool_response)
      - 1 full-completion sample (the final answer turn)

    Each sample's advantage = base_adv × credit_multiplier(turn).

    If a rollout has no search turns (went straight to answer), emits 1 sample
    with advantage = base_adv × 0.8 (final answer turn credit).
    """
    stats = {
        "skipped_small": 0,
        "skipped_no_var": 0,
        "skipped_easy": 0,
        "total_samples": 0,
        "prefix_samples": 0,
        "full_samples": 0,
    }

    for question, rollouts in rollouts_by_question.items():
        if len(rollouts) < min_group:
            stats["skipped_small"] += 1
            continue

        rewards = [r.get("reward", 0.0) for r in rollouts]

        if len(set(rewards)) == 1:
            stats["skipped_no_var"] += 1
            continue

        if max_correct_frac < 1.0:
            correct_frac = sum(1 for r in rewards if r > 0) / len(rewards)
            if correct_frac > max_correct_frac:
                stats["skipped_easy"] += 1
                continue

        advantages = group_normalize(rewards)
        prompt     = build_prompt(question)

        for rollout, base_adv in zip(rollouts, advantages):
            messages     = rollout.get("messages", [])
            turn_credits = rollout.get("turn_credits", [])  # from collect --plan C

            full_completion = messages_to_completion(messages)
            if not full_completion.strip():
                continue

            n_search_turns = len(turn_credits)

            if n_search_turns == 0:
                # No searches performed: emit one full-completion sample with
                # the "no search" credit (0.8). This gently down-weights rollouts
                # that skipped searching entirely.
                adv = base_adv * get_turn_credit(gt_found=False, did_search=False)
                record = {
                    "prompt":      prompt,
                    "completion":  full_completion,
                    "advantage":   round(adv, 6),
                    "reward":      rollout.get("reward", 0.0),
                    "sample_type": "full_no_search",
                    "turn_idx":    -1,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["total_samples"] += 1
                stats["full_samples"]  += 1
                continue

            # Emit N prefix samples (one per search turn)
            for t_idx, tc in enumerate(turn_credits):
                # Prefix completion ends after search turn (t_idx + 1)
                prefix_completion = messages_to_completion_up_to_search(
                    messages, t_idx + 1
                )
                if not prefix_completion.strip():
                    continue

                gt_found = tc.get("gt_keywords_found", False)
                credit   = get_turn_credit(gt_found=gt_found, did_search=True)
                adv      = base_adv * credit

                record = {
                    "prompt":            prompt,
                    "completion":        prefix_completion,
                    "advantage":         round(adv, 6),
                    "reward":            rollout.get("reward", 0.0),
                    "sample_type":       "prefix_search",
                    "turn_idx":          t_idx,
                    "gt_keywords_found": gt_found,
                    "credit":            credit,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["total_samples"]  += 1
                stats["prefix_samples"] += 1

            # Emit 1 full-completion sample (final answer turn, credit = 0.8)
            final_credit = get_turn_credit(gt_found=False, did_search=False)
            final_adv    = base_adv * final_credit
            record = {
                "prompt":      prompt,
                "completion":  full_completion,
                "advantage":   round(final_adv, 6),
                "reward":      rollout.get("reward", 0.0),
                "sample_type": "full_with_search",
                "turn_idx":    n_search_turns,   # indicates "after all searches"
                "credit":      final_credit,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["total_samples"] += 1
            stats["full_samples"]  += 1

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert agentic rollouts to GRPO training data. "
            "Plan A/B: uniform advantage per rollout. "
            "Plan C: per-turn advantages via SALT-lite credit assignment."
        )
    )
    parser.add_argument(
        "--plan", choices=["A", "B", "C"], default="A",
        help=(
            "Reward plan of the input rollouts: "
            "A = basic+format_bonus, "
            "B = multi-component, "
            "C = SALT-lite (uses turn_credits field)"
        ),
    )
    parser.add_argument(
        "--rollout_dir", required=True,
        help="Directory containing rank*.jsonl rollout files",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSONL path for training data",
    )
    parser.add_argument(
        "--min_group", type=int, default=2,
        help="Minimum rollouts per question to include in training (default: 2)",
    )
    parser.add_argument(
        "--max_correct_frac", type=float, default=1.0,
        help=(
            "Skip questions where >this fraction of rollouts are correct. "
            "E.g. 0.875 skips 'easy' questions (≥7/8 correct). Default: 1.0 (keep all)."
        ),
    )
    parser.add_argument(
        "--reward_fn", choices=["original", "pure_f1"], default="original",
        help=(
            "Reward function to use for advantage computation. "
            "'original' = use stored reward field as-is (default). "
            "'pure_f1'  = recompute as type-aware F1 only (no efficiency/format bonus). "
            "Requires rollouts to have 'prediction' and 'answer' fields."
        ),
    )
    parser.add_argument(
        "--n_questions", type=int, default=0,
        help=(
            "Limit to the first N unique questions (sorted by question text for reproducibility). "
            "0 = use all questions (default)."
        ),
    )
    args = parser.parse_args()

    # ── Load all rollouts ──────────────────────────────────────────────────────
    rollout_dir  = Path(args.rollout_dir)
    all_rollouts = []
    for fpath in sorted(rollout_dir.glob("rank*.jsonl")):
        for line in open(fpath):
            try:
                all_rollouts.append(json.loads(line))
            except Exception:
                pass
    print(f"Loaded {len(all_rollouts)} rollout records from {rollout_dir}")

    # Plan C needs turn_credits: warn if missing
    if args.plan == "C":
        with_credits = sum(1 for r in all_rollouts if "turn_credits" in r)
        if with_credits == 0:
            print(
                "WARNING: --plan C selected but no rollouts have 'turn_credits' field. "
                "Make sure to collect rollouts with --plan C in collect_agentic_rollouts.py."
            )
        else:
            print(f"  {with_credits}/{len(all_rollouts)} rollouts have turn_credits")

    # ── Group by question ──────────────────────────────────────────────────────
    by_question = defaultdict(list)
    for r in all_rollouts:
        # Skip error records (empty messages)
        if r.get("messages"):
            by_question[r["question"]].append(r)
    print(f"Unique questions: {len(by_question)}")

    # Limit to first N questions if requested (preserves reproducibility)
    if args.n_questions > 0:
        all_qs = sorted(by_question.keys())[:args.n_questions]
        by_question = {q: by_question[q] for q in all_qs}
        print(f"  → Limited to {len(by_question)} questions (--n_questions {args.n_questions})")

    if args.reward_fn == "pure_f1":
        print("  → Reward function: pure type-aware F1 (recomputing from prediction/answer)")

    # ── Write training data ────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as out_f:
        if args.plan in ("A", "B"):
            stats = process_plan_ab(
                by_question, args.min_group, args.max_correct_frac, out_f,
                reward_fn=args.reward_fn,
            )
        else:
            stats = process_plan_c(
                by_question, args.min_group, args.max_correct_frac, out_f
            )

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n--- Plan {args.plan} data generation complete ---")
    print(f"Skipped (group too small <{args.min_group}): {stats['skipped_small']}")
    print(f"Skipped (zero reward variance):               {stats['skipped_no_var']}")
    print(f"Skipped (too easy >{args.max_correct_frac:.0%}):              {stats['skipped_easy']}")
    print(f"Total training samples written:               {stats['total_samples']}")

    if args.plan == "C":
        print(f"  - Prefix (search turn) samples: {stats.get('prefix_samples', 0)}")
        print(f"  - Full completion samples:       {stats.get('full_samples', 0)}")

    # Overall reward / correctness statistics
    rewards_all = [
        r.get("reward", 0.0) for rollouts in by_question.values() for r in rollouts
    ]
    if rewards_all:
        correct_rate = sum(1 for r in rewards_all if r > 0) / len(rewards_all)
        mean_reward  = sum(rewards_all) / len(rewards_all)
        print(
            f"\nRollout stats over all {len(rewards_all)} records:"
            f"  correct_rate={correct_rate:.1%}"
            f"  mean_reward={mean_reward:.4f}"
        )

    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
