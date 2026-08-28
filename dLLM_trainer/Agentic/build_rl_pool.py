#!/usr/bin/env python3
"""
Phase 1: 构建 RL Question Pool

从 HotpotQA / 2WikiMultiHopQA / Musique 的 train 分区采样，
保证与 eval_600 零重叠，生成嵌套子集：
    rl_pool_256.jsonl ⊂ rl_pool_512.jsonl ⊂ rl_pool_full.jsonl

每条记录格式：
{
  "question_id":    "hotpot_train_00123",
  "dataset":        "hotpotqa",
  "question":       "...",
  "short_answers":  ["Donald O'Connor"],
  "aliases":        ["Donald O'Connor", "Donald David Dixon Ronald O'Connor"],
  "answer_type":    "entity",   # entity / yes_no / number / date / list / other_short_text
  "project_split":  "rl_train"
}
"""

import json, re, random, string, argparse, os
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

# ─── 路径 ────────────────────────────────────────────────────────────────────
ROOT      = Path("/research/cbim/vast/mz751/Projects/DLLM-Searcher")
EVAL_FILE = ROOT / "dLLM_trainer/VRPO/data/eval_600.jsonl"
OUT_DIR   = ROOT / "dLLM_trainer/VRPO/data/rl_pool"

# ─── 分层采样配置 ─────────────────────────────────────────────────────────────
# 每个数据集在 full pool 里的目标题数
# Musique 暂时跳过（HuggingFace 名称未知），后续找到本地文件后补充
DATASET_TARGETS = {
    "hotpotqa":        1000,   # 增大以弥补 musique 缺失
    "2wikimultihopqa": 1000,
}
# 256 / 512 子集的构成（均分）
SUBSETS = {256: {"hotpotqa": 128, "2wikimultihopqa": 128},
           512: {"hotpotqa": 256, "2wikimultihopqa": 256}}

# ─── answer_type 分类 ─────────────────────────────────────────────────────────
_NUM_PAT  = re.compile(r'^-?\d[\d,\.]*$')
_DATE_PAT = re.compile(r'\b(19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')

def classify_answer_type(answer: str) -> str:
    a = answer.strip().lower()
    if a in {"yes", "no", "true", "false"}:
        return "yes_no"
    if _NUM_PAT.match(a.replace(",", "")):
        return "number"
    if _DATE_PAT.search(a) and len(a.split()) <= 4:
        return "date"
    if "," in a and len(a.split(",")) >= 2 and len(a.split()) <= 12:
        return "list"
    if len(a.split()) <= 8:
        return "entity"
    return "other_short_text"

# ─── answer normalization (SQuAD-style) ──────────────────────────────────────
def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch if ch not in set(string.punctuation) else " " for ch in s)
    return " ".join(s.split())

# ─── 构建 aliases ──────────────────────────────────────────────────────────────
def make_aliases(answer: str, raw_item: dict, dataset: str) -> list:
    """提取别名：从原始数据的 aliases / answer_aliases / alternative_answers 字段。"""
    aliases = {answer.strip()}
    for field in ["aliases", "answer_aliases", "alternative_answers", "answers"]:
        val = raw_item.get(field)
        if isinstance(val, list):
            for v in val:
                if isinstance(v, str) and v.strip():
                    aliases.add(v.strip())
        elif isinstance(val, str) and val.strip():
            aliases.add(val.strip())
    # Musique: `answer_aliases` 可能藏在 `answer_aliases` 字段
    if dataset == "musique" and "answer_aliases" in raw_item:
        for a in (raw_item["answer_aliases"] or []):
            if isinstance(a, str):
                aliases.add(a.strip())
    return sorted(aliases)

# ─── 加载并处理数据集 ─────────────────────────────────────────────────────────
def load_and_process(dataset_name: str, eval_qs: set, target: int, seed: int = 42) -> list:
    """从 HuggingFace 加载 train split，过滤 eval_600，分类，采样到 target 题。"""
    print(f"\n[{dataset_name}] 加载 train split...")

    if dataset_name == "hotpotqa":
        ds = load_dataset("hotpot_qa", "distractor", split="train")
        def to_item(raw):
            q   = raw["question"].strip()
            ans = raw["answer"].strip()
            return {"question": q, "answer": ans, "raw": raw}

    elif dataset_name == "2wikimultihopqa":
        ds = load_dataset("xanhho/2WikiMultiHopQA", split="train")
        def to_item(raw):
            q   = raw["question"].strip()
            ans = raw["answer"].strip()
            return {"question": q, "answer": ans, "raw": raw}

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"  原始条数: {len(ds)}")

    # 转换 + 过滤
    items = []
    for raw in ds:
        try:
            item = to_item(raw)
        except Exception:
            continue
        if not item["question"] or not item["answer"]:
            continue
        if item["question"].lower() in eval_qs:
            continue
        items.append(item)

    print(f"  过滤 eval_600 后: {len(items)}")

    # 按 answer_type 分层采样
    by_type = defaultdict(list)
    for item in items:
        atype = classify_answer_type(item["answer"])
        by_type[atype].append(item)

    print(f"  answer_type 分布: { {k: len(v) for k, v in by_type.items()} }")

    # 分层采样：每种类型按比例取到 target
    total_before = sum(len(v) for v in by_type.values())
    sampled = []
    rng = random.Random(seed)
    for atype, pool in by_type.items():
        n = max(1, round(len(pool) / total_before * target))
        n = min(n, len(pool))
        sampled.extend(rng.sample(pool, n))

    # 若不足 target，从剩余随机补足
    if len(sampled) < target:
        sampled_qs = {i["question"] for i in sampled}
        rest = [i for i in items if i["question"] not in sampled_qs]
        rng.shuffle(rest)
        sampled.extend(rest[:target - len(sampled)])

    # 截断到 target
    rng.shuffle(sampled)
    sampled = sampled[:target]
    print(f"  采样到: {len(sampled)} 题")
    return sampled

# ─── 构造 pool record ─────────────────────────────────────────────────────────
def make_record(item: dict, dataset: str, idx: int) -> dict:
    q   = item["question"]
    ans = item["answer"]
    raw = item["raw"]
    qid = f"{dataset.replace('hotpotqa','hotpot').replace('2wikimultihopqa','2wiki').replace('musique','musique')}_train_{idx:05d}"
    return {
        "question_id":   qid,
        "dataset":       dataset,
        "question":      q,
        "short_answers": [ans],
        "aliases":       make_aliases(ans, raw, dataset),
        "answer_type":   classify_answer_type(ans),
        "project_split": "rl_train",
    }

# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载 eval_600 问题（用于过滤）
    eval_qs = set()
    with open(EVAL_FILE) as f:
        for line in f:
            r = json.loads(line)
            eval_qs.add(r["question"].strip().lower())
    print(f"eval_600 问题数（用于过滤）: {len(eval_qs)}")

    # 加载并采样每个数据集
    all_items = {}
    for ds_name, target in DATASET_TARGETS.items():
        items = load_and_process(ds_name, eval_qs, target, seed=args.seed)
        all_items[ds_name] = items

    # 构造 record，并按嵌套子集写出
    # 先确定 full pool 的顺序（按数据集，各自内部打乱）
    full_records = []
    counters = defaultdict(int)
    for ds_name, items in all_items.items():
        for item in items:
            counters[ds_name] += 1
            rec = make_record(item, ds_name, counters[ds_name])
            full_records.append(rec)

    # 打乱 full pool
    rng = random.Random(args.seed)
    rng.shuffle(full_records)

    # 写 full pool
    full_path = OUT_DIR / "rl_pool_full.jsonl"
    with open(full_path, "w") as f:
        for r in full_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✓ rl_pool_full.jsonl  {len(full_records)} 题 → {full_path}")

    # 构建 512 ⊂ full
    # 按数据集分层取前 N 题，保证子集关系
    def build_subset(n_per_ds: dict) -> list:
        subset = []
        ds_seen = defaultdict(int)
        for rec in full_records:
            ds = rec["dataset"]
            if ds_seen[ds] < n_per_ds.get(ds, 0):
                subset.append(rec)
                ds_seen[ds] += 1
        return subset

    for subset_size, n_per_ds in sorted(SUBSETS.items()):
        subset = build_subset(n_per_ds)
        path = OUT_DIR / f"rl_pool_{subset_size}.jsonl"
        with open(path, "w") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✓ rl_pool_{subset_size}.jsonl  {len(subset)} 题  "
              f"( {dict(defaultdict(int, {r['dataset']: 0 for r in subset}))} ) → {path}")

    # 统计汇报
    print(f"\n=== Pool 统计 ===")
    for ds_name in DATASET_TARGETS:
        recs = [r for r in full_records if r["dataset"] == ds_name]
        type_dist = defaultdict(int)
        for r in recs:
            type_dist[r["answer_type"]] += 1
        print(f"  {ds_name}: {len(recs)} 题  types={dict(type_dist)}")

    # 确认与 eval_600 零重叠
    pool_qs = {r["question"].lower() for r in full_records}
    overlap = pool_qs & eval_qs
    print(f"\n✓ RL pool ∩ eval_600 = {len(overlap)} 题 （应为 0）")
    if overlap:
        print(f"  [WARNING] 有重叠: {list(overlap)[:5]}")

    print("\n完成！")

if __name__ == "__main__":
    main()
