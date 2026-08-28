#!/bin/bash
# =============================================================================
# Pilot 验证脚本
# 检查生成的 rollout 是否：
#   1. 包含搜索调用（<tool_call>）
#   2. 答案格式正确（<|box_start|>/<|box_end|>）
#   3. query 是 array 格式
#   4. 完整的 turns 结构
# =============================================================================

OUTPUT_DIR=${1:-/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/round0_pilot}
PYTHON=/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python

echo "================================================================"
echo "Pilot 验证: $OUTPUT_DIR"
echo "================================================================"

$PYTHON << EOF
import json, sys
from pathlib import Path
from collections import Counter

out_dir = Path("$OUTPUT_DIR") / "round0"

# 收集所有 rank 文件
all_records = []
for f in sorted(out_dir.glob("rollouts_rank*.jsonl")):
    with open(f) as fp:
        recs = [json.loads(l) for l in fp if l.strip()]
    print(f"  {f.name}: {len(recs)} records")
    all_records.extend(recs)

if not all_records:
    print("\n[ERROR] 没有找到任何 records！")
    print("  可能原因：模型还在运行，或脚本报错")
    sys.exit(1)

print(f"\n总 records 数: {len(all_records)}")

# ── 关键指标 ────────────────────────────────────────────────────────────────
with_search     = sum(1 for r in all_records if r.get("num_searches", 0) > 0)
with_answer     = sum(1 for r in all_records if r.get("valid_format", False))
mean_search     = sum(r.get("num_searches", 0) for r in all_records) / len(all_records)
terminations    = Counter(r.get("terminated_by", "unknown") for r in all_records)

print(f"\n=== 搜索行为（最重要！之前修复前此处全为 0）===")
print(f"  有搜索调用: {with_search}/{len(all_records)} = {with_search/len(all_records)*100:.0f}%")
print(f"  平均搜索次数: {mean_search:.2f}")

print(f"\n=== 答案格式 ===")
print(f"  有效答案 (<|box_start|>): {with_answer}/{len(all_records)} = {with_answer/len(all_records)*100:.0f}%")

print(f"\n=== 终止原因分布 ===")
for reason, cnt in sorted(terminations.items(), key=lambda x: -x[1]):
    print(f"  {reason:30s}: {cnt}")

# ── 详细检查第一条 record ────────────────────────────────────────────────────
print(f"\n=== 第 1 条 Record 详情 ===")
r = all_records[0]
print(f"  question_id  : {r.get('question_id')}")
print(f"  question     : {r.get('question', '')[:80]}...")
print(f"  num_searches : {r.get('num_searches')}")
print(f"  terminated_by: {r.get('terminated_by')}")
print(f"  final_answer : {str(r.get('final_answer', ''))[:80]}")
print(f"  valid_format : {r.get('valid_format')}")
print(f"  gold         : {r.get('gold', {}).get('short_answers', [])[:3]}")
print(f"  turns ({len(r.get('turns', []))}个):")
for t in r.get("turns", []):
    tid    = t.get("turn_id")
    tc     = t.get("tool_call")
    tr     = t.get("tool_response")
    ans    = t.get("final_answer")
    think  = (t.get("think") or "")[:60]
    print(f"    Turn {tid}: think='{think}...'")
    if tc:
        q = tc.get("query")
        print(f"             tool_call query={q}")
        if tr:
            print(f"             tool_response: {str(tr)[:100]}...")
    if ans:
        print(f"             final_answer: {ans[:80]}")

# ── array query 验证 ─────────────────────────────────────────────────────────
print(f"\n=== Query 格式验证（应为 list）===")
query_ok = 0
query_bad = 0
for r in all_records:
    for t in r.get("turns", []):
        tc = t.get("tool_call")
        if tc:
            q = tc.get("query")
            if isinstance(q, list):
                query_ok += 1
            else:
                query_bad += 1
                print(f"  [BAD] rank? qid={r.get('question_id')} query type={type(q).__name__}: {q}")

print(f"  array 格式: {query_ok} 条  非 array: {query_bad} 条")

# ── 最终判断 ─────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
search_ok  = with_search / len(all_records) >= 0.5   # ≥50% 有搜索
format_ok  = with_answer / len(all_records) >= 0.3   # ≥30% 有答案
array_ok   = query_bad == 0                           # query 全是 list

PASS = search_ok and array_ok
print(f"  搜索行为  : {'✓ PASS' if search_ok  else '✗ FAIL'} ({with_search/len(all_records)*100:.0f}% ≥ 50%?)")
print(f"  有效答案  : {'✓ PASS' if format_ok  else '✗ FAIL (低于预期，但可接受)'} ({with_answer/len(all_records)*100:.0f}% ≥ 30%?)")
print(f"  Query 格式: {'✓ PASS' if array_ok   else '✗ FAIL'} (全 list)")
print(f"{'='*64}")
if PASS:
    print(f"  ✅ PILOT 通过！可以全量运行 run_round0_rollout.sh")
else:
    print(f"  ❌ PILOT 失败，请检查上方错误后重新运行")
    sys.exit(1)
EOF

EXIT=$?
if [ $EXIT -eq 0 ]; then
    echo ""
    echo "✅ Pilot 验证通过！"
    echo "下一步（全量运行）："
    echo "  1. 清理 pilot 输出（可选）: rm -rf $OUTPUT_DIR"
    echo "  2. 全量启动: screen -dmS round0_master bash $(dirname $0)/run_round0_rollout.sh"
else
    echo ""
    echo "❌ Pilot 验证失败，请检查 log 后再决定是否全量运行"
fi
