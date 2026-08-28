#!/usr/bin/env python3
"""
把 eval 结果 JSONL 渲染成可读的 HTML 报告。

用法:
  python my_eval/show_results.py \
    --files x0pred:path/to/x0pred.jsonl latent:path/to/latent.jsonl \
    --output my_eval/results_report.html

  # 单文件
  python my_eval/show_results.py --files x0pred:x0pred_ckpt6_100.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ── 判断预测是否正确（简单词覆盖，用于显示颜色，不影响 LLM Judge 分数） ──────
def quick_judge(prediction, answer) -> bool:
    if not prediction or not answer:
        return False
    pred_l  = prediction.lower()
    ans_words = re.findall(r'\w+', answer.lower())
    key_words = [w for w in ans_words if len(w) > 3][:6]
    if not key_words:
        return False
    hits = sum(1 for w in key_words if w in pred_l)
    return hits >= len(key_words) * 0.5


def parse_messages(messages):
    """从 messages 列表提取对话轮次：[(role, content), ...]，跳过 system/首个user。"""
    turns = []
    for m in messages:
        if m["role"] == "system":
            continue
        turns.append((m["role"], m["content"]))
    return turns


def render_content(text: str) -> str:
    """把 think/tool_call/tool_response/box 标签渲染成带样式的 HTML。"""
    if not text:
        return "<em>(空)</em>"
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 还原 HTML 转义后，用正则处理标签（先转义保护再还原）
    # 这里直接在原始文本上处理，不 escape
    def _esc(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def highlight(raw: str) -> str:
        # think
        raw = re.sub(
            r"&lt;think&gt;(.*?)&lt;/think&gt;",
            lambda m: f'<span class="think"><b>&lt;think&gt;</b>{m.group(1)}<b>&lt;/think&gt;</b></span>',
            raw, flags=re.DOTALL
        )
        # tool_call
        raw = re.sub(
            r"&lt;tool_call&gt;(.*?)&lt;/tool_call&gt;",
            lambda m: f'<span class="tool-call"><b>&lt;tool_call&gt;</b><code>{m.group(1)}</code><b>&lt;/tool_call&gt;</b></span>',
            raw, flags=re.DOTALL
        )
        # tool_response
        raw = re.sub(
            r"&lt;tool_response&gt;(.*?)&lt;/tool_response&gt;",
            lambda m: f'<span class="tool-response"><b>&lt;tool_response&gt;</b>{m.group(1)}<b>&lt;/tool_response&gt;</b></span>',
            raw, flags=re.DOTALL
        )
        # box
        raw = re.sub(
            r"&lt;\|box_start\|&gt;(.*?)&lt;\|box_end\|&gt;",
            lambda m: f'<span class="answer-box"><b>【最终答案】</b>{m.group(1)}</span>',
            raw, flags=re.DOTALL
        )
        return raw

    return highlight(_esc(text))


def build_html(datasets: list[tuple[str, list[dict]]]) -> str:
    """
    datasets: [(label, records), ...]
    records: list of dicts with keys question/answer/prediction/num_turns/messages
    """
    n = max(len(r) for _, r in datasets)

    tab_heads = ""
    tab_contents = ""
    for di, (label, records) in enumerate(datasets):
        active = "active" if di == 0 else ""
        tab_heads += f'<li class="nav-item"><a class="nav-link {active}" data-bs-toggle="tab" href="#tab{di}">{label} ({len(records)}题)</a></li>\n'

        answered = sum(1 for r in records if r.get("prediction"))
        correct  = sum(1 for r in records if quick_judge(r.get("prediction"), r.get("answer")))
        avg_t    = sum(r.get("num_turns", 0) for r in records) / max(len(records), 1)

        rows = ""
        for i, rec in enumerate(records):
            q      = rec.get("question", "")
            ans    = rec.get("answer", "")
            pred   = rec.get("prediction") or ""
            turns  = rec.get("num_turns", 0)
            term   = rec.get("termination_reason", "")
            msgs   = rec.get("messages", [])

            ok = quick_judge(pred, ans)
            row_cls = "table-success" if ok else ("table-danger" if pred else "table-warning")
            badge   = '<span class="badge bg-success">✓</span>' if ok else (
                      '<span class="badge bg-danger">✗</span>' if pred else
                      '<span class="badge bg-warning text-dark">无答案</span>')

            # 对话轮次折叠
            conv_id = f"conv{di}_{i}"
            conv_html = ""
            for role, content in parse_messages(msgs):
                role_cls  = "role-user" if role == "user" else "role-assistant"
                role_name = "用户/工具" if role == "user" else "模型"
                conv_html += f"""
                <div class="message {role_cls}">
                  <div class="role-label">{role_name}</div>
                  <div class="message-body">{render_content(content)}</div>
                </div>"""

            rows += f"""
            <tr class="{row_cls}">
              <td class="text-center">{i+1}</td>
              <td>{badge}</td>
              <td class="q-cell">{q}</td>
              <td class="ans-cell">{ans}</td>
              <td class="pred-cell">{pred or '<em>无</em>'}</td>
              <td class="text-center">{turns}<br><small class="text-muted">{term}</small></td>
              <td>
                <button class="btn btn-sm btn-outline-secondary"
                        onclick="toggleConv('{conv_id}')">展开</button>
                <div id="{conv_id}" class="conv-panel" style="display:none">
                  {conv_html}
                </div>
              </td>
            </tr>"""

        tab_contents += f"""
        <div class="tab-pane fade {'show active' if di == 0 else ''}" id="tab{di}">
          <div class="stats-bar">
            <span>📊 有答案：<b>{answered}/{len(records)}</b></span>
            <span>✓ 词覆盖估算：<b>{correct}/{len(records)}</b></span>
            <span>🔄 平均轮数：<b>{avg_t:.2f}</b></span>
          </div>
          <table class="table table-bordered table-hover table-sm align-middle">
            <thead class="table-dark">
              <tr>
                <th>#</th><th>判断</th><th>问题</th>
                <th>标准答案</th><th>模型预测</th>
                <th>轮次</th><th>对话</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>LLaDA Eval 结果报告</title>
  <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
  <style>
    body {{ font-size: 14px; padding: 16px; }}
    .stats-bar {{ display:flex; gap:24px; padding:10px 0 14px; font-size:15px; }}
    .q-cell   {{ max-width:260px; font-weight:500; }}
    .ans-cell {{ max-width:220px; color:#555; font-size:12px; }}
    .pred-cell{{ max-width:220px; font-size:12px; }}
    .conv-panel {{ margin-top:10px; border-top:1px solid #ddd; padding-top:8px; }}
    .message  {{ margin:6px 0; padding:8px 10px; border-radius:6px; }}
    .role-user      {{ background:#f0f4ff; }}
    .role-assistant {{ background:#f6fff0; }}
    .role-label     {{ font-size:11px; font-weight:700; color:#888; margin-bottom:3px; }}
    .message-body   {{ white-space:pre-wrap; font-size:12px; line-height:1.6; }}
    .think        {{ display:block; background:#fffbe6; border-left:3px solid #f0a; padding:4px 8px; margin:4px 0; }}
    .tool-call    {{ display:block; background:#e8f4fd; border-left:3px solid #09f; padding:4px 8px; margin:4px 0; }}
    .tool-response{{ display:block; background:#f9f9f9; border-left:3px solid #aaa; padding:4px 8px; margin:4px 0;
                     max-height:160px; overflow-y:auto; }}
    .answer-box   {{ display:block; background:#e6ffe6; border-left:3px solid #0c0; padding:6px 10px; margin:4px 0;
                     font-weight:600; font-size:13px; }}
    code {{ font-size:11px; }}
  </style>
</head>
<body>
  <h4 class="mb-3">LLaDA Eval 结果报告</h4>
  <ul class="nav nav-tabs mb-3">
    {tab_heads}
  </ul>
  <div class="tab-content">
    {tab_contents}
  </div>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    function toggleConv(id) {{
      var el = document.getElementById(id);
      el.style.display = el.style.display === 'none' ? 'block' : 'none';
    }}
  </script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", nargs="+", required=True,
        metavar="LABEL:PATH",
        help="格式: label:path.jsonl，可传多个"
    )
    parser.add_argument(
        "--output", default="my_eval/results_report.html",
        help="输出 HTML 文件路径"
    )
    parser.add_argument(
        "--max", type=int, default=0,
        help="每个文件最多显示 N 题（0=全部）"
    )
    args = parser.parse_args()

    datasets = []
    for spec in args.files:
        if ":" not in spec:
            print(f"格式错误（需要 label:path）: {spec}", file=sys.stderr)
            sys.exit(1)
        label, path = spec.split(":", 1)
        p = Path(path)
        if not p.exists():
            print(f"文件不存在: {path}", file=sys.stderr)
            sys.exit(1)
        records = []
        with open(p) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        if args.max > 0:
            records = records[:args.max]
        datasets.append((label, records))
        print(f"  加载 [{label}]: {len(records)} 题  来自 {path}")

    html = build_html(datasets)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\n报告已生成: {out.resolve()}")


if __name__ == "__main__":
    main()
