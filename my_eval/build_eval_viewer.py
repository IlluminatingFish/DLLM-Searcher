"""
Build an interactive HTML viewer for sft_llada_ckpt6_forced eval results.
600 questions, per-round think/tool call, error analysis.
"""
import json, re, html
from pathlib import Path

EVAL_INPUT  = Path("/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/data/eval_600.jsonl")
EVAL_OUTPUT = Path("/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/output/llada_eval/sft_llada_ckpt6_forced_600_shortgt.jsonl")
OUT_HTML    = Path("/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/output/llada_eval/eval_viewer.html")

# ---------- load ----------
input_rows = {r["question"]: r for r in (json.loads(l) for l in EVAL_INPUT.open() if l.strip())}
output_rows = [json.loads(l) for l in EVAL_OUTPUT.open() if l.strip()]

# ---------- helpers ----------
def extract_think(text):
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    return m.group(1).strip() if m else ""

def extract_tool_call(text):
    m = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            queries = obj.get("arguments", {}).get("query", [])
            return queries if isinstance(queries, list) else [str(queries)]
        except:
            return [m.group(1).strip()[:200]]
    return []

def extract_tool_response(text):
    m = re.search(r'<tool_response>(.*?)</tool_response>', text, re.DOTALL)
    return m.group(1).strip() if m else ""

def parse_rounds(messages):
    rounds = []
    for msg in messages:
        role = msg.get("role", "")
        content = str(msg.get("content", ""))
        if role == "assistant":
            think = extract_think(content)
            queries = extract_tool_call(content)
            # final answer: strip think+tool_call, get remainder
            answer_part = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            answer_part = re.sub(r'<tool_call>.*?</tool_call>', '', answer_part, flags=re.DOTALL)
            answer_part = answer_part.strip()
            rounds.append({"role": "assistant", "think": think,
                           "queries": queries, "answer_text": answer_part})
        elif role == "user" and "<tool_response>" in content:
            resp = extract_tool_response(content)
            rounds.append({"role": "tool_response", "content": resp[:600]})
    return rounds

def is_correct(row):
    gt   = str(row.get("answer", "")).strip()
    pred = str(row.get("prediction", "")).strip()
    if not gt or not pred or pred == "None": return False
    return gt.lower() in pred.lower()

def error_category(row, rounds):
    pred = str(row.get("prediction", "")).strip()
    gt   = str(row.get("answer", "")).strip().lower()
    if pred == "None" or not pred:
        return "无答案（超时）"
    has_search = any(r["role"] == "assistant" and r["queries"] for r in rounds)
    if not has_search:
        return "未搜索直接回答"
    # check if gt appears in any tool_response
    all_results = " ".join(r["content"] for r in rounds if r["role"] == "tool_response").lower()
    if gt in all_results:
        return "搜到答案但提取失败"
    return "搜索结果未包含答案"

# ---------- build records ----------
records = []
for row in output_rows:
    q = row["question"]
    inp = input_rows.get(q, {})
    source = inp.get("source", "unknown")
    rounds = parse_rounds(row.get("messages", []))
    correct = is_correct(row)
    pred = str(row.get("prediction", "")).strip()
    gt = str(row.get("answer", "")).strip()
    err = "" if correct else error_category(row, rounds)
    records.append({
        "question": q,
        "gt": gt,
        "source": source,
        "prediction": pred if pred != "None" else "",
        "correct": correct,
        "error": err,
        "num_turns": row.get("num_turns", 0),
        "termination": row.get("termination_reason", ""),
        "rounds": rounds,
    })

# ---------- stats ----------
n = len(records)
n_correct = sum(1 for r in records if r["correct"])
sources = sorted(set(r["source"] for r in records))
errors = sorted(set(r["error"] for r in records if r["error"]))

# ---------- HTML generation ----------
def e(s): return html.escape(str(s))

def render_rounds(rounds):
    parts = []
    turn = 0
    for rd in rounds:
        if rd["role"] == "assistant":
            turn += 1
            parts.append(f'<div class="turn"><div class="turn-label">Turn {turn}</div>')
            if rd["think"]:
                parts.append(f'<div class="think-block"><span class="block-tag tag-think">think</span><div class="block-body">{e(rd["think"])}</div></div>')
            if rd["queries"]:
                for q in rd["queries"]:
                    parts.append(f'<div class="tool-block"><span class="block-tag tag-tool">search</span><div class="block-body">{e(q)}</div></div>')
            if rd["answer_text"] and not rd["queries"]:
                parts.append(f'<div class="ans-block"><span class="block-tag tag-ans">answer</span><div class="block-body">{e(rd["answer_text"][:300])}</div></div>')
            parts.append('</div>')
        elif rd["role"] == "tool_response":
            parts.append(f'<div class="tool-resp"><span class="block-tag tag-resp">results</span><div class="block-body resp-body">{e(rd["content"])}</div></div>')
    return "".join(parts)

cards_json = json.dumps(records, ensure_ascii=False)

src_btns = "".join(
    '<button class="filter-btn" onclick="setFilter(\'source\',\'' + e(s) + '\')">' + e(s) + '</button>'
    for s in sources)
err_btns = "".join(
    '<button class="filter-btn" onclick="setFilter(\'error\',\'' + e(s) + '\')">' + e(s) + '</button>'
    for s in errors)

# ---------- HTML template (use __PLACEHOLDER__ tokens, no f-string) ----------
TMPL = open(__file__).read()  # not used; template is inline below
STYLE = """
:root {
  --bg:#F2F4F8; --surface:#fff; --surface2:#F7F9FC;
  --border:#DDE3EE; --text:#1A2035; --muted:#6B7490;
  --accent:#2D5BE8; --accent-light:#EEF2FF;
  --good:#1A8A52; --good-bg:#E5F7EE;
  --bad:#C42A2A;  --bad-bg:#FCEAEA;
  --think:#6B3FB5; --think-bg:#F3EEFF;
  --tool:#1565C0;  --tool-bg:#E3F0FF;
  --resp:#1A6A5A;  --resp-bg:#E5F5F1;
  --ans:#7A5200;   --ans-bg:#FFF8E5;
  --radius:8px; --shadow:0 1px 3px rgba(0,0,0,.08);
}
@media(prefers-color-scheme:dark){:root{
  --bg:#0E1422; --surface:#171F33; --surface2:#1C2640;
  --border:#27334D; --text:#CDD5EE; --muted:#7080A0;
  --accent:#5B84F5; --accent-light:#1A2540;
  --good:#2DC47A; --good-bg:#0B2A1D;
  --bad:#E05555;  --bad-bg:#2B0E0E;
  --think:#A07AE8; --think-bg:#1E1530;
  --tool:#5AABF5;  --tool-bg:#0D1E33;
  --resp:#3DC4A8;  --resp-bg:#0A2020;
  --ans:#E8C060;   --ans-bg:#251C00;
}}
:root[data-theme=light]{--bg:#F2F4F8;--surface:#fff;--surface2:#F7F9FC;--border:#DDE3EE;--text:#1A2035;--muted:#6B7490;--accent:#2D5BE8;--accent-light:#EEF2FF;--good:#1A8A52;--good-bg:#E5F7EE;--bad:#C42A2A;--bad-bg:#FCEAEA;--think:#6B3FB5;--think-bg:#F3EEFF;--tool:#1565C0;--tool-bg:#E3F0FF;--resp:#1A6A5A;--resp-bg:#E5F5F1;--ans:#7A5200;--ans-bg:#FFF8E5;}
:root[data-theme=dark]{--bg:#0E1422;--surface:#171F33;--surface2:#1C2640;--border:#27334D;--text:#CDD5EE;--muted:#7080A0;--accent:#5B84F5;--accent-light:#1A2540;--good:#2DC47A;--good-bg:#0B2A1D;--bad:#E05555;--bad-bg:#2B0E0E;--think:#A07AE8;--think-bg:#1E1530;--tool:#5AABF5;--tool-bg:#0D1E33;--resp:#3DC4A8;--resp-bg:#0A2020;--ans:#E8C060;--ans-bg:#251C00;}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:13.5px;line-height:1.6;color:var(--text);background:var(--bg);}
.layout{display:grid;grid-template-columns:260px 1fr;min-height:100vh;}
.sidebar{background:var(--surface);border-right:1px solid var(--border);padding:20px 16px;position:sticky;top:0;height:100vh;overflow-y:auto;}
.main{padding:24px 28px;}
.logo{font-size:13px;font-weight:800;color:var(--accent);letter-spacing:.04em;margin-bottom:4px;}
.logo-sub{font-size:11px;color:var(--muted);margin-bottom:20px;}
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:20px;}
.stat-box{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:10px;text-align:center;}
.stat-val{font-size:22px;font-weight:800;color:var(--accent);font-variant-numeric:tabular-nums;}
.stat-lbl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;}
.filter-section{margin-bottom:16px;}
.filter-label{font-size:10.5px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;}
.filter-btn{display:block;width:100%;text-align:left;padding:6px 10px;border:none;background:transparent;color:var(--text);cursor:pointer;border-radius:5px;font-size:12.5px;margin-bottom:2px;}
.filter-btn:hover{background:var(--accent-light);}
.filter-btn.active{background:var(--accent);color:#fff;font-weight:600;}
.search-box{width:100%;padding:7px 10px;border:1px solid var(--border);border-radius:6px;background:var(--surface2);color:var(--text);font-size:13px;margin-bottom:16px;}
.search-box:focus{outline:2px solid var(--accent);border-color:transparent;}
.main-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;}
.result-count{font-size:12.5px;color:var(--muted);}
.page-btns{display:flex;gap:6px;}
.page-btn{padding:5px 12px;border:1px solid var(--border);background:var(--surface);color:var(--text);border-radius:5px;cursor:pointer;font-size:12px;}
.page-btn:hover{background:var(--accent-light);}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);margin-bottom:10px;box-shadow:var(--shadow);overflow:hidden;}
.card-header{padding:12px 16px;cursor:pointer;display:flex;align-items:flex-start;gap:10px;}
.card-header:hover{background:var(--surface2);}
.card-idx{font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap;padding-top:1px;}
.card-q{flex:1;font-size:13.5px;font-weight:500;line-height:1.45;}
.badge{display:inline-flex;align-items:center;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:700;white-space:nowrap;flex-shrink:0;}
.badge-ok{background:var(--good-bg);color:var(--good);}
.badge-err{background:var(--bad-bg);color:var(--bad);}
.badge-src{background:var(--accent-light);color:var(--accent);margin-left:4px;}
.card-body{display:none;padding:0 16px 14px;border-top:1px solid var(--border);}
.card-body.open{display:block;}
.meta-row{display:flex;flex-wrap:wrap;gap:8px;padding:10px 0;font-size:12px;color:var(--muted);border-bottom:1px solid var(--border);margin-bottom:10px;}
.ans-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px;}
.ans-box{border:1px solid var(--border);border-radius:6px;padding:10px;}
.ans-box-title{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-bottom:4px;}
.ans-box-val{font-size:13px;font-weight:600;color:var(--text);}
.ans-box-val.correct{color:var(--good);}
.ans-box-val.wrong{color:var(--bad);}
.error-tag{display:inline-block;padding:3px 10px;border-radius:4px;font-size:12px;font-weight:600;background:var(--bad-bg);color:var(--bad);margin-bottom:10px;}
.rounds-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-bottom:8px;}
.turn{margin-bottom:6px;}
.turn-label{font-size:10.5px;font-weight:700;color:var(--muted);margin-bottom:4px;}
.think-block,.tool-block,.tool-resp,.ans-block{border-radius:5px;margin-bottom:4px;overflow:hidden;}
.think-block{background:var(--think-bg);}
.tool-block{background:var(--tool-bg);}
.tool-resp{background:var(--resp-bg);margin-bottom:6px;}
.ans-block{background:var(--ans-bg);}
.block-tag{display:inline-block;font-size:10px;font-weight:700;padding:2px 8px;letter-spacing:.06em;text-transform:uppercase;}
.tag-think{color:var(--think);}
.tag-tool{color:var(--tool);}
.tag-resp{color:var(--resp);}
.tag-ans{color:var(--ans);}
.block-body{padding:6px 10px 8px;font-size:12.5px;line-height:1.55;white-space:pre-wrap;word-break:break-word;}
.resp-body{max-height:140px;overflow-y:auto;font-size:11.5px;}
.chevron{font-size:12px;color:var(--muted);margin-left:4px;transition:transform .15s;flex-shrink:0;}
.chevron.open{transform:rotate(180deg);}
.case-note-wrap{margin-top:12px;border-top:1px solid var(--border);padding-top:12px;}
.case-note-label{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#C42A2A;margin-bottom:6px;display:flex;align-items:center;gap:6px;}
.case-note-label::before{content:'✏';font-size:12px;}
.case-note-label .saved-hint{font-size:10px;font-weight:400;color:var(--muted);letter-spacing:0;text-transform:none;opacity:0;transition:opacity .4s;}
.case-note-label .saved-hint.show{opacity:1;}
.case-note-area{width:100%;min-height:72px;padding:9px 12px;border:1.5px solid #E8A8A8;border-radius:6px;background:#FFF8F8;color:#8B1A1A;font-size:13px;font-family:inherit;line-height:1.6;resize:vertical;outline:none;}
.case-note-area:focus{border-color:#C42A2A;background:#fff;}
.case-note-area::placeholder{color:#D4A0A0;}
@media(prefers-color-scheme:dark){
.case-note-area{background:#1E0A0A;border-color:#6B2020;color:#F0A0A0;}
.case-note-area:focus{border-color:#E05555;background:#240E0E;}
.case-note-area::placeholder{color:#7A3A3A;}
}
:root[data-theme=dark] .case-note-area{background:#1E0A0A;border-color:#6B2020;color:#F0A0A0;}
:root[data-theme=dark] .case-note-area:focus{border-color:#E05555;background:#240E0E;}
"""

JS = r"""
const PAGE_SIZE = 25;
let filters = {correct:'all', source:'all', error:'all'};
let search = '';
let page = 0;

function setFilter(key, val) {
  filters[key] = val;
  page = 0;
  document.querySelectorAll('.filter-btn').forEach(b => {
    const oc = b.getAttribute('onclick') || '';
    if (oc.startsWith("setFilter('" + key + "'")) {
      b.classList.toggle('active', oc === "setFilter('" + key + "','" + val + "')");
    }
  });
  render();
}

document.getElementById('searchbox').addEventListener('input', ev => {
  search = ev.target.value.toLowerCase();
  page = 0;
  render();
});

function filtered() {
  return DATA.filter(r => {
    if (filters.correct === 'ok'    && !r.correct) return false;
    if (filters.correct === 'wrong' &&  r.correct) return false;
    if (filters.source !== 'all' && r.source !== filters.source) return false;
    if (filters.error  !== 'all' && r.error  !== filters.error)  return false;
    if (search && !r.question.toLowerCase().includes(search) &&
        !r.gt.toLowerCase().includes(search) &&
        !r.prediction.toLowerCase().includes(search)) return false;
    return true;
  });
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function renderRounds(rounds) {
  let h = '';
  let turn = 0;
  for (const rd of rounds) {
    if (rd.role === 'assistant') {
      turn++;
      h += '<div class="turn"><div class="turn-label">Turn ' + turn + '</div>';
      if (rd.think) h += '<div class="think-block"><span class="block-tag tag-think">think</span><div class="block-body">' + esc(rd.think) + '</div></div>';
      for (const q of rd.queries) h += '<div class="tool-block"><span class="block-tag tag-tool">search</span><div class="block-body">' + esc(q) + '</div></div>';
      if (rd.answer_text && !rd.queries.length) h += '<div class="ans-block"><span class="block-tag tag-ans">answer</span><div class="block-body">' + esc(rd.answer_text.slice(0,300)) + '</div></div>';
      h += '</div>';
    } else {
      h += '<div class="tool-resp"><span class="block-tag tag-resp">results</span><div class="block-body resp-body">' + esc(rd.content) + '</div></div>';
    }
  }
  return h;
}

function render() {
  const rows = filtered();
  const total = rows.length;
  const maxPage = Math.max(0, Math.ceil(total / PAGE_SIZE) - 1);
  if (page > maxPage) page = maxPage;
  const slice = rows.slice(page * PAGE_SIZE, (page+1) * PAGE_SIZE);

  document.getElementById('result-count').textContent = '显示 ' + total + ' 题';

  const pi = document.getElementById('page-info');
  const pb = document.getElementById('page-btns');
  pi.textContent = total > PAGE_SIZE ? ('第 ' + (page+1) + ' / ' + (maxPage+1) + ' 页') : '';
  let phtml = '';
  if (page > 0) phtml += '<button class="page-btn" onclick="goPage(' + (page-1) + ')">‹ 上一页</button>';
  if (page < maxPage) phtml += '<button class="page-btn" onclick="goPage(' + (page+1) + ')">下一页 ›</button>';
  pb.innerHTML = phtml;

  const globalIdx = page * PAGE_SIZE;
  const cards = slice.map((r, i) => {
    const idx = globalIdx + i + 1;
    const bodyId = 'body-' + idx;
    const chevId = 'chev-' + idx;
    const okBadge = r.correct
      ? '<span class="badge badge-ok">✓ 正确</span>'
      : '<span class="badge badge-err">✗ 错误</span>';
    const srcBadge = '<span class="badge badge-src">' + esc(r.source) + '</span>';
    const predClass = r.correct ? 'correct' : 'wrong';
    const errHtml = r.error ? '<div class="error-tag">错误原因：' + esc(r.error) + '</div>' : '';
    const predText = r.prediction ? esc(r.prediction.slice(0,200)) : '<em>无答案</em>';
    return '<div class="card">'
      + '<div class="card-header" onclick="toggle(\'' + bodyId + '\',\'' + chevId + '\')">'
      + '<span class="card-idx">#' + idx + '</span>'
      + '<span class="card-q">' + esc(r.question) + '</span>'
      + okBadge + srcBadge
      + '<span class="chevron" id="' + chevId + '">▼</span>'
      + '</div>'
      + '<div class="card-body" id="' + bodyId + '">'
      + '<div class="meta-row">'
      + '<span><strong>轮次</strong> ' + r.num_turns + ' turns</span>'
      + '<span><strong>终止原因</strong> ' + esc(r.termination) + '</span>'
      + '<span><strong>来源</strong> ' + esc(r.source) + '</span>'
      + '</div>'
      + '<div class="ans-row">'
      + '<div class="ans-box"><div class="ans-box-title">Golden Answer</div><div class="ans-box-val">' + esc(r.gt) + '</div></div>'
      + '<div class="ans-box"><div class="ans-box-title">模型 Prediction</div><div class="ans-box-val ' + predClass + '">' + predText + '</div></div>'
      + '</div>'
      + errHtml
      + '<div class="rounds-title">推理过程</div>'
      + renderRounds(r.rounds)
      + renderCaseNote(r.question, idx)
      + '</div></div>';
  }).join('');
  document.getElementById('cards').innerHTML = cards;
}

function toggle(bodyId, chevId) {
  const body = document.getElementById(bodyId);
  const chev = document.getElementById(chevId);
  const open = body.classList.toggle('open');
  chev.classList.toggle('open', open);
}

function goPage(p) { page = p; render(); window.scrollTo(0,0); }

// ── Case notes (localStorage) ──────────────────────────────────────────────
const NOTE_PREFIX = 'eval_note_';
function noteKey(q) { return NOTE_PREFIX + btoa(unescape(encodeURIComponent(q))).slice(0,40); }
function getNote(q) { try { return localStorage.getItem(noteKey(q)) || ''; } catch(e) { return ''; } }
function setNote(q, v) { try { localStorage.setItem(noteKey(q), v); } catch(e) {} }

function saveNote(q, areaId, hintId) {
  const area = document.getElementById(areaId);
  const hint = document.getElementById(hintId);
  if (!area) return;
  setNote(q, area.value);
  if (hint) { hint.classList.add('show'); setTimeout(() => hint.classList.remove('show'), 1200); }
}

function renderCaseNote(q, uid) {
  const areaId = 'note-area-' + uid;
  const hintId = 'note-hint-' + uid;
  const saved  = getNote(q).replace(/"/g, '&quot;');
  return '<div class="case-note-wrap">'
    + '<div class="case-note-label">Case Analysis'
    + '<span class="saved-hint" id="' + hintId + '">已保存</span>'
    + '</div>'
    + '<textarea class="case-note-area" id="' + areaId + '" placeholder="在此记录分析备注..."'
    + ' oninput="saveNote(' + JSON.stringify(q) + ',\'' + areaId + '\',\'' + hintId + '\')">'
    + saved + '</textarea>'
    + '</div>';
}

render();
"""

parts = []
parts.append("<!doctype html><html lang='zh'><head><meta charset='utf-8'>")
parts.append("<title>SFT LLaDA ckpt_6 强制推理 — Eval Viewer</title>")
parts.append("<style>" + STYLE + "</style></head><body>")
parts.append("<div class='layout'>")
parts.append("<aside class='sidebar'>")
parts.append("<div class='logo'>SFT LLaDA ckpt_6</div>")
parts.append("<div class='logo-sub'>强制推理 · 600题 Eval Viewer</div>")
parts.append("<div class='stat-grid'>")
parts.append("<div class='stat-box'><div class='stat-val'>" + str(n) + "</div><div class='stat-lbl'>总题数</div></div>")
parts.append("<div class='stat-box'><div class='stat-val' style='color:var(--good)'>" + f"{n_correct/n*100:.1f}%" + "</div><div class='stat-lbl'>ACC_R</div></div>")
parts.append("<div class='stat-box'><div class='stat-val' style='color:var(--good)'>" + str(n_correct) + "</div><div class='stat-lbl'>答对</div></div>")
parts.append("<div class='stat-box'><div class='stat-val' style='color:var(--bad)'>" + str(n - n_correct) + "</div><div class='stat-lbl'>答错</div></div>")
parts.append("</div>")
parts.append("<input class='search-box' type='text' id='searchbox' placeholder='搜索题目...'>")
parts.append("<div class='filter-section'><div class='filter-label'>正确性</div>")
parts.append("<button class='filter-btn active' onclick=\"setFilter('correct','all')\">全部</button>")
parts.append("<button class='filter-btn' onclick=\"setFilter('correct','ok')\">✓ 答对</button>")
parts.append("<button class='filter-btn' onclick=\"setFilter('correct','wrong')\">✗ 答错</button>")
parts.append("</div>")
parts.append("<div class='filter-section'><div class='filter-label'>数据集来源</div>")
parts.append("<button class='filter-btn active' onclick=\"setFilter('source','all')\">全部</button>")
parts.append(src_btns)
parts.append("</div>")
parts.append("<div class='filter-section'><div class='filter-label'>错误类型</div>")
parts.append("<button class='filter-btn active' onclick=\"setFilter('error','all')\">全部</button>")
parts.append(err_btns)
parts.append("</div></aside>")
parts.append("<main class='main'>")
parts.append("<div class='main-header'><span class='result-count' id='result-count'></span>")
parts.append("<div style='display:flex;align-items:center;gap:12px;'>")
parts.append("<span style='font-size:12px;color:var(--muted);' id='page-info'></span>")
parts.append("<div class='page-btns' id='page-btns'></div></div></div>")
parts.append("<div id='cards'></div></main></div>")
parts.append("<script>const DATA = " + cards_json + ";" + JS + "</script>")
parts.append("</body></html>")

html_out = "".join(parts)
OUT_HTML.write_text(html_out, encoding="utf-8")
print(f"Generated: {OUT_HTML}")
print(f"Total records: {len(records)}, Correct: {n_correct}/{n} ({100*n_correct/n:.1f}%)")
from collections import Counter
print("Error breakdown:", Counter(r["error"] for r in records if not r["correct"]))
print("Source breakdown:", Counter(r["source"] for r in records))
