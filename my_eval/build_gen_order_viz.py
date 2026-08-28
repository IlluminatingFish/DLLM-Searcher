"""
Build HTML visualization of token generation order for dLLM inference.
Data is embedded as JSON; JS renders token spans interactively.
"""
import json
from pathlib import Path

SFT_JSON = Path("/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/output/gen_order/sft_llada_order.json")
X0P_JSON = Path("/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/output/gen_order/x0pred_order.json")
OUT_HTML = Path("/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/output/gen_order/gen_order_viz.html")


def compute_summary(data):
    """Aggregate stats for one model's 30 questions."""
    block_counts = [len(q['blocks']) for q in data]
    all_steps = []
    hist = [0] * 8  # 8 buckets: 1-8, 9-16, ..., 57-64
    for q in data:
        for b in q['blocks']:
            for s in b['reveal_step']:
                all_steps.append(s)
                hist[min((s - 1) // 8, 7)] += 1
    import statistics as _s
    return {
        'n': len(data),
        'avg_blocks': round(sum(block_counts) / len(block_counts), 2),
        'block_dist': {str(k): block_counts.count(k) for k in sorted(set(block_counts))},
        'step_mean': round(_s.mean(all_steps), 1),
        'step_stdev': round(_s.stdev(all_steps), 1),
        'total_tokens': len(all_steps),
        'hist': hist,
    }


def slim_data(data):
    """Keep only what JS needs — question, answer, source, blocks."""
    out = []
    for q in data:
        out.append({
            'question': q['question'],
            'answer': q.get('answer', ''),
            'source': q.get('source', ''),
            'blocks': [
                {
                    'token_ids': b['token_ids'],
                    'tokens': b['tokens'],
                    'reveal_step': b['reveal_step'],
                }
                for b in q['blocks']
            ],
        })
    return out


CSS = r"""
:root {
  --bg: #090c15;
  --surface: #101525;
  --surface-2: #16203a;
  --border: #1e2a45;
  --border-soft: #151f35;
  --text: #c4cae8;
  --text-muted: #4e5878;
  --text-dim: #8892b0;
  --sft: #4d9ef6;
  --sft-glow: rgba(77,158,246,.18);
  --sft-bg: rgba(77,158,246,.08);
  --x0p: #a06af0;
  --x0p-glow: rgba(160,106,240,.18);
  --x0p-bg: rgba(160,106,240,.08);
  --accent: #4d9ef6;
  font-size: 14px;
}
[data-theme="light"] {
  --bg: #f0f2fa;
  --surface: #ffffff;
  --surface-2: #e8ecf8;
  --border: #c8d0e8;
  --border-soft: #dde2f2;
  --text: #1a2040;
  --text-muted: #8090b8;
  --text-dim: #4a5580;
  --sft-bg: rgba(77,158,246,.1);
  --x0p-bg: rgba(160,106,240,.1);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── Top bar ── */
#topbar {
  position: sticky; top: 0; z-index: 200;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  padding: 0 20px;
  display: flex; align-items: center; gap: 16px;
  height: 52px;
  backdrop-filter: blur(12px);
}
#topbar h1 {
  font-size: 15px; font-weight: 700; letter-spacing: -.3px;
  color: var(--text);
  white-space: nowrap;
}
#topbar h1 span { color: var(--sft); }
.badge {
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 4px; padding: 2px 8px; font-size: 11px;
  color: var(--text-dim); font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.spacer { flex: 1; }
#theme-btn {
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 6px; padding: 5px 12px; font-size: 12px;
  color: var(--text-dim); cursor: pointer;
}
#theme-btn:hover { border-color: var(--accent); color: var(--text); }

/* ── Stats panel ── */
#stats-panel {
  display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
  padding: 16px 20px; border-bottom: 1px solid var(--border-soft);
}
.stat-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 18px;
}
.stat-card h3 {
  font-size: 12px; font-weight: 700; letter-spacing: .05em;
  text-transform: uppercase; margin-bottom: 10px;
}
.stat-card.sft h3 { color: var(--sft); }
.stat-card.x0p h3 { color: var(--x0p); }
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.stat-item { }
.stat-label { font-size: 10px; color: var(--text-muted); letter-spacing: .04em; text-transform: uppercase; }
.stat-val { font-size: 20px; font-weight: 700; color: var(--text); font-variant-numeric: tabular-nums; line-height: 1.2; }
.stat-sub { font-size: 11px; color: var(--text-dim); }
.hist-wrap { margin-top: 10px; }
.hist-label { font-size: 10px; color: var(--text-muted); margin-bottom: 4px; }
.hist-bars { display: flex; align-items: flex-end; gap: 2px; height: 28px; }
.hist-bar {
  flex: 1; border-radius: 2px 2px 0 0; min-height: 2px;
  opacity: .85;
}

/* ── Nav bar ── */
#nav-bar {
  display: flex; gap: 6px; overflow-x: auto; padding: 10px 20px;
  border-bottom: 1px solid var(--border-soft);
  scrollbar-width: thin;
}
#nav-bar::-webkit-scrollbar { height: 4px; }
#nav-bar::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
.nav-btn {
  flex-shrink: 0; background: var(--surface); border: 1px solid var(--border);
  border-radius: 5px; padding: 4px 10px; font-size: 11px;
  color: var(--text-dim); cursor: pointer; font-variant-numeric: tabular-nums;
}
.nav-btn:hover { border-color: var(--accent); color: var(--text); }
.nav-btn.active { background: var(--sft-bg); border-color: var(--sft); color: var(--sft); }

/* ── Legend ── */
#legend {
  padding: 8px 20px 12px;
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
}
.leg-label { font-size: 11px; color: var(--text-muted); }
.leg-swatch {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 11px; color: var(--text-dim);
}
.leg-color {
  width: 28px; height: 12px; border-radius: 2px;
}
.leg-sep { color: var(--border); font-size: 10px; }

/* ── Main content ── */
#main { padding: 12px 20px 40px; display: flex; flex-direction: column; gap: 14px; }

.q-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px; overflow: hidden;
}
.q-meta {
  padding: 12px 16px 8px;
  border-bottom: 1px solid var(--border-soft);
  display: flex; gap: 10px; align-items: flex-start;
}
.q-num-badge {
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 5px; padding: 2px 8px; font-size: 11px; font-weight: 700;
  color: var(--text-dim); white-space: nowrap; flex-shrink: 0;
}
.q-src-badge {
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 5px; padding: 2px 8px; font-size: 10px; font-weight: 600;
  color: var(--text-muted); white-space: nowrap; flex-shrink: 0;
  letter-spacing: .04em; text-transform: uppercase;
}
.q-text { font-size: 13px; color: var(--text); line-height: 1.5; flex: 1; }
.q-ans {
  padding: 6px 16px 10px; font-size: 12px;
  color: var(--text-dim); background: var(--surface);
}
.q-ans strong { color: var(--text); }

.models-row {
  display: grid; grid-template-columns: 1fr 1fr;
  border-top: 1px solid var(--border-soft);
}
.model-col {
  padding: 12px 14px; overflow-x: auto;
}
.model-col:first-child { border-right: 1px solid var(--border-soft); }
.model-head {
  font-size: 11px; font-weight: 700; letter-spacing: .06em;
  text-transform: uppercase; margin-bottom: 10px;
  display: flex; align-items: center; gap: 8px;
}
.model-head.sft { color: var(--sft); }
.model-head.x0p { color: var(--x0p); }
.model-head .block-count {
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 4px; padding: 1px 7px; font-size: 10px;
  color: var(--text-muted); font-weight: 400;
}

/* ── Token blocks ── */
.block-section { margin-bottom: 14px; }
.block-title {
  font-size: 10px; color: var(--text-muted); letter-spacing: .05em;
  text-transform: uppercase; margin-bottom: 4px;
}
.tok-row {
  display: flex; flex-wrap: wrap; gap: 2px; align-items: flex-start;
}
.tok {
  display: inline-block;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 11px; line-height: 1.5;
  padding: 1px 4px; border-radius: 3px;
  color: rgba(255,255,255,.95);
  cursor: default;
  white-space: pre;
  transition: transform .1s;
}
.tok:hover { transform: scale(1.1); z-index: 10; outline: 1px solid rgba(255,255,255,.5); }

/* Light theme token text */
[data-theme="light"] .tok { color: rgba(0,0,0,.85); }

/* View toggle */
.view-toggle {
  display: flex; gap: 0; margin-bottom: 10px; border-radius: 6px; overflow: hidden;
  border: 1px solid var(--border); width: fit-content;
}
.vtog-btn {
  padding: 4px 12px; font-size: 11px; cursor: pointer;
  background: var(--surface-2); color: var(--text-dim);
  border: none; border-right: 1px solid var(--border);
}
.vtog-btn:last-child { border-right: none; }
.vtog-btn.active { background: var(--sft-bg); color: var(--sft); }
.model-col:last-child .vtog-btn.active { background: var(--x0p-bg); color: var(--x0p); }

/* ── Responsive ── */
@media (max-width: 800px) {
  .models-row { grid-template-columns: 1fr; }
  #stats-panel { grid-template-columns: 1fr; }
  .model-col:first-child { border-right: none; border-bottom: 1px solid var(--border-soft); }
}
"""

JS = r"""
// ─── Data ───────────────────────────────────────────────────────────────────
const SFT = window.__SFT__;
const X0P = window.__X0P__;
const SFT_STATS = window.__SFT_STATS__;
const X0P_STATS = window.__X0P_STATS__;

// ─── Color mapping ──────────────────────────────────────────────────────────
const STOPS = [
  { s: 1,  h: 0,   sat: 88, lig: 52 },
  { s: 14, h: 25,  sat: 88, lig: 52 },
  { s: 25, h: 45,  sat: 85, lig: 48 },
  { s: 36, h: 100, sat: 72, lig: 44 },
  { s: 50, h: 175, sat: 70, lig: 42 },
  { s: 64, h: 220, sat: 75, lig: 52 },
];

function stepColor(step) {
  let lo = STOPS[0], hi = STOPS[STOPS.length - 1];
  for (let i = 0; i < STOPS.length - 1; i++) {
    if (step >= STOPS[i].s && step <= STOPS[i+1].s) {
      lo = STOPS[i]; hi = STOPS[i+1]; break;
    }
  }
  const t = (step - lo.s) / Math.max(hi.s - lo.s, 1);
  const h = lo.h + t * (hi.h - lo.h);
  const sat = lo.sat + t * (hi.sat - lo.sat);
  const lig = lo.lig + t * (hi.lig - lo.lig);
  return `hsl(${h.toFixed(0)},${sat.toFixed(0)}%,${lig.toFixed(0)}%)`;
}

function stepGlow(step) {
  if (step < 50) return '';
  const strength = (step - 50) / 14 * 0.45;
  return `box-shadow:0 0 4px ${strength.toFixed(2)}px ${stepColor(step)}`;
}

// ─── Token rendering ─────────────────────────────────────────────────────────
function renderTokens(tokens, steps, orderByReveal) {
  const indices = orderByReveal
    ? Array.from({length: tokens.length}, (_, i) => i).sort((a, b) => steps[a] - steps[b])
    : Array.from({length: tokens.length}, (_, i) => i);

  return indices.map(i => {
    const tok = tokens[i];
    const step = steps[i];
    const display = tok.replace(/\n/g, '↵').replace(/\t/g, '→') || '·';
    const safe = display.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    const bg = stepColor(step);
    const glow = stepGlow(step);
    const styleStr = `background:${bg};${glow}`;
    const title = `pos=${i+1} | step=${step}`;
    return `<span class="tok" style="${styleStr}" title="${title}">${safe}</span>`;
  }).join('');
}

function renderModelCol(data, idx, label, cls, view) {
  const q = data[idx];
  const blocks = q.blocks;
  const blockCountText = blocks.length === 1 ? '1 block' : `${blocks.length} blocks`;
  const orderByReveal = (view === 'reveal');

  let blocksHTML = blocks.map((b, bi) => {
    const tokHTML = renderTokens(b.tokens, b.reveal_step, orderByReveal);
    const titleText = orderByReveal
      ? `Block ${bi+1} — reveal order (left=first, right=last)`
      : `Block ${bi+1} — positional order`;
    return `<div class="block-section">
      <div class="block-title">${titleText}</div>
      <div class="tok-row">${tokHTML}</div>
    </div>`;
  }).join('');

  return `<div class="model-col">
    <div class="model-head ${cls}">${label} <span class="block-count">${blockCountText}</span></div>
    ${blocksHTML}
  </div>`;
}

// ─── Question card ───────────────────────────────────────────────────────────
const viewState = {}; // qIdx -> {sft: 'positional'|'reveal', x0p: ...}

function renderCard(idx) {
  const qs = SFT[idx], qx = X0P[idx];
  const state = viewState[idx] || { sft: 'positional', x0p: 'positional' };

  const srcBadge = qs.source
    ? `<span class="q-src-badge">${qs.source.replace(/<.*?>/g,'')}</span>`
    : '';
  const qText = qs.question.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const ans = qs.answer.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

  const sftViewBtns = [
    `<button class="vtog-btn${state.sft==='positional'?' active':''}" onclick="setView(${idx},'sft','positional')">位置顺序</button>`,
    `<button class="vtog-btn${state.sft==='reveal'?' active':''}" onclick="setView(${idx},'sft','reveal')">揭示顺序</button>`,
  ].join('');
  const x0pViewBtns = [
    `<button class="vtog-btn${state.x0p==='positional'?' active':''}" onclick="setView(${idx},'x0p','positional')">位置顺序</button>`,
    `<button class="vtog-btn${state.x0p==='reveal'?' active':''}" onclick="setView(${idx},'x0p','reveal')">揭示顺序</button>`,
  ].join('');

  const sftCol = `<div class="model-col">
    <div class="model-head sft">sft_llada ckpt_6 <span class="block-count">${qs.blocks.length} block${qs.blocks.length>1?'s':''}</span></div>
    <div class="view-toggle">${sftViewBtns}</div>
    ${qs.blocks.map((b,bi) => {
      const tokHTML = renderTokens(b.tokens, b.reveal_step, state.sft==='reveal');
      const t = state.sft==='reveal' ? `Block ${bi+1} — 揭示时序（左=先，右=后）` : `Block ${bi+1} — 位置顺序`;
      return `<div class="block-section"><div class="block-title">${t}</div><div class="tok-row">${tokHTML}</div></div>`;
    }).join('')}
  </div>`;

  const x0pCol = `<div class="model-col">
    <div class="model-head x0p">x0pred ckpt_6 <span class="block-count">${qx.blocks.length} block${qx.blocks.length>1?'s':''}</span></div>
    <div class="view-toggle">${x0pViewBtns}</div>
    ${qx.blocks.map((b,bi) => {
      const tokHTML = renderTokens(b.tokens, b.reveal_step, state.x0p==='reveal');
      const t = state.x0p==='reveal' ? `Block ${bi+1} — 揭示时序（左=先，右=后）` : `Block ${bi+1} — 位置顺序`;
      return `<div class="block-section"><div class="block-title">${t}</div><div class="tok-row">${tokHTML}</div></div>`;
    }).join('')}
  </div>`;

  return `<div class="q-card" id="qcard${idx}">
    <div class="q-meta">
      <span class="q-num-badge">Q${idx+1}</span>
      ${srcBadge}
      <span class="q-text">${qText}</span>
    </div>
    <div class="q-ans">答案 · <strong>${ans}</strong></div>
    <div class="models-row">${sftCol}${x0pCol}</div>
  </div>`;
}

function setView(idx, model, view) {
  if (!viewState[idx]) viewState[idx] = { sft: 'positional', x0p: 'positional' };
  viewState[idx][model] = view;
  const card = document.getElementById('qcard' + idx);
  if (card) card.outerHTML = renderCard(idx);
  // Re-attach the card (outerHTML replaces the element)
  const newCard = document.getElementById('qcard' + idx);
  // update nav
  updateNav(currentQ);
}

// ─── Stats rendering ──────────────────────────────────────────────────────────
function histBars(hist, color) {
  const max = Math.max(...hist);
  return hist.map((v, i) => {
    const pct = max > 0 ? (v / max * 100).toFixed(1) : 0;
    const bg = stepColor(i * 8 + 4);
    return `<div class="hist-bar" style="height:${pct}%;background:${bg}" title="step ${i*8+1}-${i*8+8}: ${v} tokens"></div>`;
  }).join('');
}

function renderStats() {
  const panel = document.getElementById('stats-panel');

  function card(stats, label, cls, color) {
    const bd = Object.entries(stats.block_dist)
      .map(([k,v]) => `${k}块 ${v}题`).join('  /  ');
    return `<div class="stat-card ${cls}">
      <h3>${label}</h3>
      <div class="stat-grid">
        <div class="stat-item">
          <div class="stat-label">平均块数</div>
          <div class="stat-val">${stats.avg_blocks}</div>
        </div>
        <div class="stat-item">
          <div class="stat-label">平均揭示步</div>
          <div class="stat-val">${stats.step_mean}</div>
          <div class="stat-sub">±${stats.step_stdev}</div>
        </div>
        <div class="stat-item" style="grid-column:1/-1">
          <div class="stat-label">块数分布</div>
          <div class="stat-sub" style="font-size:12px;margin-top:2px">${bd}</div>
        </div>
      </div>
      <div class="hist-wrap">
        <div class="hist-label">Token 揭示步分布（step 1-64，8 档）</div>
        <div class="hist-bars">${histBars(stats.hist, color)}</div>
        <div style="display:flex;justify-content:space-between;font-size:9px;color:var(--text-muted);margin-top:2px">
          <span>step 1</span><span>step 32</span><span>step 64</span>
        </div>
      </div>
    </div>`;
  }

  panel.innerHTML =
    card(SFT_STATS, 'sft_llada ckpt_6', 'sft', '#4d9ef6') +
    card(X0P_STATS, 'x0pred ckpt_6', 'x0p', '#a06af0');
}

// ─── Legend ───────────────────────────────────────────────────────────────────
function renderLegend() {
  const leg = document.getElementById('legend');
  const stepLabels = [1, 10, 20, 32, 45, 55, 64];
  const swatches = stepLabels.map(s => {
    const bg = stepColor(s);
    return `<span class="leg-swatch"><span class="leg-color" style="background:${bg}"></span>step ${s}</span>`;
  });
  leg.innerHTML = `<span class="leg-label">揭示时序：</span>` + swatches.join('<span class="leg-sep">→</span>');
}

// ─── Navigation ──────────────────────────────────────────────────────────────
let currentQ = 0;

function updateNav(active) {
  document.querySelectorAll('.nav-btn').forEach((b, i) => {
    b.classList.toggle('active', i === active);
  });
}

function scrollToQ(idx) {
  currentQ = idx;
  updateNav(idx);
  const el = document.getElementById('qcard' + idx);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderNav() {
  const bar = document.getElementById('nav-bar');
  bar.innerHTML = Array.from({length: SFT.length}, (_, i) =>
    `<button class="nav-btn${i===0?' active':''}" onclick="scrollToQ(${i})">Q${i+1}</button>`
  ).join('');
}

// ─── Main render ─────────────────────────────────────────────────────────────
function renderAll() {
  renderStats();
  renderLegend();
  renderNav();
  const main = document.getElementById('main');
  main.innerHTML = SFT.map((_, i) => renderCard(i)).join('');
}

// ─── Theme ───────────────────────────────────────────────────────────────────
function initTheme() {
  const root = document.documentElement;
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  const saved = localStorage.getItem('theme');
  const theme = saved || (mq.matches ? 'dark' : 'light');
  root.setAttribute('data-theme', theme);
  const btn = document.getElementById('theme-btn');
  btn.textContent = theme === 'dark' ? '☀ Light' : '☾ Dark';
  btn.onclick = () => {
    const cur = root.getAttribute('data-theme');
    const next = cur === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    btn.textContent = next === 'dark' ? '☀ Light' : '☾ Dark';
  };
}

// ─── Boot ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  renderAll();
});
"""


def build_html(sft_data, x0p_data):
    sft_slim = slim_data(sft_data)
    x0p_slim = slim_data(x0p_data)
    sft_stats = compute_summary(sft_data)
    x0p_stats = compute_summary(x0p_data)

    sft_json = json.dumps(sft_slim, ensure_ascii=False)
    x0p_json = json.dumps(x0p_slim, ensure_ascii=False)
    sft_stats_json = json.dumps(sft_stats)
    x0p_stats_json = json.dumps(x0p_stats)

    return f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>dLLM Token 生成顺序可视化</title>
<style>{CSS}</style>
</head>
<body>

<div id="topbar">
  <h1>dLLM <span>Token 生成顺序</span></h1>
  <span class="badge">30 questions · 2 models · LLaDA block diffusion</span>
  <div class="spacer"></div>
  <button id="theme-btn">☾ Dark</button>
</div>

<div id="stats-panel"></div>

<div id="legend"></div>

<div id="nav-bar"></div>

<div id="main"></div>

<script>
window.__SFT__ = {sft_json};
window.__X0P__ = {x0p_json};
window.__SFT_STATS__ = {sft_stats_json};
window.__X0P_STATS__ = {x0p_stats_json};
</script>
<script>{JS}</script>
</body>
</html>"""


def main():
    with open(SFT_JSON) as f:
        sft = json.load(f)
    with open(X0P_JSON) as f:
        x0p = json.load(f)

    html_content = build_html(sft, x0p)
    with open(OUT_HTML, 'w') as f:
        f.write(html_content)
    print(f"[saved] {OUT_HTML}  ({len(sft)} questions, {len(html_content)//1024}KB)")


if __name__ == '__main__':
    main()
