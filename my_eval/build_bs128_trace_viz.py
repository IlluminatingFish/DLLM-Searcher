#!/usr/bin/env python3
"""
build_bs128_trace_viz.py
========================
把 run_llada_eval_trace_bs128.py 输出的 rank*.jsonl 合并，
生成与 π₃ Live Search Trace 格式相同的交互式 HTML 可视化。

用法:
  python my_eval/build_bs128_trace_viz.py \
      --trace_dir my_eval/results/bs128/trace \
      --out /tmp/bs128_trace_viz.html
"""

import argparse, json, re
from pathlib import Path


# ─── GPT-2 BPE 字节映射 ───────────────────────────────────────────────────────
def _build_unicode_to_bytes():
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

_U2B = _build_unicode_to_bytes()

def decode_bpe(tok):
    """把 GPT-2 BPE token（含 Ġ/Ċ 等）解码为可读 UTF-8 文字。"""
    try:
        raw = bytes([_U2B.get(c, ord(c) if ord(c) < 256 else 32) for c in tok])
        return raw.decode("utf-8", errors="replace").replace("�", "?")
    except Exception:
        return tok


# ─── 统计 ─────────────────────────────────────────────────────────────────────
def compute_stats(cases):
    block_counts = [sum(len(t["blocks"]) for t in c["turns"]) for c in cases]
    all_steps = []
    hist = [0] * 16  # 16 buckets for 128 steps
    for c in cases:
        for t in c["turns"]:
            for b in t["blocks"]:
                for s in b["reveal_step"]:
                    all_steps.append(s)
                    hist[min(s // 8, 15)] += 1
    import statistics as _s
    return {
        "n": len(cases),
        "avg_blocks": round(sum(block_counts) / max(len(block_counts), 1), 2),
        "step_mean":  round(_s.mean(all_steps), 1) if all_steps else 0,
        "step_stdev": round(_s.stdev(all_steps), 1) if len(all_steps) > 1 else 0,
        "total_tokens": len(all_steps),
        "hist": hist,
        "regen_correct": sum(1 for c in cases if c.get("regen_correct", False)),
    }


def slim_case(c):
    """精简到 JS 需要的字段，减少 HTML 体积。"""
    turns_slim = []
    for t in c["turns"]:
        blocks_slim = []
        for b in t["blocks"]:
            blocks_slim.append({
                "tokens":      [decode_bpe(t) for t in b["tokens"]],
                "token_ids":   b["token_ids"],
                "reveal_step": b["reveal_step"],
            })
        turns_slim.append({
            "turn_idx":     t["turn_idx"],
            "is_toolcall":  t.get("is_toolcall", False),
            "force_answer": t.get("force_answer", False),
            "text":         t.get("text", "")[:800],  # 截断长文本
            "search_result": t.get("search_result", "")[:2000],
            "blocks":       blocks_slim,
        })
    return {
        "question":   c["question"],
        "gt":         c["gt"],
        "cat":        c["cat"],
        "prediction": c.get("prediction", ""),
        "regen_correct": c.get("regen_correct", False),
        "termination_reason": c.get("termination_reason", ""),
        "turns": turns_slim,
    }


def build_html(cases, stats):
    cases_json  = json.dumps([slim_case(c) for c in cases], ensure_ascii=False)
    stats_json  = json.dumps(stats)

    cat_counts = {}
    for c in cases:
        cat_counts[c["cat"]] = cat_counts.get(c["cat"], 0) + 1

    correct_count  = stats["regen_correct"]
    total          = stats["n"]
    regen_acc      = f"{correct_count/total*100:.1f}%" if total else "—"

    return f'''<title>bs128 ckpt_4 Token 生成轨迹</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root {{
  --bg:#0C1116;--s1:#151D26;--s2:#1C2533;--s3:#243040;
  --bd:#2A3747;--bd2:#384D66;
  --tx:#DDE5EF;--mu:#7A8FA6;--dim:#4A6078;
  --ac:#4FA3E8;
  --green:#3FB950;--green-bg:rgba(63,185,80,.13);
  --red:#F47067;--red-bg:rgba(244,112,103,.13);
  --amber:#D09A20;--amber-bg:rgba(208,154,32,.13);
  --purple:#B98AEF;--purple-bg:rgba(185,138,239,.13);
  --teal:#45B8C2;--teal-bg:rgba(69,184,194,.13);
  --mono:'JetBrains Mono','Fira Code',ui-monospace,monospace;
  --r:5px;
}}
@media(prefers-color-scheme:light){{
  :root:not([data-theme="dark"]){{
    --bg:#F2F5F9;--s1:#FFF;--s2:#EBF0F6;--s3:#E2EAF2;
    --bd:#C8D6E5;--bd2:#B0C4D8;
    --tx:#1A2433;--mu:#5A6B7E;--dim:#8A9AAC;--ac:#1A6EBD;
    --green:#1B7F37;--green-bg:rgba(27,127,55,.1);
    --red:#C0392B;--red-bg:rgba(192,57,43,.1);
    --amber:#9A6700;--amber-bg:rgba(154,103,0,.1);
    --purple:#6B3FBF;--purple-bg:rgba(107,63,191,.1);
    --teal:#1A7F8B;--teal-bg:rgba(26,127,139,.1);
  }}
}}
:root[data-theme="light"]{{
  --bg:#F2F5F9;--s1:#FFF;--s2:#EBF0F6;--s3:#E2EAF2;
  --bd:#C8D6E5;--bd2:#B0C4D8;
  --tx:#1A2433;--mu:#5A6B7E;--dim:#8A9AAC;--ac:#1A6EBD;
  --green:#1B7F37;--green-bg:rgba(27,127,55,.1);
  --red:#C0392B;--red-bg:rgba(192,57,43,.1);
  --amber:#9A6700;--amber-bg:rgba(154,103,0,.1);
  --purple:#6B3FBF;--purple-bg:rgba(107,63,191,.1);
  --teal:#1A7F8B;--teal-bg:rgba(26,127,139,.1);
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--tx);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;font-size:13px;line-height:1.6}}

/* ── Header ── */
.hdr{{position:sticky;top:0;z-index:100;background:var(--bg);border-bottom:1px solid var(--bd);padding:9px 16px}}
.hdr-r1{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:7px}}
.hdr-title{{font-size:14px;font-weight:700}}
.hdr-sub{{font-size:11px;color:var(--mu)}}
.pills{{display:flex;gap:5px;flex-wrap:wrap}}
.pill{{display:inline-flex;align-items:center;gap:4px;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid;cursor:pointer;transition:opacity .12s;user-select:none}}
.pill.dim{{opacity:.28}}
.pill-all{{border-color:var(--bd2);color:var(--tx);background:var(--s3)}}
.pill-rm{{border-color:var(--amber);color:var(--amber);background:var(--amber-bg)}}
.pill-bq{{border-color:var(--red);color:var(--red);background:var(--red-bg)}}
.pill-fs{{border-color:var(--purple);color:var(--purple);background:var(--purple-bg)}}
.pill-ns{{border-color:var(--teal);color:var(--teal);background:var(--teal-bg)}}
.hdr-r2{{display:flex;gap:7px;align-items:center}}
.sbox{{flex:1;max-width:360px;padding:5px 12px;border-radius:var(--r);border:1px solid var(--bd);background:var(--s1);color:var(--tx);font-size:12px;outline:none}}
.sbox::placeholder{{color:var(--dim)}}
.sbox:focus{{border-color:var(--ac)}}
.cnt{{color:var(--mu);font-size:11px;margin-left:auto;white-space:nowrap}}
.tbtn{{padding:4px 10px;border-radius:var(--r);border:1px solid var(--bd);background:var(--s2);color:var(--mu);font-size:11px;cursor:pointer;font-family:inherit}}
.tbtn:hover{{color:var(--tx)}}
.theme-btn{{padding:4px 10px;border-radius:var(--r);border:1px solid var(--bd);background:var(--s2);color:var(--mu);font-size:11px;cursor:pointer;font-family:inherit}}

/* ── Stats bar ── */
.stats{{display:flex;gap:8px;padding:8px 16px;border-bottom:1px solid var(--bd);flex-wrap:wrap;align-items:center;font-size:11px}}
.chip{{display:flex;align-items:center;gap:4px;padding:3px 10px;border-radius:var(--r);border:1px solid var(--bd);background:var(--s1)}}
.chip b{{font-variant-numeric:tabular-nums;font-size:12px}}
.chip.regen b{{color:var(--ac)}}
.chip.rm b{{color:var(--amber)}} .chip.bq b{{color:var(--red)}}
.chip.fs b{{color:var(--purple)}} .chip.ns b{{color:var(--teal)}}
/* step histogram */
.hist-wrap{{display:flex;align-items:center;gap:6px;margin-left:auto}}
.hist-label{{font-size:10px;color:var(--mu)}}
.hist-bars{{display:flex;align-items:flex-end;gap:1px;height:20px}}
.hbar{{width:6px;border-radius:1px 1px 0 0;min-height:2px}}

/* ── Legend ── */
.legend{{display:flex;align-items:center;gap:6px;padding:5px 16px;border-bottom:1px solid var(--bd);flex-wrap:wrap}}
.leg-lbl{{font-size:10px;color:var(--mu)}}
.leg-sw{{display:inline-flex;align-items:center;gap:4px;font-size:10px;color:var(--dim)}}
.leg-c{{width:22px;height:10px;border-radius:2px}}

/* ── Card ── */
main{{padding:10px 16px;max-width:1080px;margin:0 auto}}
.card{{background:var(--s1);border:1px solid var(--bd);border-radius:var(--r);margin-bottom:8px;overflow:hidden}}
.card:hover{{border-color:var(--bd2)}}
.card[data-cat="result_misread"]{{border-left:3px solid var(--amber)}}
.card[data-cat="bad_query"]     {{border-left:3px solid var(--red)}}
.card[data-cat="fake_search"]   {{border-left:3px solid var(--purple)}}
.card[data-cat="no_search"]     {{border-left:3px solid var(--teal)}}
.card.regen-ok{{box-shadow:inset 0 0 0 1px rgba(63,185,80,.3)}}

/* card header */
.ch{{display:flex;align-items:flex-start;gap:8px;padding:9px 11px 6px;cursor:pointer;user-select:none}}
.cidx{{flex-shrink:0;min-width:28px;height:26px;background:var(--s2);border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;color:var(--mu);padding:0 4px;font-variant-numeric:tabular-nums}}
.cq{{font-size:13px;font-weight:600;flex:1;line-height:1.45}}
.cbadge{{flex-shrink:0;padding:2px 8px;border-radius:3px;font-size:10px;font-weight:700;border:1px solid;text-transform:uppercase;letter-spacing:.05em;margin-top:1px}}
.cbadge.rm{{border-color:var(--amber);color:var(--amber);background:var(--amber-bg)}}
.cbadge.bq{{border-color:var(--red);color:var(--red);background:var(--red-bg)}}
.cbadge.fs{{border-color:var(--purple);color:var(--purple);background:var(--purple-bg)}}
.cbadge.ns{{border-color:var(--teal);color:var(--teal);background:var(--teal-bg)}}
.regen-badge{{padding:2px 7px;border-radius:3px;font-size:10px;font-weight:700;border:1px solid;margin-top:1px;margin-left:4px}}
.regen-badge.ok{{border-color:var(--green);color:var(--green);background:var(--green-bg)}}
.regen-badge.fail{{border-color:var(--red);color:var(--red);background:var(--red-bg)}}

/* body */
.body{{display:none;padding:0 11px 10px;border-top:1px solid var(--bd)}}
.card.open .body{{display:block}}

/* ans comparison */
.ansbar{{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:8px 0 6px}}
@media(max-width:560px){{.ansbar{{grid-template-columns:1fr}}}}
.abox{{padding:6px 10px;border-radius:var(--r);border:1px solid var(--bd);background:var(--s2)}}
.albl{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px}}
.albl.gt{{color:var(--green)}} .albl.pr{{color:var(--amber)}}
.aval{{font-size:12px;line-height:1.5;word-break:break-word}}

/* turn list */
.turns{{display:flex;flex-direction:column;gap:5px;margin-top:4px}}
.turn{{border:1px solid var(--bd);border-radius:var(--r);overflow:hidden}}
.turn-hdr{{display:flex;align-items:center;gap:6px;padding:4px 10px;background:var(--s2);cursor:pointer;user-select:none}}
.t-role{{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;padding:1px 7px;border-radius:3px;flex-shrink:0}}
.role-a{{background:color-mix(in srgb,var(--ac) 15%,transparent);color:var(--ac)}}
.t-sum{{font-size:11px;color:var(--mu);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.t-blk{{font-size:10px;color:var(--dim);flex-shrink:0}}
.turn-body{{display:none;background:var(--bg)}}
.turn.topen .turn-body{{display:block}}

/* Token blocks */
.block-sec{{padding:6px 10px;border-top:1px solid var(--bd)}}
.block-sec:first-child{{border-top:none}}
.blk-title{{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px;display:flex;align-items:center;gap:8px}}
.view-tog{{display:inline-flex;border:1px solid var(--bd);border-radius:3px;overflow:hidden;font-size:9px}}
.vtog{{padding:1px 7px;cursor:pointer;background:var(--s2);color:var(--dim);border:none;font-family:inherit;font-size:9px}}
.vtog.active{{background:var(--ac);color:#fff}}
.tok-row{{display:flex;flex-wrap:wrap;gap:2px;align-items:flex-start;line-height:1}}
.tok{{display:inline-block;font-family:var(--mono);font-size:10.5px;padding:1px 3px;border-radius:2px;cursor:default;white-space:pre;transition:transform .08s}}
.tok:hover{{transform:scale(1.12);z-index:10;outline:1px solid rgba(255,255,255,.4)}}
[data-theme="light"] .tok{{color:rgba(0,0,0,.85)}}

/* ── Search result inset ── */
.search-res{{margin:6px 10px;border:1px solid var(--bd2);border-radius:var(--r);overflow:hidden}}
.sr-hdr{{display:flex;align-items:center;gap:6px;padding:5px 10px;background:var(--s3);font-size:11px;font-weight:700;color:var(--ac)}}
.sr-body{{padding:8px 10px;font-size:11px;color:var(--mu);font-family:var(--mono);white-space:pre-wrap;line-height:1.55;max-height:140px;overflow-y:auto}}
.sr-body::-webkit-scrollbar{{width:4px;height:4px}} .sr-body::-webkit-scrollbar-thumb{{background:var(--bd2);border-radius:2px}}
/* highlight result titles */
.sr-result{{margin-bottom:5px}}
.sr-result-title{{color:var(--ac);font-size:10.5px;font-weight:600}}
.sr-result-snip{{color:var(--tx);font-size:10.5px;margin-top:1px}}
</style>

<div class="hdr">
  <div class="hdr-r1">
    <span class="hdr-title">bs128 ckpt_4 · Token 生成轨迹</span>
    <span class="hdr-sub">{total} 道错误题 · block_size=128 · 重新推理</span>
    <div class="pills">
      <span class="pill pill-all" data-f="all">全部 <b>{total}</b></span>
      <span class="pill pill-rm"  data-f="result_misread">结果误读 <b>{cat_counts.get("result_misread",0)}</b></span>
      <span class="pill pill-bq"  data-f="bad_query">查询失败 <b>{cat_counts.get("bad_query",0)}</b></span>
      <span class="pill pill-fs"  data-f="fake_search">幻觉生成 <b>{cat_counts.get("fake_search",0)}</b></span>
      <span class="pill pill-ns"  data-f="no_search">无搜索 <b>{cat_counts.get("no_search",0)}</b></span>
    </div>
  </div>
  <div class="hdr-r2">
    <input class="sbox" type="search" placeholder="搜索题目/答案…" id="q">
    <span class="cnt" id="cnt"></span>
    <button class="tbtn" id="xbtn">展开全部</button>
    <button class="tbtn" id="tbtn">展开轨迹</button>
    <button class="theme-btn" id="theme-btn">☀ Light</button>
  </div>
</div>

<div class="stats">
  <div class="chip regen"><span>重跑准确率</span><b>{regen_acc}</b><small style="color:var(--mu)">({correct_count}/{total})</small></div>
  <div class="chip rm"><span>结果误读</span><b>{cat_counts.get("result_misread",0)}</b></div>
  <div class="chip bq"><span>查询失败</span><b>{cat_counts.get("bad_query",0)}</b></div>
  <div class="chip fs"><span>幻觉生成</span><b>{cat_counts.get("fake_search",0)}</b></div>
  <div class="chip ns"><span>无搜索</span><b>{cat_counts.get("no_search",0)}</b></div>
  <div class="hist-wrap" id="hist-wrap">
    <span class="hist-label">Token 揭示步分布 (step 0→127)</span>
    <div class="hist-bars" id="hist-bars"></div>
  </div>
</div>

<div class="legend" id="legend"></div>

<main id="main"></main>

<script>
const CASES = {cases_json};
const STATS  = {stats_json};

// ─── Color: hsl gradient over 128 steps ─────────────────────────────────────
const STOPS = [
  {{s:0,   h:0,   sat:88, lig:52}},
  {{s:20,  h:25,  sat:88, lig:52}},
  {{s:50,  h:45,  sat:85, lig:48}},
  {{s:80,  h:100, sat:72, lig:44}},
  {{s:110, h:175, sat:70, lig:42}},
  {{s:127, h:220, sat:75, lig:52}},
];
function stepColor(step) {{
  let lo = STOPS[0], hi = STOPS[STOPS.length-1];
  for(let i=0;i<STOPS.length-1;i++) {{
    if(step>=STOPS[i].s && step<=STOPS[i+1].s) {{lo=STOPS[i];hi=STOPS[i+1];break;}}
  }}
  const t = (step-lo.s)/Math.max(hi.s-lo.s,1);
  const h   = lo.h   + t*(hi.h-lo.h);
  const sat = lo.sat + t*(hi.sat-lo.sat);
  const lig = lo.lig + t*(hi.lig-lo.lig);
  return `hsl(${{h.toFixed(0)}},${{sat.toFixed(0)}}%,${{lig.toFixed(0)}}%)`;
}}

// ─── Token rendering ─────────────────────────────────────────────────────────
function renderToks(tokens, steps, orderByReveal) {{
  const indices = orderByReveal
    ? Array.from({{length:tokens.length}},(_,i)=>i).sort((a,b)=>steps[a]-steps[b])
    : Array.from({{length:tokens.length}},(_,i)=>i);
  return indices.map(i => {{
    const tok   = tokens[i];
    const step  = steps[i];
    const disp  = tok.replace(/\\n/g,'↵').replace(/\\t/g,'→') || '·';
    const safe  = disp.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    const bg    = stepColor(step);
    return `<span class="tok" style="background:${{bg}}" title="pos=${{i+1}} step=${{step}}">${{safe}}</span>`;
  }}).join('');
}}

// view state per (caseIdx, turnIdx, blockIdx)
const viewState = {{}};

function getViewKey(ci,ti,bi){{ return `${{ci}}_${{ti}}_${{bi}}`; }}

function setView(ci,ti,bi,v) {{
  viewState[getViewKey(ci,ti,bi)] = v;
  // re-render just that block section
  const el = document.getElementById(`bsec_${{ci}}_${{ti}}_${{bi}}`);
  if(el) el.outerHTML = renderBlockSec(ci, CASES[ci].turns[ti].blocks[bi], ti, bi);
}}

function renderBlockSec(ci, blk, ti, bi) {{
  const key = getViewKey(ci,ti,bi);
  const view = viewState[key] || 'pos';
  const tokHTML = renderToks(blk.tokens, blk.reveal_step, view==='rev');
  return `<div class="block-sec" id="bsec_${{ci}}_${{ti}}_${{bi}}">
    <div class="blk-title">
      Block ${{bi+1}}
      <div class="view-tog">
        <button class="vtog${{view==='pos'?' active':''}}" onclick="setView(${{ci}},${{ti}},${{bi}},'pos')">位置</button>
        <button class="vtog${{view==='rev'?' active':''}}" onclick="setView(${{ci}},${{ti}},${{bi}},'rev')">揭示序</button>
      </div>
    </div>
    <div class="tok-row">${{tokHTML}}</div>
  </div>`;
}}

function renderTurn(ci, t, ti) {{
  const blkCnt = t.blocks.length;
  const sumParts = [];
  if(t.is_toolcall) {{
    const m = t.text.match(/"query"\\s*:\\s*\\[([^\\]]+)\\]/);
    if(m) sumParts.push('🔍 ' + m[1].slice(0,60));
    else sumParts.push('🔍 search');
  }}
  if(t.force_answer) sumParts.push('✎ forced');
  const boxM = t.text.match(/<\\|box_start\\|>(.*?)<\\|box_end\\|>/s);
  if(boxM) sumParts.push('→ ' + boxM[1].trim().slice(0,50));
  if(!sumParts.length) sumParts.push(t.text.slice(0,60));

  const openCls = tracesExpanded ? ' topen' : '';
  const blocksHTML = t.blocks.map((b,bi) => renderBlockSec(ci,b,ti,bi)).join('');

  let srHTML = '';
  if(t.search_result && t.search_result.trim()) {{
    const raw = t.search_result;
    const lines = raw.split('\\n');
    const hdr = lines[0] || '';
    const bodyHTML = lines.slice(1).map(l => {{
      const m = l.match(/^(\\d+)\\.\\s*\\[([^\\]]*?)\\]\\s*(.*)/);
      if(m) return '<div class="sr-result"><span class="sr-result-title">[' +
        m[2].replace(/&/g,'&amp;').replace(/</g,'&lt;') + ']</span> ' +
        (m[3]||'').slice(0,200).replace(/&/g,'&amp;').replace(/</g,'&lt;') + '</div>';
      if(l.trim()) return '<div class="sr-result-snip">' + l.replace(/&/g,'&amp;').replace(/</g,'&lt;') + '</div>';
      return '';
    }}).join('');
    srHTML = '<div class="search-res"><div class="sr-hdr">🔍 ' +
      hdr.replace(/&/g,'&amp;').replace(/</g,'&lt;') +
      '</div><div class="sr-body">' + bodyHTML + '</div></div>';
  }}

  return `<div class="turn${{openCls}}" id="turn_${{ci}}_${{ti}}">
    <div class="turn-hdr" onclick="toggleTurn(${{ci}},${{ti}})">
      <span class="t-role role-a">Turn ${{ti+1}}</span>
      <span class="t-sum">${{sumParts.join(' · ').replace(/</g,'&lt;').replace(/>/g,'&gt;')}}</span>
      <span class="t-blk">${{blkCnt}} block${{blkCnt!==1?'s':''}}</span>
    </div>
    <div class="turn-body">${{srHTML}}${{blocksHTML}}</div>
  </div>`;
}}

function renderCard(ci, c) {{
  const CAT_CLS = {{result_misread:'rm',bad_query:'bq',fake_search:'fs',no_search:'ns'}};
  const CAT_LBL = {{result_misread:'结果误读',bad_query:'查询失败',fake_search:'幻觉生成',no_search:'无搜索'}};
  const cls = CAT_CLS[c.cat] || '';
  const regenBadge = c.regen_correct
    ? '<span class="regen-badge ok">重跑✓</span>'
    : '<span class="regen-badge fail">重跑✗</span>';
  const turnsHTML = c.turns.map((t,ti) => renderTurn(ci,t,ti)).join('');

  return `<div class="card${{c.regen_correct?' regen-ok':''}}" data-cat="${{c.cat}}"
      data-search="${{(c.question+' '+c.gt).toLowerCase().replace(/"/g,'')}}">
    <div class="ch" onclick="toggleCard(this)">
      <div class="cidx">${{ci+1}}</div>
      <div class="cq">${{c.question.replace(/</g,'&lt;').replace(/>/g,'&gt;')}}</div>
      <div class="cbadge ${{cls}}">${{CAT_LBL[c.cat]||c.cat}}</div>
      ${{regenBadge}}
    </div>
    <div class="body">
      <div class="ansbar">
        <div class="abox"><div class="albl gt">✓ 标准答案</div><div class="aval">${{c.gt.replace(/</g,'&lt;')}}</div></div>
        <div class="abox"><div class="albl pr">⚑ 重跑预测</div><div class="aval">${{(c.prediction||'').replace(/</g,'&lt;').slice(0,200)}}</div></div>
      </div>
      <div class="turns">${{turnsHTML}}</div>
    </div>
  </div>`;
}}

function toggleCard(hdr) {{ hdr.closest('.card').classList.toggle('open'); }}
function toggleTurn(ci,ti) {{
  document.getElementById(`turn_${{ci}}_${{ti}}`).classList.toggle('topen');
}}

// ─── Filter & render ──────────────────────────────────────────────────────────
let activeF = 'all', searchQ = '', allExpanded = true;  // 默认展开所有卡片
let tracesExpanded = false;  // turn 详情默认折叠（点击 turn header 展开）

function render() {{
  const main = document.getElementById('main');
  const filtered = CASES.map((c,i)=>{{c._i=i;return c;}}).filter(c => {{
    if(activeF !== 'all' && c.cat !== activeF) return false;
    if(searchQ && !c.question.toLowerCase().includes(searchQ) && !(c.gt||'').toLowerCase().includes(searchQ)) return false;
    return true;
  }});
  document.getElementById('cnt').textContent = filtered.length + ' 条';
  if(!filtered.length) {{ main.innerHTML='<div style="text-align:center;padding:48px;color:var(--dim)">无匹配</div>'; return; }}
  main.innerHTML = filtered.map(c => renderCard(c._i, c)).join('');
  if(allExpanded) main.querySelectorAll('.card').forEach(c=>c.classList.add('open'));
}}

// Pills
document.querySelectorAll('.pill').forEach(p => {{
  p.addEventListener('click', () => {{
    activeF = p.dataset.f;
    document.querySelectorAll('.pill').forEach(x => x.classList.toggle('dim', x!==p));
    render();
  }});
}});
document.getElementById('q').addEventListener('input', e => {{ searchQ=e.target.value.toLowerCase().trim(); render(); }});
document.getElementById('xbtn').addEventListener('click', () => {{
  allExpanded=!allExpanded;
  document.getElementById('xbtn').textContent=allExpanded?'收起全部':'展开全部';
  document.querySelectorAll('.card').forEach(c=>c.classList.toggle('open',allExpanded));
}});
document.getElementById('tbtn').addEventListener('click', () => {{
  tracesExpanded=!tracesExpanded;
  document.getElementById('tbtn').textContent=tracesExpanded?'收起轨迹':'展开轨迹';
  document.querySelectorAll('.turn').forEach(t=>t.classList.toggle('topen',tracesExpanded));
}});

// ─── Histogram ───────────────────────────────────────────────────────────────
function renderHist() {{
  const hist = STATS.hist;
  const max  = Math.max(...hist);
  const bars = hist.map((v,i) => {{
    const pct = max>0 ? (v/max*100).toFixed(1) : 0;
    const bg  = stepColor(i*8+4);
    return `<div class="hbar" style="height:${{pct}}%;background:${{bg}}" title="step ${{i*8}}-${{i*8+7}}: ${{v}} tokens"></div>`;
  }}).join('');
  document.getElementById('hist-bars').innerHTML = bars;
}}

// ─── Legend ──────────────────────────────────────────────────────────────────
function renderLegend() {{
  const steps = [0,20,50,80,110,127];
  const swatches = steps.map(s =>
    `<span class="leg-sw"><span class="leg-c" style="background:${{stepColor(s)}}"></span>step ${{s}}</span>`
  );
  document.getElementById('legend').innerHTML =
    '<span class="leg-lbl">揭示步：</span>' +
    swatches.join('<span style="color:var(--dim);font-size:9px">→</span>');
}}

// ─── Theme ───────────────────────────────────────────────────────────────────
function initTheme() {{
  const root = document.documentElement;
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  let theme;
  try {{ theme = localStorage.getItem('theme'); }} catch {{}}
  theme = theme || (mq.matches?'dark':'light');
  root.setAttribute('data-theme', theme);
  const btn = document.getElementById('theme-btn');
  btn.textContent = theme==='dark'?'☀ Light':'☾ Dark';
  btn.onclick = () => {{
    const cur = root.getAttribute('data-theme');
    const next = cur==='dark'?'light':'dark';
    root.setAttribute('data-theme', next);
    try {{ localStorage.setItem('theme', next); }} catch {{}}
    btn.textContent = next==='dark'?'☀ Light':'☾ Dark';
  }};
}}

// 等 DOM 准备好再渲染
if(document.readyState === 'loading') {{
  document.addEventListener('DOMContentLoaded', function() {{
    initTheme(); renderHist(); renderLegend(); render();
    // 默认展开所有卡片
    document.querySelectorAll('.card').forEach(c=>c.classList.add('open'));
  }});
}} else {{
  initTheme(); renderHist(); renderLegend(); render();
  document.querySelectorAll('.card').forEach(c=>c.classList.add('open'));
}}
</script>
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", default="/research/cbim/vast/mz751/Projects/DLLM-Searcher/my_eval/results/bs128/trace")
    ap.add_argument("--out", default="/research/cbim/vast/mz751/Projects/DLLM-Searcher/my_eval/results/bs128/trace/bs128_trace_viz.html")
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    rank_files = sorted(trace_dir.glob("rank*.jsonl"))
    if not rank_files:
        print(f"[ERROR] 未找到 rank*.jsonl 文件于 {trace_dir}")
        return

    cases = []
    for rf in rank_files:
        for line in open(rf):
            line = line.strip()
            if line:
                try: cases.append(json.loads(line))
                except: pass
    print(f"共加载 {len(cases)} 个 trace 案例（来自 {len(rank_files)} 个 rank 文件）")

    # 按原始 category 排序
    cat_order = ["result_misread", "bad_query", "fake_search", "no_search"]
    cases.sort(key=lambda c: (cat_order.index(c.get("cat","no_search")) if c.get("cat") in cat_order else 99, c.get("idx",0)))

    stats = compute_stats(cases)
    print(f"统计: n={stats['n']}, avg_blocks={stats['avg_blocks']}, "
          f"step_mean={stats['step_mean']}, regen_correct={stats['regen_correct']}")

    html = build_html(cases, stats)
    with open(args.out, "w") as f:
        f.write(html)
    sz = Path(args.out).stat().st_size
    print(f"[saved] {args.out}  ({sz//1024} KB)")


if __name__ == "__main__":
    main()
