# DLLM-Searcher Project Notes

Last updated: 2026-05-14

---

## Session Log

### 2026-05-14 — First session with Claude Code

**What we did:**

1. **Explored the project structure** — read all READMEs and understood the 4-step
   pipeline: Dataroller (data collection) → SFT training → VRPO training → Evaluation.

2. **Located all trained checkpoints:**
   - SFT: `sft_sdar/ckpt_{0,2,4,6,8}/optimized` (every 2 epochs over 10 total)
   - VRPO: multiple runs under `output/dpo_dpo_<timestamp>/`, with checkpoints saved
     in: 20260308 (ckpt-90, ckpt-130), 20260413 (ckpt-50), 20260513 (ckpt-50),
     and several runs today (20260514).

3. **Clarified the eval pipeline** — confirmed that `run_test.sh` is the rollout
   (generates predictions) and `cal_acc.py` is the actual scoring step.

4. **Analyzed existing rollout results** (no GPU needed, read files from disk):
   - Hotpot 100 samples: CEM-1 = 5/100 = 5.0%, null predictions = 72/100,
     max_turns termination = 61/100. Very poor performance.

5. **Identified root causes** (two problems):
   - **Problem 1 (extract_answer):** The `answer_rl` strategy forces `<|box_start|>`
     at position 63 in the block, but the model generates another `<|box_start|>`
     right after (as part of `<|box_start|>assistant\n<think>...`). The old
     `split()[1]` grabbed the empty text between the two tokens → null prediction.
   - **Problem 2 (answer_rl strategy):** The model was never trained to understand
     that "position 63 = give answer now," so it ignores the forced token and keeps
     generating thinking content. This is the deeper issue.

6. **Fixed extract_answer (Fix 1)** in `dLLM_trainer/VRPO/my_train/my_test.py`:
   - Added `import re`
   - Replaced single `split()[1]` with a loop scanning all box pairs, filtering
     out garbage (role tokens, unclosed think tags, tool_call markers)
   - Also discovered the old 5% CEM-1 was inflated: predictions were thinking text
     that accidentally contained answer keywords. True honest accuracy is ~1%.

7. **Wrote this Claude.md** with all findings, run commands, and next steps.

---

## Key Paths

- SFT checkpoints: `dLLM_trainer/SFT/dLLM-RL/sft_sdar/ckpt_{0,2,4,6,8}/optimized`
- VRPO checkpoints: `dLLM_trainer/VRPO/output/dpo_dpo_<timestamp>/checkpoints/checkpoint-<N>`
- Eval rollout output: `dLLM_trainer/VRPO/output/preact_eval/<dataset>/`
- VRPO training data: `dLLM_trainer/VRPO/data/train.jsonl` (2236 samples)
- SFT training data: `dLLM_trainer/SFT/data/data.json` (3977 samples)

## GPU Note
GPUs 0,1,2,3 may be occupied. Check with `nvidia-smi` before running.
Use `CUDA_VISIBLE_DEVICES=4,5,6,7` to use the free GPUs.

---

## Step 4: Evaluation

### Step 4a — Rollout (requires GPU, run from dLLM_trainer/VRPO/)
```bash
cd dLLM_trainer/VRPO

# Use free GPUs (4,5,6,7). Override model and dataset as needed.
CUDA_VISIBLE_DEVICES=4,5,6,7 \
GPU_NUM=4 \
MODEL_PATH="/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/output/dpo_dpo_<timestamp>/checkpoints/checkpoint-<N>" \
DATASETS="hotpot" \
MAX_SAMPLES=100 \
bash ../../my_eval/run_test.sh
```

Output goes to: `output/preact_eval/<dataset>/rollout_results_rank*_<timestamp>.jsonl`

### Step 4b — Merge rollout files from all ranks
```bash
cat dLLM_trainer/VRPO/output/preact_eval/hotpot/rollout_results_rank*_<timestamp>.jsonl \
  > /tmp/hotpot_merged.jsonl
```

### Step 4c — Score with CEM-1 + LLM Judge (no GPU needed)
```bash
cd my_eval
python cal_acc.py --data /tmp/hotpot_merged.jsonl
```

---

## Baseline Results (before any fixes)
- Dataset: Hotpot, 100 samples, SFT ckpt_8 model, run March 2026
- Reported CEM-1: 5/100 = 5.0% — but this was INFLATED
  - The old `extract_answer` was returning full thinking text which accidentally
    contained the answer keywords, fooling CEM-1's weak word-coverage check
- True clean accuracy: ~1% (only 1 case had a genuinely clean answer)
- Null/empty predictions: 72/100
- Termination by max_turns (no answer given): 61/100

---

## Root Cause Analysis

### Problem 1: extract_answer double-box_start bug
**File:** `dLLM_trainer/VRPO/my_train/my_test.py` → `extract_answer()`

The `answer_rl` remasking strategy in `jetengine/engine/scheduler.py` forces
`<|box_start|>` at hardcoded position 63 and `<|box_end|>` at position 126 in the
generation block. The model then generates `<|box_start|>assistant\n<think>...` in
positions 64-125 (role token + more thinking), producing TWO `<|box_start|>` tokens.

Old code used `content.split(ANSWER_START)[1]` which grabbed the text BETWEEN the
two `<|box_start|>` tokens → empty string → null prediction.

### Problem 2: answer_rl strategy (root cause, TODO)
The model was never trained to associate "position 63 in the block = give answer now."
So the `answer_rl` forcing is ignored: the model generates thinking content inside the
answer region instead of a clean answer. This is the deeper fix needed.

---

## Fixes Applied

### Fix 1: extract_answer — DONE (2026-05-14)
**File:** `dLLM_trainer/VRPO/my_train/my_test.py`

Replaced single `split()[1]` with a loop that scans ALL `<|box_start|>...<|box_end|>`
pairs and returns the first one that passes a garbage filter (no `<think>`, `<tool_call>`,
`<|im_end|>`, `<|im_start|>`). Also strips role tokens and closed `<think>` blocks.

Result on existing hotpot data (re-extraction without re-running rollout):
- Null predictions: 72 → 95 (more filtered, but cleaner)
- CEM-1: 5% → 1% (5% was inflated; 1% is the honest baseline)

The extraction is now correct. Further accuracy gain requires Fix 2.

### Fix 2: answer_rl strategy — PLANNED, NOT YET IMPLEMENTED

**Root cause:**
`jetengine/engine/scheduler.py` lines 160-170 — the `answer_rl` strategy forces
`<|box_start|>` (token 151648) at hardcoded position 63 and `<|box_end|>` (token
151649) at position 126 within the generation block. The model was never trained to
associate "position 63 = give answer now," so it ignores the forced token and generates
`<|box_start|>assistant\n<think>...` (role token + more thinking) in positions 64-125.
Result: the answer region contains thinking, not a clean answer.

**Chosen solution: Option A — Prepend `<|box_start|>` to the context prompt**
Do NOT modify the jetengine. Instead, change the rollout in `my_test.py` so that on
the final answer turn, `<|box_start|>` is appended directly to the context string as
part of the prefix. The model then sees it as an already-started answer and naturally
completes it. Use `low_confidence_static` strategy (normal denoising) instead of
`answer_rl`.

**Exactly 4 changes needed, all in one file:**
`dLLM_trainer/VRPO/my_train/my_test.py`

---

**Change 1 — Strategy switch (line ~385)**

Find this block in `rollout_batch`:
```python
if turn >= 3:
    sampling_params.remasking_strategy = "answer_rl"
else:
    sampling_params.remasking_strategy = "toolcall_pre_rl"
```
Change to:
```python
if turn >= 3:
    sampling_params.remasking_strategy = "low_confidence_static"
else:
    sampling_params.remasking_strategy = "toolcall_pre_rl"
```

---

**Change 2 — Append forcing prompt to context before final-turn generation**

In `rollout_batch`, just before `prompts_for_generation` is built, add logic so that
on the final answer turn (turn >= 3) we append to each active sample's context:
```python
ANSWER_FORCING_SUFFIX = (
    "<|im_start|>user\n"
    "Based on your research above, give your final answer.\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<|box_start|>"
)
```
So the model's prefix ends with `<|box_start|>` and it only needs to generate the
answer text + `<|box_end|>`.

Important: mark each sample with a flag (e.g. `sample.answer_forcing = True`) so
downstream logic knows to handle detection differently.

---

**Change 3 — Answer detection for forced-prefix turns**

Currently `has_answer` and `extract_answer` both look for `<|box_start|>` inside
`new_text`. After Change 2, `<|box_start|>` is in the prefix (context), NOT in
`new_text`. So `has_answer(new_text)` will return False even on a correct answer.

On forced-prefix turns, detection changes to:
- Check for `<|box_end|>` in `new_text`
- Extract answer as: `new_text.split("<|box_end|>")[0].strip()`

Logic to add in the per-sample processing loop in `rollout_batch`:
```python
if getattr(sample, 'answer_forcing', False):
    # <|box_start|> was already in prefix; model generates answer + <|box_end|>
    if self.ANSWER_END in new_text:
        prediction = new_text.split(self.ANSWER_END)[0].strip()
        completed_results[sample.idx] = self._create_result(
            sample, prediction if prediction else None, "answer"
        )
        continue
```
This block should come BEFORE the existing `if self.has_answer(new_text):` check.

---

**Change 4 — Add `<|box_end|>` as a stop word on the final turn**

Token ID for `<|box_end|>` is `151649`. On the answer turn, add it to stop_words
so generation halts as soon as the model closes the answer box.

Before the `llm.generate_streaming(...)` call, on turn >= 3:
```python
if turn >= 3:
    sampling_params.stop_words = [151645, 151658, 151649]  # add <|box_end|>
else:
    sampling_params.stop_words = [151645, 151658]  # original
```
Remember to restore after the turn so it doesn't carry over.

---

**After implementing Fix 2, re-run evaluation:**
```bash
# From dLLM_trainer/VRPO/
CUDA_VISIBLE_DEVICES=4,5,6,7 \
GPU_NUM=4 \
MODEL_PATH="<latest VRPO checkpoint>" \
DATASETS="hotpot" \
MAX_SAMPLES=100 \
bash ../../my_eval/run_test.sh

# Then merge and score
cat output/preact_eval/hotpot/rollout_results_rank*_<timestamp>.jsonl > /tmp/hotpot_merged.jsonl
cd ../../my_eval && python cal_acc.py --data /tmp/hotpot_merged.jsonl
```

---

## Other Known Issues

- `run_test.sh` default MODEL_PATH points to `ckpt_7` which does NOT exist.
  Always override with `MODEL_PATH=...` when running.
- VRPO `run_dpo.sh` uses `learning_rate=5e-6` but `dpo.yaml` says `5e-7` (10x conflict).
  The shell script takes precedence. Check which is intended before next VRPO run.
- 255/2236 VRPO rejected samples lack `<|box_start|>` format (11% of rejected).
