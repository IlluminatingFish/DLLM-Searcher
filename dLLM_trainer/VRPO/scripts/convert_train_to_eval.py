#!/usr/bin/env python3
"""
Convert SFT/DPO training data into eval format {question, answer}.

- question : extracted from the "User: ..." line at end of the prompt
- answer   : verbose chosen/response box content (for LLM-judge eval only,
             NOT suitable for CEM-1 / exact-match metrics)

Usage:
  # From DPO train.jsonl  (format: {prompt, chosen, rejected})
  python scripts/convert_train_to_eval.py \
    --input data/train.jsonl --answer_from chosen \
    --output data/train_as_eval_v2.jsonl

  # From SFT data.json    (format: {prompt, response})
  python scripts/convert_train_to_eval.py \
    --input ../../SFT/data/data.json --answer_from response \
    --output data/sft_as_eval_v2.jsonl

  # 200-sample subset for quick testing
  python scripts/convert_train_to_eval.py \
    --input data/train.jsonl --answer_from chosen \
    --output data/train_as_eval_200_v2.jsonl --max_samples 200
"""

import argparse
import json
import re
import sys
from pathlib import Path


def extract_question(prompt: str) -> str | None:
    """Extract the plain question from the prompt's 'User: ...' line."""
    m = re.search(r'User:\s*(.*?)\s*<\|im_end\|>', prompt, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_answer(text: str) -> str | None:
    """Extract content between <|box_start|> and <|box_end|>."""
    m = re.search(r'<\|box_start\|>(.*?)<\|box_end\|>', text, re.DOTALL)
    return m.group(1).strip() if m else None


def load_samples(path: str, answer_from: str):
    p = Path(path)
    if p.suffix == '.json':
        with open(p) as f:
            data = json.load(f)
    else:  # .jsonl
        with open(p) as f:
            data = [json.loads(line) for line in f if line.strip()]

    samples = []
    skipped = 0
    for obj in data:
        q = extract_question(obj.get('prompt', ''))
        raw_ans = obj.get(answer_from, '')
        a = extract_answer(raw_ans)

        if not q:
            skipped += 1
            continue
        if not a:
            # Fall back to full response text (trimmed)
            a = raw_ans.strip()[:1000]

        samples.append({'question': q, 'answer': a})

    print(f"Loaded {len(samples)} samples, skipped {skipped} (no question found).",
          file=sys.stderr)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input file (.json or .jsonl)')
    parser.add_argument('--answer_from', default='chosen',
                        choices=['chosen', 'response', 'rejected'],
                        help='Which field to extract the answer from')
    parser.add_argument('--output', required=True, help='Output .jsonl file')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Limit samples (0 = all)')
    args = parser.parse_args()

    samples = load_samples(args.input, args.answer_from)

    if args.max_samples > 0:
        samples = samples[:args.max_samples]
        print(f"Trimmed to {len(samples)} samples.", file=sys.stderr)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"Wrote {len(samples)} samples → {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
