#!/usr/bin/env python3
"""Convert Hotpot parquet to question/answer jsonl for DLLM-Searcher eval."""
import argparse
import pandas as pd
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet file (e.g. data/hotpot/val_data.parquet)")
    parser.add_argument("--output", required=True, help="Output jsonl file")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples (0 = all)")
    parser.add_argument("--skip_empty_answer", action="store_true", help="Skip rows where answer is empty")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError(f"Expected 'question' and 'answer' columns, got {list(df.columns)}")

    if args.skip_empty_answer:
        df = df[df["answer"].notna() & (df["answer"].astype(str).str.strip() != "")]
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    out_dir = __import__("os").path.dirname(args.output)
    if out_dir:
        __import__("os").makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            obj = {"question": str(row["question"]), "answer": str(row["answer"])}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
