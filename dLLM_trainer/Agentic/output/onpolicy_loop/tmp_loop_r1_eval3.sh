#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export CUDA_VISIBLE_DEVICES=3
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_mz751
/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python -u /research/cbim/vast/mz751/Projects/DLLM-Searcher/my_eval/run_llada_eval.py     --model    "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop/pi2/model"     --input    "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/data/eval_600.jsonl"     --output   "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop/pi2/eval/shard3.jsonl"     --offset   225     --max_samples 75     2>&1 | tee "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop/pi2/eval/shard3.log"
