#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export CUDA_VISIBLE_DEVICES=4
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751_bs128
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_mz751_bs128
/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python -u /research/cbim/vast/mz751/Projects/DLLM-Searcher/my_eval/run_llada_eval.py     --model    "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop_bs128/pi5/model"     --input    "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/data/eval_600.jsonl"     --output   "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop_bs128/pi5/eval/shard4.jsonl"     --offset   300     --max_samples 75     --block_size 128     --num_steps  128     2>&1 | tee "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop_bs128/pi5/eval/shard4.log"
