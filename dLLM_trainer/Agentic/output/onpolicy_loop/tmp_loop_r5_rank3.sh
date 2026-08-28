#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export CUDA_VISIBLE_DEVICES=3
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_mz751
/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python -u /research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/collect_round0_rollouts.py     --rank 3 --world 8     --model "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop/pi5/model"     --input "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop/pi6/pool.jsonl"     --output "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop/pi6/rollouts"     --n_rolls 8     --round_id 5     2>&1 | tee "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop/pi6/rollouts/logs/rank3.log"
