#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export CUDA_VISIBLE_DEVICES=7
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751_offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_mz751_offline
/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python -u /research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/collect_round0_rollouts.py     --rank 7 --world 8     --model "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL/sft_llada_x0pred/ckpt_6/optimized"     --input "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/offline_loop/offline_rollouts/round4/pool.jsonl"     --output "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/offline_loop/offline_rollouts/round4/rollouts"     --n_rolls 8     --round_id 4     2>&1 | tee "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/offline_loop/offline_rollouts/round4/rollouts/logs/rank7.log"
