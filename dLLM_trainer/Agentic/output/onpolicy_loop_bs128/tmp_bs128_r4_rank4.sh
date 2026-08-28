#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export CUDA_VISIBLE_DEVICES=4
export TRITON_CACHE_DIR=/tmp/triton_cache_mz751_bs128
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache_mz751_bs128
/research/cbim/vast/mz751/miniforge3/envs/espo/bin/python -u /research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/collect_round0_rollouts.py     --rank 4 --world 8     --model "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop_bs128/pi4/model"     --input "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop_bs128/pi5/pool.jsonl"     --output "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop_bs128/pi5/rollouts"     --n_rolls 8     --round_id 4     --block_size 128     --num_steps 128     2>&1 | tee "/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/Agentic/output/onpolicy_loop_bs128/pi5/rollouts/logs/rank4.log"
