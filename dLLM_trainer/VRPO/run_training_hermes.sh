#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/espo/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export NUM_GPUS=7
cd /research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO
exec bash recipes/run_dpo.sh
