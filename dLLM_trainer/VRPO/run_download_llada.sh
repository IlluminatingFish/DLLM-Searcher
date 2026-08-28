#!/bin/bash
export PATH=/research/cbim/vast/mz751/miniforge3/envs/llada/bin:$PATH
export HF_HOME=/research/cbim/vast/mz751/.cache/huggingface
exec huggingface-cli download GSAI-ML/LLaDA-1.5 \
  --local-dir /research/cbim/vast/mz751/.cache/huggingface/hub/models--GSAI-ML--LLaDA-1.5 \
  --local-dir-use-symlinks False
