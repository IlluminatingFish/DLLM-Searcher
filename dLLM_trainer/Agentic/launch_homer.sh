#!/bin/bash
# Plan C on homer: SALT-lite per-turn credit assignment
# 用法: bash launch_homer.sh
set -e
echo "Launching Plan C on homer..."
ssh -o StrictHostKeyChecking=no homer.cs.rutgers.edu \
    "cd /research/cbim/vast/mz751/Projects/DLLM-Searcher && \
     tmux new-session -d -s agentic_plan_c \
     'export PLAN=C SERVER=homer MAX_ROUNDS=15; \
      bash dLLM_trainer/Agentic/run_agentic_loop.sh 2>&1 | \
      tee dLLM_trainer/Agentic/output/plan_c/logs/tmux_main.log; \
      echo DONE' && \
     echo 'tmux session agentic_plan_c started on homer'"
echo "Use: ssh homer.cs.rutgers.edu 'tmux attach -t agentic_plan_c'"
