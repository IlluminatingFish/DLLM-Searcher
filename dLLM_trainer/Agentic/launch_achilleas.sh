#!/bin/bash
# Plan B on achilleas: multi-component reward (F1 + format + search_utility)
# 用法: bash launch_achilleas.sh
set -e
echo "Launching Plan B on achilleas..."
ssh -o StrictHostKeyChecking=no achilleas.cs.rutgers.edu \
    "cd /research/cbim/vast/mz751/Projects/DLLM-Searcher && \
     tmux new-session -d -s agentic_plan_b \
     'export PLAN=B SERVER=achilleas MAX_ROUNDS=15; \
      bash dLLM_trainer/Agentic/run_agentic_loop.sh 2>&1 | \
      tee dLLM_trainer/Agentic/output/plan_b/logs/tmux_main.log; \
      echo DONE' && \
     echo 'tmux session agentic_plan_b started on achilleas'"
echo "Use: ssh achilleas.cs.rutgers.edu 'tmux attach -t agentic_plan_b'"
