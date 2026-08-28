#!/bin/bash
# Plan A on hermes: 标准 online GRPO，F1+format reward，8 rollouts/题
# 用法: bash launch_hermes.sh
set -e
echo "Launching Plan A on hermes..."
ssh -o StrictHostKeyChecking=no hermes.cs.rutgers.edu \
    "cd /research/cbim/vast/mz751/Projects/DLLM-Searcher && \
     tmux new-session -d -s agentic_plan_a \
     'export PLAN=A SERVER=hermes MAX_ROUNDS=15; \
      bash dLLM_trainer/Agentic/run_agentic_loop.sh 2>&1 | \
      tee dLLM_trainer/Agentic/output/plan_a/logs/tmux_main.log; \
      echo DONE' && \
     echo 'tmux session agentic_plan_a started on hermes'"
echo "Use: ssh hermes.cs.rutgers.edu 'tmux attach -t agentic_plan_a'"
