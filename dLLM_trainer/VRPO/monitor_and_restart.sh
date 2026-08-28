#!/bin/bash
# Monitor DPO training and restart on same GPUs if it dies

WORKDIR="/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO"
LOG="$WORKDIR/gpu_monitor.log"
GPUS="0,1,2,3"

echo "[$(date)] Monitor started. Watching training on CUDA_VISIBLE_DEVICES=$GPUS" | tee -a "$LOG"

while true; do
    # Check if training is still alive
    ALIVE=$(pgrep -f "run_dpo.sh" | wc -l)

    if [ "$ALIVE" -eq 0 ]; then
        echo "[$(date)] Training process not found. Restarting on CUDA_VISIBLE_DEVICES=$GPUS ..." | tee -a "$LOG"
        sleep 10  # brief pause before restart

        cd "$WORKDIR"
        export CUDA_VISIBLE_DEVICES="$GPUS"
        bash recipes/run_dpo.sh 2>&1 | tee -a "$LOG"
        EXIT_CODE=$?
        echo "[$(date)] Training exited with code $EXIT_CODE." | tee -a "$LOG"

        if [ "$EXIT_CODE" -eq 0 ]; then
            echo "[$(date)] Training completed successfully. Monitor exiting." | tee -a "$LOG"
            break
        else
            echo "[$(date)] Training crashed (exit $EXIT_CODE). Will retry in 30s..." | tee -a "$LOG"
            sleep 30
        fi
    else
        echo "[$(date)] Training alive (run_dpo.sh found). Checking again in 60s." | tee -a "$LOG"
        sleep 60
    fi
done
