#!/bin/bash
# Monitor GPUs and auto-launch DPO training when 4 GPUs are free

WORKDIR="/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO"
LOG="$WORKDIR/gpu_monitor.log"
FREE_THRESHOLD=2000  # MiB — GPU considered free if used memory < this

echo "[$(date)] GPU monitor started. Waiting for 4 free GPUs..." | tee -a "$LOG"

while true; do
    FREE_GPUS=()
    for i in 0 1 2 3 4 5 6 7; do
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i 2>/dev/null)
        if [ -n "$USED" ] && [ "$USED" -lt "$FREE_THRESHOLD" ]; then
            FREE_GPUS+=($i)
        fi
    done

    echo "[$(date)] Free GPUs (< ${FREE_THRESHOLD}MiB used): [${FREE_GPUS[*]}] — ${#FREE_GPUS[@]}/8 free" | tee -a "$LOG"

    if [ ${#FREE_GPUS[@]} -ge 4 ]; then
        SELECTED="${FREE_GPUS[0]},${FREE_GPUS[1]},${FREE_GPUS[2]},${FREE_GPUS[3]}"
        echo "[$(date)] Found 4 free GPUs: [$SELECTED]. Launching DPO training in 10s..." | tee -a "$LOG"
        sleep 10

        # Double-check they're still free before launching
        STILL_FREE=0
        for gpu in ${FREE_GPUS[0]} ${FREE_GPUS[1]} ${FREE_GPUS[2]} ${FREE_GPUS[3]}; do
            USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu 2>/dev/null)
            if [ -n "$USED" ] && [ "$USED" -lt "$FREE_THRESHOLD" ]; then
                STILL_FREE=$((STILL_FREE + 1))
            fi
        done

        if [ "$STILL_FREE" -lt 4 ]; then
            echo "[$(date)] GPUs snatched before launch, re-checking..." | tee -a "$LOG"
            sleep 60
            continue
        fi

        echo "[$(date)] Launching: CUDA_VISIBLE_DEVICES=$SELECTED bash recipes/run_dpo.sh" | tee -a "$LOG"
        cd "$WORKDIR"
        export CUDA_VISIBLE_DEVICES="$SELECTED"
        bash recipes/run_dpo.sh 2>&1 | tee -a "$LOG"
        echo "[$(date)] Training finished (exit code: $?)." | tee -a "$LOG"
        break
    fi

    sleep 120  # Check every 2 minutes
done
