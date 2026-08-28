#!/bin/bash
# Multi-node LLaDA SFT: hermes (rank 0) + hercules (rank 1), 16 GPUs total
MASTER_ADDR="hermes.cs.rutgers.edu"
MASTER_PORT=29600
WORKDIR="/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/SFT/dLLM-RL"
PYTHON="/research/cbim/vast/mz751/miniforge3/envs/dllm-rl/bin"
ACCEL_CFG="accelerate_configs/2_nodes_16_gpus_deepspeed_zero3.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_HERMES="${WORKDIR}/llada_sft_hermes_${TIMESTAMP}.log"
LOG_HERCULES="${WORKDIR}/llada_sft_hercules_${TIMESTAMP}.log"

echo "=============================="
echo "Multi-node LLaDA SFT (16 GPUs)"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "hermes log:   $LOG_HERMES"
echo "hercules log: $LOG_HERCULES"
echo "=============================="

# 先启动 worker (hercules, rank 1)
ssh -o BatchMode=yes -o StrictHostKeyChecking=no hercules \
  "cd $WORKDIR && PATH=$PYTHON:\$PATH \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   $PYTHON/accelerate launch \
     --num_machines 2 --num_processes 16 \
     --machine_rank 1 \
     --main_process_ip $MASTER_ADDR \
     --main_process_port $MASTER_PORT \
     --config_file $ACCEL_CFG \
     train/sft_llada.py config=configs/sft_llada.yaml \
   > $LOG_HERCULES 2>&1" &
HERCULES_SSH=$!
echo "hercules started (ssh bg pid=$HERCULES_SSH)"

sleep 5

# 启动 master (hermes, rank 0)
ssh -o BatchMode=yes -o StrictHostKeyChecking=no hermes \
  "cd $WORKDIR && PATH=$PYTHON:\$PATH \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   $PYTHON/accelerate launch \
     --num_machines 2 --num_processes 16 \
     --machine_rank 0 \
     --main_process_ip $MASTER_ADDR \
     --main_process_port $MASTER_PORT \
     --config_file $ACCEL_CFG \
     train/sft_llada.py config=configs/sft_llada.yaml \
   > $LOG_HERMES 2>&1" &
HERMES_SSH=$!
echo "hermes started (ssh bg pid=$HERMES_SSH)"
echo "LOG_HERMES=$LOG_HERMES"
echo "LOG_HERCULES=$LOG_HERCULES"
wait
