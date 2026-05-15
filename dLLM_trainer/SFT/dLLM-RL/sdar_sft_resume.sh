#!/bin/bash
# 从 ckpt_1 继续训练
# ckpt_1 = 已完成 epoch 1，从 epoch 2 继续
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 减少显存碎片
accelerate launch --debug \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 8888 \
  --config_file accelerate_configs/1_node_4_gpus_deepspeed_zero3.yaml \
  train/sft_sdar.py \
  config=configs/sft_sdar.yaml \
  experiment.resume_from_checkpoint=sft_sdar/ckpt_8/optimized \
  training.resume_epoch=9
