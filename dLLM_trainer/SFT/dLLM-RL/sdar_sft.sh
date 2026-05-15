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
  config=configs/sft_sdar.yaml