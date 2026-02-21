# Simplified fine-tuning script for OpenVLA-OFT with L1 regression + FSDP.
# Global batch size < 16 may lead to unsuccessful training.

project_root=/zixiao/code/AiClass/RoboTwin

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 增加 NCCL 稳定性环境变量，防止 8 卡死锁或通信超时
export NCCL_TIMEOUT=1200000
# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune_simple.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir ${project_root}/data/oft-rlds \
  --dataset_name aloha_beat_block_hammer_mix \
  --run_root_dir ${project_root}/ckpt \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --gradient_checkpointing False \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 10000 \
  --max_steps 20000 \
  --grad_accumulation_steps 1 \
  --run_id_note $(date +%Y%m%d_%H%M%S) \
  --use_val_set True \
  --val_freq 2000 \
  --save_freq 2000 \
  --image_aug True \
  --lora_rank 32 \
  --log_dir ${project_root}/runs \
  --log_freq 10 \
  ### Example: resume a training run
  # --resume True \
  # --resume_step 5000 \
  # --resume_base_model_path openvla/openvla-7b
