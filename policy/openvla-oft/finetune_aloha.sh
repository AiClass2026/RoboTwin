# In experiments, global batch size of less than 16 will easily lead to unsuccessful training, where the training and validation 
# loss would not converge low enough, and the final policy would repeat one trajectory regardless of the visual and language inputs.

# 项目根目录
project_root=/zixiao/code/AiClass/RoboTwin

export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir ${project_root}/data/oft-rlds \
  --dataset_name aloha_beat_block_hammer \
  --run_root_dir ${project_root}/ckpt \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_fsdp True \
  --gradient_checkpointing True \
  --grad_accumulation_steps 4 \
  --use_proprio True \
  --batch_size 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100005 \
  --use_val_set True \
  --val_freq 1000 \
  --save_freq 1 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "your_wandb_identity" \
  --wandb_project "you_wandb_object" \
  --run_id_override "Optional" \
  --run_id_note some_run_id_note \
  ### example usage for resuming a training process
  # --resume True\
  # --resume_step 5000 \
  # --resume_base_model_path openvla/openvla-7b
