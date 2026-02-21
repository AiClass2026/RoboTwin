#!/bin/bash

base_checkpoint=${1:-openvla/openvla-7b}
lora_finetuned_checkpoint_dir=${2:-/zixiao/code/AiClass/RoboTwin/ckpt/20260220133352--openvla-7b+aloha_beat_block_hammer_mix+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--26000_chkpt}
save_path=${3:-$lora_finetuned_checkpoint_dir}

python vla-scripts/merge_lora_weights_and_save.py \
  --base_checkpoint "$base_checkpoint" \
  --lora_finetuned_checkpoint_dir "$lora_finetuned_checkpoint_dir" \
  --save_path "$save_path"
