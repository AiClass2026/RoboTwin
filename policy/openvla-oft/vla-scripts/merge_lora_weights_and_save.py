"""
Loads a checkpoint that only has a LoRA adapter (no merged model) and merges the adapter
into the base OpenVLA model. Saves the final merged model to `save_path` (or to
`lora_finetuned_checkpoint_dir` if `save_path` is not specified).

Make sure to specify the correct base checkpoint when running this script. For example,
- if you fine-tuned the default OpenVLA-7B model without modifications, then `--base_checkpoint=="openvla/openvla-7b"`
- if you fine-tuned a different model or resumed fine-tuning from a different checkpoint, then specify that base checkpoint
- if you fine-tuned the default OpenVLA-7B model with modifications to `modeling_prismatic.py` (OpenVLA class definition),
  then the base checkpoint path should point to the checkpoint containing the modifications

Usage:
    python vla-scripts/merge_lora_weights_and_save.py \
        --base_checkpoint openvla/openvla-7b \
        --lora_finetuned_checkpoint_dir /PATH/TO/CHECKPOINT/DIR/ \
        --save_path /PATH/TO/MERGED/MODEL/          # optional; defaults to lora_finetuned_checkpoint_dir
"""

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

import prismatic.extern.hf.configuration_prismatic as _configuration_prismatic_module
import prismatic.extern.hf.modeling_prismatic as _modeling_prismatic_module
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


@dataclass
class ConvertConfig:
    # fmt: off

    base_checkpoint: Union[str, Path] = ""                    # Base model checkpoint path/dir
    lora_finetuned_checkpoint_dir: Union[str, Path] = ""      # Checkpoint directory containing the LoRA adapter
    save_path: Optional[Union[str, Path]] = None              # Where to save the merged model; defaults to lora_finetuned_checkpoint_dir

    # fmt: on


@draccus.wrap()
def main(cfg: ConvertConfig) -> None:
    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    output_dir = Path(cfg.save_path) if cfg.save_path else Path(cfg.lora_finetuned_checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Model using HF AutoClasses
    print(f"Loading base model: {cfg.base_checkpoint}")
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.base_checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Load LoRA weights and merge into base model, then save final checkpoint
    print("Merging LoRA weights into base model...")
    start_time = time.time()
    merged_vla = PeftModel.from_pretrained(
        vla, os.path.join(cfg.lora_finetuned_checkpoint_dir, "lora_adapter")
    ).to("cuda")
    merged_vla = merged_vla.merge_and_unload()
    merged_vla.save_pretrained(output_dir)
    print(f"\nMerging complete! Time elapsed (sec): {time.time() - start_time}")

    for mod in [_configuration_prismatic_module, _modeling_prismatic_module]:
        shutil.copy2(mod.__file__, output_dir / Path(mod.__file__).name)
        print(f"Copied {Path(mod.__file__).name} to {output_dir}")

    print(f"\nSaved merged model checkpoint at:\n{output_dir}")


if __name__ == "__main__":
    main()
