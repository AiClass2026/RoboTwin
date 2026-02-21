"""
finetune_simple.py

Simplified version of finetune.py for fine-tuning OpenVLA-OFT via LoRA.
Uses L1 regression action head, FSDP for VLA backbone, and TensorBoard for logging.
"""

import gc
import os
import random
import time
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import draccus
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ======================== Configuration ========================


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (HuggingFace Hub or local)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset
    run_root_dir: Path = Path("runs")                # Directory to store checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size

    # Architecture
    use_film: bool = False                           # Use FiLM to infuse language into visual features
    num_images_in_input: int = 1                     # Number of images in VLA input
    use_proprio: bool = False                        # Include robot proprioceptive state in input

    # Training
    batch_size: int = 8                              # Batch size per device
    learning_rate: float = 5e-4
    lr_warmup_steps: int = 0                         # Steps to warm up LR (10% -> 100%)
    num_steps_before_decay: int = 100_000            # Steps before LR decays by 10x
    grad_accumulation_steps: int = 1
    max_steps: int = 200_000
    use_val_set: bool = False                        # Use validation set
    val_freq: int = 10_000                           # Validation frequency in steps
    val_time_limit: int = 180                        # Time limit (seconds) for validation
    save_freq: int = 10_000                          # Checkpoint save frequency (-1 = no mid-training saves)
    resume: bool = False                             # Resume from checkpoint
    resume_step: Optional[int] = None                # Step number to resume from
    resume_base_model_path: Optional[str] = None     # Base model path when resuming
    image_aug: bool = True                           # Train with image augmentations

    # Parallelism & memory
    gradient_checkpointing: bool = False             # Trade compute for memory

    # LoRA
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0

    # Logging
    log_dir: Path = Path("logs")                     # TensorBoard log directory
    log_freq: int = 10                               # TensorBoard logging frequency in steps
    run_id_note: Optional[str] = None                # Extra note to append to run_id (e.g. timestamp)
    # fmt: on


# ======================== Helpers ========================


def remove_ddp_prefix(state_dict: dict) -> dict:
    """Strip 'module.' prefix from DDP-saved state dicts."""
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """Load a module checkpoint and strip DDP prefix."""
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_prefix(state_dict)


def get_run_id(cfg: FinetuneConfig) -> str:
    """Generate an experiment run ID."""
    if cfg.resume:
        run_id = cfg.vla_path.split("/")[-1]
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id = f"{cfg.run_id_note}--{run_id}"
    return run_id


def count_parameters(module: nn.Module, name: str) -> None:
    """Print trainable parameter count."""
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Random seed set to {seed}")


# ======================== Module Wrappers ========================


class IdentityWrapper(nn.Module):
    """Wraps a module with a `.module` attribute, mimicking DDP interface for single-GPU use."""
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _get_llm_transformer_layer_cls():
    """Auto-detect LLM decoder layer class for FSDP wrapping."""
    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        return LlamaDecoderLayer
    except ImportError:
        raise RuntimeError("Cannot detect LLM decoder layer class for FSDP wrapping.")


def wrap_fsdp(module: nn.Module, device_id: int) -> FSDP:
    """Wrap module with FSDP (full sharding, bf16 mixed precision)."""
    layer_cls = _get_llm_transformer_layer_cls()
    auto_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    return FSDP(
        module,
        auto_wrap_policy=auto_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device_id,
        use_orig_params=True,
        limit_all_gathers=True,
    )


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> nn.Module:
    """Wrap with DDP if distributed is initialized, otherwise use IdentityWrapper for single-GPU."""
    if dist.is_available() and dist.is_initialized():
        return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)
    else:
        print("[INFO] Distributed not initialized. Using single-GPU IdentityWrapper.")
        return IdentityWrapper(module)


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> nn.Module:
    """Initialize a module, optionally load checkpoint, move to device, and wrap with DDP."""
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)


# ======================== Forward Pass (L1 Regression Only) ========================


def move_to_device(x, device):
    """Recursively move tensors in nested dicts/lists to device."""
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    return x


def run_forward_pass(
    vla,
    action_head,
    proprio_projector,
    batch,
    device_id,
    use_proprio,
    use_film,
    num_patches,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute forward pass with L1 regression action head. Returns (loss, metrics_dict)."""
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)

    device = next(vla.parameters()).device
    batch = move_to_device(batch, device)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            use_film=use_film,
        )

    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # Extract action hidden states from LLM output
    last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
    text_hidden_states = last_hidden_states[:, num_patches:-1]
    batch_size = batch["input_ids"].shape[0]
    actions_hidden_states = (
        text_hidden_states[current_action_mask | next_actions_mask]
        .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )  # (B, act_chunk_len, D)

    # L1 regression: predict actions and compute loss
    predicted_actions = action_head.module.predict_action(actions_hidden_states)
    loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)

    # Detailed per-step L1 metrics
    curr_action_l1 = torch.nn.L1Loss()(ground_truth_actions[:, 0], predicted_actions[:, 0])
    next_actions_l1 = torch.nn.L1Loss()(ground_truth_actions[:, 1:], predicted_actions[:, 1:])

    metrics = {
        "loss": loss.item(),
        "curr_action_l1": curr_action_l1.item(),
        "next_actions_l1": next_actions_l1.item(),
    }
    return loss, metrics


# ======================== Smoothened Metrics ========================


def compute_smoothened_metrics(metrics_deques: dict) -> dict:
    """Compute averaged metrics from recent deques."""
    return {name: sum(d) / len(d) for name, d in metrics_deques.items() if len(d) > 0}


# ======================== Checkpoint Saving (FSDP) ========================


def save_training_checkpoint(
    cfg: FinetuneConfig,
    run_dir: Path,
    log_step: int,
    vla,
    processor,
    action_head,
    proprio_projector,
    train_dataset,
    distributed_state,
    optimizer,
    scheduler,
) -> None:
    """Save checkpoint: FSDP state dict for VLA, DDP state dict for small modules."""
    checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
    checkpoint_name_suffix = f"{log_step}_checkpoint.pt"
    adapter_dir = checkpoint_dir / "lora_adapter"

    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving checkpoint for step {log_step}")

    if distributed_state.num_processes > 1:
        dist.barrier()

    # Gather FSDP-sharded VLA params to rank 0
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(vla, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = vla.state_dict()

    if distributed_state.is_main_process:
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir, state_dict=cpu_state)

        if cfg.use_film:
            vb_prefix = "base_model.model.vision_backbone."
            vb_sd = {k[len(vb_prefix):]: v for k, v in cpu_state.items() if k.startswith(vb_prefix)}
            torch.save(vb_sd, checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}")

        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

    # Each rank saves its own optimizer/scheduler state (FSDP-sharded)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        checkpoint_dir / f"training_state_rank{dist.get_rank()}.pt",
    )

    del cpu_state
    gc.collect()
    torch.cuda.empty_cache()

    if distributed_state.num_processes > 1:
        dist.barrier()


# ======================== Validation ========================


def run_validation(
    vla,
    action_head,
    proprio_projector,
    val_dataloader,
    device_id,
    cfg: FinetuneConfig,
    num_patches: int,
    log_step: int,
    distributed_state,
    writer: Optional[SummaryWriter],
) -> None:
    """Run validation loop and log metrics to TensorBoard."""
    val_start_time = time.time()
    vla.eval()
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                device_id=device_id,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
            )
            all_val_metrics.append(metrics)
            if time.time() - val_start_time > cfg.val_time_limit:
                break

    # Average metrics
    avg_metrics = {}
    for key in all_val_metrics[0]:
        values = [m[key] for m in all_val_metrics if key in m]
        if values:
            avg_metrics[key] = sum(values) / len(values)

    if distributed_state.is_main_process:
        print(f"[Val step {log_step}] batches={len(all_val_metrics)}, " +
              ", ".join(f"{k}={v:.4f}" for k, v in avg_metrics.items()))
        if writer is not None:
            for k, v in avg_metrics.items():
                writer.add_scalar(f"val/{k}", v, log_step)


# ======================== Main Training Loop ========================


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """Fine-tune OpenVLA via LoRA with L1 regression action head and FSDP."""
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"

    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA `{cfg.vla_path}` on `{cfg.dataset_name}`")

    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # TensorBoard (main process only)
    writer = None
    if distributed_state.is_main_process:
        tb_log_dir = cfg.log_dir / run_id
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        print(f"TensorBoard logs: {tb_log_dir}")

    print(
        f"Constants: NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}, ACTION_DIM={ACTION_DIM}, "
        f"PROPRIO_DIM={PROPRIO_DIM}, NORM_TYPE={ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Load or register model
    if model_is_on_hf_hub(cfg.vla_path):
        cfg.vla_path = snapshot_download(repo_id=cfg.vla_path)
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)

    if distributed_state.num_processes > 1:
        dist.barrier()

    # Load processor and VLA (keep on CPU for FSDP)
    _base_path = cfg.resume_base_model_path if cfg.resume else cfg.vla_path
    processor = AutoProcessor.from_pretrained(_base_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        _base_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    )

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # LoRA setup
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)
    vla.print_trainable_parameters()

    if cfg.resume:
        assert cfg.resume_step is not None, "resume_step must be set when resume=True"
        adapter_dir = os.path.join(cfg.vla_path, "lora_adapter")
        if os.path.exists(adapter_dir):
            print(f"[Resume] Loading LoRA adapter from: {adapter_dir}")
            vla.load_adapter(adapter_dir, adapter_name="default", is_trainable=True)
            vla.set_adapter("default")
        else:
            print(f"[WARNING] lora_adapter not found at: {adapter_dir}")

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vision_backbone (original)")
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone, llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vision_backbone (post-FiLM)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)

    # Gradient checkpointing (must be before FSDP wrapping)
    if cfg.gradient_checkpointing:
        vla.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[INFO] Gradient checkpointing enabled")

    # Wrap VLA with FSDP
    vla = vla.to(dtype=torch.bfloat16)
    vla = wrap_fsdp(vla, device_id)
    print("[INFO] VLA wrapped with FSDP (FULL_SHARD)")

    # Initialize small modules (DDP-wrapped)
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector, "proprio_projector", cfg, device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    action_head = init_module(
        L1RegressionActionHead, "action_head", cfg, device_id,
        {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
        to_bf16=True,
    )

    # Compute number of vision patches
    NUM_PATCHES = (
        vla.module.vision_backbone.get_num_patches()
        * vla.module.vision_backbone.get_num_images_in_input()
    )
    if cfg.use_proprio:
        NUM_PATCHES += 1

    # Optimizer
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    trainable_params += [p for p in action_head.parameters() if p.requires_grad]
    if cfg.use_proprio:
        trainable_params += [p for p in proprio_projector.parameters() if p.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")

    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    original_lr = optimizer.param_groups[0]["lr"]

    scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)

    # Resume optimizer/scheduler state
    if cfg.resume:
        rank = dist.get_rank()
        train_state_path = Path(cfg.vla_path) / f"training_state_rank{rank}.pt"
        if train_state_path.exists():
            print(f"[Resume] Loading optimizer/scheduler from: {train_state_path}")
            ckpt = torch.load(train_state_path, map_location=f"cuda:{device_id}")
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            print(f"[Warning] {train_state_path.name} not found")
        print(f"[Resume] Resumed from step {cfg.resume_step}")

    # Dataset & DataLoader
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    use_wrist_image = cfg.num_images_in_input > 1

    batch_transform = RLDSBatchTransform(
        action_tokenizer, processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir, cfg.dataset_name, batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir, cfg.dataset_name, batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right",
    )
    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0)
    val_dataloader = None
    if cfg.use_val_set:
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0)

    # Smoothened metrics deques
    recent_metrics = {
        "loss": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # ===================== Training Loop =====================
    log_step = 0
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                device_id=device_id,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
            )

            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            for name, value in metrics.items():
                if name in recent_metrics:
                    recent_metrics[name].append(value)

            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            log_step = gradient_step_idx if not cfg.resume else (cfg.resume_step or 0) + gradient_step_idx

            # LR warmup
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

            # Optimizer step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # TensorBoard logging
            if distributed_state.is_main_process and log_step % cfg.log_freq == 0:
                smoothened = compute_smoothened_metrics(recent_metrics)
                if writer is not None:
                    for k, v in smoothened.items():
                        writer.add_scalar(f"train/{k}", v, log_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], log_step)
                progress.set_postfix({k: f"{v:.4f}" for k, v in smoothened.items()})

            # Checkpoint saving
            if cfg.save_freq > 0 and gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg, run_dir=run_dir, log_step=log_step, vla=vla,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    train_dataset=train_dataset, distributed_state=distributed_state,
                    optimizer=optimizer, scheduler=scheduler,
                )

            # Validation
            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla, action_head=action_head,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    val_dataloader=val_dataloader, device_id=device_id,
                    cfg=cfg, num_patches=NUM_PATCHES, log_step=log_step,
                    distributed_state=distributed_state, writer=writer,
                )
                vla.train()

            # Stop at max_steps
            if log_step >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break

    # Save final checkpoint
    final_step = log_step
    print(f"Saving final checkpoint at step {final_step}...")
    save_training_checkpoint(
        cfg=cfg, run_dir=run_dir, log_step=final_step, vla=vla,
        processor=processor, action_head=action_head,
        proprio_projector=proprio_projector if cfg.use_proprio else None,
        train_dataset=train_dataset, distributed_state=distributed_state,
        optimizer=optimizer, scheduler=scheduler,
    )

    if writer is not None:
        writer.close()
    print("Training complete.")


if __name__ == "__main__":
    set_seed(seed=0)
    finetune()
