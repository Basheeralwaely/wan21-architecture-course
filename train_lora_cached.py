"""
Step 2: Train LoRA with Cached Data
====================================
This script trains LoRA on pre-cached latents/embeddings.
Much faster than train_full_pipeline.py because:
- No VAE encoding (expensive!)
- No text encoding
- No CLIP encoding
- Only DiT forward/backward passes

Usage:
1. First run: python preprocess_cache.py --dataset_base_path ... --output_path ./cache
2. Then run: python train_lora_cached.py --cache_path ./cache --output_path ./lora_output
"""

import torch
import torch.nn as nn
import os
import argparse
import json
import accelerate
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import save_file

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import FlowMatchScheduler


class CachedLatentDataset(Dataset):
    """Dataset that loads pre-cached latents and embeddings."""

    def __init__(self, cache_path, repeat=1):
        self.cache_path = Path(cache_path)
        with open(self.cache_path / "cache_index.json", "r") as f:
            self.index = json.load(f)
        self.repeat = repeat

    def __len__(self):
        return len(self.index) * self.repeat

    def __getitem__(self, idx):
        actual_idx = idx % len(self.index)
        sample_info = self.index[actual_idx]
        sample_path = Path(sample_info["path"])

        # If the stored path doesn't exist, resolve relative to cache_path
        if not sample_path.exists():
            sample_path = self.cache_path / sample_path.name

        # Load cached tensors
        data = {}
        data["input_latents"] = torch.load(sample_path / "input_latents.pt", weights_only=True)
        data["context"] = torch.load(sample_path / "context.pt", weights_only=True)

        # Load optional tensors
        if (sample_path / "control_latents.pt").exists():
            data["control_latents"] = torch.load(sample_path / "control_latents.pt", weights_only=True)
        if (sample_path / "reference_latents.pt").exists():
            data["reference_latents"] = torch.load(sample_path / "reference_latents.pt", weights_only=True)
        if (sample_path / "clip_feature.pt").exists():
            data["clip_feature"] = torch.load(sample_path / "clip_feature.pt", weights_only=True)

        # Load metadata
        with open(sample_path / "metadata.json", "r") as f:
            data["metadata"] = json.load(f)

        return data


def collate_fn(batch):
    """Collate cached data into batches."""
    result = {}
    for key in batch[0].keys():
        if key == "metadata":
            result[key] = [b[key] for b in batch]
        else:
            result[key] = torch.cat([b[key] for b in batch], dim=0)
    return result


def load_dit_for_training(args, device):
    """Load only the DiT model for training."""
    model_configs = []
    for model_id_with_origin_path in args.model_id_with_origin_paths.split(","):
        model_id, origin_file_pattern = model_id_with_origin_path.split(":")
        # Only load DiT weights, skip VAE/text encoder/CLIP
        if "diffusion_pytorch_model" in origin_file_pattern:
            model_configs.append(ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern))

    # Create minimal pipeline (only DiT)
    pipe = WanVideoPipeline(device=device, torch_dtype=torch.bfloat16)
    model_pool = pipe.download_and_load_models(model_configs)
    pipe.dit = model_pool.fetch_model("wan_video_dit")

    return pipe


def setup_lora(model, target_modules, rank):
    """Inject LoRA adapters into the model."""
    target_module_list = [m.strip() for m in target_modules.split(",")]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=target_module_list,
    )

    model = inject_adapter_in_model(lora_config, model)
    return model


def get_trainable_params(model):
    """Get only LoRA parameters for training."""
    trainable_params = []
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    return trainable_params


def compute_flow_match_loss(dit, latents, noise, context, timestep, y=None, clip_feature=None, reference_latents=None,
                            use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False):
    """Compute flow matching loss for training."""
    # Interpolate between noise and latents based on timestep
    # t=0 -> pure latents, t=1 -> pure noise
    t = timestep.view(-1, 1, 1, 1, 1)
    noisy_latents = (1 - t) * latents + t * noise

    # Target is the velocity field: noise - data (matches scheduler.training_target convention)
    target = noise - latents

    # Build conditioning inputs for DiT
    # Prepare y tensor (control + reference latents)
    if y is None:
        batch_size = latents.shape[0]
        num_frames = latents.shape[2]
        height = latents.shape[3]
        width = latents.shape[4]
        y_dim = dit.in_dim - latents.shape[1]
        if reference_latents is not None:
            y_dim -= reference_latents.shape[1]
        y = torch.zeros(batch_size, y_dim, num_frames, height, width,
                       dtype=latents.dtype, device=latents.device)

    if clip_feature is None:
        clip_feature = torch.zeros((latents.shape[0], 257, 1280),
                                   dtype=latents.dtype, device=latents.device)

    # Forward pass through DiT
    pred = dit(
        noisy_latents,
        timestep=timestep * 1000,  # DiT expects timestep in [0, 1000]
        context=context,
        y=y,
        clip_feature=clip_feature,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )

    # MSE loss between prediction and target velocity (compute in float32 for stability)
    loss = torch.nn.functional.mse_loss(pred.float(), target.float())
    return loss


def prepare_y_tensor(batch, dit, device, dtype):
    """Prepare the y conditioning tensor from cached data."""
    input_latents = batch["input_latents"].to(device=device, dtype=dtype)
    batch_size = input_latents.shape[0]
    num_frames = input_latents.shape[2]
    height = input_latents.shape[3]
    width = input_latents.shape[4]

    control_latents = batch.get("control_latents")
    reference_latents = batch.get("reference_latents")

    # Calculate y dimension based on DiT's expected input
    latent_dim = input_latents.shape[1]
    total_in_dim = dit.in_dim

    y_components = []

    # Add control latents if present
    if control_latents is not None:
        control_latents = control_latents.to(device=device, dtype=dtype)
        y_components.append(control_latents)

    # Add reference latents if present (expanded to match video frames)
    if reference_latents is not None:
        reference_latents = reference_latents.to(device=device, dtype=dtype)
        # Reference latents have shape [B, C, 1, H, W], expand to [B, C, F, H, W]
        reference_latents_expanded = reference_latents.expand(-1, -1, num_frames, -1, -1)
        y_components.append(reference_latents_expanded)

    # Calculate remaining dimensions for zero padding
    used_dim = latent_dim
    if control_latents is not None:
        used_dim += control_latents.shape[1]
    if reference_latents is not None:
        used_dim += reference_latents.shape[1]

    remaining_dim = total_in_dim - used_dim
    if remaining_dim > 0:
        zeros = torch.zeros(batch_size, remaining_dim, num_frames, height, width,
                           dtype=dtype, device=device)
        y_components.append(zeros)

    if y_components:
        y = torch.cat(y_components, dim=1)
    else:
        y = None

    return y, reference_latents


def train_epoch(model, dataloader, optimizer, scheduler, accelerator, epoch, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
    for batch in pbar:
        with accelerator.accumulate(model):
            # Move data to device
            input_latents = batch["input_latents"].to(accelerator.device, dtype=torch.bfloat16)
            context = batch["context"].to(accelerator.device, dtype=torch.bfloat16)

            # Prepare conditioning
            y, reference_latents = prepare_y_tensor(batch, model, accelerator.device, torch.bfloat16)

            clip_feature = batch.get("clip_feature")
            if clip_feature is not None:
                clip_feature = clip_feature.to(accelerator.device, dtype=torch.bfloat16)

            if reference_latents is not None:
                reference_latents = reference_latents.to(accelerator.device, dtype=torch.bfloat16)

            # Sample random timestep
            batch_size = input_latents.shape[0]
            timestep = torch.rand(batch_size, device=accelerator.device, dtype=torch.bfloat16)

            # Sample noise
            noise = torch.randn_like(input_latents)

            # Compute loss
            loss = compute_flow_match_loss(
                model, input_latents, noise, context, timestep,
                y=y, clip_feature=clip_feature, reference_latents=reference_latents,
                use_gradient_checkpointing=args.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
            )

            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss/num_batches:.4f}"})

    return total_loss / num_batches


def save_lora_weights(model, output_path, prefix=""):
    """Save only the LoRA weights."""
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            save_name = name
            if prefix:
                save_name = name.replace(prefix, "", 1)
            lora_state_dict[save_name] = param.cpu()

    save_file(lora_state_dict, output_path)
    return lora_state_dict


def main():
    parser = argparse.ArgumentParser(description="Train LoRA with cached latents")
    parser.add_argument("--cache_path", type=str, required=True, help="Path to cached data from preprocess_cache.py")
    parser.add_argument("--model_id_with_origin_paths", type=str, required=True, help="Model paths (only DiT will be loaded)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save LoRA weights")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataset_repeat", type=int, default=50)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true",
                        help="Enable gradient checkpointing to reduce VRAM usage (recommended)")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true",
                        help="Offload gradient checkpointing to CPU (saves more VRAM but slower)")
    args = parser.parse_args()

    # Setup accelerator
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading DiT model...")
    pipe = load_dit_for_training(args, accelerator.device)
    dit = pipe.dit

    print("Setting up LoRA...")
    dit = setup_lora(dit, args.lora_target_modules, args.lora_rank)

    # Convert LoRA parameters to bfloat16 to match model dtype
    for name, param in dit.named_parameters():
        if "lora" in name.lower():
            param.data = param.data.to(torch.bfloat16)

    trainable_params = get_trainable_params(dit)

    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in dit.parameters())
    print(f"Trainable parameters: {num_trainable:,} / {num_total:,} ({100*num_trainable/num_total:.2f}%)")

    print(f"\nCached data directory: {args.cache_path}")
    print(f"Contents:")
    for item in sorted(Path(args.cache_path).iterdir()):
        if item.is_dir():
            sub_items = list(item.iterdir())
            print(f"  {item.name}/ ({len(sub_items)} files)")
        else:
            print(f"  {item.name}")

    print("\nLoading cached dataset...")
    dataset = CachedLatentDataset(args.cache_path, repeat=args.dataset_repeat)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Dataset size: {len(dataset)} (with {args.dataset_repeat}x repeat)")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    num_training_steps = len(dataloader) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)

    # Prepare for distributed training
    dit, optimizer, dataloader, scheduler = accelerator.prepare(dit, optimizer, dataloader, scheduler)

    print(f"\nStarting training for {args.num_epochs} epochs...")
    print(f"Total training steps: {num_training_steps}")

    for epoch in range(args.num_epochs):
        avg_loss = train_epoch(dit, dataloader, optimizer, scheduler, accelerator, epoch, args)
        print(f"Epoch {epoch+1}/{args.num_epochs} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every_n_epochs == 0:
            checkpoint_path = output_path / f"lora_epoch_{epoch+1}.safetensors"
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(dit)
                save_lora_weights(unwrapped_model, checkpoint_path, prefix=args.remove_prefix_in_ckpt)
                print(f"Saved checkpoint: {checkpoint_path}")

    # Save final weights
    if accelerator.is_main_process:
        final_path = output_path / "lora_final.safetensors"
        unwrapped_model = accelerator.unwrap_model(dit)
        save_lora_weights(unwrapped_model, final_path, prefix=args.remove_prefix_in_ckpt)
        print(f"\nTraining complete! Final weights saved to: {final_path}")


if __name__ == "__main__":
    main()
