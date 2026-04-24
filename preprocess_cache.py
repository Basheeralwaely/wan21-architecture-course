"""
Step 1: Preprocess and Cache
=============================
This script pre-computes all expensive encodings (VAE, text, CLIP) and saves them to disk.
Run this ONCE before training. Then use train_lora_cached.py for fast training iterations.

Cached data per sample:
- input_latents: VAE-encoded video (target for training)
- control_latents: VAE-encoded control video
- reference_latents: VAE-encoded reference image
- clip_feature: CLIP-encoded reference image
- context: Text-encoded prompt
- metadata: height, width, num_frames
"""

import torch
import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from PIL import Image

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, ImageCropAndResize, ToAbsolutePath, LoadImage


def create_cache_dataset(args):
    """Create dataset for preprocessing."""
    return UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=1,  # No repeat for caching - we process each sample once
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
    )


def load_models_for_caching(args, device):
    """Load only the models needed for caching (no DiT needed)."""
    model_configs = []
    for model_id_with_origin_path in args.model_id_with_origin_paths.split(","):
        model_id, origin_file_pattern = model_id_with_origin_path.split(":")
        model_configs.append(ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern))

    tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    return pipe


def preprocess_video(pipe, video_frames):
    """Preprocess video frames to tensor."""
    video_tensor = torch.stack([
        torch.tensor(frame.resize((pipe.width_division_factor * (frame.size[0] // pipe.width_division_factor),
                                   pipe.height_division_factor * (frame.size[1] // pipe.height_division_factor))).convert("RGB").getdata(),
                    dtype=torch.float32).reshape(frame.size[1], frame.size[0], 3).permute(2, 0, 1) / 255.0 * 2 - 1
        for frame in video_frames
    ], dim=1).unsqueeze(0)
    return video_tensor


@torch.no_grad()
def encode_sample(pipe, data, device, extra_inputs):
    """Encode a single sample and return cached tensors."""
    cache = {}

    # Get video frames
    video_frames = data["video"]
    height = video_frames[0].size[1]
    width = video_frames[0].size[0]
    num_frames = len(video_frames)

    # Store metadata
    cache["metadata"] = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "prompt": data["prompt"]
    }

    # 1. Encode input video (target latents)
    pipe.load_models_to_device(("vae",))
    video_tensor = pipe.preprocess_video(video_frames)
    input_latents = pipe.vae.encode(video_tensor, device=device, tiled=False)
    cache["input_latents"] = input_latents.cpu().to(torch.float16)

    # 2. Encode prompt
    pipe.load_models_to_device(("text_encoder",))
    ids, mask = pipe.tokenizer(data["prompt"], return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    context = pipe.text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        context[:, v:] = 0
    cache["context"] = context.cpu().to(torch.float16)

    # 3. Encode control video if present
    if "control_video" in extra_inputs and "control_video" in data:
        pipe.load_models_to_device(("vae",))
        control_frames = data["control_video"]
        control_tensor = pipe.preprocess_video(control_frames)
        control_latents = pipe.vae.encode(control_tensor, device=device, tiled=False)
        cache["control_latents"] = control_latents.cpu().to(torch.float16)

    # 4. Encode reference image if present
    if "reference_image" in extra_inputs and "reference_image" in data:
        ref_image = data["reference_image"]
        if isinstance(ref_image, list):
            ref_image = ref_image[0]
        ref_image = ref_image.resize((width, height))

        # VAE encode reference
        pipe.load_models_to_device(("vae",))
        ref_tensor = pipe.preprocess_video([ref_image])
        reference_latents = pipe.vae.encode(ref_tensor, device=device)
        cache["reference_latents"] = reference_latents.cpu().to(torch.float16)

        # CLIP encode reference
        if pipe.image_encoder is not None:
            pipe.load_models_to_device(("image_encoder",))
            clip_input = pipe.preprocess_image(ref_image).to(device)
            clip_feature = pipe.image_encoder.encode_image([clip_input])
            cache["clip_feature"] = clip_feature.cpu().to(torch.float16)

    return cache


def save_cache(cache, output_path, index):
    """Save cached tensors to disk."""
    sample_dir = output_path / f"sample_{index:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Save tensors
    for key, value in cache.items():
        if key == "metadata":
            with open(sample_dir / "metadata.json", "w") as f:
                json.dump(value, f, indent=2)
        else:
            torch.save(value, sample_dir / f"{key}.pt")

    return sample_dir


def main():
    parser = argparse.ArgumentParser(description="Preprocess and cache training data")
    parser.add_argument("--dataset_base_path", type=str, required=True)
    parser.add_argument("--dataset_metadata_path", type=str, required=True)
    parser.add_argument("--data_file_keys", type=str, default="video,control_video,reference_image")
    parser.add_argument("--model_id_with_origin_paths", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True, help="Path to save cached data")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--extra_inputs", type=str, default="control_video,reference_image")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    extra_inputs = args.extra_inputs.split(",") if args.extra_inputs else []

    print("Loading models for caching...")
    pipe = load_models_for_caching(args, args.device)

    print("Creating dataset...")
    dataset = create_cache_dataset(args)

    print(f"Processing {len(dataset)} samples...")
    cache_index = []

    for idx in tqdm(range(len(dataset)), desc="Caching"):
        try:
            data = dataset[idx]
            cache = encode_sample(pipe, data, args.device, extra_inputs)
            sample_dir = save_cache(cache, output_path, idx)
            cache_index.append({
                "index": idx,
                "path": str(sample_dir),
                "prompt": data["prompt"],
            })
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    # Save cache index
    with open(output_path / "cache_index.json", "w") as f:
        json.dump(cache_index, f, indent=2)

    print(f"\nCaching complete!")
    print(f"  Cached samples: {len(cache_index)}")
    print(f"  Output path: {output_path}")
    print(f"\nNext step: Run train_lora_cached.py with --cache_path {output_path}")


if __name__ == "__main__":
    main()
