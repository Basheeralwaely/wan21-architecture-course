import torch, os
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from tqdm import tqdm
import time

main_path = "/home/basheer/Signapse/Codes/stand_alone_Wan_lora_training"

# ── Configuration ──────────────────────────────────────────────────────────
num_frames = 113
height = 512
width = 512
seed = 1
cfg_scale = 1.0 # 5.0
num_inference_steps = 1  # 50
sigma_shift = 5.0
tiled = True
tile_size = (30, 52)
tile_stride = (15, 26)
lora_alpha = 1
signer = "Olivia_depth"
lora_path = f"./output/{signer}/Wan2.1-Fun-V1.1-1.3B-Control_lora/epoch-4.safetensors"

# Prompt / reference (only used when computing cache)
prompt = "Adult Woman signing, black turtleneck, purple background"
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
reference_image_path = f"./data/{signer}/reference/HAPPY@JOY.jpg"

# Control video (changes every run)
control_video_path = f"./data/test_depth/depth_pose/GUIDELINE@RULE.mp4"
output_path = f"./results/GUIDELINE@RULE.mp4"
# Cache directory
cache_dir = f"./cache/fast_inference/{signer}"

# ── Cache helpers ──────────────────────────────────────────────────────────
CACHE_FILES = {
    "context_posi": os.path.join(cache_dir, "context_posi.pt"),
    "context_nega": os.path.join(cache_dir, "context_nega.pt"),
    "clip_feature": os.path.join(cache_dir, "clip_feature.pt"),
    "reference_latents": os.path.join(cache_dir, "reference_latents.pt"),
}


def cache_exists():
    return all(os.path.exists(f) for f in CACHE_FILES.values())


def load_cache(device, dtype):
    print("Loading cached tensors (skipping T5 + CLIP computation)...")
    return {
        k: torch.load(v, map_location=device, weights_only=True).to(dtype=dtype)
        for k, v in CACHE_FILES.items()
    }


def save_cache(tensors):
    os.makedirs(cache_dir, exist_ok=True)
    for k, v in tensors.items():
        path = CACHE_FILES[k]
        torch.save(v.cpu(), path)
        print(f"  Saved {k} → {path}  {tuple(v.shape)}")


# ── Main ───────────────────────────────────────────────────────────────────
os.chdir(main_path)
t_start = time.time()
use_cache = cache_exists()

# Build model configs — skip T5 & CLIP when cache is available
model_configs = [
    ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
    ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="Wan2.1_VAE.pth"),
]
if not use_cache:
    print("Cache not found — loading T5 + CLIP to compute and save embeddings.")
    model_configs += [
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ]
else:
    print("Cache found — skipping T5 + CLIP model loading.")

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=model_configs,
)
pipe.load_lora(pipe.dit, lora_path, alpha=lora_alpha)
t_model_loaded = time.time()
print(f"Model loaded in {t_model_loaded - t_start:.2f} seconds.")

# ── Obtain prompt / reference embeddings ───────────────────────────────────

def encode_prompt(pipe, text):
    """Encode a single prompt with T5, matching WanVideoUnit_PromptEmbedder."""
    ids, mask = pipe.tokenizer(text, return_mask=True, add_special_tokens=True)
    ids, mask = ids.to(pipe.device), mask.to(pipe.device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_emb = pipe.text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        prompt_emb[:, v:] = 0
    return prompt_emb

if use_cache:
    cached = load_cache("cuda", pipe.torch_dtype)
    context_posi = cached["context_posi"]
    context_nega = cached["context_nega"]
    clip_feature = cached["clip_feature"]
    reference_latents = cached["reference_latents"]
else:
    with torch.no_grad():
        # Encode prompts with T5
        pipe.load_models_to_device(("text_encoder",))
        context_posi = encode_prompt(pipe, prompt)
        context_nega = encode_prompt(pipe, negative_prompt)

        # Encode reference image with VAE + CLIP
        ref_img = Image.open(reference_image_path).convert("RGB").resize((width, height))

        pipe.load_models_to_device(("vae", "image_encoder"))
        reference_latents = pipe.preprocess_video([ref_img])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)

        clip_input = pipe.preprocess_image(ref_img).to(pipe.device)
        clip_feature = pipe.image_encoder.encode_image([clip_input])

    # Save for next time
    save_cache({
        "context_posi": context_posi,
        "context_nega": context_nega,
        "clip_feature": clip_feature,
        "reference_latents": reference_latents,
    })
    print("Cache saved. Next run will skip T5 + CLIP entirely.")

t_cache = time.time()
print(f"Embeddings ready in {t_cache - t_model_loaded:.2f} seconds.")

# ── Prepare control video latents (changes every run) ─────────────────────
video_frames = VideoData(control_video_path, height=height, width=width)
video_frames = [video_frames[i].resize((width, height)) for i in range(num_frames)]

with torch.no_grad():
    pipe.load_models_to_device(("vae",))
    control_video_tensor = pipe.preprocess_video(video_frames)
    control_latents = pipe.vae.encode(
        control_video_tensor, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
    ).to(dtype=pipe.torch_dtype, device=pipe.device)
del video_frames, control_video_tensor

# ── Build y (control conditioning tensor) ──────────────────────────────────
# Replicate WanVideoUnit_FunControl logic:
# Since input_image is None for this pipeline, y starts as zeros.
length = (num_frames - 1) // 4 + 1
noise = pipe.generate_noise(
    (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor),
    seed=seed, rand_device="cpu",
)
latents = noise  # no input_video, so latents = noise

pipe.dit = torch.compile(pipe.dit, mode="max-autotune")   # Compile Dit for faster inference
y_dim = pipe.dit.in_dim - control_latents.shape[1] - latents.shape[1]
y = torch.zeros(
    (1, y_dim, length, height // 8, width // 8),
    dtype=pipe.torch_dtype, device=pipe.device,
)
y = torch.concat([control_latents, y], dim=1)
del control_latents

t_control = time.time()
print(f"Control latents ready in {t_control - t_cache:.2f} seconds.")

# ── Denoise ────────────────────────────────────────────────────────────────
pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=sigma_shift)

pipe.load_models_to_device(("dit",))
models = {"dit": pipe.dit}
# Fill unused model slots expected by model_fn
for name in ("motion_controller", "vace", "animate_adapter", "vap"):
    models[name] = getattr(pipe, name, None)

# Shared inputs for every denoising step
inputs_shared = {
    "latents": latents,
    "clip_feature": clip_feature.to(dtype=pipe.torch_dtype, device=pipe.device),
    "y": y,
    "reference_latents": reference_latents.to(dtype=pipe.torch_dtype, device=pipe.device),
    "cfg_merge": False,  #(set to True for faster CFG only if you keep cfg_scale > 1), may cause OOM 
    "cfg_scale": cfg_scale,
    # Unused but expected by model_fn
    "vace_context": None, "vace_scale": 1.0,
    "audio_embeds": None, "motion_latents": None, "s2v_pose_latents": None,
    "vap_hidden_state": None, "vap_clip_feature": None, "context_vap": None,
    "drop_motion_frames": True, "tea_cache": None,
    "use_unified_sequence_parallel": False, "motion_bucket_id": None,
    "pose_latents": None, "face_pixel_values": None, "longcat_latents": None,
    "sliding_window_size": None, "sliding_window_stride": None,
    "use_gradient_checkpointing": False, "use_gradient_checkpointing_offload": False,
    "control_camera_latents_input": None, "fuse_vae_embedding_in_latents": False,
}
inputs_posi = {"context": context_posi.to(dtype=pipe.torch_dtype, device=pipe.device)}
inputs_nega = {"context": context_nega.to(dtype=pipe.torch_dtype, device=pipe.device)}

with torch.no_grad():
    for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps, desc="Denoising")):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

        noise_pred_posi = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
        if cfg_scale != 1.0:
            noise_pred_nega = pipe.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi

        inputs_shared["latents"] = pipe.scheduler.step(
            noise_pred, pipe.scheduler.timesteps[progress_id], inputs_shared["latents"]
        )

t_denoise = time.time()
print(f"Denoising done in {t_denoise - t_control:.2f} seconds.")

# ── Decode & save ──────────────────────────────────────────────────────────
with torch.no_grad():
    pipe.load_models_to_device(("vae",))
    video = pipe.vae.decode(
        inputs_shared["latents"], device=pipe.device,
        tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
    )
    video = pipe.vae_output_to_video(video)
pipe.load_models_to_device([])

os.makedirs(os.path.dirname(output_path), exist_ok=True)
save_video(video, output_path, fps=30, quality=5)

t_end = time.time()
print(f"Decode + save in {t_end - t_denoise:.2f} seconds.")
print(f"Total: {t_end - t_start:.2f} seconds.")
