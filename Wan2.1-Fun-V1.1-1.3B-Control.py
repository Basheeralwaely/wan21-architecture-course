import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
# from modelscope import dataset_snapshot_download
import time

t_start = time.time()
num_frames = 17   # 81
height = 1024      # 480
width = 1024       # 832

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
)
pipe.load_lora(pipe.dit, "models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora/epoch-4.safetensors", alpha=1)
t_model_loaded = time.time()
print(f"Model loaded in {t_model_loaded - t_start:.2f} seconds.")

video = VideoData("/home/basheer/Signapse/Codes/DiffSynth-Studio/my_training/data/example_video_dataset/depth_pose/H2_0021.mp4", height=height, width=width)
video = [video[i].resize((width, height)) for i in range(num_frames)]  # Use first 17 frames, resized
# reference_image = "/home/basheer/Signapse/Codes/DiffSynth-Studio/my_training/data/example_video_dataset/reference/H2_0002.jpg" # VideoData("data/example_video_dataset/video1.mp4", height=480, width=832)[0]
reference_image = Image.open("/home/basheer/Signapse/Codes/DiffSynth-Studio/my_training/data/example_video_dataset/reference/H2_0002.jpg").convert("RGB").resize((width, height))

# Control video
video = pipe(
    prompt="Adult Woman signing, black turtleneck, green background",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    control_video=video, reference_image=reference_image,
    num_frames=num_frames,
    height=height, width=width,
    seed=1, tiled=True
)
save_video(video, "/home/basheer/Signapse/Codes/DiffSynth-Studio/outputs/trained_video_Wan2.1-Fun-V1.1-1.3B-Control.mp4", fps=15, quality=5)
t_end = time.time()
print(f"Video generated in {t_end - t_model_loaded:.2f} seconds.")
