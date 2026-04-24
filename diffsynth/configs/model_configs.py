wan_series = [
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": "diffsynth.models.wan_video_text_encoder.WanTextEncoder",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth")
        "model_hash": "ccc42284ea13e1ad04693284c7a09be6",
        "model_name": "wan_video_vae",
        "model_class": "diffsynth.models.wan_video_vae.WanVideoVAE",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_vae.WanVideoVAEStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="meituan-longcat/LongCat-Video", origin_file_pattern="dit/diffusion_pytorch_model*.safetensors")
        "model_hash": "8b27900f680d7251ce44e2dc8ae1ffef",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.longcat_video_dit.LongCatVideoTransformer3DModel",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
        "model_hash": "5941c53e207d62f20f9025686193c40b",
        "model_name": "wan_video_image_encoder",
        "model_class": "diffsynth.models.wan_video_image_encoder.WanImageEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_image_encoder.WanImageEncoderStateDictConverter"
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1", origin_file_pattern="model.safetensors")
        "model_hash": "dbd5ec76bbf977983f972c151d545389",
        "model_name": "wan_video_motion_controller",
        "model_class": "diffsynth.models.wan_video_motion_controller.WanMotionControllerModel",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "9269f8db9040a9d860eaca435be61814",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.1-Fun-1.3B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "349723183fc063b2bfc10bb2835cf677",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 48, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        # THIS IS THE PRIMARY MODEL FOR YOUR TRAINING
        "model_hash": "70ddad9d3a133785da5ea371aae09504",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 48, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06, 'has_ref_conv': True}
    },
    {
        # Example: ModelConfig(model_id="iic/VACE-Wan2.1-1.3B-Preview", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "a61453409b67cd3246cf0c3bebad47ba",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="iic/VACE-Wan2.1-1.3B-Preview", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "a61453409b67cd3246cf0c3bebad47ba",
        "model_name": "wan_video_vace",
        "model_class": "diffsynth.models.wan_video_vace.VaceWanModel",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_vace.VaceWanModelDictConverter"
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/model.safetensors")
        "model_hash": "06be60f3a4526586d8431cd038a71486",
        "model_name": "wans2v_audio_encoder",
        "model_class": "diffsynth.models.wav2vec.WanS2VAudioEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wans2v_audio_encoder.WanS2VAudioEncoderStateDictConverter",
    },
]

MODEL_CONFIGS = wan_series
