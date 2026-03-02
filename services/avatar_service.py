"""LiveAvatar generation service wrapper."""
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import torch
import tempfile
import torch.distributed as dist

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liveavatar.models.wan.causal_s2v_pipeline import WanS2V
from liveavatar.models.wan.wan_2_2.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from liveavatar.models.wan.wan_2_2.utils.utils import merge_video_audio, save_video
from liveavatar.utils.args_config import parse_args_for_training_config

from config.settings import settings

logger = logging.getLogger(__name__)


class AvatarService:
    """Service wrapper for LiveAvatar generation."""
    
    def __init__(self):
        """Initialize Avatar service."""
        self.pipeline = None
        self.config = None
        self.training_config = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the LiveAvatar pipeline (async wrapper)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._initialize_sync)
    
    def _initialize_sync(self):
        """Initialize the LiveAvatar pipeline (synchronous)."""
        if self.is_initialized:
            logger.info("Avatar service already initialized")
            return
        
        try:
            logger.info("Initializing LiveAvatar pipeline...")
            
            # Set environment variables for optimization
            os.environ['ENABLE_COMPILE'] = str(settings.enable_compile).lower()
            os.environ['ENABLE_FP8'] = str(settings.enable_fp8).lower()
            os.environ['CUDNN_BENCHMARK'] = '1'
            
            # Handle Distributed setup
            is_dist = dist.is_initialized()
            world_size = dist.get_world_size() if is_dist else 1
            rank = dist.get_rank() if is_dist else 0
            
            # Set CUDA device
            device_id = rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            logger.info(f"[Rank {rank}/{world_size}] Setting device to cuda:{device_id}")
            
            # Load configuration
            self.config = WAN_CONFIGS["s2v-14B"]
            
            # Load training configuration
            training_config_path = "liveavatar/configs/s2v_causal_sft.yaml"
            self.training_config = parse_args_for_training_config(training_config_path)
            
            logger.info(f"[Rank {rank}] Creating WanS2V pipeline...")
            
            # Create pipeline
            self.pipeline = WanS2V(
                config=self.config,
                checkpoint_dir=settings.liveavatar_ckpt_dir,
                device_id=device_id,
                rank=rank,
                t5_fsdp=False,
                dit_fsdp=False,
                use_sp=False,
                sp_size=1,
                t5_cpu=False,
                convert_model_dtype=True,
                single_gpu=not is_dist,
                offload_kv_cache=False,
            )
            
            # Load LoRA weights
            logger.info(f"Loading LoRA weights from {settings.liveavatar_lora_path}...")
            self.pipeline.noise_model = self.pipeline.add_lora_to_model(
                self.pipeline.noise_model,
                lora_rank=self.training_config['lora_rank'],
                lora_alpha=self.training_config['lora_alpha'],
                lora_target_modules=self.training_config['lora_target_modules'],
                init_lora_weights=self.training_config['init_lora_weights'],
                pretrained_lora_path=settings.liveavatar_lora_path,
                load_lora_weight_only=False,
            )
            
            # Apply FP8 quantization if enabled
            if settings.enable_fp8 and hasattr(torch, "_scaled_mm"):
                logger.info("Applying FP8 quantization...")
                from liveavatar.utils.fp8_linear import replace_linear_with_scaled_fp8
                replace_linear_with_scaled_fp8(
                    self.pipeline.noise_model,
                    ignore_keys=[
                        'text_embedding', 'time_embedding',
                        'time_projection', 'head.head',
                        'casual_audio_encoder.encoder.final_linear',
                    ]
                )
            
            # Ensure VAE is on GPU
            logger.info("Moving VAE to GPU...")
            if hasattr(self.pipeline, 'vae') and self.pipeline.vae is not None:
                if hasattr(self.pipeline.vae, 'model'):
                    self.pipeline.vae.model = self.pipeline.vae.model.to('cuda')
            
            self.is_initialized = True
            logger.info("LiveAvatar pipeline initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize avatar service: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def generate_avatar_video(
        self,
        audio_path: str,
        prompt: Optional[str] = None,
        reference_image: Optional[str] = None,
        num_clips: int = 1,
        seed: int = 420
    ) -> str:
        """
        Generate avatar video from audio.
        
        Args:
            audio_path: Path to audio file
            prompt: Optional text prompt for the scene
            reference_image: Optional path to reference image
            num_clips: Number of video clips to generate
            seed: Random seed for generation
        
        Returns:
            Path to generated video file
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Use defaults if not provided
        prompt = prompt or settings.default_avatar_prompt
        reference_image = reference_image or settings.default_avatar_image
        
        logger.info(f"Generating avatar video with {num_clips} clips...")
        
        # Run generation in executor to avoid blocking
        loop = asyncio.get_event_loop()
        video_path = await loop.run_in_executor(
            None,
            self._generate_sync,
            audio_path,
            prompt,
            reference_image,
            num_clips,
            seed
        )
        
        return video_path
    
    def _generate_sync(
        self,
        audio_path: str,
        prompt: str,
        reference_image: str,
        num_clips: int,
        seed: int
    ) -> str:
        """Synchronous video generation."""
        try:
            is_dist = dist.is_initialized()
            rank = dist.get_rank() if is_dist else 0
            world_size = dist.get_world_size() if is_dist else 1
            
            # In multi-GPU TPP mode, 1 GPU is used for VAE, the rest for DiT
            num_gpus_dit = max(1, world_size - 1) if is_dist else 1
            # VAE parallel is enabled if we have dedicated DiT GPUs
            enable_vae_parallel = is_dist and world_size > 1
            # The rank that actually saves the video in TPP mode is usually rank 0 or the VAE rank
            # based on pipeline logic, Rank 0 is safe for returning the path
            
            logger.info(f"[Rank {rank}] Participating in video generation... (clips: {num_clips}, gpus_dit: {num_gpus_dit}, vae_parallel: {enable_vae_parallel})")
            
            # Broadcast parameters from Rank 0 to workers in distributed mode
            if is_dist and world_size > 1:
                if rank == 0:
                    inputs = ["infer", audio_path, prompt, reference_image, num_clips, seed]
                    logger.info("Rank 0 broadcasting inference request to workers...")
                    dist.broadcast_object_list(inputs, src=0)
            
            # Generate video (All participating ranks block here until completion)
            video, dataset_info = self.pipeline.generate(
                input_prompt=prompt,
                ref_image_path=reference_image,
                audio_path=audio_path,
                enable_tts=False,
                tts_prompt_audio=None,
                tts_prompt_text=None,
                tts_text=None,
                num_repeat=num_clips,
                pose_video=None,
                generate_size=settings.liveavatar_size,
                max_area=MAX_AREA_CONFIGS[settings.liveavatar_size],
                infer_frames=settings.liveavatar_infer_frames,
                shift=self.config.sample_shift,
                sample_solver="euler",
                sampling_steps=settings.liveavatar_sample_steps,
                guide_scale=0,
                seed=seed,
                offload_model=False,
                init_first_frame=False,
                use_dataset=False,
                dataset_sample_idx=0,
                drop_motion_noisy=False,
                num_gpus_dit=num_gpus_dit,
                enable_vae_parallel=enable_vae_parallel,
                input_video_for_sam2=None,
            )
            
            # Only rank 0 handles file saving and returning the path
            if rank == 0:
                logger.info("Video generation completed, saving on Rank 0...")
                
                # Create output path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = "output/interactive"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"avatar_{timestamp}.mp4")
                
                # Save video
                save_video(
                    tensor=video[None],
                    save_file=output_path,
                    fps=self.config.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                )
                
                # Merge with audio
                merge_video_audio(video_path=output_path, audio_path=audio_path)
                
                logger.info(f"Avatar video saved to: {output_path}")
                video_ret = output_path
            else:
                logger.info(f"[Rank {rank}] Generation logic completed. Returning empty string.")
                video_ret = ""
                
            # Clean up on all ranks
            if video is not None:
                del video
            torch.cuda.empty_cache()
            
            return video_ret
            
        except Exception as e:
            logger.error(f"Avatar generation error: {e}")
            import traceback
            traceback.print_exc()
            raise
