#!/usr/bin/env python3
import argparse
import subprocess
import sys
import math
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import shutil
import random
import collections.abc

# ============================================
# Numpy compatibility for madmom (MUST be before numpy import)
# Requires numpy >= 1.24 where these aliases are fully removed
# ============================================
import numpy as np

# Check numpy version
_np_version = tuple(map(int, np.__version__.split('.')[:2]))
if _np_version < (1, 24):
    print(f"[Warning] numpy {np.__version__} detected. Recommend numpy >= 1.24.0 to avoid FutureWarnings.")
    print("          Run: pip install 'numpy>=1.24.0'")

# For numpy >= 1.24, these attributes are removed, so we can safely add them
# For numpy < 1.24, they exist but are deprecated (will show warning)
# This is required for madmom compatibility
_NP_TYPE_ALIASES = {
    'float': np.float64,
    'int': np.int_,
    'bool': np.bool_,
    'complex': np.complex128,
    'object': np.object_,
    'str': np.str_,
    'long': np.int_,
    'unicode': np.str_,
}

for _name, _dtype in _NP_TYPE_ALIASES.items():
    if not hasattr(np, _name):
        setattr(np, _name, _dtype)

# Collections compatibility for older libraries
if not hasattr(collections, 'MutableSequence'):
    collections.MutableSequence = collections.abc.MutableSequence
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable
# ============================================

# Add BiM-VFI to path
sys.path.append(str(Path(__file__).parent / "BiM-VFI"))

import cv2
from PIL import Image
import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

# FLF2V imports
try:
    from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
    from transformers import CLIPVisionModel
    FLF2V_AVAILABLE = True
except ImportError:
    print("[Warning] FLF2V dependencies not found. Install with: pip install diffusers transformers")
    FLF2V_AVAILABLE = False

# BiM-VFI imports
try:
    from modules.components import make_components
    import torch.nn.functional as F
    BIM_VFI_AVAILABLE = True
except ImportError:
    print("[Warning] BiM-VFI modules not found. Make sure you cloned BiM-VFI correctly.")
    BIM_VFI_AVAILABLE = False

# Madmom and librosa for beat detection
try:
    import madmom
    import librosa
except ImportError as exc:
    print(f"Error importing dependencies: {exc}")
    print("Install with: pip install madmom librosa")
    sys.exit(1)

# HuggingFace Hub for authentication
try:
    from huggingface_hub import login as hf_login, HfFolder
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
# -----------------------------------------------------------


# ==========================================
# 1. Beat Detection (Madmom-based)
# ==========================================
def detect_beats_madmom(audio_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect beats using madmom's RNN-based beat tracker.
    
    Returns:
        beat_times: Array of beat times in seconds
        beat_types: Array of beat types (1=downbeat, 2=regular beat)
        beat_strengths: Array of beat strengths (0.0-1.0)
    """
    print(f"[Analysis] Extracting beats from {audio_path}...")
    
    proc = madmom.features.downbeats.RNNDownBeatProcessor()
    act = proc(audio_path)
    
    proc_beat = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    combined_act = np.max(act, axis=1)
    beat_times = proc_beat(combined_act)

    beat_types = np.full(len(beat_times), 2, dtype=np.int32)
    beat_strengths = np.zeros(len(beat_times), dtype=np.float64)
    
    if act.shape[1] >= 2:
        fps = 100.0
        for i, t in enumerate(beat_times):
            frame = int(t * fps)
            window = 2 
            start_f = max(0, frame - window)
            end_f = min(len(act), frame + window + 1)
            
            if start_f < end_f:
                local_act = act[start_f:end_f]
                max_activation = np.max(local_act)
                beat_strengths[i] = float(max_activation)
                local_beat_score = np.sum(local_act[:, 0])
                local_downbeat_score = np.sum(local_act[:, 1])
                if local_downbeat_score > local_beat_score:
                    beat_types[i] = 1 

    print(f" -> Detected {len(beat_times)} beats.")
    return beat_times, beat_types, beat_strengths


# ==========================================
# 2. Wan Model Generator (T2V + FLF2V + BiM-VFI Hybrid)
# ==========================================
class WanVideoGenerator:
    def __init__(
        self, 
        output_dir: Path, 
        model_id: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers", 
        width=832, 
        height=480,
        interpolation_mode: str = "hybrid",
        hybrid_threshold: float = 1.5,
        flf2v_model_id: str = "Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers",
        max_flf2v_count: int = -1,
        hf_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        cpu_offload_mode: str = "model",
        device_map: Optional[str] = None
    ):
        """
        Initialize the video generator.
        
        Args:
            output_dir: Directory for output files
            model_id: Wan T2V model ID (HuggingFace repo or local path, default: Wan2.2-14B)
            width: Video width
            height: Video height
            interpolation_mode: "hybrid", "flf2v", or "bim"
                - hybrid: Use FLF2V for long transitions (>= hybrid_threshold), BiM-VFI for short
                - flf2v: Always use FLF2V (Wan 2.2 First-Last-Frame to Video)
                - bim: Always use BiM-VFI (fast interpolation)
            hybrid_threshold: Duration threshold (seconds) for switching between methods in hybrid mode
            flf2v_model_id: Wan FLF2V model ID (Note: FLF2V only available in Wan2.1)
            max_flf2v_count: Maximum number of FLF2V generations allowed (-1 for unlimited)
            hf_token: HuggingFace API token for downloading gated/large models
            cache_dir: HuggingFace cache directory (use SSD for better performance)
            local_files_only: Only use locally cached models, don't download
            cpu_offload_mode: "model" (faster) or "sequential" (less VRAM) or "none"
            device_map: "auto" or "balanced" for multi-GPU, None for single GPU
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self.flf2v_model_id = flf2v_model_id
        self.width = width
        self.height = height
        self.interpolation_mode = interpolation_mode
        self.hybrid_threshold = hybrid_threshold
        self.max_flf2v_count = max_flf2v_count
        self.flf2v_usage_count = 0  # Track FLF2V usage
        self.hf_token = hf_token
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.cpu_offload_mode = cpu_offload_mode
        self.device_map = device_map
        
        # Default prompts optimized for Wan T2V models
        # Peak prompt: Energetic, visually striking for beat drops
        self.peak_prompt = "Cinematic abstract visual explosion, vibrant neon particles bursting outward, dynamic rapid camera zoom through swirling geometric shapes, electric blue and magenta energy waves pulsating rhythmically, dramatic lens flares and light rays piercing through darkness, high contrast dramatic lighting, fluid motion blur effects, professional music video aesthetic, photorealistic CGI rendering, 8K ultra HD, 60fps smooth motion, dolby vision HDR colors"
        
        # Transition prompt: Smooth morphing between scenes
        self.transition_prompt = "Seamless abstract morphing transition, fluid organic shapes slowly transforming, ethereal dreamlike atmosphere with soft gradient colors blending, gentle camera drift through luminous particles and soft bokeh lights, smooth flowing liquid metal textures reflecting ambient light, cinematic slow motion with graceful movement, professional film color grading, subtle film grain texture, 8K ultra HD, buttery smooth interpolation, photorealistic quality"
        
        self.t2v_pipe = None
        self.flf2v_pipe = None
        self.bim_model = None
        
        # Validate interpolation mode
        if interpolation_mode == "flf2v" and not FLF2V_AVAILABLE:
            raise RuntimeError("FLF2V mode requested but dependencies not available. "
                             "Install with: pip install diffusers transformers")
        if interpolation_mode == "bim" and not BIM_VFI_AVAILABLE:
            raise RuntimeError("BiM-VFI mode requested but dependencies not available. "
                             "Make sure BiM-VFI is cloned correctly.")
        if interpolation_mode == "hybrid" and not (FLF2V_AVAILABLE and BIM_VFI_AVAILABLE):
            print("[Warning] Hybrid mode requires both FLF2V and BiM-VFI. Falling back to available method.")
            if FLF2V_AVAILABLE:
                self.interpolation_mode = "flf2v"
            elif BIM_VFI_AVAILABLE:
                self.interpolation_mode = "bim"
            else:
                raise RuntimeError("No interpolation method available!")

    def _get_t2v_pipe(self):
        """Load Wan T2V pipeline for peak clip generation."""
        if self.t2v_pipe is None:
            print(f"[Wan T2V] Loading {self.model_id}...")
            
            # Check if model_id is a local path
            is_local = Path(self.model_id).exists()
            
            if is_local:
                print(f"  -> Loading from local path")
            elif self.local_files_only:
                print(f"  -> Loading from cache only (local_files_only=True)")
            else:
                print(f"  -> Loading from HuggingFace (may download if not cached)")
            
            load_kwargs = {
                "torch_dtype": torch.bfloat16
            }
            
            # Add HuggingFace options for remote models
            if not is_local:
                if self.hf_token:
                    load_kwargs["token"] = self.hf_token
                if self.cache_dir:
                    load_kwargs["cache_dir"] = self.cache_dir
                if self.local_files_only:
                    load_kwargs["local_files_only"] = True
            
            # Multi-GPU support
            if self.device_map:
                load_kwargs["device_map"] = self.device_map
                print(f"  -> Using device_map: {self.device_map}")
            
            import time
            start_time = time.time()
            
            self.t2v_pipe = WanPipeline.from_pretrained(
                self.model_id,
                **load_kwargs
            )
            
            load_time = time.time() - start_time
            print(f"  -> Model loaded in {load_time:.1f}s")
            
            # Apply CPU offload based on mode (only if not using device_map)
            if not self.device_map:
                if self.cpu_offload_mode == "sequential":
                    print(f"  -> Enabling sequential CPU offload (lowest VRAM, slower)")
                    self.t2v_pipe.enable_sequential_cpu_offload()
                elif self.cpu_offload_mode == "model":
                    print(f"  -> Enabling model CPU offload (balanced)")
                    self.t2v_pipe.enable_model_cpu_offload()
                else:
                    print(f"  -> No CPU offload, moving to CUDA")
                    self.t2v_pipe.to("cuda")
            
        return self.t2v_pipe

    def _get_flf2v_pipe(self):
        """Load Wan FLF2V (First-Last-Frame to Video) pipeline."""
        if self.flf2v_pipe is None:
            if not FLF2V_AVAILABLE:
                raise RuntimeError("FLF2V not available. Install required dependencies.")
            
            print(f"[Wan FLF2V] Loading {self.flf2v_model_id}...")
            
            # Check if model_id is a local path
            is_local = Path(self.flf2v_model_id).exists()
            
            if is_local:
                print(f"  -> Loading from local path")
            elif self.local_files_only:
                print(f"  -> Loading from cache only (local_files_only=True)")
            else:
                print(f"  -> Loading from HuggingFace (may download if not cached)")
            
            # Prepare kwargs for HuggingFace
            hf_kwargs = {}
            if not is_local:
                if self.hf_token:
                    hf_kwargs["token"] = self.hf_token
                if self.cache_dir:
                    hf_kwargs["cache_dir"] = self.cache_dir
                if self.local_files_only:
                    hf_kwargs["local_files_only"] = True
            
            import time
            start_time = time.time()
            
            # Load components with correct dtypes
            image_encoder = CLIPVisionModel.from_pretrained(
                self.flf2v_model_id, 
                subfolder="image_encoder", 
                torch_dtype=torch.float32,
                **hf_kwargs
            )
            vae = AutoencoderKLWan.from_pretrained(
                self.flf2v_model_id, 
                subfolder="vae", 
                torch_dtype=torch.float32,
                **hf_kwargs
            )
            
            # Multi-GPU support
            pipe_kwargs = {
                "vae": vae,
                "image_encoder": image_encoder,
                "torch_dtype": torch.bfloat16,
                **hf_kwargs
            }
            
            if self.device_map:
                pipe_kwargs["device_map"] = self.device_map
                print(f"  -> Using device_map: {self.device_map}")
            
            self.flf2v_pipe = WanImageToVideoPipeline.from_pretrained(
                self.flf2v_model_id,
                **pipe_kwargs
            )
            
            # Apply CPU offload based on mode (only if not using device_map)
            if not self.device_map:
                if self.cpu_offload_mode == "sequential":
                    print(f"  -> Enabling sequential CPU offload (lowest VRAM, slower)")
                    self.flf2v_pipe.enable_sequential_cpu_offload()
                elif self.cpu_offload_mode == "model":
                    print(f"  -> Enabling model CPU offload (balanced)")
                    self.flf2v_pipe.enable_model_cpu_offload()
                else:
                    print(f"  -> No CPU offload, moving to CUDA")
                    self.flf2v_pipe.to("cuda")
            
            load_time = time.time() - start_time
            print(f"  -> Model loaded in {load_time:.1f}s")
            
        return self.flf2v_pipe

    def _get_bim_model(self):
        """Load BiM-VFI interpolation model."""
        if self.bim_model is None:
            if not BIM_VFI_AVAILABLE:
                raise RuntimeError("BiM-VFI not available. Make sure it's cloned correctly.")
            
            print("[BiM-VFI] Loading Interpolation Model...")
            
            # Model Config
            model_spec = {
                'name': 'bim_vfi',
                'args': {
                    'pyr_level': 3,
                    'feat_channels': 32
                }
            }
            
            # Init Model
            self.bim_model = make_components(model_spec)
            
            # Load Checkpoint
            ckpt_path = Path(__file__).parent / "BiM-VFI/pretrained/bim_vfi.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Model not found at {ckpt_path}")
            
            # PyTorch 2.6+ requires weights_only=False for checkpoints with custom objects
            checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            self.bim_model.load_state_dict(checkpoint['model'])
            
            self.bim_model.cuda()
            self.bim_model.eval()
            
        return self.bim_model

    def _unload_t2v_pipe(self):
        """Unload T2V pipeline to free VRAM."""
        if self.t2v_pipe is not None:
            print("[Wan T2V] Unloading to free VRAM...")
            del self.t2v_pipe
            self.t2v_pipe = None
            torch.cuda.empty_cache()

    def _unload_flf2v_pipe(self):
        """Unload FLF2V pipeline to free VRAM."""
        if self.flf2v_pipe is not None:
            print("[Wan FLF2V] Unloading to free VRAM...")
            del self.flf2v_pipe
            self.flf2v_pipe = None
            torch.cuda.empty_cache()

    def _unload_bim_model(self):
        """Unload BiM-VFI model to free VRAM."""
        if self.bim_model is not None:
            print("[BiM-VFI] Unloading to free VRAM...")
            del self.bim_model
            self.bim_model = None
            torch.cuda.empty_cache()

    def extract_frame(self, video_path: Path, at_start: bool = True) -> Image.Image:
        """Extract first or last frame from a video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        if not at_start:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
            
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Try seeking a bit earlier if last frame failed
            if not at_start:
                 cap = cv2.VideoCapture(str(video_path))
                 frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                 cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 5))
                 ret, frame = cap.read()
                 cap.release()
                 if not ret:
                     raise IOError(f"Cannot read frame from {video_path}")
            else:
                 raise IOError(f"Cannot read frame from {video_path}")
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    def generate_peak_clips_batch(self, segments: List[dict], batch_size: int = 1) -> List[Path]:
        """Generate peak clips using Wan T2V."""
        if not segments:
            return []

        print(f"[Wan T2V] Generating {len(segments)} peak clips (Batch Size: {batch_size})...")
        
        pipe = self._get_t2v_pipe()
        generated_paths = []
        
        for i in range(0, len(segments), batch_size):
            batch = segments[i : i + batch_size]
            batch_indices = [s['index'] for s in batch]
            duration = max(s['duration'] for s in batch)
            
            num_frames = int(duration * 30)
            num_frames = (num_frames // 4) * 4 + 1
            
            print(f"  -> Processing Batch {i//batch_size + 1}: Indices {batch_indices} (Frames: {num_frames})")

            prompts = [self.peak_prompt] * len(batch)
            
            # Random seeds for diversity
            generators = [torch.Generator(device="cpu").manual_seed(random.randint(0, 2**32 - 1)) for _ in range(len(batch))]
            
            output = pipe(
                prompt=prompts,
                height=self.height,
                width=self.width,
                num_frames=num_frames,
                num_inference_steps=30, 
                guidance_scale=6.0,
                generator=generators
            )
            
            for j, frames in enumerate(output.frames):
                idx = batch_indices[j]
                filename = self.output_dir / f"peak_{idx:03d}.mp4"
                export_to_video(frames, str(filename), fps=30)
                generated_paths.append(filename)
                
        return generated_paths

    def generate_peak_clips_from_single_video(
        self, 
        num_clips: int, 
        clip_duration: float,
        randomize_order: bool = True
    ) -> List[Path]:
        """
        Generate peak clips by creating ONE long T2V video and slicing it.
        
        This method generates a single continuous video and then splits it into 
        num_clips segments of clip_duration each.
        
        Args:
            num_clips: Number of clips to extract from the video
            clip_duration: Duration of each clip in seconds
            randomize_order: If True, shuffle the clips for random assignment to beat peaks
        
        Returns:
            List of paths to the sliced peak clips
        """
        if num_clips <= 0:
            return []
        
        pipe = self._get_t2v_pipe()
        
        fps = 30
        # Wan models have a maximum frame limit (81-121 frames typically)
        # We generate ONE video at max capacity and slice it into clips
        MAX_FRAMES_PER_GENERATION = 121  # Conservative limit for 14B model
        
        # Calculate how many frames we can generate in one pass
        max_duration_single_pass = (MAX_FRAMES_PER_GENERATION - 1) / fps  # ~4 seconds
        
        # Calculate ideal total duration
        ideal_total_duration = num_clips * clip_duration
        
        # Determine actual generation strategy
        if ideal_total_duration <= max_duration_single_pass:
            # Can generate everything in one pass
            total_frames = int(ideal_total_duration * fps)
            total_frames = (total_frames // 4) * 4 + 1
            total_frames = max(17, total_frames)  # Minimum 17 frames for T2V
            actual_duration = ideal_total_duration
            print(f"[Wan T2V] Single pass: {total_frames} frames ({actual_duration:.2f}s) -> {num_clips} clips of {clip_duration:.2f}s")
        else:
            # Generate max frames possible, then slice into as many clips as we can
            total_frames = MAX_FRAMES_PER_GENERATION
            actual_duration = (total_frames - 1) / fps
            # Recalculate how many full clips we can get
            actual_num_clips = int(actual_duration / clip_duration)
            if actual_num_clips < num_clips:
                print(f"[Wan T2V] WARNING: Can only generate {actual_num_clips} clips of {clip_duration:.2f}s")
                print(f"          (Max single generation: {actual_duration:.2f}s, requested: {ideal_total_duration:.2f}s)")
                print(f"          Clips will be reused to fill {num_clips} slots.")
            print(f"[Wan T2V] Single pass: {total_frames} frames ({actual_duration:.2f}s) -> slicing into clips")
        
        long_video_path = self.output_dir / "peak_source_long.mp4"
        
        # Generate single video
        print(f"  -> Generating {total_frames} frames...")
        
        generator = torch.Generator(device="cpu").manual_seed(random.randint(0, 2**32 - 1))
        
        output = pipe(
            prompt=self.peak_prompt,
            height=self.height,
            width=self.width,
            num_frames=total_frames,
            num_inference_steps=30,
            guidance_scale=6.0,
            generator=generator
        )
        
        export_to_video(output.frames[0], str(long_video_path), fps=fps)
        
        # Get actual video duration
        actual_video_duration = (total_frames - 1) / fps
        
        # Calculate how many full clips we can extract
        max_possible_clips = int(actual_video_duration / clip_duration)
        clips_to_create = min(max_possible_clips, num_clips)
        
        if clips_to_create == 0:
            # Video too short for even one clip, use the whole video as one clip
            print(f"  -> Video too short ({actual_video_duration:.2f}s) for {clip_duration:.2f}s clips")
            print(f"  -> Using entire video as single clip")
            clips_to_create = 1
            effective_clip_duration = actual_video_duration
        else:
            effective_clip_duration = clip_duration
        
        # Now slice the long video into clips
        print(f"  -> Slicing into {clips_to_create} clips of {effective_clip_duration:.2f}s each...")
        
        sliced_paths = []
        for i in range(clips_to_create):
            start_time = i * effective_clip_duration
            output_clip_path = self.output_dir / f"peak_{i:03d}.mp4"
            
            # Use FFmpeg to extract segment
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", str(long_video_path),
                "-t", str(effective_clip_duration),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-an",  # No audio
                str(output_clip_path)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if output_clip_path.exists():
                sliced_paths.append(output_clip_path)
                print(f"    -> Created peak_{i:03d}.mp4 (start: {start_time:.2f}s)")
            else:
                print(f"    [Warning] Failed to create peak_{i:03d}.mp4")
        
        # If we created fewer clips than requested, duplicate to fill the pool
        if len(sliced_paths) < num_clips and len(sliced_paths) > 0:
            print(f"  -> Created {len(sliced_paths)} clips, but {num_clips} requested.")
            print(f"  -> Clips will be reused when assigned to peaks.")
        
        # Optionally randomize the order for more variety when assigning to beats
        if randomize_order and len(sliced_paths) > 1:
            random.shuffle(sliced_paths)
            print(f"  -> Shuffled clip order for random beat assignment")
        
        # Keep the source video for reference (optional: delete if not needed)
        print(f"  -> Source video saved: {long_video_path}")
        
        return sliced_paths

    def generate_transition_clips_batch(self, tasks: List[dict], batch_size: int = 1) -> Dict[str, Path]:
        """
        Generates transition clips using hybrid interpolation.
        
        In hybrid mode:
        - Long transitions (>= hybrid_threshold): Use FLF2V for better quality
        - Short transitions (< hybrid_threshold): Use BiM-VFI for speed
        """
        if not tasks: 
            return {}
        
        mode = self.interpolation_mode
        print(f"[Interpolation] Generating {len(tasks)} transition clips (Mode: {mode})...")
        
        results = {}
        
        # Separate tasks by method in hybrid mode
        if mode == "hybrid":
            flf2v_tasks = []
            bim_tasks = []
            
            for t in tasks:
                if t['duration'] >= self.hybrid_threshold:
                    flf2v_tasks.append(t)
                else:
                    bim_tasks.append(t)
            
            print(f"  -> FLF2V tasks (>= {self.hybrid_threshold}s): {len(flf2v_tasks)}")
            print(f"  -> BiM-VFI tasks (< {self.hybrid_threshold}s): {len(bim_tasks)}")
            
            # Process FLF2V tasks first (usually fewer but higher quality needed)
            if flf2v_tasks:
                flf2v_results = self._generate_transitions_flf2v(flf2v_tasks)
                results.update(flf2v_results)
            
            # Then process BiM-VFI tasks
            if bim_tasks:
                bim_results = self._generate_transitions_bim(bim_tasks)
                results.update(bim_results)
                
        elif mode == "flf2v":
            results = self._generate_transitions_flf2v(tasks)
        else:  # bim
            results = self._generate_transitions_bim(tasks)
                
        return results

    def _generate_transitions_flf2v(self, tasks: List[dict]) -> Dict[str, Path]:
        """Generate transitions using Wan FLF2V (First-Last-Frame to Video)."""
        if not tasks:
            return {}
        
        print(f"[Wan FLF2V] Generating {len(tasks)} transition clips...")
        if self.max_flf2v_count >= 0:
            remaining = self.max_flf2v_count - self.flf2v_usage_count
            print(f"  -> FLF2V limit: {self.max_flf2v_count}, Used: {self.flf2v_usage_count}, Remaining: {remaining}")
        
        # Unload T2V pipe if loaded to free VRAM for FLF2V
        self._unload_t2v_pipe()
        
        results = {}
        
        for i, t in enumerate(tasks):
            # Check if we've hit the FLF2V limit
            if self.max_flf2v_count >= 0 and self.flf2v_usage_count >= self.max_flf2v_count:
                print(f"  -> [Limit Reached] FLF2V count ({self.flf2v_usage_count}) reached max ({self.max_flf2v_count})")
                print(f"  -> Falling back to BiM-VFI for remaining {len(tasks) - i} transitions...")
                
                # Unload FLF2V before using BiM-VFI
                self._unload_flf2v_pipe()
                
                # Process remaining tasks with BiM-VFI
                remaining_tasks = tasks[i:]
                bim_results = self._generate_transitions_bim(remaining_tasks)
                results.update(bim_results)
                break
            
            print(f"  -> Processing FLF2V Transition {i + 1}/{len(tasks)} (Duration: {t['duration']:.2f}s)")
            
            fname = self.output_dir / f"transition_{t['index']}.mp4"
            
            # Check if already exists
            if fname.exists():
                results[t['index']] = fname
                continue
            
            try:
                # Load FLF2V pipe (lazy loading)
                pipe = self._get_flf2v_pipe()
                
                path = self._generate_single_flf2v_transition(
                    t['index'], t['start_img'], t['end_img'], t['duration']
                )
                results[t['index']] = path
                self.flf2v_usage_count += 1  # Increment usage counter
            except torch.cuda.OutOfMemoryError as e:
                print(f"    [Error] FLF2V CUDA OOM: {str(e)[:100]}...")
                print(f"    [Fallback] Unloading FLF2V and using BiM-VFI...")
                
                # CRITICAL: Unload FLF2V to free VRAM before fallback
                self._unload_flf2v_pipe()
                torch.cuda.empty_cache()
                
                # Fallback to BiM-VFI
                path = self._generate_single_bim_transition(
                    t['index'], t['start_img'], t['end_img'], t['duration']
                )
                results[t['index']] = path
            except Exception as e:
                print(f"    [Error] FLF2V failed: {e}")
                print(f"    [Fallback] Unloading FLF2V and using BiM-VFI...")
                
                # CRITICAL: Unload FLF2V to free VRAM before fallback
                self._unload_flf2v_pipe()
                torch.cuda.empty_cache()
                
                # Fallback to BiM-VFI
                path = self._generate_single_bim_transition(
                    t['index'], t['start_img'], t['end_img'], t['duration']
                )
                results[t['index']] = path
        
        return results

    def _generate_single_flf2v_transition(
        self, 
        index: str, 
        start_img: Image.Image, 
        end_img: Image.Image, 
        duration: float
    ) -> Path:
        """Generate a single transition using Wan FLF2V."""
        filename = self.output_dir / f"transition_{index}.mp4"
        if filename.exists():
            return filename
        
        print(f"[Wan FLF2V] Generating Transition {index} (Duration: {duration:.2f}s)...")
        
        pipe = self._get_flf2v_pipe()
        
        # FLF2V specific dimensions (720p model)
        flf2v_width = 1280
        flf2v_height = 720
        
        # Resize images for FLF2V
        start_img_resized = start_img.resize((flf2v_width, flf2v_height), Image.Resampling.LANCZOS)
        end_img_resized = end_img.resize((flf2v_width, flf2v_height), Image.Resampling.LANCZOS)
        
        # Ensure RGB mode
        if start_img_resized.mode != 'RGB':
            start_img_resized = start_img_resized.convert('RGB')
        if end_img_resized.mode != 'RGB':
            end_img_resized = end_img_resized.convert('RGB')
        
        # Calculate frames (FLF2V generates at ~24fps typically)
        fps = 24
        num_frames = int(duration * fps)
        # FLF2V requires specific frame counts (multiples of 4 + 1)
        num_frames = max(17, (num_frames // 4) * 4 + 1)  # Minimum 17 frames
        num_frames = min(num_frames, 81)  # Maximum 81 frames for 14B model
        
        print(f"    -> Generating {num_frames} frames at {fps}fps...")
        
        # Generate with FLF2V
        generator = torch.Generator(device="cuda").manual_seed(random.randint(0, 2**32 - 1))
        
        output = pipe(
            image=start_img_resized,
            last_image=end_img_resized,
            prompt=self.transition_prompt,
            negative_prompt="blurry, distorted, low quality, artifacts",
            height=flf2v_height,
            width=flf2v_width,
            num_frames=num_frames,
            num_inference_steps=30,
            guidance_scale=5.0,
            generator=generator
        )
        
        # Export video
        temp_filename = self.output_dir / f"transition_{index}_temp.mp4"
        export_to_video(output.frames[0], str(temp_filename), fps=fps)
        
        # Resize back to target dimensions and adjust duration
        self._resize_and_adjust_video(temp_filename, filename, self.width, self.height, duration)
        
        # Cleanup temp file
        if temp_filename.exists():
            temp_filename.unlink()
        
        return filename

    def _generate_transitions_bim(self, tasks: List[dict]) -> Dict[str, Path]:
        """Generate transitions using BiM-VFI."""
        if not tasks:
            return {}
        
        print(f"[BiM-VFI] Generating {len(tasks)} transition clips...")
        
        results = {}
        
        for i, t in enumerate(tasks):
            print(f"  -> Processing BiM-VFI Transition {i + 1}/{len(tasks)} (Duration: {t['duration']:.2f}s)")
            
            fname = self.output_dir / f"transition_{t['index']}.mp4"
            
            if fname.exists():
                results[t['index']] = fname
            else:
                path = self._generate_single_bim_transition(
                    t['index'], t['start_img'], t['end_img'], t['duration']
                )
                results[t['index']] = path
                
        return results

    def _generate_single_bim_transition(
        self, 
        index: str, 
        start_img: Image.Image, 
        end_img: Image.Image, 
        duration: float
    ) -> Path:
        """Generate a single transition using BiM-VFI interpolation."""
        filename = self.output_dir / f"transition_{index}.mp4"
        if filename.exists(): 
            return filename

        print(f"[BiM-VFI] Generating Transition {index} (Duration: {duration:.2f}s)...")
        
        model = self._get_bim_model()
        
        start_img = start_img.resize((self.width, self.height))
        end_img = end_img.resize((self.width, self.height))
        
        # Ensure images are in RGB mode
        if start_img.mode != 'RGB':
            start_img = start_img.convert('RGB')
        if end_img.mode != 'RGB':
            end_img = end_img.convert('RGB')
            
        img0_np = np.array(start_img)
        img1_np = np.array(end_img)
        
        img0 = (torch.tensor(img0_np.transpose(2, 0, 1).copy()).float() / 255.0).unsqueeze(0).cuda()
        img1 = (torch.tensor(img1_np.transpose(2, 0, 1).copy()).float() / 255.0).unsqueeze(0).cuda()
        
        fps = 30
        num_frames = int(duration * fps)
        if num_frames < 2: 
            num_frames = 2  # Need at least 2 frames for interpolation
        
        frames = []
        
        with torch.no_grad():
            h, w = img0.shape[2], img0.shape[3]
            if h >= 2160: scale_factor = 0.25; pyr_level = 7
            elif h >= 1080: scale_factor = 0.5; pyr_level = 6
            else: scale_factor = 1; pyr_level = 5
            
            for frame_idx in range(num_frames):
                # Avoid t=0 (identical to start_img) by using a small offset
                t = (frame_idx + 0.5) / float(num_frames)
                
                time_step = torch.tensor([t]).view(1, 1, 1, 1).cuda()
                dis0 = torch.ones((1, 1, h, w), device=img0.device) * t
                dis1 = 1 - dis0
                
                results_dict = model(
                    img0=img0, img1=img1, 
                    time_step=time_step, 
                    dis0=dis0, dis1=dis1, 
                    scale_factor=scale_factor,
                    ratio=1.0, 
                    pyr_level=pyr_level, 
                    nr_lvl_skipped=0
                )
                
                pred = results_dict['imgt_pred']
                pred = torch.clip(pred, 0, 1)
                
                frame_np = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                frames.append(frame_np)
        
        out = cv2.VideoWriter(str(filename), cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.width, self.height))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        
        return filename

    def _resize_and_adjust_video(
        self, 
        input_path: Path, 
        output_path: Path, 
        target_width: int, 
        target_height: int, 
        target_duration: float
    ):
        """Resize video and adjust duration using FFmpeg."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", f"scale={target_width}:{target_height}",
            "-t", str(target_duration),
            "-r", "30",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"    [Warning] FFmpeg resize failed: {result.stderr.decode()[:200]}")

    def generate_transition_clip(
        self, 
        index: str, 
        start_img: Image.Image, 
        end_img: Image.Image, 
        duration: float
    ) -> Path:
        """
        Generate a single transition clip using the configured interpolation method.
        """
        mode = self.interpolation_mode
        
        # Check FLF2V limit
        flf2v_available = True
        if self.max_flf2v_count >= 0 and self.flf2v_usage_count >= self.max_flf2v_count:
            flf2v_available = False
            if mode in ["hybrid", "flf2v"]:
                print(f"  -> [Limit Reached] FLF2V count ({self.flf2v_usage_count}) reached max ({self.max_flf2v_count})")
                print(f"  -> Using BiM-VFI instead")
        
        if mode == "hybrid":
            # In hybrid mode, choose based on duration (and FLF2V availability)
            if duration >= self.hybrid_threshold and flf2v_available:
                path = self._generate_single_flf2v_transition(index, start_img, end_img, duration)
                self.flf2v_usage_count += 1
                return path
            else:
                return self._generate_single_bim_transition(index, start_img, end_img, duration)
        elif mode == "flf2v":
            if flf2v_available:
                path = self._generate_single_flf2v_transition(index, start_img, end_img, duration)
                self.flf2v_usage_count += 1
                return path
            else:
                return self._generate_single_bim_transition(index, start_img, end_img, duration)
        else:  # bim
            return self._generate_single_bim_transition(index, start_img, end_img, duration)


# ==========================================
# 3. Timeline Scheduler & Assembly
# ==========================================
class SceneScheduler:
    def __init__(self, duration: float, beat_times: np.ndarray, beat_strengths: np.ndarray):
        self.total_duration = duration
        self.beat_times = beat_times
        self.beat_strengths = beat_strengths
        self.timeline = [] 

    def plan_scenes(self, peak_threshold: float = 0.7, peak_clip_len: float = 2.0):
        print(f"[Planner] Scheduling scenes (Threshold: {peak_threshold}, Base Peak Len: {peak_clip_len}s)...")
        print(f"  -> Total audio duration: {self.total_duration:.2f}s")
        print(f"  -> Beat times: {self.beat_times[:10]}..." if len(self.beat_times) > 10 else f"  -> Beat times: {self.beat_times}")
        print(f"  -> Beat strengths: {self.beat_strengths[:10]}..." if len(self.beat_strengths) > 10 else f"  -> Beat strengths: {self.beat_strengths}")
        
        # 1. Identify Peak Intervals
        peaks = []
        min_transition_len = 0.1  # Minimum gap between peaks (reduced for BiM-VFI which is fast)
        
        for t, strength in zip(self.beat_times, self.beat_strengths):
            if strength >= peak_threshold:
                start_t = t
                end_t = t + peak_clip_len
                
                # Check if we have enough space from previous peak
                if peaks:
                    prev_peak = peaks[-1]
                    gap_avail = start_t - prev_peak['end']
                    
                    if gap_avail < min_transition_len:
                        # Previous peak ends too late. Shorten it.
                        new_prev_end = start_t - min_transition_len
                        prev_dur = new_prev_end - prev_peak['start']
                        
                        # Only update if duration stays positive
                        if prev_dur > 0.05:
                            prev_peak['end'] = new_prev_end
                            prev_peak['duration'] = prev_dur
                        else:
                            # Skip this beat if it would make previous peak too short
                            continue
                
                if start_t >= self.total_duration: 
                    continue
                    
                end_t = min(end_t, self.total_duration)
                duration = end_t - start_t
                
                # Only add if duration is positive
                if duration > 0.05:
                    peaks.append({
                        'type': 'peak',
                        'start': start_t,
                        'end': end_t,
                        'duration': duration
                    })

        print(f"  -> Found {len(peaks)} valid peaks")

        # 2. Build Full Timeline
        self.timeline = []
        current_time = 0.0
        
        for p in peaks:
            # Gap before this peak
            gap_duration = p['start'] - current_time
            if gap_duration > 0.01:
                self.timeline.append({
                    'type': 'gap',
                    'start': current_time,
                    'end': p['start'],
                    'duration': gap_duration
                })
            
            # Peak itself
            self.timeline.append(p)
            current_time = p['end']
            
        # Final Gap (from last peak to end of audio)
        final_gap = self.total_duration - current_time
        if final_gap > 0.01:
            self.timeline.append({
                'type': 'gap',
                'start': current_time,
                'end': self.total_duration,
                'duration': final_gap
            })

        # Debug: Print timeline
        total_timeline_dur = sum(item['duration'] for item in self.timeline)
        print(f" -> Scheduled {len(self.timeline)} segments ({len(peaks)} peaks).")
        print(f" -> Timeline total duration: {total_timeline_dur:.2f}s (expected: {self.total_duration:.2f}s)")
        
        for i, item in enumerate(self.timeline):
            print(f"    [{i}] {item['type']}: {item['start']:.2f}s - {item['end']:.2f}s (dur: {item['duration']:.2f}s)")
        
        return self.timeline


def assemble_video(timeline: List[dict], output_path: Path, fps=30):
    """
    Assembles the clips using ffmpeg. 
    Simply concatenates all clips - each clip should already have the correct duration.
    """
    print(f"[Assembly] Processing segments and stitching...")
    print(f"  -> Total segments: {len(timeline)}")
    
    total_expected_duration = sum(item['duration'] for item in timeline)
    print(f"  -> Expected total duration: {total_expected_duration:.2f}s")
    
    temp_dir = output_path.parent / "segments_processed"
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    
    for i, item in enumerate(timeline):
        src_file = item['file']
        out_file = temp_dir / f"seg_{i:04d}.mp4"
        target_dur = item['duration']
        
        print(f"  -> Segment {i}: {item['type']} | target={target_dur:.2f}s | file={src_file.name}")
        
        # Skip segments with invalid duration
        if target_dur <= 0:
            print(f"    [Warning] Skipping segment {i} with duration {target_dur}")
            continue
        
        # 1. Probe Source Duration
        try:
            probe = subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", str(src_file)
            ])
            src_dur = float(probe.strip())
        except Exception as e:
            print(f"    [Warning] Probe failed: {e}. Using target duration.")
            src_dur = target_dur
        
        # Check for trim parameters (used for loop transition splitting)
        trim_start = item.get('trim_start', 0.0)
        trim_duration = item.get('trim_duration', None)
        
        print(f"    Source duration: {src_dur:.2f}s, trim_start: {trim_start:.2f}s, trim_duration: {trim_duration}")
        
        # 2. Build FFmpeg command
        # If trim parameters are set, we're extracting a specific portion of the source
        if trim_duration is not None:
            # Extract specific portion from source (for loop transition)
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(trim_start),
                "-i", str(src_file),
                "-t", str(trim_duration),
                "-r", str(fps),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(out_file)
            ]
        elif src_dur < target_dur * 0.9:
            # Source is significantly shorter than target - slow it down
            speed_factor = src_dur / target_dur
            filter_str = f"setpts={1/speed_factor:.6f}*PTS"
            cmd = [
                "ffmpeg", "-y",
                "-i", str(src_file),
                "-filter:v", filter_str,
                "-t", str(target_dur),
                "-r", str(fps),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(out_file)
            ]
        elif src_dur > target_dur * 1.1:
            # Source is significantly longer than target - speed it up
            speed_factor = src_dur / target_dur
            filter_str = f"setpts={1/speed_factor:.6f}*PTS"
            cmd = [
                "ffmpeg", "-y",
                "-i", str(src_file),
                "-filter:v", filter_str,
                "-t", str(target_dur),
                "-r", str(fps),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(out_file)
            ]
        else:
            # Source duration is close to target - just trim/pad
            cmd = [
                "ffmpeg", "-y",
                "-i", str(src_file),
                "-t", str(target_dur),
                "-r", str(fps),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(out_file)
            ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"    [Error] FFmpeg failed: {result.stderr.decode()[:200]}")
        
        # Verify output exists and has content
        if out_file.exists() and out_file.stat().st_size > 0:
            processed_files.append(out_file)
        else:
            print(f"    [Error] Output file not created or empty!")

    if not processed_files:
        print("[Error] No segments were processed!")
        return output_path.with_name("temp_no_audio.mp4")

    # Create file list for concatenation
    list_file = temp_dir / "files.txt"
    with open(list_file, "w") as f:
        for p in processed_files:
            f.write(f"file '{p.resolve()}'\n")

    temp_video = output_path.with_name("temp_no_audio.mp4")
    
    # Concatenate all segments
    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(temp_video)
    ]
    result = subprocess.run(cmd_concat, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"[Error] Concat failed: {result.stderr.decode()[:200]}")
    
    # Verify final output
    if temp_video.exists():
        try:
            probe = subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(temp_video)
            ])
            final_dur = float(probe.strip())
            print(f"  -> Final video duration: {final_dur:.2f}s")
        except:
            pass
    
    return temp_video


def mux_audio(video_path: Path, audio_path: Path, output_path: Path):
    print(f"[Muxing] Combining audio and video...")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✓ Final video saved: {output_path}")


def trim_audio(input_path: Path, output_path: Path, start: float, end: Optional[float] = None):
    """
    Trim audio file to specified time range.
    
    Args:
        input_path: Source audio file
        output_path: Output trimmed audio file
        start: Start time in seconds
        end: End time in seconds (None = until end of file)
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ss", str(start),
    ]
    if end is not None:
        duration = end - start
        cmd.extend(["-t", str(duration)])
        end_str = f"{end:.1f}s"
        dur_str = f"{duration:.1f}s"
    else:
        end_str = "end"
        dur_str = "remaining"
    
    cmd.extend(["-c", "copy", str(output_path)])
    
    print(f"[Audio] Trimming: {start:.1f}s ~ {end_str} (duration: {dur_str})")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print(f"[Warning] Audio trim may have issues: {result.stderr.decode()[:100]}")
    
    return output_path


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Beat-Reactive Video Generator (Wan T2V + Hybrid Interpolation: FLF2V + BiM-VFI)"
    )
    parser.add_argument("--audio", required=True, type=Path, help="Input audio file")
    parser.add_argument("--output", type=Path, default=Path("final_output.mp4"), help="Output video file")
    parser.add_argument("--peak-thresh", type=float, default=0.5, help="0.0-1.0, threshold for Peak detection")
    parser.add_argument("--peak-len", type=float, default=2.0, help="Duration of peak clips (seconds)")
    parser.add_argument("--num-peak-clips", type=int, default=5, help="Number of unique peak clips to generate and reuse")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for T2V generation")
    parser.add_argument("--model", type=str, default="wan2.2-14b", 
                       choices=["wan2.1-1.3b", "wan2.2-5b", "wan2.2-14b"], 
                       help="Model to use for T2V generation (default: wan2.2-14b)")
    parser.add_argument("--peak-prompt", type=str, 
                       default=None,
                       help="Prompt for beat peak video generation. If not set, uses optimized default.")
    parser.add_argument("--prompt-style", type=str, default="energetic",
                       choices=["energetic", "nature", "abstract", "cyberpunk", "elegant", "cosmic"],
                       help="Preset prompt style (overridden by --peak-prompt/--transition-prompt if set)")
    parser.add_argument("--audio-start", type=float, default=0.0, 
                       help="Start time in seconds. Only process audio from this point. (default: 0.0)")
    parser.add_argument("--audio-end", type=float, default=None, 
                       help="End time in seconds. Only process audio until this point. (default: end of audio)")
    parser.add_argument("--work-dir", type=Path, default=Path("workspace_wan"), 
                       help="Directory to save intermediate and final results")
    
    # New interpolation arguments
    parser.add_argument("--interpolation-mode", type=str, default="hybrid",
                       choices=["hybrid", "flf2v", "bim"],
                       help="Interpolation mode: hybrid (FLF2V for long, BiM-VFI for short), "
                            "flf2v (always use Wan FLF2V), bim (always use BiM-VFI)")
    parser.add_argument("--hybrid-threshold", type=float, default=1.5,
                       help="Duration threshold (seconds) for hybrid mode. "
                            "Transitions >= threshold use FLF2V, shorter use BiM-VFI")
    parser.add_argument("--flf2v-model", type=str, default="Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers",
                       help="Wan FLF2V model ID (Note: FLF2V is only available in Wan2.1)")
    parser.add_argument("--transition-prompt", type=str,
                       default=None,
                       help="Prompt for FLF2V transition generation. If not set, uses optimized default.")
    parser.add_argument("--max-flf2v-count", type=int, default=-1,
                       help="Maximum number of FLF2V generations allowed. "
                            "-1 for unlimited. When limit is reached, falls back to BiM-VFI.")
    
    # HuggingFace and model path arguments
    parser.add_argument("--hf-token", type=str, default=None,
                       help="HuggingFace API token for downloading gated/large models. "
                            "Can also be set via HF_TOKEN environment variable.")
    parser.add_argument("--t2v-model-path", type=str, default=None,
                       help="Local path to T2V model (overrides --model if set)")
    parser.add_argument("--flf2v-model-path", type=str, default=None,
                       help="Local path to FLF2V model (overrides --flf2v-model if set)")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="HuggingFace cache directory (default: ~/.cache/huggingface). "
                            "Use fast SSD path for better performance.")
    parser.add_argument("--local-files-only", action="store_true",
                       help="Only use locally cached models, don't download from HuggingFace")
    parser.add_argument("--cpu-offload", type=str, default="model",
                       choices=["none", "model", "sequential"],
                       help="CPU offload mode: none (all on GPU), model (balanced, default), "
                            "sequential (lowest VRAM but slower)")
    parser.add_argument("--device-map", type=str, default=None,
                       choices=["balanced"],
                       help="Multi-GPU device mapping: 'balanced' distributes model across GPUs. "
                            "Use with multiple GPUs (e.g., CUDA_VISIBLE_DEVICES=0,1)")
    
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"File not found: {args.audio}")
        return

    # 0. Setup
    work_dir = args.work_dir.resolve()  # Get absolute path
    work_dir.mkdir(parents=True, exist_ok=True)

    # Update output path if relative and just a filename
    if not args.output.is_absolute() and len(args.output.parts) == 1:
        args.output = work_dir / args.output
    args.output = args.output.resolve()  # Get absolute path

    # Handle HuggingFace token
    import os
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", None)
    
    # Display GPU information
    print(f"\n[GPU Settings]")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"  Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"    GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        print(f"  CPU Offload Mode: {args.cpu_offload}")
        print(f"  Device Map: {args.device_map or 'None (single GPU)'}")
        if num_gpus > 1 and not args.device_map:
            print(f"  [Tip] Multiple GPUs detected. Use --device-map auto for multi-GPU inference")
    else:
        print(f"  No CUDA GPUs available!")
    
    # Display cache settings
    default_cache = Path.home() / ".cache" / "huggingface"
    cache_dir = args.cache_dir or os.environ.get("HF_HOME", str(default_cache))
    print(f"\n[HuggingFace Settings]")
    print(f"  Cache Directory: {cache_dir}")
    print(f"  Local Files Only: {args.local_files_only}")
    
    if hf_token and HF_HUB_AVAILABLE:
        print(f"  Token: Provided")
        try:
            hf_login(token=hf_token, add_to_git_credential=False)
        except Exception as e:
            print(f"  [Warning] HuggingFace login failed: {e}")
    else:
        print(f"  Token: Not provided")
        if not args.local_files_only and not args.t2v_model_path:
            print(f"  [Tip] For faster loading, use --local-files-only if models are already cached")
            print(f"        Or use --t2v-model-path / --flf2v-model-path for local models")

    # Determine model paths (local path takes priority)
    model_map = {
        "wan2.1-1.3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "wan2.2-5b": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "wan2.2-14b": "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    }
    
    if args.t2v_model_path:
        selected_model_id = args.t2v_model_path
        print(f"[Model] Using local T2V model: {selected_model_id}")
    else:
        selected_model_id = model_map[args.model]
        print(f"[Model] Using HuggingFace T2V model: {selected_model_id}")
    
    if args.flf2v_model_path:
        flf2v_model_id = args.flf2v_model_path
        print(f"[Model] Using local FLF2V model: {flf2v_model_id}")
    else:
        flf2v_model_id = args.flf2v_model
        print(f"[Model] Using HuggingFace FLF2V model: {flf2v_model_id}")

    # Print path information
    print(f"\n{'='*60}")
    print(f"[Path Information]")
    print(f"  Working Directory: {work_dir}")
    print(f"  Peak Clips:        {work_dir}/peak_*.mp4")
    print(f"  Transitions:       {work_dir}/transition_*.mp4")
    print(f"  Final Output:      {args.output}")
    print(f"{'='*60}\n")

    # Initialize generator with hybrid interpolation
    generator = WanVideoGenerator(
        output_dir=work_dir, 
        model_id=selected_model_id,
        interpolation_mode=args.interpolation_mode,
        hybrid_threshold=args.hybrid_threshold,
        flf2v_model_id=flf2v_model_id,
        max_flf2v_count=args.max_flf2v_count,
        hf_token=hf_token,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        cpu_offload_mode=args.cpu_offload,
        device_map=args.device_map
    )
    
    # Prompt presets optimized for Wan T2V models
    PROMPT_PRESETS = {
        "energetic": {
            "peak": (
                "Cinematic abstract visual explosion, vibrant neon particles bursting outward, "
                "dynamic rapid camera zoom through swirling geometric shapes, "
                "electric blue and magenta energy waves pulsating rhythmically, "
                "dramatic lens flares and light rays piercing through darkness, "
                "high contrast dramatic lighting, fluid motion blur effects, "
                "professional music video aesthetic, photorealistic CGI rendering, "
                "8K ultra HD, 60fps smooth motion, dolby vision HDR colors"
            ),
            "transition": (
                "Seamless abstract morphing transition, fluid organic shapes slowly transforming, "
                "ethereal dreamlike atmosphere with soft gradient colors blending, "
                "gentle camera drift through luminous particles and soft bokeh lights, "
                "smooth flowing liquid metal textures reflecting ambient light, "
                "cinematic slow motion with graceful movement, "
                "professional film color grading, subtle film grain texture, "
                "8K ultra HD, buttery smooth interpolation, photorealistic quality"
            )
        },
        "nature": {
            "peak": (
                "Breathtaking nature scene with dramatic weather, "
                "powerful ocean waves crashing against rocky cliffs in golden hour light, "
                "dynamic aerial drone shot sweeping over misty mountain peaks, "
                "lush green forests with sunbeams piercing through canopy, "
                "wildlife in motion captured with telephoto lens, "
                "National Geographic documentary cinematography style, "
                "8K ultra HD, vivid natural colors, cinematic color grading"
            ),
            "transition": (
                "Serene nature timelapse transition, gentle clouds flowing over valleys, "
                "soft morning mist dissolving to reveal pristine landscapes, "
                "delicate flower petals floating on crystal clear water surface, "
                "peaceful sunrise colors gradually warming the scene, "
                "smooth dolly shot through enchanted forest, "
                "nature documentary aesthetic, professional color science, "
                "8K ultra HD, seamless natural motion"
            )
        },
        "abstract": {
            "peak": (
                "Hypnotic abstract generative art in motion, "
                "complex fractal patterns exploding and reforming endlessly, "
                "kaleidoscopic symmetry with infinite recursive depth, "
                "fluid simulation with iridescent colors colliding and mixing, "
                "mathematical visualization of higher dimensions unfolding, "
                "procedural art aesthetic, perfect symmetry and balance, "
                "8K ultra HD, vibrant saturated colors, mesmerizing visual flow"
            ),
            "transition": (
                "Meditative abstract transition, gentle morphing of organic forms, "
                "soft cellular automata patterns evolving peacefully, "
                "watercolor pigments diffusing through liquid medium, "
                "aurora borealis colors dancing and interweaving slowly, "
                "zen garden sand patterns being drawn by invisible forces, "
                "contemplative art house aesthetic, subtle movements, "
                "8K ultra HD, calming color palette, hypnotic smoothness"
            )
        },
        "cyberpunk": {
            "peak": (
                "Cyberpunk neon cityscape at night with heavy rain, "
                "holographic advertisements flickering on towering skyscrapers, "
                "high speed chase through crowded futuristic streets, "
                "glitching digital artifacts and data streams visualized, "
                "dramatic camera movements through neon-lit alleyways, "
                "Blade Runner aesthetic, pink and cyan color scheme, "
                "8K ultra HD, ray-traced reflections, cinematic lens flares"
            ),
            "transition": (
                "Digital glitch transition with pixel sorting effects, "
                "hologram flickering between different cyberpunk scenes, "
                "data visualization morphing into cityscapes, "
                "smooth camera push through virtual reality environments, "
                "neon signs reflecting on rain-soaked streets, "
                "synthwave aesthetic, retrofuturistic atmosphere, "
                "8K ultra HD, moody atmospheric lighting"
            )
        },
        "elegant": {
            "peak": (
                "Luxurious high fashion editorial cinematography, "
                "silk fabrics flowing in slow motion with dramatic studio lighting, "
                "gold and marble textures with perfect reflections, "
                "graceful ballet dancer movements captured with precision, "
                "sophisticated camera movements revealing artistic compositions, "
                "Vogue magazine aesthetic, timeless elegance, "
                "8K ultra HD, rich deep blacks, pristine whites, film look"
            ),
            "transition": (
                "Elegant dissolve transition with soft bokeh effects, "
                "champagne bubbles rising through golden light, "
                "delicate lace patterns fading between scenes, "
                "smooth crane shot through opulent interior spaces, "
                "gentle fabric movements creating visual poetry, "
                "luxury brand commercial aesthetic, refined taste, "
                "8K ultra HD, creamy skin tones, sophisticated color palette"
            )
        },
        "cosmic": {
            "peak": (
                "Epic cosmic journey through deep space, "
                "massive nebula clouds with vibrant stellar nurseries, "
                "supernova explosion sending shockwaves across galaxy, "
                "warp speed travel through starfields with motion blur, "
                "alien planet landscapes with multiple moons rising, "
                "NASA visualization meets Hollywood sci-fi aesthetic, "
                "8K ultra HD, infinite depth of field, awe-inspiring scale"
            ),
            "transition": (
                "Celestial transition through cosmic phenomena, "
                "gentle drift past Saturn's rings catching sunlight, "
                "aurora dancing on distant exoplanet atmosphere, "
                "smooth orbit around crystalline asteroid formations, "
                "time dilation effect near black hole event horizon, "
                "contemplative space documentary aesthetic, "
                "8K ultra HD, perfect darkness of space, stellar illumination"
            )
        }
    }
    
    # Apply prompt preset first, then override with custom prompts if provided
    preset = PROMPT_PRESETS.get(args.prompt_style, PROMPT_PRESETS["energetic"])
    generator.peak_prompt = preset["peak"]
    generator.transition_prompt = preset["transition"]
    
    # Override with custom prompts if explicitly provided
    if args.peak_prompt is not None:
        generator.peak_prompt = args.peak_prompt
    if args.transition_prompt is not None:
        generator.transition_prompt = args.transition_prompt
    
    # Display prompts being used
    print(f"\n[Prompts] Style: {args.prompt_style}")
    print(f"  Peak: {generator.peak_prompt[:80]}...")
    print(f"  Transition: {generator.transition_prompt[:80]}...")

    # Handle Audio Trimming
    process_audio_path = args.audio
    temp_audio = None
    
    if args.audio_start > 0 or args.audio_end is not None:
        temp_audio = work_dir / "trimmed_audio.mp3"
        end_str = f"{args.audio_end:.1f}s" if args.audio_end else "end of audio"
        print(f"\n[Audio Range] Processing only {args.audio_start:.1f}s ~ {end_str}")
        trim_audio(args.audio, temp_audio, args.audio_start, args.audio_end)
        process_audio_path = temp_audio
    else:
        print(f"\n[Audio Range] Processing entire audio file")

    # 1. Analyze Audio
    beat_times, beat_types, beat_strengths = detect_beats_madmom(str(process_audio_path))
    duration = librosa.get_duration(path=process_audio_path)
    
    # 2. Plan Timeline
    scheduler = SceneScheduler(duration, beat_times, beat_strengths)
    timeline = scheduler.plan_scenes(peak_threshold=args.peak_thresh, peak_clip_len=args.peak_len)

    if not timeline:
        print("No timeline generated. Check thresholds.")
        return

    # 3. Generate Assets
    
    # A. Generate Pool of Peak Clips (T2V) - NEW METHOD: Generate single video and slice
    print(f"[Generator] Creating a pool of {args.num_peak_clips} unique peak clips...")
    print(f"  -> Method: Generate ONE long video and slice into {args.num_peak_clips} segments")
    print(f"  -> Each clip duration: {args.peak_len}s")
    print(f"  -> Total T2V generation: {args.num_peak_clips * args.peak_len:.2f}s")
    
    # Use the new slicing method instead of batch generation with different seeds
    peak_pool_paths = generator.generate_peak_clips_from_single_video(
        num_clips=args.num_peak_clips,
        clip_duration=args.peak_len,
        randomize_order=True  # Shuffle for variety when assigning to beats
    )
    
    # B. Assign Peaks to Timeline
    # Pre-assign files to peaks so we can calculate Loop Transitions
    print("[Generator] Assigning peaks to timeline...")
    for item in timeline:
        if item['type'] == 'peak':
            item['file'] = random.choice(peak_pool_paths)

    # C. Generate Transitions (Loop Aware)
    # For a perfect loop: 
    #   - The video should start and end at the SAME point in a continuous transition
    #   - Intro Gap + Outro Gap = One continuous transition from Last Peak End -> First Peak Start
    #   - Intro uses the SECOND half, Outro uses the FIRST half
    
    first_peak_idx = next((i for i, x in enumerate(timeline) if x['type'] == 'peak'), None)
    last_peak_idx = next((i for i, x in enumerate(reversed(timeline)) if x['type'] == 'peak'), None)
    if last_peak_idx is not None: last_peak_idx = len(timeline) - 1 - last_peak_idx
    
    if first_peak_idx is not None and last_peak_idx is not None:
        # Get frames
        first_peak_start_frame = generator.extract_frame(timeline[first_peak_idx]['file'], at_start=True)
        last_peak_end_frame = generator.extract_frame(timeline[last_peak_idx]['file'], at_start=False)
        
        has_intro_gap = (timeline[0]['type'] == 'gap')
        has_outro_gap = (timeline[-1]['type'] == 'gap')
        
        intro_dur = timeline[0]['duration'] if has_intro_gap else 0.0
        outro_dur = timeline[-1]['duration'] if has_outro_gap else 0.0
        
        # Total loop transition duration (Outro flows into Intro seamlessly)
        total_loop_dur = intro_dur + outro_dur
        
        if total_loop_dur > 0:
            print(f"[Generator] Creating Loop Transition (Outro: {outro_dur:.2f}s + Intro: {intro_dur:.2f}s = {total_loop_dur:.2f}s)...")
            print(f"  -> This creates a seamless loop: Last Peak End -> First Peak Start")
            print(f"  -> Using interpolation mode: {args.interpolation_mode}")
            
            # Generate ONE continuous transition video
            loop_clip = generator.generate_transition_clip(
                index="loop_combined",
                start_img=last_peak_end_frame,
                end_img=first_peak_start_frame,
                duration=total_loop_dur
            )
            
            # Split the loop clip:
            # - Outro uses frames 0 to outro_dur (first part)
            # - Intro uses frames outro_dur to end (second part)
            # This way: Video End (Outro) -> Loop Point -> Video Start (Intro)
            
            if has_outro_gap:
                timeline[-1]['file'] = loop_clip
                timeline[-1]['trim_start'] = 0.0
                timeline[-1]['trim_duration'] = outro_dur
                
            if has_intro_gap:
                timeline[0]['file'] = loop_clip
                timeline[0]['trim_start'] = outro_dur
                timeline[0]['trim_duration'] = intro_dur
                
    # D. Generate Standard Transitions (Inner Gaps)
    previous_last_frame = None
    transition_tasks = []
    
    # 1. Collect all transition tasks first
    print("[Generator] Preparing transition tasks for batch execution...")
    
    for i, item in enumerate(timeline):
        if item['type'] == 'peak':
            previous_last_frame = generator.extract_frame(item['file'], at_start=False)
            
        elif item['type'] == 'gap':
            # Skip if it's Intro/Outro already handled
            if 'file' in item: continue
            
            # Standard Gap: Prev Peak End -> Next Peak Start
            if previous_last_frame is None:
                # Should not happen if Intro is handled, but safety: Black
                previous_last_frame = Image.new('RGB', (generator.width, generator.height), color=(0, 0, 0))
                
            # Find Next Peak Start Frame
            next_first_frame = None
            if i + 1 < len(timeline) and timeline[i+1]['type'] == 'peak':
                next_first_frame = generator.extract_frame(timeline[i+1]['file'], at_start=True)
            else:
                # Should be Outro, handled above. If we are here, something weird.
                next_first_frame = Image.new('RGB', (generator.width, generator.height), color=(0, 0, 0))
            
            # Add to task list
            transition_tasks.append({
                'timeline_idx': i,
                'index': f"{i:04d}",
                'start_img': previous_last_frame,
                'end_img': next_first_frame,
                'duration': item['duration']
            })

    # 2. Run Batch Generation (with Hybrid mode support)
    if transition_tasks:
        print(f"[Generator] Processing {len(transition_tasks)} transitions with {args.interpolation_mode} mode...")
        generated_files = generator.generate_transition_clips_batch(transition_tasks, batch_size=args.batch_size)
        
        # 3. Assign files back to timeline
        for task in transition_tasks:
            key = task['index']
            idx = task['timeline_idx']
            if key in generated_files:
                timeline[idx]['file'] = generated_files[key]

    # 4. Assemble
    temp_video = assemble_video(timeline, args.output)
    
    # 5. Mux Audio
    mux_audio(temp_video, process_audio_path, args.output)
    
    # Cleanup
    if temp_video.exists():
        temp_video.unlink()
    
    if temp_audio and temp_audio.exists():
        temp_audio.unlink()
    
    print(f"\n{'='*60}")
    print(f"[COMPLETE] Video generation finished!")
    print(f"  Final Output:      {args.output}")
    print(f"  Working Directory: {work_dir}")
    # Audio range info
    if args.audio_start > 0 or args.audio_end is not None:
        end_str = f"{args.audio_end:.1f}s" if args.audio_end else "end"
        print(f"  Audio Range: {args.audio_start:.1f}s ~ {end_str}")
    print(f"  Processed Duration: {duration:.2f}s")
    print(f"  Prompt Style: {args.prompt_style}")
    print(f"  Mode: {args.interpolation_mode}")
    print(f"  CPU Offload: {args.cpu_offload}")
    print(f"  Device Map: {args.device_map or 'single GPU'}")
    if args.interpolation_mode == "hybrid":
        print(f"  Threshold: {args.hybrid_threshold}s (FLF2V for longer, BiM-VFI for shorter)")
    if args.interpolation_mode in ["hybrid", "flf2v"]:
        limit_str = "unlimited" if args.max_flf2v_count < 0 else str(args.max_flf2v_count)
        print(f"  FLF2V Usage: {generator.flf2v_usage_count} (limit: {limit_str})")
    print(f"\n[Intermediate Files] (in {work_dir})")
    print(f"  - peak_*.mp4       : Generated peak clips")
    print(f"  - transition_*.mp4 : Generated transition clips")
    print(f"  - segments_processed/ : Processed segments for assembly")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()