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
import re

import numpy as np

_np_version = tuple(map(int, np.__version__.split('.')[:2]))
if _np_version < (1, 24):
    print(f"[Warning] numpy {np.__version__} detected. Recommend numpy >= 1.24.0")

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

if not hasattr(collections, 'MutableSequence'):
    collections.MutableSequence = collections.abc.MutableSequence
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable

sys.path.append(str(Path(__file__).parent / "BiM-VFI"))

import cv2
from PIL import Image
import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

try:
    from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
    from transformers import CLIPVisionModel
    FLF2V_AVAILABLE = True
except ImportError:
    print("[Warning] FLF2V dependencies not found.")
    FLF2V_AVAILABLE = False

try:
    from modules.components import make_components
    import torch.nn.functional as F
    BIM_VFI_AVAILABLE = True
except ImportError:
    print("[Warning] BiM-VFI modules not found.")
    BIM_VFI_AVAILABLE = False

try:
    import madmom
    import librosa
except ImportError as exc:
    print(f"Error importing dependencies: {exc}")
    sys.exit(1)

try:
    from huggingface_hub import login as hf_login, HfFolder
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


# ============================================
# LLM-based Prompt Enhancement System
# ============================================
class PromptEnhancer:
    """
    LLM-based prompt enhancement system for Wan 2.2 video generation.
    
    Features:
    - Expands simple keywords into detailed 80-120 word prompts
    - Dynamically adjusts CFG based on prompt complexity
    - Provides optimized negative prompts based on Wan 2.2 best practices
    """
    
    # Wan 2.2 optimized negative prompts based on official recommendations
    NEGATIVE_PROMPTS = {
        "default": (
            "bright colors, overexposed, static, blurred details, subtitles, "
            "style, artwork, painting, picture, still, overall gray, "
            "worst quality, low quality, JPEG compression residue, ugly, incomplete, "
            "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
            "malformed limbs, fused fingers, still picture, cluttered background, "
            "three legs, many people in the background, walking backwards"
        ),
        "motion_focused": (
            "static, frozen, no movement, still image, jittery motion, "
            "discontinuous motion, teleporting, abrupt cuts, flickering, "
            "motion blur artifacts, ghosting, frame skipping, stuttering, "
            "blurry, distorted, low quality, watermark, text overlay"
        ),
        "cinematic": (
            "amateur, home video, low resolution, pixelated, "
            "overexposed, underexposed, washed out colors, flat lighting, "
            "shaky camera, dutch angle, lens flare artifacts, "
            "bad composition, cluttered frame, distracting elements, "
            "worst quality, low quality, blurry, out of focus"
        ),
        "abstract": (
            "realistic, photorealistic, human faces, recognizable objects, "
            "text, logos, watermarks, static patterns, boring, monotonous, "
            "low contrast, muddy colors, worst quality, low quality, "
            "pixelated, compression artifacts"
        )
    }
    
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://gateway.letsur.ai/v1",
        model: str = "claude-sonnet-4-5-20250929",
        enable_extension: bool = True,
        min_prompt_words: int = 15  # Prompts shorter than this get extended
    ):
        """
        Initialize the prompt enhancer.
        
        Args:
            api_key: API key for the LLM service
            base_url: Base URL for the API endpoint
            model: Model identifier to use
            enable_extension: Whether to enable automatic prompt extension
            min_prompt_words: Minimum word count before extension is applied
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.enable_extension = enable_extension
        self.min_prompt_words = min_prompt_words
        self.client = None
        
        if enable_extension and api_key:
            try:
                import openai
                self.client = openai.OpenAI(
                    base_url=base_url,
                    api_key=api_key
                )
                print(f"[PromptEnhancer] Initialized with model: {model}")
            except ImportError:
                print("[Warning] openai package not found. Install with: pip install openai")
                self.enable_extension = False
            except Exception as e:
                print(f"[Warning] Failed to initialize LLM client: {e}")
                self.enable_extension = False
    
    def count_words(self, text: str) -> int:
        """Count words in a prompt."""
        return len(text.split())
    
    def analyze_prompt_complexity(self, prompt: str) -> Dict:
        """
        Analyze prompt complexity to determine optimal generation parameters.
        
        Returns:
            Dict with:
            - word_count: Number of words
            - has_camera_motion: Whether camera motion is specified
            - has_lighting: Whether lighting is specified
            - has_style: Whether visual style is specified
            - has_motion: Whether motion/action is specified
            - complexity_score: 0-100 score
            - recommended_cfg: Recommended CFG value
        """
        word_count = self.count_words(prompt)
        prompt_lower = prompt.lower()
        
        # Check for various elements
        camera_keywords = ['camera', 'pan', 'tilt', 'dolly', 'zoom', 'tracking', 'crane', 'aerial', 'pov', 'close-up', 'wide shot']
        lighting_keywords = ['light', 'lighting', 'illuminat', 'shadow', 'glow', 'neon', 'sunset', 'sunrise', 'golden hour', 'backlit']
        style_keywords = ['cinematic', 'photorealistic', 'aesthetic', '8k', '4k', 'hdr', 'film', 'documentary', 'professional']
        motion_keywords = ['moving', 'walking', 'running', 'flowing', 'flying', 'floating', 'dancing', 'exploding', 'morphing', 'transforming']
        
        has_camera = any(kw in prompt_lower for kw in camera_keywords)
        has_lighting = any(kw in prompt_lower for kw in lighting_keywords)
        has_style = any(kw in prompt_lower for kw in style_keywords)
        has_motion = any(kw in prompt_lower for kw in motion_keywords)
        
        # Calculate complexity score (0-100)
        score = 0
        score += min(40, word_count * 0.5)  # Word count contribution (max 40)
        score += 15 if has_camera else 0
        score += 15 if has_lighting else 0
        score += 15 if has_style else 0
        score += 15 if has_motion else 0
        
        complexity_score = min(100, score)
        
        # Determine CFG based on complexity
        # Lower complexity = higher CFG to force model to follow simple prompts more strictly
        # Higher complexity = slightly lower CFG to give model room to interpret details
        if complexity_score < 30:
            recommended_cfg = 7.0  # Simple prompt - higher guidance
        elif complexity_score < 50:
            recommended_cfg = 6.5
        elif complexity_score < 70:
            recommended_cfg = 6.0
        else:
            recommended_cfg = 5.5  # Complex prompt - more freedom
        
        return {
            'word_count': word_count,
            'has_camera_motion': has_camera,
            'has_lighting': has_lighting,
            'has_style': has_style,
            'has_motion': has_motion,
            'complexity_score': complexity_score,
            'recommended_cfg': recommended_cfg
        }
    
    def extend_prompt(self, simple_prompt: str, style: str = "energetic") -> str:
        """
        Extend a simple prompt into a detailed Wan 2.2 optimized prompt.
        
        Args:
            simple_prompt: The user's simple/keyword prompt
            style: Style hint (energetic, nature, abstract, cyberpunk, elegant, cosmic)
        
        Returns:
            Extended prompt optimized for Wan 2.2 (80-120 words)
        """
        if not self.client or not self.enable_extension:
            return simple_prompt
        
        # Check if extension is needed
        word_count = self.count_words(simple_prompt)
        if word_count >= self.min_prompt_words:
            print(f"[PromptEnhancer] Prompt already detailed ({word_count} words), skipping extension")
            return simple_prompt
        
        print(f"[PromptEnhancer] Extending simple prompt ({word_count} words) -> 80-120 words")
        
        system_message = """You are an expert prompt engineer for Wan 2.2, an AI video generation model.
Your task is to expand simple keywords or short prompts into detailed, high-quality video generation prompts.

IMPORTANT GUIDELINES for Wan 2.2:
1. Target 80-120 words for optimal results
2. Structure: [Opening Scene] → [Camera Motion] → [Visual Details] → [Style/Atmosphere]
3. Include specific camera movements: pan, tilt, dolly, zoom, tracking, crane, orbital
4. Specify motion/speed: slow-motion, time-lapse, smooth glide, rapid movement
5. Add lighting details: volumetric, rim light, golden hour, neon, backlit
6. Include visual style: cinematic, photorealistic, 8K, HDR, film grain, color grading
7. Describe what changes over time - videos are about MOTION

DO NOT include:
- Negative terms (these go in negative prompts)
- Technical parameters like resolution numbers
- Multiple unrelated scenes

OUTPUT ONLY THE ENHANCED PROMPT, nothing else."""

        style_hints = {
            "energetic": "Focus on dynamic energy, vibrant colors, explosive motion, music video aesthetics",
            "nature": "Focus on natural beauty, organic textures, wildlife, landscape cinematography",
            "abstract": "Focus on geometric patterns, fluid simulations, hypnotic visuals, generative art",
            "cyberpunk": "Focus on neon lights, rain, futuristic city, holographics, Blade Runner aesthetics",
            "elegant": "Focus on luxury, fashion, graceful movement, sophisticated lighting, high-end aesthetics",
            "cosmic": "Focus on space, nebulae, galaxies, celestial bodies, epic scale"
        }
        
        user_message = f"""Expand this simple prompt into a detailed Wan 2.2 video generation prompt (80-120 words):

Simple prompt: "{simple_prompt}"

Style direction: {style_hints.get(style, style_hints['energetic'])}

Remember: Focus on describing motion and camera work - videos are about what MOVES and CHANGES over time."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            extended_prompt = response.choices[0].message.content.strip()
            
            # Clean up any quotes or extra formatting
            extended_prompt = extended_prompt.strip('"\'')
            
            new_word_count = self.count_words(extended_prompt)
            print(f"[PromptEnhancer] Extended prompt: {new_word_count} words")
            
            return extended_prompt
            
        except Exception as e:
            print(f"[Warning] Prompt extension failed: {e}")
            return simple_prompt
    
    def get_negative_prompt(self, style: str = "default") -> str:
        """
        Get optimized negative prompt for Wan 2.2.
        
        Args:
            style: Style type (default, motion_focused, cinematic, abstract)
        
        Returns:
            Negative prompt string
        """
        return self.NEGATIVE_PROMPTS.get(style, self.NEGATIVE_PROMPTS["default"])
    
    def process_prompt(
        self,
        prompt: str,
        style: str = "energetic",
        negative_style: str = "default",
        custom_negative: Optional[str] = None
    ) -> Dict:
        """
        Process a prompt and return all generation parameters.
        
        Args:
            prompt: User's prompt (simple or detailed)
            style: Visual style for extension
            negative_style: Style for negative prompt selection
            custom_negative: Optional custom negative prompt to append
        
        Returns:
            Dict with:
            - enhanced_prompt: Extended/original prompt
            - negative_prompt: Optimized negative prompt
            - recommended_cfg: Recommended CFG value
            - analysis: Full prompt analysis
        """
        # Extend prompt if needed
        enhanced_prompt = self.extend_prompt(prompt, style)
        
        # Analyze complexity
        analysis = self.analyze_prompt_complexity(enhanced_prompt)
        
        # Get negative prompt
        negative_prompt = self.get_negative_prompt(negative_style)
        if custom_negative:
            negative_prompt = f"{negative_prompt}, {custom_negative}"
        
        return {
            'enhanced_prompt': enhanced_prompt,
            'negative_prompt': negative_prompt,
            'recommended_cfg': analysis['recommended_cfg'],
            'analysis': analysis
        }


# ==========================================
# Beat Detection (Madmom-based)
# ==========================================
def detect_beats_madmom(audio_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect beats using madmom's RNN-based beat tracker."""
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
# Wan Model Generator with LLM Enhancement
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
        device_map: Optional[str] = None,
        # NEW: LLM Enhancement parameters
        prompt_enhancer: Optional[PromptEnhancer] = None,
        auto_cfg: bool = True,
        base_cfg: float = 6.0,
        use_negative_prompt: bool = True
    ):
        """
        Initialize the video generator with LLM enhancement support.
        
        New Args:
            prompt_enhancer: PromptEnhancer instance for LLM-based prompt extension
            auto_cfg: Whether to automatically adjust CFG based on prompt complexity
            base_cfg: Base CFG value (used when auto_cfg is False)
            use_negative_prompt: Whether to use negative prompts
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
        self.flf2v_usage_count = 0
        self.hf_token = hf_token
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.cpu_offload_mode = cpu_offload_mode
        self.device_map = device_map
        
        # LLM Enhancement settings
        self.prompt_enhancer = prompt_enhancer
        self.auto_cfg = auto_cfg
        self.base_cfg = base_cfg
        self.use_negative_prompt = use_negative_prompt
        
        # Current generation parameters (updated per generation)
        self.current_cfg = base_cfg
        self.current_negative_prompt = ""
        
        # Track actual CFG values used (for logging)
        self.actual_cfg_values = {
            "peak": None,
            "transition": None
        }
        
        # Track actual enhanced prompts used (for metadata)
        self.actual_enhanced_prompts = {
            "peak": None,
            "peak_negative": None,
            "transition": None,
            "transition_negative": None
        }
        
        # Default prompts optimized for Wan T2V models
        self.peak_prompt = "Cinematic abstract visual explosion, vibrant neon particles bursting outward, dynamic rapid camera zoom through swirling geometric shapes, electric blue and magenta energy waves pulsating rhythmically, dramatic lens flares and light rays piercing through darkness, high contrast dramatic lighting, fluid motion blur effects, professional music video aesthetic, photorealistic CGI rendering, 8K ultra HD, 60fps smooth motion, dolby vision HDR colors"
        self.transition_prompt = "Seamless abstract morphing transition, fluid organic shapes slowly transforming, ethereal dreamlike atmosphere with soft gradient colors blending, gentle camera drift through luminous particles and soft bokeh lights, smooth flowing liquid metal textures reflecting ambient light, cinematic slow motion with graceful movement, professional film color grading, subtle film grain texture, 8K ultra HD, buttery smooth interpolation, photorealistic quality"
        
        self.t2v_pipe = None
        self.flf2v_pipe = None
        self.bim_model = None
        
        if interpolation_mode == "flf2v" and not FLF2V_AVAILABLE:
            raise RuntimeError("FLF2V mode requested but dependencies not available.")
        if interpolation_mode == "bim" and not BIM_VFI_AVAILABLE:
            raise RuntimeError("BiM-VFI mode requested but dependencies not available.")
        if interpolation_mode == "hybrid" and not (FLF2V_AVAILABLE and BIM_VFI_AVAILABLE):
            print("[Warning] Hybrid mode requires both FLF2V and BiM-VFI. Falling back to available method.")
            if FLF2V_AVAILABLE:
                self.interpolation_mode = "flf2v"
            elif BIM_VFI_AVAILABLE:
                self.interpolation_mode = "bim"
            else:
                raise RuntimeError("No interpolation method available!")

    def prepare_prompt(self, prompt: str, style: str = "energetic", is_transition: bool = False) -> Tuple[str, str, float]:
        """
        Prepare a prompt for generation with LLM enhancement.
        
        Args:
            prompt: Original prompt
            style: Style for enhancement
            is_transition: Whether this is for a transition clip
        
        Returns:
            Tuple of (enhanced_prompt, negative_prompt, cfg_value)
        """
        if self.prompt_enhancer:
            negative_style = "motion_focused" if is_transition else "cinematic"
            result = self.prompt_enhancer.process_prompt(
                prompt=prompt,
                style=style,
                negative_style=negative_style
            )
            
            enhanced_prompt = result['enhanced_prompt']
            negative_prompt = result['negative_prompt'] if self.use_negative_prompt else ""
            cfg = result['recommended_cfg'] if self.auto_cfg else self.base_cfg
            
            # Track actual CFG value used
            cfg_key = "transition" if is_transition else "peak"
            self.actual_cfg_values[cfg_key] = cfg
            
            # Log the enhancement
            analysis = result['analysis']
            print(f"[PromptEnhancer] Complexity: {analysis['complexity_score']:.0f}/100")
            print(f"  - Camera motion: {'✓' if analysis['has_camera_motion'] else '✗'}")
            print(f"  - Lighting: {'✓' if analysis['has_lighting'] else '✗'}")
            print(f"  - Style: {'✓' if analysis['has_style'] else '✗'}")
            print(f"  - Motion: {'✓' if analysis['has_motion'] else '✗'}")
            print(f"  - CFG: {cfg}")
            
            return enhanced_prompt, negative_prompt, cfg
        else:
            # No enhancer - use defaults
            negative_prompt = PromptEnhancer.NEGATIVE_PROMPTS["default"] if self.use_negative_prompt else ""
            cfg = self.base_cfg
            
            # Track actual CFG value used
            cfg_key = "transition" if is_transition else "peak"
            self.actual_cfg_values[cfg_key] = cfg
            
            return prompt, negative_prompt, cfg

    def _get_t2v_pipe(self):
        """Load Wan T2V pipeline for peak clip generation."""
        if self.t2v_pipe is None:
            print(f"[Wan T2V] Loading {self.model_id}...")
            
            is_local = Path(self.model_id).exists()
            
            if is_local:
                print(f"  -> Loading from local path")
            elif self.local_files_only:
                print(f"  -> Loading from cache only (local_files_only=True)")
            else:
                print(f"  -> Loading from HuggingFace (may download if not cached)")
            
            load_kwargs = {"torch_dtype": torch.bfloat16}
            
            if not is_local:
                if self.hf_token:
                    load_kwargs["token"] = self.hf_token
                if self.cache_dir:
                    load_kwargs["cache_dir"] = self.cache_dir
                if self.local_files_only:
                    load_kwargs["local_files_only"] = True
            
            if self.device_map:
                load_kwargs["device_map"] = self.device_map
                print(f"  -> Using device_map: {self.device_map}")
            
            import time
            start_time = time.time()
            
            self.t2v_pipe = WanPipeline.from_pretrained(self.model_id, **load_kwargs)
            
            load_time = time.time() - start_time
            print(f"  -> Model loaded in {load_time:.1f}s")
            
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
        """Load Wan FLF2V pipeline."""
        if self.flf2v_pipe is None:
            if not FLF2V_AVAILABLE:
                raise RuntimeError("FLF2V not available.")
            
            print(f"[Wan FLF2V] Loading {self.flf2v_model_id}...")
            
            is_local = Path(self.flf2v_model_id).exists()
            
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
            
            pipe_kwargs = {
                "vae": vae,
                "image_encoder": image_encoder,
                "torch_dtype": torch.bfloat16,
                **hf_kwargs
            }
            
            if self.device_map:
                pipe_kwargs["device_map"] = self.device_map
            
            self.flf2v_pipe = WanImageToVideoPipeline.from_pretrained(
                self.flf2v_model_id,
                **pipe_kwargs
            )
            
            if not self.device_map:
                if self.cpu_offload_mode == "sequential":
                    self.flf2v_pipe.enable_sequential_cpu_offload()
                elif self.cpu_offload_mode == "model":
                    self.flf2v_pipe.enable_model_cpu_offload()
                else:
                    self.flf2v_pipe.to("cuda")
            
            load_time = time.time() - start_time
            print(f"  -> Model loaded in {load_time:.1f}s")
            
        return self.flf2v_pipe

    def _get_bim_model(self):
        """Load BiM-VFI interpolation model."""
        if self.bim_model is None:
            if not BIM_VFI_AVAILABLE:
                raise RuntimeError("BiM-VFI not available.")
            
            print("[BiM-VFI] Loading Interpolation Model...")
            
            model_spec = {
                'name': 'bim_vfi',
                'args': {'pyr_level': 3, 'feat_channels': 32}
            }
            
            self.bim_model = make_components(model_spec)
            
            ckpt_path = Path(__file__).parent / "BiM-VFI/pretrained/bim_vfi.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Model not found at {ckpt_path}")
            
            checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            self.bim_model.load_state_dict(checkpoint['model'])
            
            self.bim_model.cuda()
            self.bim_model.eval()
            
        return self.bim_model

    def _unload_t2v_pipe(self):
        if self.t2v_pipe is not None:
            print("[Wan T2V] Unloading to free VRAM...")
            del self.t2v_pipe
            self.t2v_pipe = None
            torch.cuda.empty_cache()

    def _unload_flf2v_pipe(self):
        if self.flf2v_pipe is not None:
            print("[Wan FLF2V] Unloading to free VRAM...")
            del self.flf2v_pipe
            self.flf2v_pipe = None
            torch.cuda.empty_cache()

    def _unload_bim_model(self):
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

    def generate_peak_clips_from_single_video(
        self, 
        num_clips: int, 
        clip_duration: float,
        randomize_order: bool = True,
        style: str = "energetic"
    ) -> List[Path]:
        """
        Generate peak clips with LLM-enhanced prompts.
        """
        if num_clips <= 0:
            return []
        
        # Prepare enhanced prompt
        enhanced_prompt, negative_prompt, cfg = self.prepare_prompt(
            self.peak_prompt, 
            style=style, 
            is_transition=False
        )
        
        # Store actual enhanced prompts for metadata
        self.actual_enhanced_prompts["peak"] = enhanced_prompt
        self.actual_enhanced_prompts["peak_negative"] = negative_prompt
        
        print(f"\n[Wan T2V] Using enhanced prompt:")
        print(f"  Prompt: {enhanced_prompt[:100]}...")
        print(f"  Negative: {negative_prompt[:80]}...")
        print(f"  CFG: {cfg}")
        
        pipe = self._get_t2v_pipe()
        
        fps = 30
        MAX_FRAMES_PER_GENERATION = 121
        
        max_duration_single_pass = (MAX_FRAMES_PER_GENERATION - 1) / fps
        ideal_total_duration = num_clips * clip_duration
        
        if ideal_total_duration <= max_duration_single_pass:
            total_frames = int(ideal_total_duration * fps)
            total_frames = (total_frames // 4) * 4 + 1
            total_frames = max(17, total_frames)
            actual_duration = ideal_total_duration
            print(f"[Wan T2V] Single pass: {total_frames} frames ({actual_duration:.2f}s) -> {num_clips} clips of {clip_duration:.2f}s")
        else:
            total_frames = MAX_FRAMES_PER_GENERATION
            actual_duration = (total_frames - 1) / fps
            actual_num_clips = int(actual_duration / clip_duration)
            if actual_num_clips < num_clips:
                print(f"[Wan T2V] WARNING: Can only generate {actual_num_clips} clips of {clip_duration:.2f}s")
            print(f"[Wan T2V] Single pass: {total_frames} frames ({actual_duration:.2f}s) -> slicing into clips")
        
        long_video_path = self.output_dir / "peak_source_long.mp4"

        # Check if source video already exists
        if long_video_path.exists():
            print(f"  -> Found existing peak source video: {long_video_path}")
            print(f"  -> Skipping T2V generation, using cached video")
        else:
            print(f"  -> Generating {total_frames} frames...")

            # Generator setup: use CPU generator to avoid device mismatch issues
            # The pipeline will handle device placement automatically even with device_map
            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device="cpu").manual_seed(seed)

            # Generate with enhanced parameters
            output = pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt if self.use_negative_prompt else None,
                height=self.height,
                width=self.width,
                num_frames=total_frames,
                num_inference_steps=30,
                guidance_scale=cfg,
                generator=generator
            )

            export_to_video(output.frames[0], str(long_video_path), fps=fps)
            print(f"  -> Source video saved: {long_video_path}")
        
        actual_video_duration = (total_frames - 1) / fps
        max_possible_clips = int(actual_video_duration / clip_duration)
        clips_to_create = min(max_possible_clips, num_clips)
        
        if clips_to_create == 0:
            print(f"  -> Video too short ({actual_video_duration:.2f}s) for {clip_duration:.2f}s clips")
            print(f"  -> Using entire video as single clip")
            clips_to_create = 1
            effective_clip_duration = actual_video_duration
        else:
            effective_clip_duration = clip_duration
        
        print(f"  -> Slicing into {clips_to_create} clips of {effective_clip_duration:.2f}s each...")

        sliced_paths = []
        for i in range(clips_to_create):
            start_time = i * effective_clip_duration
            output_clip_path = self.output_dir / f"peak_{i:03d}.mp4"

            # Check if clip already exists
            if output_clip_path.exists():
                sliced_paths.append(output_clip_path)
                print(f"    -> Using cached peak_{i:03d}.mp4")
            else:
                # Create new clip
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start_time),
                    "-i", str(long_video_path),
                    "-t", str(effective_clip_duration),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    "-an",
                    str(output_clip_path)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                if output_clip_path.exists():
                    sliced_paths.append(output_clip_path)
                    print(f"    -> Created peak_{i:03d}.mp4 (start: {start_time:.2f}s)")
        
        if len(sliced_paths) < num_clips and len(sliced_paths) > 0:
            print(f"  -> Created {len(sliced_paths)} clips, but {num_clips} requested. Clips will be reused.")
        
        if randomize_order and len(sliced_paths) > 1:
            random.shuffle(sliced_paths)
            print(f"  -> Shuffled clip order for random beat assignment")

        return sliced_paths

    def generate_transition_clips_batch(self, tasks: List[dict], batch_size: int = 1) -> Dict[str, Path]:
        """Generates transition clips using hybrid interpolation."""
        if not tasks: 
            return {}
        
        mode = self.interpolation_mode
        print(f"[Interpolation] Generating {len(tasks)} transition clips (Mode: {mode})...")
        
        results = {}
        
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
            
            if flf2v_tasks:
                flf2v_results = self._generate_transitions_flf2v(flf2v_tasks)
                results.update(flf2v_results)
            
            if bim_tasks:
                bim_results = self._generate_transitions_bim(bim_tasks)
                results.update(bim_results)
                
        elif mode == "flf2v":
            results = self._generate_transitions_flf2v(tasks)
        else:
            results = self._generate_transitions_bim(tasks)
                
        return results

    def _generate_transitions_flf2v(self, tasks: List[dict]) -> Dict[str, Path]:
        """Generate transitions using Wan FLF2V with enhanced prompts."""
        if not tasks:
            return {}
        
        print(f"[Wan FLF2V] Generating {len(tasks)} transition clips...")
        
        self._unload_t2v_pipe()
        
        results = {}
        
        for i, t in enumerate(tasks):
            if self.max_flf2v_count >= 0 and self.flf2v_usage_count >= self.max_flf2v_count:
                print(f"  -> [Limit Reached] Falling back to BiM-VFI for remaining {len(tasks) - i} transitions...")
                self._unload_flf2v_pipe()
                remaining_tasks = tasks[i:]
                bim_results = self._generate_transitions_bim(remaining_tasks)
                results.update(bim_results)
                break
            
            print(f"  -> Processing FLF2V Transition {i + 1}/{len(tasks)} (Duration: {t['duration']:.2f}s)")
            
            fname = self.output_dir / f"transition_{t['index']}.mp4"
            
            if fname.exists():
                results[t['index']] = fname
                continue
            
            try:
                pipe = self._get_flf2v_pipe()
                
                path = self._generate_single_flf2v_transition(
                    t['index'], t['start_img'], t['end_img'], t['duration']
                )
                results[t['index']] = path
                self.flf2v_usage_count += 1
            except torch.cuda.OutOfMemoryError as e:
                print(f"    [Error] FLF2V CUDA OOM, falling back to BiM-VFI...")
                self._unload_flf2v_pipe()
                torch.cuda.empty_cache()
                path = self._generate_single_bim_transition(
                    t['index'], t['start_img'], t['end_img'], t['duration']
                )
                results[t['index']] = path
            except Exception as e:
                print(f"    [Error] FLF2V failed: {e}, falling back to BiM-VFI...")
                self._unload_flf2v_pipe()
                torch.cuda.empty_cache()
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
        """Generate a single transition using Wan FLF2V with enhanced prompts."""
        filename = self.output_dir / f"transition_{index}.mp4"
        if filename.exists():
            return filename
        
        # Prepare enhanced prompt for transition
        enhanced_prompt, negative_prompt, cfg = self.prepare_prompt(
            self.transition_prompt,
            style="elegant",
            is_transition=True
        )
        
        # Store actual enhanced prompts for metadata (update with latest transition)
        self.actual_enhanced_prompts["transition"] = enhanced_prompt
        self.actual_enhanced_prompts["transition_negative"] = negative_prompt
        
        print(f"[Wan FLF2V] Generating Transition {index} (Duration: {duration:.2f}s)...")
        print(f"  -> CFG: {cfg}")
        
        pipe = self._get_flf2v_pipe()
        
        flf2v_width = 1280
        flf2v_height = 720
        
        start_img_resized = start_img.resize((flf2v_width, flf2v_height), Image.Resampling.LANCZOS)
        end_img_resized = end_img.resize((flf2v_width, flf2v_height), Image.Resampling.LANCZOS)
        
        if start_img_resized.mode != 'RGB':
            start_img_resized = start_img_resized.convert('RGB')
        if end_img_resized.mode != 'RGB':
            end_img_resized = end_img_resized.convert('RGB')
        
        fps = 24
        num_frames = int(duration * fps)
        num_frames = max(17, (num_frames // 4) * 4 + 1)
        num_frames = min(num_frames, 81)
        
        print(f"    -> Generating {num_frames} frames at {fps}fps...")

        # Generator setup: prefer using generator object for better reproducibility
        # Even with device_map, we can use a generator on CPU (pipeline will handle device placement)
        seed = random.randint(0, 2**32 - 1)
        # Use CPU generator to avoid device mismatch issues with device_map
        # The pipeline will handle device placement automatically
        generator = torch.Generator(device="cpu").manual_seed(seed)

        output = pipe(
            image=start_img_resized,
            last_image=end_img_resized,
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt if self.use_negative_prompt else "blurry, distorted, low quality, artifacts",
            height=flf2v_height,
            width=flf2v_width,
            num_frames=num_frames,
            num_inference_steps=30,
            guidance_scale=cfg,
            generator=generator
        )
        
        # Export video to temporary file first
        temp_filename = self.output_dir / f"transition_{index}_temp.mp4"
        export_to_video(output.frames[0], str(temp_filename), fps=fps)
        
        # Resize back to target dimensions and adjust duration
        self._resize_and_adjust_video(temp_filename, filename, self.width, self.height, duration)
        
        # Cleanup temp file
        if temp_filename.exists():
            temp_filename.unlink()
        
        print(f"    -> Saved: {filename}")
        
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
# Scene Scheduler
# ==========================================
class SceneScheduler:
    def __init__(self, duration: float, beat_times: np.ndarray, beat_strengths: np.ndarray):
        self.total_duration = duration
        self.beat_times = beat_times
        self.beat_strengths = beat_strengths
        self.timeline = []

    def plan_scenes(self, peak_threshold: float = 0.7, peak_clip_len: float = 2.0) -> List[Dict]:
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


# ==========================================
# Video Assembly
# ==========================================
def assemble_video(timeline: List[Dict], output_path: Path, fps=30) -> Path:
    """
    Assembles the clips using ffmpeg with speed adjustment to match beat timing.
    Each clip is adjusted to match the target duration exactly using setpts filter.
    """
    print(f"[Assembly] Processing segments and stitching...")
    print(f"  -> Total segments: {len(timeline)}")
    
    total_expected_duration = sum(item['duration'] for item in timeline)
    print(f"  -> Expected total duration: {total_expected_duration:.2f}s")
    
    temp_dir = output_path.parent / "segments_processed"
    if temp_dir.exists(): 
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    temp_video = output_path.with_suffix('.temp.mp4')

    for i, item in enumerate(timeline):
        if 'file' not in item:
            print(f"  [Warning] Segment {i} has no file, skipping")
            continue

        src_file = item['file']
        if not Path(src_file).exists():
            print(f"  [Warning] File not found: {src_file}")
            continue

        out_file = temp_dir / f"seg_{i:04d}.mp4"
        target_dur = item['duration']
        
        print(f"  -> Segment {i}: {item['type']} | target={target_dur:.2f}s | file={Path(src_file).name}")
        
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
            # After extraction, we may still need to adjust speed to match target_dur
            extracted_dur = trim_duration
            if extracted_dur < target_dur * 0.9:
                # Extracted portion is too short - slow it down
                speed_factor = extracted_dur / target_dur
                filter_str = f"setpts={1/speed_factor:.6f}*PTS"
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(trim_start),
                    "-i", str(src_file),
                    "-t", str(trim_duration),
                    "-filter:v", filter_str,
                    "-t", str(target_dur),
                    "-r", str(fps),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    str(out_file)
                ]
            elif extracted_dur > target_dur * 1.1:
                # Extracted portion is too long - speed it up
                speed_factor = extracted_dur / target_dur
                filter_str = f"setpts={1/speed_factor:.6f}*PTS"
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(trim_start),
                    "-i", str(src_file),
                    "-t", str(trim_duration),
                    "-filter:v", filter_str,
                    "-t", str(target_dur),
                    "-r", str(fps),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    str(out_file)
                ]
            else:
                # Extracted portion is close to target - just extract
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
        return temp_video
    
    # Create file list for concatenation
    list_file = temp_dir / "files.txt"
    with open(list_file, "w") as f:
        for p in processed_files:
            f.write(f"file '{p.resolve()}'\n")
    
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
    """Combine video and audio - exact copy from original working version."""
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
    """Trim audio file."""
    end_str = f"{end:.1f}" if end else "end"
    print(f"[Audio] Trimming: {start:.1f}s ~ {end_str}s")

    if not input_path.exists():
        print(f"[Error] Audio input not found: {input_path}")
        return None

    cmd = ["ffmpeg", "-y", "-i", str(input_path), "-ss", str(start)]
    if end is not None:
        cmd.extend(["-t", str(end - start)])
    cmd.extend(["-c", "copy", str(output_path)])

    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0:
        print(f"[Error] Audio trim failed: {result.stderr.decode()[:200]}")
        return None

    if not output_path.exists():
        print(f"[Error] Trimmed audio not created: {output_path}")
        return None

    print(f"  -> Trimmed audio saved: {output_path}")
    return output_path


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Beat-Reactive Video Generator with LLM-Enhanced Prompt Quality Control"
    )
    parser.add_argument("--audio", required=True, type=Path, help="Input audio file")
    parser.add_argument("--output", type=Path, default=Path("final_output.mp4"), help="Output video file")
    parser.add_argument("--peak-thresh", type=float, default=0.5, help="0.0-1.0, threshold for Peak detection")
    parser.add_argument("--peak-len", type=float, default=2.0, help="Duration of peak clips (seconds)")
    parser.add_argument("--num-peak-clips", type=int, default=5, help="Number of unique peak clips to generate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for T2V generation")
    parser.add_argument("--model", type=str, default="wan2.2-14b", 
                       choices=["wan2.1-1.3b", "wan2.2-5b", "wan2.2-14b"], 
                       help="Model to use for T2V generation")
    parser.add_argument("--peak-prompt", type=str, default=None,
                       help="Prompt for beat peak video generation")
    parser.add_argument("--prompt-style", type=str, default="energetic",
                       choices=["energetic", "nature", "abstract", "cyberpunk", "elegant", "cosmic"],
                       help="Preset prompt style")
    parser.add_argument("--audio-start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--audio-end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--work-dir", type=Path, default=Path("workspace_wan"), 
                       help="Directory to save intermediate results")
    
    # Interpolation arguments
    parser.add_argument("--interpolation-mode", type=str, default="hybrid",
                       choices=["hybrid", "flf2v", "bim"],
                       help="Interpolation mode")
    parser.add_argument("--hybrid-threshold", type=float, default=1.5,
                       help="Duration threshold for hybrid mode")
    parser.add_argument("--flf2v-model", type=str, default="Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers",
                       help="Wan FLF2V model ID")
    parser.add_argument("--transition-prompt", type=str, default=None,
                       help="Prompt for FLF2V transition generation")
    parser.add_argument("--max-flf2v-count", type=int, default=-1,
                       help="Maximum number of FLF2V generations (-1 for unlimited)")
    
    # HuggingFace arguments
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument("--t2v-model-path", type=str, default=None, help="Local T2V model path")
    parser.add_argument("--flf2v-model-path", type=str, default=None, help="Local FLF2V model path")
    parser.add_argument("--cache-dir", type=str, default=None, help="HuggingFace cache directory")
    parser.add_argument("--local-files-only", action="store_true", help="Only use cached models")
    parser.add_argument("--cpu-offload", type=str, default="model",
                       choices=["none", "model", "sequential"], help="CPU offload mode")
    parser.add_argument("--device-map", type=str, default=None,
                       choices=["balanced"], help="Multi-GPU device mapping")
    
    # NEW: LLM Enhancement arguments
    parser.add_argument("--llm-api-key", type=str, default=None,
                       help="API key for LLM prompt enhancement (also reads LLM_API_KEY env var)")
    parser.add_argument("--llm-base-url", type=str, default="https://gateway.letsur.ai/v1",
                       help="Base URL for LLM API")
    parser.add_argument("--llm-model", type=str, default="claude-3-7-sonnet-20250219",
                       help="LLM model to use for prompt enhancement")
    parser.add_argument("--enable-prompt-extension", action="store_true",
                       help="Enable LLM-based prompt extension for simple prompts")
    parser.add_argument("--auto-cfg", action="store_true",
                       help="Automatically adjust CFG based on prompt complexity")
    parser.add_argument("--base-cfg", type=float, default=6.0,
                       help="Base CFG value (used when auto-cfg is disabled)")
    parser.add_argument("--use-negative-prompt", action="store_true", default=True,
                       help="Use optimized negative prompts (default: True)")
    parser.add_argument("--no-negative-prompt", action="store_false", dest="use_negative_prompt",
                       help="Disable negative prompts")
    parser.add_argument("--min-prompt-words", type=int, default=15,
                       help="Minimum words before prompt extension is applied")
    parser.add_argument("--assemble-only", action="store_true",
                       help="Skip generation, only assemble existing clips and mux audio")

    args = parser.parse_args()

    if not args.audio.exists():
        print(f"File not found: {args.audio}")
        return

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if not args.output.is_absolute() and len(args.output.parts) == 1:
        args.output = work_dir / args.output
    args.output = args.output.resolve()

    import os
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", None)
    llm_api_key = args.llm_api_key or os.environ.get("LLM_API_KEY", None)
    
    # Initialize Prompt Enhancer if API key is provided
    prompt_enhancer = None
    if args.enable_prompt_extension and llm_api_key:
        prompt_enhancer = PromptEnhancer(
            api_key=llm_api_key,
            base_url=args.llm_base_url,
            model=args.llm_model,
            enable_extension=True,
            min_prompt_words=args.min_prompt_words
        )
        print(f"\n[LLM Enhancement] Enabled")
        print(f"  Model: {args.llm_model}")
        print(f"  Auto CFG: {args.auto_cfg}")
        print(f"  Negative Prompts: {args.use_negative_prompt}")
        print(f"  Min Words for Extension: {args.min_prompt_words}")
    elif args.enable_prompt_extension and not llm_api_key:
        print("\n[Warning] --enable-prompt-extension specified but no API key provided")
        print("  Set LLM_API_KEY environment variable or use --llm-api-key")
    
    # Display settings
    print(f"\n[GPU Settings]")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"  Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"    GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    # Model selection
    model_map = {
        "wan2.1-1.3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "wan2.2-5b": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "wan2.2-14b": "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    }
    
    selected_model_id = args.t2v_model_path if args.t2v_model_path else model_map[args.model]
    flf2v_model_id = args.flf2v_model_path if args.flf2v_model_path else args.flf2v_model
    
    print(f"[Model] T2V: {selected_model_id}")
    print(f"[Model] FLF2V: {flf2v_model_id}")

    # Initialize generator
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
        device_map=args.device_map,
        prompt_enhancer=prompt_enhancer,
        auto_cfg=args.auto_cfg,
        base_cfg=args.base_cfg,
        use_negative_prompt=args.use_negative_prompt
    )
    
    # Prompt presets
    PROMPT_PRESETS = {
        "energetic": {
            "peak": "Cinematic abstract visual explosion, vibrant neon particles bursting outward, dynamic rapid camera zoom through swirling geometric shapes, electric blue and magenta energy waves pulsating rhythmically, dramatic lens flares and light rays piercing through darkness, high contrast dramatic lighting, fluid motion blur effects, professional music video aesthetic, photorealistic CGI rendering, 8K ultra HD, 60fps smooth motion",
            "transition": "Seamless abstract morphing transition, fluid organic shapes slowly transforming, ethereal dreamlike atmosphere with soft gradient colors blending, gentle camera drift through luminous particles and soft bokeh lights, smooth flowing liquid metal textures reflecting ambient light, cinematic slow motion with graceful movement"
        },
        "nature": {
            "peak": "Breathtaking nature scene with dramatic weather, powerful ocean waves crashing against rocky cliffs in golden hour light, dynamic aerial drone shot sweeping over misty mountain peaks, lush green forests with sunbeams piercing through canopy, National Geographic documentary cinematography style, 8K ultra HD",
            "transition": "Serene nature timelapse transition, gentle clouds flowing over valleys, soft morning mist dissolving to reveal pristine landscapes, delicate flower petals floating on crystal clear water surface, smooth dolly shot through enchanted forest"
        },
        "abstract": {
            "peak": "Hypnotic abstract generative art in motion, complex fractal patterns exploding and reforming endlessly, kaleidoscopic symmetry with infinite recursive depth, fluid simulation with iridescent colors colliding and mixing, mathematical visualization of higher dimensions unfolding, procedural art aesthetic",
            "transition": "Meditative abstract transition, gentle morphing of organic forms, soft cellular automata patterns evolving peacefully, watercolor pigments diffusing through liquid medium, aurora borealis colors dancing and interweaving slowly"
        },
        "cyberpunk": {
            "peak": "Cyberpunk neon cityscape at night with heavy rain, holographic advertisements flickering on towering skyscrapers, high speed chase through crowded futuristic streets, glitching digital artifacts and data streams visualized, Blade Runner aesthetic, pink and cyan color scheme",
            "transition": "Digital glitch transition with pixel sorting effects, hologram flickering between different cyberpunk scenes, data visualization morphing into cityscapes, smooth camera push through virtual reality environments"
        },
        "elegant": {
            "peak": "Luxurious high fashion editorial cinematography, silk fabrics flowing in slow motion with dramatic studio lighting, gold and marble textures with perfect reflections, graceful ballet dancer movements captured with precision, Vogue magazine aesthetic",
            "transition": "Elegant dissolve transition with soft bokeh effects, champagne bubbles rising through golden light, delicate lace patterns fading between scenes, smooth crane shot through opulent interior spaces"
        },
        "cosmic": {
            "peak": "Epic cosmic journey through deep space, massive nebula clouds with vibrant stellar nurseries, supernova explosion sending shockwaves across galaxy, warp speed travel through starfields with motion blur, alien planet landscapes with multiple moons rising",
            "transition": "Celestial transition through cosmic phenomena, gentle drift past Saturn's rings catching sunlight, aurora dancing on distant exoplanet atmosphere, smooth orbit around crystalline asteroid formations"
        }
    }
    
    preset = PROMPT_PRESETS.get(args.prompt_style, PROMPT_PRESETS["energetic"])
    generator.peak_prompt = preset["peak"]
    generator.transition_prompt = preset["transition"]
    
    if args.peak_prompt is not None:
        generator.peak_prompt = args.peak_prompt
    if args.transition_prompt is not None:
        generator.transition_prompt = args.transition_prompt
    
    print(f"\n[Prompts] Style: {args.prompt_style}")
    print(f"  Peak: {generator.peak_prompt[:80]}...")
    print(f"  Transition: {generator.transition_prompt[:80]}...")

    # Handle Audio Trimming
    process_audio_path = args.audio
    temp_audio = None

    if args.audio_start > 0 or args.audio_end is not None:
        temp_audio = work_dir / "trimmed_audio.mp3"

        # Check if trimmed audio already exists (for --assemble-only mode)
        if temp_audio.exists():
            print(f"[Audio] Using existing trimmed audio: {temp_audio}")
            process_audio_path = temp_audio
        else:
            result = trim_audio(args.audio, temp_audio, args.audio_start, args.audio_end)
            if result is None:
                print("[Error] Failed to trim audio. Aborting.")
                return
            process_audio_path = temp_audio

    # Verify audio file exists
    if not process_audio_path.exists():
        print(f"[Error] Audio file not found: {process_audio_path}")
        return

    # Analyze Audio
    beat_times, beat_types, beat_strengths = detect_beats_madmom(str(process_audio_path))
    duration = librosa.get_duration(path=process_audio_path)
    
    # Plan Timeline
    scheduler = SceneScheduler(duration, beat_times, beat_strengths)
    timeline = scheduler.plan_scenes(peak_threshold=args.peak_thresh, peak_clip_len=args.peak_len)

    if not timeline:
        print("No timeline generated. Check thresholds.")
        return

    # --assemble-only mode: skip generation, use existing files
    if args.assemble_only:
        print("\n[Assemble-Only Mode] Skipping generation, using existing files...")

        # Find existing peak clips
        existing_peaks = sorted(work_dir.glob("peak_*.mp4"))
        existing_peaks = [p for p in existing_peaks if "source" not in p.name]

        if not existing_peaks:
            print("[Error] No existing peak clips found in work_dir")
            return

        print(f"  -> Found {len(existing_peaks)} peak clips")

        # Assign existing peaks to timeline
        for item in timeline:
            if item['type'] == 'peak':
                item['file'] = random.choice(existing_peaks)

        # Find existing transition clips
        for i, item in enumerate(timeline):
            if item['type'] == 'gap':
                # Check for loop_combined transition (intro/outro)
                if i == 0 or i == len(timeline) - 1:
                    loop_clip = work_dir / "transition_loop_combined.mp4"
                    if loop_clip.exists():
                        item['file'] = loop_clip
                        if i == len(timeline) - 1:  # outro
                            item['trim_start'] = 0.0
                            item['trim_duration'] = item['duration']
                        else:  # intro
                            # Calculate outro duration for trim_start
                            outro_dur = timeline[-1]['duration'] if timeline[-1]['type'] == 'gap' else 0
                            item['trim_start'] = outro_dur
                            item['trim_duration'] = item['duration']
                        continue

                # Check for regular transition
                trans_file = work_dir / f"transition_{i:04d}.mp4"
                if trans_file.exists():
                    item['file'] = trans_file
                else:
                    print(f"  [Warning] Missing transition file: {trans_file}")

        # Go directly to assembly
        temp_video = assemble_video(timeline, args.output)

        if not temp_video.exists():
            print(f"[Error] Assembly failed - temp video not found: {temp_video}")
            return

        # Mux Audio
        mux_success = mux_audio(temp_video, process_audio_path, args.output)

        if not mux_success:
            print(f"[Warning] Muxing may have failed.")

        # Cleanup temp video (keep trimmed_audio for future reuse)
        if temp_video.exists():
            temp_video.unlink()

        print(f"\n[COMPLETE] Assemble-only finished!")
        print(f"  Final Output: {args.output}")
        return

    # Generate Peak Clips
    print(f"[Generator] Creating pool of {args.num_peak_clips} unique peak clips...")
    
    peak_pool_paths = generator.generate_peak_clips_from_single_video(
        num_clips=args.num_peak_clips,
        clip_duration=args.peak_len,
        randomize_order=True,
        style=args.prompt_style
    )
    
    # Assign Peaks to Timeline
    print("[Generator] Assigning peaks to timeline...")
    for item in timeline:
        if item['type'] == 'peak':
            item['file'] = random.choice(peak_pool_paths)

    # Generate Loop Transitions
    first_peak_idx = next((i for i, x in enumerate(timeline) if x['type'] == 'peak'), None)
    last_peak_idx = next((i for i, x in enumerate(reversed(timeline)) if x['type'] == 'peak'), None)
    if last_peak_idx is not None: 
        last_peak_idx = len(timeline) - 1 - last_peak_idx
    
    if first_peak_idx is not None and last_peak_idx is not None:
        first_peak_start_frame = generator.extract_frame(timeline[first_peak_idx]['file'], at_start=True)
        last_peak_end_frame = generator.extract_frame(timeline[last_peak_idx]['file'], at_start=False)
        
        has_intro_gap = (timeline[0]['type'] == 'gap')
        has_outro_gap = (timeline[-1]['type'] == 'gap')
        
        intro_dur = timeline[0]['duration'] if has_intro_gap else 0.0
        outro_dur = timeline[-1]['duration'] if has_outro_gap else 0.0
        
        total_loop_dur = intro_dur + outro_dur
        
        if total_loop_dur > 0:
            print(f"[Generator] Creating Loop Transition ({total_loop_dur:.2f}s)...")
            
            loop_clip = generator.generate_transition_clip(
                index="loop_combined",
                start_img=last_peak_end_frame,
                end_img=first_peak_start_frame,
                duration=total_loop_dur
            )
            
            if has_outro_gap:
                timeline[-1]['file'] = loop_clip
                timeline[-1]['trim_start'] = 0.0
                timeline[-1]['trim_duration'] = outro_dur
                
            if has_intro_gap:
                timeline[0]['file'] = loop_clip
                timeline[0]['trim_start'] = outro_dur
                timeline[0]['trim_duration'] = intro_dur

    # Generate Standard Transitions
    previous_last_frame = None
    transition_tasks = []
    
    for i, item in enumerate(timeline):
        if item['type'] == 'peak':
            previous_last_frame = generator.extract_frame(item['file'], at_start=False)
            
        elif item['type'] == 'gap':
            if 'file' in item: 
                continue
            
            if previous_last_frame is None:
                previous_last_frame = Image.new('RGB', (generator.width, generator.height), color=(0, 0, 0))
                
            next_first_frame = None
            if i + 1 < len(timeline) and timeline[i+1]['type'] == 'peak':
                next_first_frame = generator.extract_frame(timeline[i+1]['file'], at_start=True)
            else:
                next_first_frame = Image.new('RGB', (generator.width, generator.height), color=(0, 0, 0))
            
            transition_tasks.append({
                'timeline_idx': i,
                'index': f"{i:04d}",
                'start_img': previous_last_frame,
                'end_img': next_first_frame,
                'duration': item['duration']
            })

    if transition_tasks:
        print(f"[Generator] Processing {len(transition_tasks)} transitions...")
        generated_files = generator.generate_transition_clips_batch(transition_tasks, batch_size=args.batch_size)
        
        for task in transition_tasks:
            key = task['index']
            idx = task['timeline_idx']
            if key in generated_files:
                timeline[idx]['file'] = generated_files[key]

    # Assemble
    temp_video = assemble_video(timeline, args.output)

    if not temp_video.exists():
        print(f"[Error] Assembly failed - temp video not found: {temp_video}")
        return

    # Mux Audio
    mux_success = mux_audio(temp_video, process_audio_path, args.output)

    if not mux_success:
        print(f"[Warning] Muxing may have failed. Check if output file has audio.")
        print(f"  -> temp_video exists: {temp_video.exists()}")
        print(f"  -> audio exists: {process_audio_path.exists()}")
        print(f"  -> output exists: {args.output.exists()}")
        # Continue anyway like original - don't return early
        # The output file should still be created even if there were warnings

    # Cleanup (keep trimmed_audio for future reuse with --assemble-only)
    if temp_video.exists():
        temp_video.unlink()

    # Save generation metadata
    metadata = {
        "arguments": {
            "audio": str(args.audio),
            "output": str(args.output),
            "peak_thresh": args.peak_thresh,
            "peak_len": args.peak_len,
            "num_peak_clips": args.num_peak_clips,
            "batch_size": args.batch_size,
            "model": args.model,
            "peak_prompt_original": generator.peak_prompt,
            "transition_prompt_original": generator.transition_prompt,
            "prompt_style": args.prompt_style,
            "audio_start": args.audio_start,
            "audio_end": args.audio_end,
            "work_dir": str(work_dir),
            "interpolation_mode": args.interpolation_mode,
            "hybrid_threshold": args.hybrid_threshold,
            "flf2v_model": args.flf2v_model,
            "max_flf2v_count": args.max_flf2v_count,
            "hf_token": "***" if hf_token else None,
            "t2v_model_path": args.t2v_model_path,
            "flf2v_model_path": args.flf2v_model_path,
            "cache_dir": args.cache_dir,
            "local_files_only": args.local_files_only,
            "cpu_offload": args.cpu_offload,
            "device_map": args.device_map,
            "llm_api_key": "***" if llm_api_key else None,
            "llm_base_url": args.llm_base_url,
            "llm_model": args.llm_model,
            "enable_prompt_extension": args.enable_prompt_extension,
            "auto_cfg": args.auto_cfg,
            "base_cfg": args.base_cfg,
            "use_negative_prompt": args.use_negative_prompt,
            "min_prompt_words": args.min_prompt_words,
        },
        "actual_cfg_values": generator.actual_cfg_values,
        "actual_enhanced_prompts": generator.actual_enhanced_prompts,
        "generation_info": {
            "processed_duration": duration,
            "final_output": str(args.output),
            "llm_enhancement_enabled": prompt_enhancer is not None,
            "t2v_model_id": selected_model_id,
            "flf2v_model_id": flf2v_model_id,
        },
        "timeline_summary": {
            "total_scenes": len(timeline),
            "peak_count": sum(1 for item in timeline if item['type'] == 'peak'),
            "transition_count": sum(1 for item in timeline if item['type'] == 'gap'),
        }
    }
    
    # Save metadata to work_dir (so each run has its own log)
    metadata_file = work_dir / "generation_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"[COMPLETE] Video generation finished!")
    print(f"  Final Output: {args.output}")
    print(f"  Processed Duration: {duration:.2f}s")
    print(f"  Prompt Style: {args.prompt_style}")
    print(f"  LLM Enhancement: {'Enabled' if prompt_enhancer else 'Disabled'}")
    if generator.actual_cfg_values["peak"] is not None:
        print(f"  Actual CFG (Peak): {generator.actual_cfg_values['peak']}")
    if generator.actual_cfg_values["transition"] is not None:
        print(f"  Actual CFG (Transition): {generator.actual_cfg_values['transition']}")
    print(f"  Metadata saved to: {metadata_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
