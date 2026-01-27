# lib.py - Core library code for Beat-Reactive Video Generation
# Original code wrapped for ComfyUI compatibility

import subprocess
import sys
import math
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import shutil
import random
import collections.abc
import collections

import numpy as np

# Numpy compatibility patches
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

# Lazy imports for heavy dependencies
_cv2 = None
_PIL_Image = None
_torch = None
_madmom = None
_librosa = None


def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2


def _get_pil_image():
    global _PIL_Image
    if _PIL_Image is None:
        from PIL import Image
        _PIL_Image = Image
    return _PIL_Image


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_madmom():
    global _madmom
    if _madmom is None:
        import madmom
        _madmom = madmom
    return _madmom


def _get_librosa():
    global _librosa
    if _librosa is None:
        import librosa
        _librosa = librosa
    return _librosa


# Check availability of optional dependencies
def check_flf2v_available():
    try:
        from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
        from transformers import CLIPVisionModel
        return True
    except ImportError:
        return False


def check_bim_vfi_available():
    try:
        from modules.components import make_components
        import torch.nn.functional as F
        return True
    except ImportError:
        return False


def check_hf_hub_available():
    try:
        from huggingface_hub import login as hf_login, HfFolder
        return True
    except ImportError:
        return False


# ============================================
# LLM-based Prompt Enhancement System
# ============================================
class PromptEnhancer:
    """
    LLM-based prompt enhancement system for Wan 2.2 video generation.
    """
    
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
        min_prompt_words: int = 15
    ):
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
                print("[Warning] openai package not found.")
                self.enable_extension = False
            except Exception as e:
                print(f"[Warning] Failed to initialize LLM client: {e}")
                self.enable_extension = False
    
    def count_words(self, text: str) -> int:
        return len(text.split())
    
    def analyze_prompt_complexity(self, prompt: str) -> Dict:
        word_count = self.count_words(prompt)
        prompt_lower = prompt.lower()
        
        camera_keywords = ['camera', 'pan', 'tilt', 'dolly', 'zoom', 'tracking', 'crane', 'aerial', 'pov', 'close-up', 'wide shot']
        lighting_keywords = ['light', 'lighting', 'illuminat', 'shadow', 'glow', 'neon', 'sunset', 'sunrise', 'golden hour', 'backlit']
        style_keywords = ['cinematic', 'photorealistic', 'aesthetic', '8k', '4k', 'hdr', 'film', 'documentary', 'professional']
        motion_keywords = ['moving', 'walking', 'running', 'flowing', 'flying', 'floating', 'dancing', 'exploding', 'morphing', 'transforming']
        
        has_camera = any(kw in prompt_lower for kw in camera_keywords)
        has_lighting = any(kw in prompt_lower for kw in lighting_keywords)
        has_style = any(kw in prompt_lower for kw in style_keywords)
        has_motion = any(kw in prompt_lower for kw in motion_keywords)
        
        score = 0
        score += min(40, word_count * 0.5)
        score += 15 if has_camera else 0
        score += 15 if has_lighting else 0
        score += 15 if has_style else 0
        score += 15 if has_motion else 0
        
        complexity_score = min(100, score)
        
        if complexity_score < 30:
            recommended_cfg = 7.0
        elif complexity_score < 50:
            recommended_cfg = 6.5
        elif complexity_score < 70:
            recommended_cfg = 6.0
        else:
            recommended_cfg = 5.5
        
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
        if not self.client or not self.enable_extension:
            return simple_prompt
        
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
            extended_prompt = extended_prompt.strip('"\'')
            
            new_word_count = self.count_words(extended_prompt)
            print(f"[PromptEnhancer] Extended prompt: {new_word_count} words")
            
            return extended_prompt
            
        except Exception as e:
            print(f"[Warning] Prompt extension failed: {e}")
            return simple_prompt
    
    def get_negative_prompt(self, style: str = "default") -> str:
        return self.NEGATIVE_PROMPTS.get(style, self.NEGATIVE_PROMPTS["default"])
    
    def process_prompt(
        self,
        prompt: str,
        style: str = "energetic",
        negative_style: str = "default",
        custom_negative: Optional[str] = None
    ) -> Dict:
        enhanced_prompt = self.extend_prompt(prompt, style)
        analysis = self.analyze_prompt_complexity(enhanced_prompt)
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
    madmom = _get_madmom()
    
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


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file using librosa."""
    librosa = _get_librosa()
    return librosa.get_duration(path=audio_path)


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
        
        peaks = []
        min_transition_len = 0.1
        
        for t, strength in zip(self.beat_times, self.beat_strengths):
            if strength >= peak_threshold:
                start_t = t
                end_t = t + peak_clip_len
                
                if peaks:
                    prev_peak = peaks[-1]
                    gap_avail = start_t - prev_peak['end']
                    
                    if gap_avail < min_transition_len:
                        new_prev_end = start_t - min_transition_len
                        prev_dur = new_prev_end - prev_peak['start']
                        
                        if prev_dur > 0.05:
                            prev_peak['end'] = new_prev_end
                            prev_peak['duration'] = prev_dur
                        else:
                            continue
                
                if start_t >= self.total_duration: 
                    continue
                    
                end_t = min(end_t, self.total_duration)
                duration = end_t - start_t
                
                if duration > 0.05:
                    peaks.append({
                        'type': 'peak',
                        'start': start_t,
                        'end': end_t,
                        'duration': duration
                    })

        print(f"  -> Found {len(peaks)} valid peaks")

        self.timeline = []
        current_time = 0.0
        
        for p in peaks:
            gap_duration = p['start'] - current_time
            if gap_duration > 0.01:
                self.timeline.append({
                    'type': 'gap',
                    'start': current_time,
                    'end': p['start'],
                    'duration': gap_duration
                })
            
            self.timeline.append(p)
            current_time = p['end']
            
        final_gap = self.total_duration - current_time
        if final_gap > 0.01:
            self.timeline.append({
                'type': 'gap',
                'start': current_time,
                'end': self.total_duration,
                'duration': final_gap
            })

        total_timeline_dur = sum(item['duration'] for item in self.timeline)
        print(f" -> Scheduled {len(self.timeline)} segments ({len(peaks)} peaks).")
        print(f" -> Timeline total duration: {total_timeline_dur:.2f}s (expected: {self.total_duration:.2f}s)")
        
        return self.timeline


# ==========================================
# Wan Model Generator
# ==========================================
class WanVideoGenerator:
    def __init__(
        self, 
        output_dir: Path, 
        model_id: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers", 
        width: int = 832, 
        height: int = 480,
        interpolation_mode: str = "hybrid",
        hybrid_threshold: float = 1.5,
        flf2v_model_id: str = "Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers",
        max_flf2v_count: int = -1,
        hf_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        cpu_offload_mode: str = "model",
        device_map: Optional[str] = None,
        prompt_enhancer: Optional[PromptEnhancer] = None,
        auto_cfg: bool = True,
        base_cfg: float = 6.0,
        use_negative_prompt: bool = True
    ):
        self.output_dir = Path(output_dir)
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
        
        self.prompt_enhancer = prompt_enhancer
        self.auto_cfg = auto_cfg
        self.base_cfg = base_cfg
        self.use_negative_prompt = use_negative_prompt
        
        self.current_cfg = base_cfg
        self.current_negative_prompt = ""
        
        self.actual_cfg_values = {"peak": None, "transition": None}
        self.actual_enhanced_prompts = {
            "peak": None, "peak_negative": None,
            "transition": None, "transition_negative": None
        }
        
        self.peak_prompt = "Cinematic abstract visual explosion, vibrant neon particles bursting outward, dynamic rapid camera zoom through swirling geometric shapes, electric blue and magenta energy waves pulsating rhythmically, dramatic lens flares and light rays piercing through darkness, high contrast dramatic lighting, fluid motion blur effects, professional music video aesthetic, photorealistic CGI rendering, 8K ultra HD, 60fps smooth motion, dolby vision HDR colors"
        self.transition_prompt = "Seamless abstract morphing transition, fluid organic shapes slowly transforming, ethereal dreamlike atmosphere with soft gradient colors blending, gentle camera drift through luminous particles and soft bokeh lights, smooth flowing liquid metal textures reflecting ambient light, cinematic slow motion with graceful movement, professional film color grading, subtle film grain texture, 8K ultra HD, buttery smooth interpolation, photorealistic quality"
        
        self.t2v_pipe = None
        self.flf2v_pipe = None
        self.bim_model = None
        
        FLF2V_AVAILABLE = check_flf2v_available()
        BIM_VFI_AVAILABLE = check_bim_vfi_available()
        
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
            
            cfg_key = "transition" if is_transition else "peak"
            self.actual_cfg_values[cfg_key] = cfg
            
            analysis = result['analysis']
            print(f"[PromptEnhancer] Complexity: {analysis['complexity_score']:.0f}/100")
            print(f"  - Camera motion: {'✓' if analysis['has_camera_motion'] else '✗'}")
            print(f"  - Lighting: {'✓' if analysis['has_lighting'] else '✗'}")
            print(f"  - Style: {'✓' if analysis['has_style'] else '✗'}")
            print(f"  - Motion: {'✓' if analysis['has_motion'] else '✗'}")
            print(f"  - CFG: {cfg}")
            
            return enhanced_prompt, negative_prompt, cfg
        else:
            negative_prompt = PromptEnhancer.NEGATIVE_PROMPTS["default"] if self.use_negative_prompt else ""
            cfg = self.base_cfg
            
            cfg_key = "transition" if is_transition else "peak"
            self.actual_cfg_values[cfg_key] = cfg
            
            return prompt, negative_prompt, cfg

    def _get_t2v_pipe(self):
        if self.t2v_pipe is None:
            torch = _get_torch()
            from diffusers import WanPipeline
            
            print(f"[Wan T2V] Loading {self.model_id}...")
            
            is_local = Path(self.model_id).exists()
            
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
            
            import time
            start_time = time.time()
            
            self.t2v_pipe = WanPipeline.from_pretrained(self.model_id, **load_kwargs)
            
            load_time = time.time() - start_time
            print(f"  -> Model loaded in {load_time:.1f}s")
            
            if not self.device_map:
                if self.cpu_offload_mode == "sequential":
                    self.t2v_pipe.enable_sequential_cpu_offload()
                elif self.cpu_offload_mode == "model":
                    self.t2v_pipe.enable_model_cpu_offload()
                else:
                    self.t2v_pipe.to("cuda")
            
        return self.t2v_pipe

    def _get_flf2v_pipe(self):
        if self.flf2v_pipe is None:
            if not check_flf2v_available():
                raise RuntimeError("FLF2V not available.")
            
            torch = _get_torch()
            from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
            from transformers import CLIPVisionModel
            
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
        if self.bim_model is None:
            if not check_bim_vfi_available():
                raise RuntimeError("BiM-VFI not available.")
            
            torch = _get_torch()
            from modules.components import make_components
            
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
            torch = _get_torch()
            print("[Wan T2V] Unloading to free VRAM...")
            del self.t2v_pipe
            self.t2v_pipe = None
            torch.cuda.empty_cache()

    def _unload_flf2v_pipe(self):
        if self.flf2v_pipe is not None:
            torch = _get_torch()
            print("[Wan FLF2V] Unloading to free VRAM...")
            del self.flf2v_pipe
            self.flf2v_pipe = None
            torch.cuda.empty_cache()

    def _unload_bim_model(self):
        if self.bim_model is not None:
            torch = _get_torch()
            print("[BiM-VFI] Unloading to free VRAM...")
            del self.bim_model
            self.bim_model = None
            torch.cuda.empty_cache()

    def extract_frame(self, video_path: Path, at_start: bool = True):
        cv2 = _get_cv2()
        Image = _get_pil_image()
        
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
        if num_clips <= 0:
            return []
        
        torch = _get_torch()
        from diffusers.utils import export_to_video
        
        enhanced_prompt, negative_prompt, cfg = self.prepare_prompt(
            self.peak_prompt, 
            style=style, 
            is_transition=False
        )
        
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
        else:
            total_frames = MAX_FRAMES_PER_GENERATION
            actual_duration = (total_frames - 1) / fps
        
        long_video_path = self.output_dir / "peak_source_long.mp4"

        if long_video_path.exists():
            print(f"  -> Found existing peak source video: {long_video_path}")
        else:
            print(f"  -> Generating {total_frames} frames...")

            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device="cpu").manual_seed(seed)

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
            clips_to_create = 1
            effective_clip_duration = actual_video_duration
        else:
            effective_clip_duration = clip_duration
        
        print(f"  -> Slicing into {clips_to_create} clips of {effective_clip_duration:.2f}s each...")

        sliced_paths = []
        for i in range(clips_to_create):
            start_time = i * effective_clip_duration
            output_clip_path = self.output_dir / f"peak_{i:03d}.mp4"

            if output_clip_path.exists():
                sliced_paths.append(output_clip_path)
            else:
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
        
        if randomize_order and len(sliced_paths) > 1:
            random.shuffle(sliced_paths)

        return sliced_paths

    def generate_transition_clips_batch(self, tasks: List[dict], batch_size: int = 1) -> Dict[str, Path]:
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
        if not tasks:
            return {}
        
        torch = _get_torch()
        
        print(f"[Wan FLF2V] Generating {len(tasks)} transition clips...")
        
        self._unload_t2v_pipe()
        
        results = {}
        
        for i, t in enumerate(tasks):
            if self.max_flf2v_count >= 0 and self.flf2v_usage_count >= self.max_flf2v_count:
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
            except torch.cuda.OutOfMemoryError:
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
        start_img, 
        end_img, 
        duration: float
    ) -> Path:
        torch = _get_torch()
        Image = _get_pil_image()
        from diffusers.utils import export_to_video
        
        filename = self.output_dir / f"transition_{index}.mp4"
        if filename.exists():
            return filename
        
        enhanced_prompt, negative_prompt, cfg = self.prepare_prompt(
            self.transition_prompt,
            style="elegant",
            is_transition=True
        )
        
        self.actual_enhanced_prompts["transition"] = enhanced_prompt
        self.actual_enhanced_prompts["transition_negative"] = negative_prompt
        
        print(f"[Wan FLF2V] Generating Transition {index} (Duration: {duration:.2f}s)...")
        
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

        seed = random.randint(0, 2**32 - 1)
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
        
        temp_filename = self.output_dir / f"transition_{index}_temp.mp4"
        export_to_video(output.frames[0], str(temp_filename), fps=fps)
        
        self._resize_and_adjust_video(temp_filename, filename, self.width, self.height, duration)
        
        if temp_filename.exists():
            temp_filename.unlink()
        
        return filename

    def _generate_transitions_bim(self, tasks: List[dict]) -> Dict[str, Path]:
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
        start_img,
        end_img,
        duration: float
    ) -> Path:
        torch = _get_torch()
        cv2 = _get_cv2()
        
        filename = self.output_dir / f"transition_{index}.mp4"
        if filename.exists():
            return filename

        print(f"[BiM-VFI] Generating Transition {index} (Duration: {duration:.2f}s)...")

        model = self._get_bim_model()

        start_img = start_img.resize((self.width, self.height))
        end_img = end_img.resize((self.width, self.height))

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
            num_frames = 2

        frames = []

        with torch.no_grad():
            h, w = img0.shape[2], img0.shape[3]
            if h >= 2160: scale_factor = 0.25; pyr_level = 7
            elif h >= 1080: scale_factor = 0.5; pyr_level = 6
            else: scale_factor = 1; pyr_level = 5

            for frame_idx in range(num_frames):
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
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", f"scale={target_width}:{target_height}",
            "-t", str(target_duration),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def generate_transition_clip(
        self,
        index: str,
        start_img,
        end_img,
        duration: float
    ) -> Path:
        torch = _get_torch()
        mode = self.interpolation_mode

        flf2v_available = True
        if self.max_flf2v_count >= 0 and self.flf2v_usage_count >= self.max_flf2v_count:
            flf2v_available = False

        if mode == "hybrid":
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
        else:
            return self._generate_single_bim_transition(index, start_img, end_img, duration)


# ==========================================
# Video Assembly
# ==========================================
def assemble_video(timeline: List[Dict], output_path: Path, fps: int = 30) -> Path:
    """Assembles clips using ffmpeg with speed adjustment."""
    print(f"[Assembly] Processing segments and stitching...")
    
    output_path = Path(output_path)
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
        
        if target_dur <= 0:
            continue
        
        try:
            probe = subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", str(src_file)
            ])
            src_dur = float(probe.strip())
        except Exception:
            src_dur = target_dur
        
        trim_start = item.get('trim_start', 0.0)
        trim_duration = item.get('trim_duration', None)
        
        if trim_duration is not None:
            extracted_dur = trim_duration
            if extracted_dur < target_dur * 0.9:
                speed_factor = extracted_dur / target_dur
                filter_str = f"setpts={1/speed_factor:.6f}*PTS"
                cmd = [
                    "ffmpeg", "-y", "-ss", str(trim_start), "-i", str(src_file),
                    "-t", str(trim_duration), "-filter:v", filter_str,
                    "-t", str(target_dur), "-r", str(fps), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(out_file)
                ]
            elif extracted_dur > target_dur * 1.1:
                speed_factor = extracted_dur / target_dur
                filter_str = f"setpts={1/speed_factor:.6f}*PTS"
                cmd = [
                    "ffmpeg", "-y", "-ss", str(trim_start), "-i", str(src_file),
                    "-t", str(trim_duration), "-filter:v", filter_str,
                    "-t", str(target_dur), "-r", str(fps), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(out_file)
                ]
            else:
                cmd = [
                    "ffmpeg", "-y", "-ss", str(trim_start), "-i", str(src_file),
                    "-t", str(trim_duration), "-r", str(fps), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(out_file)
                ]
        elif src_dur < target_dur * 0.9:
            speed_factor = src_dur / target_dur
            filter_str = f"setpts={1/speed_factor:.6f}*PTS"
            cmd = [
                "ffmpeg", "-y", "-i", str(src_file), "-filter:v", filter_str,
                "-t", str(target_dur), "-r", str(fps), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(out_file)
            ]
        elif src_dur > target_dur * 1.1:
            speed_factor = src_dur / target_dur
            filter_str = f"setpts={1/speed_factor:.6f}*PTS"
            cmd = [
                "ffmpeg", "-y", "-i", str(src_file), "-filter:v", filter_str,
                "-t", str(target_dur), "-r", str(fps), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(out_file)
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-i", str(src_file), "-t", str(target_dur),
                "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_file)
            ]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if out_file.exists() and out_file.stat().st_size > 0:
            processed_files.append(out_file)

    if not processed_files:
        print("[Error] No segments were processed!")
        return temp_video
    
    list_file = temp_dir / "files.txt"
    with open(list_file, "w") as f:
        for p in processed_files:
            f.write(f"file '{p.resolve()}'\n")
    
    cmd_concat = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(temp_video)
    ]
    subprocess.run(cmd_concat, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    return temp_video


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    """Combine video and audio."""
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
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✓ Final video saved: {output_path}")
    return result.returncode == 0


def trim_audio(input_path: Path, output_path: Path, start: float, end: Optional[float] = None) -> Optional[Path]:
    """Trim audio file."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"[Error] Audio input not found: {input_path}")
        return None

    cmd = ["ffmpeg", "-y", "-i", str(input_path), "-ss", str(start)]
    if end is not None:
        cmd.extend(["-t", str(end - start)])
    cmd.extend(["-c", "copy", str(output_path)])

    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0 or not output_path.exists():
        return None

    return output_path


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
