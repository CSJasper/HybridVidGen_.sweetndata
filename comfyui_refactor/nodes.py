# nodes.py - ComfyUI Custom Nodes for Beat-Reactive Video Generation
# Wraps the beat_hybrid_vidgen library for ComfyUI integration

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch

# Lazy import of library to avoid heavy loading at import time
_lib = None

def _get_lib():
    global _lib
    if _lib is None:
        from . import lib
        _lib = lib
    return _lib


# ============================================
# Custom Types for ComfyUI
# ============================================
class BeatHybridTypes:
    """Custom type definitions for this node package."""
    BEAT_DATA = "BEAT_DATA"
    TIMELINE = "TIMELINE"
    VIDEO_GENERATOR = "VIDEO_GENERATOR"
    PROMPT_ENHANCER = "PROMPT_ENHANCER"
    PEAK_CLIPS = "PEAK_CLIPS"


# ============================================
# Beat Detection Node
# ============================================
class BeatDetectionNode:
    """
    Detect beats from an audio file using madmom's RNN-based beat tracker.
    Returns beat times, types, and strengths for scene scheduling.
    """
    
    CATEGORY = "beat_hybrid_vidgen/analysis"
    RETURN_TYPES = (BeatHybridTypes.BEAT_DATA, "FLOAT")
    RETURN_NAMES = ("beat_data", "duration")
    FUNCTION = "detect_beats"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "audio_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "audio_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
            }
        }
    
    def detect_beats(self, audio_path: str, audio_start: float = 0.0, audio_end: float = 0.0):
        lib = _get_lib()
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Handle audio trimming if needed
        process_audio_path = audio_path
        if audio_start > 0 or audio_end > 0:
            work_dir = audio_path.parent / "beat_hybrid_temp"
            work_dir.mkdir(parents=True, exist_ok=True)
            trimmed_audio = work_dir / "trimmed_audio.mp3"
            
            end_val = audio_end if audio_end > 0 else None
            result = lib.trim_audio(audio_path, trimmed_audio, audio_start, end_val)
            if result:
                process_audio_path = trimmed_audio
        
        # Detect beats
        beat_times, beat_types, beat_strengths = lib.detect_beats_madmom(str(process_audio_path))
        duration = lib.get_audio_duration(str(process_audio_path))
        
        beat_data = {
            "beat_times": beat_times.tolist(),
            "beat_types": beat_types.tolist(),
            "beat_strengths": beat_strengths.tolist(),
            "audio_path": str(process_audio_path),
            "duration": duration
        }
        
        return (beat_data, duration)


# ============================================
# Scene Scheduler Node
# ============================================
class SceneSchedulerNode:
    """
    Plan video scenes based on detected beats.
    Creates a timeline with peak and transition segments.
    """
    
    CATEGORY = "beat_hybrid_vidgen/planning"
    RETURN_TYPES = (BeatHybridTypes.TIMELINE, "INT")
    RETURN_NAMES = ("timeline", "peak_count")
    FUNCTION = "plan_scenes"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "beat_data": (BeatHybridTypes.BEAT_DATA,),
                "peak_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "peak_clip_length": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1}),
            }
        }
    
    def plan_scenes(self, beat_data: Dict, peak_threshold: float, peak_clip_length: float):
        lib = _get_lib()
        
        beat_times = np.array(beat_data["beat_times"])
        beat_strengths = np.array(beat_data["beat_strengths"])
        duration = beat_data["duration"]
        
        scheduler = lib.SceneScheduler(duration, beat_times, beat_strengths)
        timeline = scheduler.plan_scenes(
            peak_threshold=peak_threshold,
            peak_clip_len=peak_clip_length
        )
        
        peak_count = sum(1 for item in timeline if item['type'] == 'peak')
        
        timeline_data = {
            "timeline": timeline,
            "audio_path": beat_data.get("audio_path", ""),
            "duration": duration,
            "peak_count": peak_count
        }
        
        return (timeline_data, peak_count)


# ============================================
# Prompt Enhancer Node
# ============================================
class PromptEnhancerNode:
    """
    Initialize LLM-based prompt enhancement system.
    Expands simple prompts into detailed video generation prompts.
    """
    
    CATEGORY = "beat_hybrid_vidgen/prompt"
    RETURN_TYPES = (BeatHybridTypes.PROMPT_ENHANCER,)
    RETURN_NAMES = ("prompt_enhancer",)
    FUNCTION = "create_enhancer"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_extension": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "base_url": ("STRING", {"default": "https://gateway.letsur.ai/v1", "multiline": False}),
                "model": ("STRING", {"default": "claude-sonnet-4-5-20250929", "multiline": False}),
                "min_prompt_words": ("INT", {"default": 15, "min": 5, "max": 100}),
            }
        }
    
    def create_enhancer(
        self, 
        enable_extension: bool,
        api_key: str = "",
        base_url: str = "https://gateway.letsur.ai/v1",
        model: str = "claude-sonnet-4-5-20250929",
        min_prompt_words: int = 15
    ):
        lib = _get_lib()
        
        # Try to get API key from environment if not provided
        if not api_key:
            api_key = os.environ.get("LLM_API_KEY", "")
        
        enhancer = lib.PromptEnhancer(
            api_key=api_key,
            base_url=base_url,
            model=model,
            enable_extension=enable_extension,
            min_prompt_words=min_prompt_words
        )
        
        return (enhancer,)


# ============================================
# Enhance Prompt Node
# ============================================
class EnhancePromptNode:
    """
    Enhance a single prompt using the LLM enhancer.
    """
    
    CATEGORY = "beat_hybrid_vidgen/prompt"
    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("enhanced_prompt", "negative_prompt", "recommended_cfg")
    FUNCTION = "enhance"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "style": (["energetic", "nature", "abstract", "cyberpunk", "elegant", "cosmic"],),
            },
            "optional": {
                "prompt_enhancer": (BeatHybridTypes.PROMPT_ENHANCER,),
                "negative_style": (["default", "motion_focused", "cinematic", "abstract"],),
            }
        }
    
    def enhance(
        self, 
        prompt: str, 
        style: str,
        prompt_enhancer=None,
        negative_style: str = "default"
    ):
        lib = _get_lib()
        
        if prompt_enhancer:
            result = prompt_enhancer.process_prompt(
                prompt=prompt,
                style=style,
                negative_style=negative_style
            )
            return (
                result['enhanced_prompt'],
                result['negative_prompt'],
                result['recommended_cfg']
            )
        else:
            # Return defaults without enhancement
            negative_prompt = lib.PromptEnhancer.NEGATIVE_PROMPTS.get(negative_style, lib.PromptEnhancer.NEGATIVE_PROMPTS["default"])
            return (prompt, negative_prompt, 6.0)


# ============================================
# Video Generator Loader Node
# ============================================
class VideoGeneratorLoaderNode:
    """
    Initialize the Wan Video Generator with model configuration.
    Supports T2V, FLF2V, and BiM-VFI interpolation.
    """
    
    CATEGORY = "beat_hybrid_vidgen/generator"
    RETURN_TYPES = (BeatHybridTypes.VIDEO_GENERATOR,)
    RETURN_NAMES = ("generator",)
    FUNCTION = "load_generator"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_dir": ("STRING", {"default": "workspace_wan", "multiline": False}),
                "model": (["wan2.1-1.3b", "wan2.2-5b", "wan2.2-14b"],),
                "width": ("INT", {"default": 832, "min": 256, "max": 1920, "step": 16}),
                "height": ("INT", {"default": 480, "min": 256, "max": 1080, "step": 16}),
                "interpolation_mode": (["hybrid", "flf2v", "bim"],),
            },
            "optional": {
                "prompt_enhancer": (BeatHybridTypes.PROMPT_ENHANCER,),
                "hybrid_threshold": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 5.0, "step": 0.1}),
                "max_flf2v_count": ("INT", {"default": -1, "min": -1, "max": 100}),
                "hf_token": ("STRING", {"default": "", "multiline": False}),
                "t2v_model_path": ("STRING", {"default": "", "multiline": False}),
                "flf2v_model_path": ("STRING", {"default": "", "multiline": False}),
                "flf2v_model": ("STRING", {"default": "Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers", "multiline": False}),
                "cache_dir": ("STRING", {"default": "", "multiline": False}),
                "local_files_only": ("BOOLEAN", {"default": False}),
                "cpu_offload": (["model", "sequential", "none"],),
                "device_map": (["none", "balanced"],),
                "auto_cfg": ("BOOLEAN", {"default": True}),
                "base_cfg": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "use_negative_prompt": ("BOOLEAN", {"default": True}),
            }
        }
    
    def load_generator(
        self,
        output_dir: str,
        model: str,
        width: int,
        height: int,
        interpolation_mode: str,
        prompt_enhancer=None,
        hybrid_threshold: float = 1.5,
        max_flf2v_count: int = -1,
        hf_token: str = "",
        t2v_model_path: str = "",
        flf2v_model_path: str = "",
        flf2v_model: str = "Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers",
        cache_dir: str = "",
        local_files_only: bool = False,
        cpu_offload: str = "model",
        device_map: str = "none",
        auto_cfg: bool = True,
        base_cfg: float = 6.0,
        use_negative_prompt: bool = True
    ):
        lib = _get_lib()
        
        # Model ID mapping
        model_map = {
            "wan2.1-1.3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "wan2.2-5b": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "wan2.2-14b": "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        }
        
        model_id = t2v_model_path if t2v_model_path else model_map.get(model, model_map["wan2.2-14b"])
        flf2v_model_id = flf2v_model_path if flf2v_model_path else flf2v_model
        
        # Get tokens from environment if not provided
        if not hf_token:
            hf_token = os.environ.get("HF_TOKEN", None)
        
        generator = lib.WanVideoGenerator(
            output_dir=Path(output_dir),
            model_id=model_id,
            width=width,
            height=height,
            interpolation_mode=interpolation_mode,
            hybrid_threshold=hybrid_threshold,
            flf2v_model_id=flf2v_model_id,
            max_flf2v_count=max_flf2v_count,
            hf_token=hf_token if hf_token else None,
            cache_dir=cache_dir if cache_dir else None,
            local_files_only=local_files_only,
            cpu_offload_mode=cpu_offload,
            device_map=device_map if device_map != "none" else None,
            prompt_enhancer=prompt_enhancer,
            auto_cfg=auto_cfg,
            base_cfg=base_cfg,
            use_negative_prompt=use_negative_prompt
        )
        
        return (generator,)


# ============================================
# Set Generator Prompts Node
# ============================================
class SetGeneratorPromptsNode:
    """
    Set custom prompts for peak and transition generation,
    or use preset styles.
    """
    
    CATEGORY = "beat_hybrid_vidgen/generator"
    RETURN_TYPES = (BeatHybridTypes.VIDEO_GENERATOR,)
    RETURN_NAMES = ("generator",)
    FUNCTION = "set_prompts"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generator": (BeatHybridTypes.VIDEO_GENERATOR,),
                "prompt_style": (["energetic", "nature", "abstract", "cyberpunk", "elegant", "cosmic", "custom"],),
            },
            "optional": {
                "custom_peak_prompt": ("STRING", {"default": "", "multiline": True}),
                "custom_transition_prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }
    
    def set_prompts(
        self,
        generator,
        prompt_style: str,
        custom_peak_prompt: str = "",
        custom_transition_prompt: str = ""
    ):
        lib = _get_lib()
        
        if prompt_style != "custom":
            preset = lib.PROMPT_PRESETS.get(prompt_style, lib.PROMPT_PRESETS["energetic"])
            generator.peak_prompt = preset["peak"]
            generator.transition_prompt = preset["transition"]
        
        if custom_peak_prompt:
            generator.peak_prompt = custom_peak_prompt
        if custom_transition_prompt:
            generator.transition_prompt = custom_transition_prompt
        
        return (generator,)


# ============================================
# Generate Peak Clips Node
# ============================================
class GeneratePeakClipsNode:
    """
    Generate peak video clips using Wan T2V model.
    """
    
    CATEGORY = "beat_hybrid_vidgen/generation"
    RETURN_TYPES = (BeatHybridTypes.PEAK_CLIPS,)
    RETURN_NAMES = ("peak_clips",)
    FUNCTION = "generate"
    OUTPUT_NODE = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generator": (BeatHybridTypes.VIDEO_GENERATOR,),
                "num_clips": ("INT", {"default": 5, "min": 1, "max": 20}),
                "clip_duration": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "style": (["energetic", "nature", "abstract", "cyberpunk", "elegant", "cosmic"],),
                "randomize_order": ("BOOLEAN", {"default": True}),
            }
        }
    
    def generate(
        self,
        generator,
        num_clips: int,
        clip_duration: float,
        style: str = "energetic",
        randomize_order: bool = True
    ):
        peak_paths = generator.generate_peak_clips_from_single_video(
            num_clips=num_clips,
            clip_duration=clip_duration,
            randomize_order=randomize_order,
            style=style
        )
        
        peak_clips = {
            "paths": [str(p) for p in peak_paths],
            "num_clips": len(peak_paths),
            "duration": clip_duration
        }
        
        return (peak_clips,)


# ============================================
# Assign Peaks to Timeline Node
# ============================================
class AssignPeaksToTimelineNode:
    """
    Assign generated peak clips to timeline peak positions.
    """
    
    CATEGORY = "beat_hybrid_vidgen/assembly"
    RETURN_TYPES = (BeatHybridTypes.TIMELINE,)
    RETURN_NAMES = ("timeline",)
    FUNCTION = "assign"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": (BeatHybridTypes.TIMELINE,),
                "peak_clips": (BeatHybridTypes.PEAK_CLIPS,),
            },
            "optional": {
                "randomize": ("BOOLEAN", {"default": True}),
            }
        }
    
    def assign(self, timeline: Dict, peak_clips: Dict, randomize: bool = True):
        import random
        
        timeline_list = timeline["timeline"]
        peak_paths = [Path(p) for p in peak_clips["paths"]]
        
        if randomize and len(peak_paths) > 1:
            random.shuffle(peak_paths)
        
        for item in timeline_list:
            if item['type'] == 'peak':
                item['file'] = random.choice(peak_paths) if randomize else peak_paths[0]
        
        timeline["timeline"] = timeline_list
        return (timeline,)


# ============================================
# Generate Transitions Node
# ============================================
class GenerateTransitionsNode:
    """
    Generate transition clips between peaks using FLF2V or BiM-VFI.
    """
    
    CATEGORY = "beat_hybrid_vidgen/generation"
    RETURN_TYPES = (BeatHybridTypes.TIMELINE,)
    RETURN_NAMES = ("timeline",)
    FUNCTION = "generate"
    OUTPUT_NODE = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generator": (BeatHybridTypes.VIDEO_GENERATOR,),
                "timeline": (BeatHybridTypes.TIMELINE,),
            },
            "optional": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
                "include_loop_transition": ("BOOLEAN", {"default": True}),
            }
        }
    
    def generate(
        self,
        generator,
        timeline: Dict,
        batch_size: int = 1,
        include_loop_transition: bool = True
    ):
        from PIL import Image
        import random
        
        timeline_list = timeline["timeline"]
        
        # Find first and last peaks for loop transition
        first_peak_idx = next((i for i, x in enumerate(timeline_list) if x['type'] == 'peak'), None)
        last_peak_idx = next((i for i, x in enumerate(reversed(timeline_list)) if x['type'] == 'peak'), None)
        if last_peak_idx is not None:
            last_peak_idx = len(timeline_list) - 1 - last_peak_idx
        
        # Generate loop transition if needed
        if include_loop_transition and first_peak_idx is not None and last_peak_idx is not None:
            first_peak_start_frame = generator.extract_frame(Path(timeline_list[first_peak_idx]['file']), at_start=True)
            last_peak_end_frame = generator.extract_frame(Path(timeline_list[last_peak_idx]['file']), at_start=False)
            
            has_intro_gap = (timeline_list[0]['type'] == 'gap')
            has_outro_gap = (timeline_list[-1]['type'] == 'gap')
            
            intro_dur = timeline_list[0]['duration'] if has_intro_gap else 0.0
            outro_dur = timeline_list[-1]['duration'] if has_outro_gap else 0.0
            
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
                    timeline_list[-1]['file'] = loop_clip
                    timeline_list[-1]['trim_start'] = 0.0
                    timeline_list[-1]['trim_duration'] = outro_dur
                    
                if has_intro_gap:
                    timeline_list[0]['file'] = loop_clip
                    timeline_list[0]['trim_start'] = outro_dur
                    timeline_list[0]['trim_duration'] = intro_dur
        
        # Generate standard transitions
        previous_last_frame = None
        transition_tasks = []
        
        for i, item in enumerate(timeline_list):
            if item['type'] == 'peak':
                previous_last_frame = generator.extract_frame(Path(item['file']), at_start=False)
                
            elif item['type'] == 'gap':
                if 'file' in item:
                    continue
                
                if previous_last_frame is None:
                    previous_last_frame = Image.new('RGB', (generator.width, generator.height), color=(0, 0, 0))
                    
                next_first_frame = None
                if i + 1 < len(timeline_list) and timeline_list[i+1]['type'] == 'peak':
                    next_first_frame = generator.extract_frame(Path(timeline_list[i+1]['file']), at_start=True)
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
            generated_files = generator.generate_transition_clips_batch(transition_tasks, batch_size=batch_size)
            
            for task in transition_tasks:
                key = task['index']
                idx = task['timeline_idx']
                if key in generated_files:
                    timeline_list[idx]['file'] = generated_files[key]
        
        timeline["timeline"] = timeline_list
        return (timeline,)


# ============================================
# Assemble Video Node
# ============================================
class AssembleVideoNode:
    """
    Assemble all clips into a single video based on timeline.
    """
    
    CATEGORY = "beat_hybrid_vidgen/assembly"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("temp_video_path",)
    FUNCTION = "assemble"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": (BeatHybridTypes.TIMELINE,),
                "output_path": ("STRING", {"default": "final_output.mp4", "multiline": False}),
            },
            "optional": {
                "fps": ("INT", {"default": 30, "min": 15, "max": 60}),
            }
        }
    
    def assemble(self, timeline: Dict, output_path: str, fps: int = 30):
        lib = _get_lib()
        
        output_path = Path(output_path)
        timeline_list = timeline["timeline"]
        
        temp_video = lib.assemble_video(timeline_list, output_path, fps=fps)
        
        return (str(temp_video),)


# ============================================
# Mux Audio Node
# ============================================
class MuxAudioNode:
    """
    Combine video with audio track to produce final output.
    """
    
    CATEGORY = "beat_hybrid_vidgen/assembly"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "mux"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"forceInput": True}),
                "output_path": ("STRING", {"default": "final_output.mp4", "multiline": False}),
            },
            "optional": {
                "audio_path": ("STRING", {"default": "", "multiline": False}),
                "timeline": (BeatHybridTypes.TIMELINE,),
            }
        }
    
    def mux(self, video_path: str, output_path: str, audio_path: str = "", timeline: Dict = None):
        lib = _get_lib()
        
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        # Get audio path from timeline if not provided
        if not audio_path and timeline:
            audio_path = timeline.get("audio_path", "")
        
        if not audio_path:
            raise ValueError("Audio path must be provided either directly or through timeline")
        
        audio_path = Path(audio_path)
        
        lib.mux_audio(video_path, audio_path, output_path)
        
        # Cleanup temp video
        if video_path.exists() and ".temp." in video_path.name:
            video_path.unlink()
        
        return (str(output_path),)


# ============================================
# Full Pipeline Node (All-in-One)
# ============================================
class BeatReactiveVideoPipelineNode:
    """
    Complete beat-reactive video generation pipeline.
    Combines all steps: beat detection, scene planning, 
    peak generation, transitions, assembly, and audio muxing.
    """
    
    CATEGORY = "beat_hybrid_vidgen"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "generate"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {"default": "", "multiline": False}),
                "output_path": ("STRING", {"default": "final_output.mp4", "multiline": False}),
                "work_dir": ("STRING", {"default": "workspace_wan", "multiline": False}),
                "model": (["wan2.1-1.3b", "wan2.2-5b", "wan2.2-14b"],),
                "prompt_style": (["energetic", "nature", "abstract", "cyberpunk", "elegant", "cosmic"],),
            },
            "optional": {
                "peak_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "peak_length": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "num_peak_clips": ("INT", {"default": 5, "min": 1, "max": 20}),
                "width": ("INT", {"default": 832, "min": 256, "max": 1920, "step": 16}),
                "height": ("INT", {"default": 480, "min": 256, "max": 1080, "step": 16}),
                "interpolation_mode": (["hybrid", "flf2v", "bim"],),
                "hybrid_threshold": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 5.0, "step": 0.1}),
                "audio_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "audio_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "cpu_offload": (["model", "sequential", "none"],),
                "base_cfg": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "custom_peak_prompt": ("STRING", {"default": "", "multiline": True}),
                "custom_transition_prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }
    
    def generate(
        self,
        audio_path: str,
        output_path: str,
        work_dir: str,
        model: str,
        prompt_style: str,
        peak_threshold: float = 0.5,
        peak_length: float = 2.0,
        num_peak_clips: int = 5,
        width: int = 832,
        height: int = 480,
        interpolation_mode: str = "hybrid",
        hybrid_threshold: float = 1.5,
        audio_start: float = 0.0,
        audio_end: float = 0.0,
        cpu_offload: str = "model",
        base_cfg: float = 6.0,
        custom_peak_prompt: str = "",
        custom_transition_prompt: str = ""
    ):
        lib = _get_lib()
        from PIL import Image
        import random
        
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        
        if not output_path.is_absolute():
            output_path = work_dir / output_path
        
        # Model mapping
        model_map = {
            "wan2.1-1.3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "wan2.2-5b": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "wan2.2-14b": "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        }
        
        hf_token = os.environ.get("HF_TOKEN", None)
        
        # Initialize generator
        generator = lib.WanVideoGenerator(
            output_dir=work_dir,
            model_id=model_map.get(model, model_map["wan2.2-14b"]),
            width=width,
            height=height,
            interpolation_mode=interpolation_mode,
            hybrid_threshold=hybrid_threshold,
            hf_token=hf_token,
            cpu_offload_mode=cpu_offload,
            base_cfg=base_cfg,
            use_negative_prompt=True
        )
        
        # Set prompts
        preset = lib.PROMPT_PRESETS.get(prompt_style, lib.PROMPT_PRESETS["energetic"])
        generator.peak_prompt = custom_peak_prompt if custom_peak_prompt else preset["peak"]
        generator.transition_prompt = custom_transition_prompt if custom_transition_prompt else preset["transition"]
        
        # Handle audio trimming
        process_audio_path = audio_path
        if audio_start > 0 or audio_end > 0:
            trimmed_audio = work_dir / "trimmed_audio.mp3"
            end_val = audio_end if audio_end > 0 else None
            result = lib.trim_audio(audio_path, trimmed_audio, audio_start, end_val)
            if result:
                process_audio_path = trimmed_audio
        
        # Detect beats
        beat_times, beat_types, beat_strengths = lib.detect_beats_madmom(str(process_audio_path))
        duration = lib.get_audio_duration(str(process_audio_path))
        
        # Plan timeline
        scheduler = lib.SceneScheduler(duration, beat_times, beat_strengths)
        timeline = scheduler.plan_scenes(peak_threshold=peak_threshold, peak_clip_len=peak_length)
        
        if not timeline:
            raise RuntimeError("No timeline generated. Check thresholds.")
        
        # Generate peak clips
        peak_paths = generator.generate_peak_clips_from_single_video(
            num_clips=num_peak_clips,
            clip_duration=peak_length,
            randomize_order=True,
            style=prompt_style
        )
        
        # Assign peaks to timeline
        for item in timeline:
            if item['type'] == 'peak':
                item['file'] = random.choice(peak_paths)
        
        # Generate loop transitions
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
        
        # Generate standard transitions
        previous_last_frame = None
        transition_tasks = []
        
        for i, item in enumerate(timeline):
            if item['type'] == 'peak':
                previous_last_frame = generator.extract_frame(item['file'], at_start=False)
            elif item['type'] == 'gap':
                if 'file' in item:
                    continue
                
                if previous_last_frame is None:
                    previous_last_frame = Image.new('RGB', (width, height), color=(0, 0, 0))
                    
                next_first_frame = None
                if i + 1 < len(timeline) and timeline[i+1]['type'] == 'peak':
                    next_first_frame = generator.extract_frame(timeline[i+1]['file'], at_start=True)
                else:
                    next_first_frame = Image.new('RGB', (width, height), color=(0, 0, 0))
                
                transition_tasks.append({
                    'timeline_idx': i,
                    'index': f"{i:04d}",
                    'start_img': previous_last_frame,
                    'end_img': next_first_frame,
                    'duration': item['duration']
                })
        
        if transition_tasks:
            generated_files = generator.generate_transition_clips_batch(transition_tasks, batch_size=1)
            for task in transition_tasks:
                key = task['index']
                idx = task['timeline_idx']
                if key in generated_files:
                    timeline[idx]['file'] = generated_files[key]
        
        # Assemble video
        temp_video = lib.assemble_video(timeline, output_path)
        
        # Mux audio
        lib.mux_audio(temp_video, process_audio_path, output_path)
        
        # Cleanup
        if temp_video.exists():
            temp_video.unlink()
        
        return (str(output_path),)


# ============================================
# Node Mappings Export
# ============================================
NODE_CLASS_MAPPINGS = {
    "beat_hybrid_vidgen.BeatDetection": BeatDetectionNode,
    "beat_hybrid_vidgen.SceneScheduler": SceneSchedulerNode,
    "beat_hybrid_vidgen.PromptEnhancer": PromptEnhancerNode,
    "beat_hybrid_vidgen.EnhancePrompt": EnhancePromptNode,
    "beat_hybrid_vidgen.VideoGeneratorLoader": VideoGeneratorLoaderNode,
    "beat_hybrid_vidgen.SetGeneratorPrompts": SetGeneratorPromptsNode,
    "beat_hybrid_vidgen.GeneratePeakClips": GeneratePeakClipsNode,
    "beat_hybrid_vidgen.AssignPeaksToTimeline": AssignPeaksToTimelineNode,
    "beat_hybrid_vidgen.GenerateTransitions": GenerateTransitionsNode,
    "beat_hybrid_vidgen.AssembleVideo": AssembleVideoNode,
    "beat_hybrid_vidgen.MuxAudio": MuxAudioNode,
    "beat_hybrid_vidgen.FullPipeline": BeatReactiveVideoPipelineNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "beat_hybrid_vidgen.BeatDetection": "Beat Detection (Madmom)",
    "beat_hybrid_vidgen.SceneScheduler": "Scene Scheduler",
    "beat_hybrid_vidgen.PromptEnhancer": "Prompt Enhancer (LLM)",
    "beat_hybrid_vidgen.EnhancePrompt": "Enhance Prompt",
    "beat_hybrid_vidgen.VideoGeneratorLoader": "Video Generator Loader",
    "beat_hybrid_vidgen.SetGeneratorPrompts": "Set Generator Prompts",
    "beat_hybrid_vidgen.GeneratePeakClips": "Generate Peak Clips",
    "beat_hybrid_vidgen.AssignPeaksToTimeline": "Assign Peaks to Timeline",
    "beat_hybrid_vidgen.GenerateTransitions": "Generate Transitions",
    "beat_hybrid_vidgen.AssembleVideo": "Assemble Video",
    "beat_hybrid_vidgen.MuxAudio": "Mux Audio",
    "beat_hybrid_vidgen.FullPipeline": "Beat-Reactive Video (Full Pipeline)",
}
