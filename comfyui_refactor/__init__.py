# __init__.py - ComfyUI Custom Node Package: Beat-Reactive Video Generation
# 
# This package wraps the beat_hybrid_vidgen library for ComfyUI integration.
# It provides nodes for beat detection, scene planning, video generation,
# and audio muxing to create beat-reactive music videos.
#
# Usage:
#   1. Install to ComfyUI custom_nodes directory
#   2. Install requirements: pip install -r requirements.txt
#   3. Restart ComfyUI
#
# Nodes provided:
#   - Beat Detection (Madmom) - Detect beats from audio
#   - Scene Scheduler - Plan video scenes based on beats
#   - Prompt Enhancer (LLM) - Initialize LLM prompt enhancement
#   - Enhance Prompt - Enhance a single prompt
#   - Video Generator Loader - Load Wan video generator
#   - Set Generator Prompts - Configure prompts for generation
#   - Generate Peak Clips - Generate peak video clips
#   - Assign Peaks to Timeline - Assign clips to timeline
#   - Generate Transitions - Generate transition clips
#   - Assemble Video - Combine clips into video
#   - Mux Audio - Add audio to video
#   - Beat-Reactive Video (Full Pipeline) - All-in-one node

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
