# Beat-Reactive Video Generation - ComfyUI Custom Nodes

A ComfyUI custom node package for generating beat-reactive music videos using Wan AI models. This package wraps the `beat_hybrid_vidgen` library to provide a visual node-based workflow for creating professional music visualizations.

## Features

- **Beat Detection**: RNN-based beat tracking using madmom
- **Scene Planning**: Automatic timeline generation based on beat strength
- **Peak Video Generation**: Wan T2V model for generating dynamic peak clips
- **Transition Generation**: Hybrid interpolation using FLF2V and BiM-VFI
- **LLM Prompt Enhancement**: Optional LLM-based prompt expansion
- **Video Assembly**: FFmpeg-based clip assembly and audio muxing

## Installation

1. Clone or copy this folder to your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone <repository_url> beat_hybrid_vidgen_nodes
   ```

2. Install dependencies:
   ```bash
   cd beat_hybrid_vidgen_nodes
   pip install -r requirements.txt
   ```

3. (Optional) For BiM-VFI interpolation, clone the BiM-VFI repository:
   ```bash
   git clone https://github.com/your-repo/BiM-VFI
   ```

4. Restart ComfyUI

## Nodes

### Analysis Nodes

#### Beat Detection (Madmom)
Detects beats from an audio file using madmom's RNN-based beat tracker.

**Inputs:**
- `audio_path` (STRING): Path to audio file
- `audio_start` (FLOAT, optional): Start time in seconds
- `audio_end` (FLOAT, optional): End time in seconds

**Outputs:**
- `beat_data`: Beat timing and strength data
- `duration`: Audio duration in seconds

---

### Planning Nodes

#### Scene Scheduler
Plans video scenes based on detected beats, creating a timeline with peak and transition segments.

**Inputs:**
- `beat_data`: From Beat Detection node
- `peak_threshold` (FLOAT): Beat strength threshold (0.0-1.0)
- `peak_clip_length` (FLOAT): Duration of peak clips in seconds

**Outputs:**
- `timeline`: Scene timeline data
- `peak_count`: Number of peaks detected

---

### Prompt Nodes

#### Prompt Enhancer (LLM)
Initializes the LLM-based prompt enhancement system.

**Inputs:**
- `enable_extension` (BOOLEAN): Enable prompt extension
- `api_key` (STRING, optional): LLM API key (or set LLM_API_KEY env var)
- `base_url` (STRING, optional): API base URL
- `model` (STRING, optional): LLM model name
- `min_prompt_words` (INT, optional): Minimum words before extension

**Outputs:**
- `prompt_enhancer`: Enhancer instance

#### Enhance Prompt
Enhances a single prompt using the LLM enhancer.

**Inputs:**
- `prompt` (STRING): Input prompt
- `style`: Visual style preset
- `prompt_enhancer` (optional): From Prompt Enhancer node
- `negative_style` (optional): Negative prompt style

**Outputs:**
- `enhanced_prompt`: Extended prompt
- `negative_prompt`: Optimized negative prompt
- `recommended_cfg`: Suggested CFG value

---

### Generator Nodes

#### Video Generator Loader
Initializes the Wan Video Generator with model configuration.

**Inputs:**
- `output_dir` (STRING): Working directory
- `model`: Model selection (wan2.1-1.3b, wan2.2-5b, wan2.2-14b)
- `width`, `height` (INT): Video dimensions
- `interpolation_mode`: Interpolation method (hybrid, flf2v, bim)
- Various optional configuration parameters

**Outputs:**
- `generator`: Video generator instance

#### Set Generator Prompts
Configures prompts for the generator using presets or custom prompts.

**Inputs:**
- `generator`: From Video Generator Loader
- `prompt_style`: Style preset
- `custom_peak_prompt` (optional): Custom peak prompt
- `custom_transition_prompt` (optional): Custom transition prompt

**Outputs:**
- `generator`: Configured generator

---

### Generation Nodes

#### Generate Peak Clips
Generates peak video clips using the Wan T2V model.

**Inputs:**
- `generator`: From Video Generator Loader
- `num_clips` (INT): Number of clips to generate
- `clip_duration` (FLOAT): Duration of each clip
- `style` (optional): Visual style
- `randomize_order` (optional): Shuffle clip order

**Outputs:**
- `peak_clips`: Generated clip paths

#### Generate Transitions
Generates transition clips between peaks using FLF2V or BiM-VFI.

**Inputs:**
- `generator`: From Video Generator Loader
- `timeline`: From Assign Peaks to Timeline
- `batch_size` (optional): Batch processing size
- `include_loop_transition` (optional): Generate intro/outro loop

**Outputs:**
- `timeline`: Updated timeline with transitions

---

### Assembly Nodes

#### Assign Peaks to Timeline
Assigns generated peak clips to timeline peak positions.

**Inputs:**
- `timeline`: From Scene Scheduler
- `peak_clips`: From Generate Peak Clips
- `randomize` (optional): Random assignment

**Outputs:**
- `timeline`: Updated timeline

#### Assemble Video
Assembles all clips into a single video based on timeline.

**Inputs:**
- `timeline`: From Generate Transitions
- `output_path` (STRING): Output video path
- `fps` (optional): Frame rate

**Outputs:**
- `temp_video_path`: Path to assembled video (without audio)

#### Mux Audio
Combines video with audio track to produce final output.

**Inputs:**
- `video_path`: From Assemble Video
- `output_path` (STRING): Final output path
- `audio_path` (optional): Audio file path
- `timeline` (optional): For automatic audio path

**Outputs:**
- `output_path`: Final video path

---

### All-in-One Node

#### Beat-Reactive Video (Full Pipeline)
Complete pipeline combining all steps in a single node.

**Inputs:**
- `audio_path`: Input audio file
- `output_path`: Output video path
- `work_dir`: Working directory
- `model`: Wan model selection
- `prompt_style`: Visual style
- Various optional parameters for fine-tuning

**Outputs:**
- `output_path`: Final video path

---

## Example Workflows

### Basic Workflow (Modular)

```
[Audio Path] → [Beat Detection] → [Scene Scheduler] → [Assign Peaks]
                                                           ↓
[Generator Loader] → [Set Prompts] → [Generate Peak Clips] ↓
                           ↓                               ↓
                    [Generate Transitions] ← ────────────────
                           ↓
                    [Assemble Video] → [Mux Audio] → [Output]
```

### Simple Workflow (All-in-One)

```
[Audio Path] → [Beat-Reactive Video (Full Pipeline)] → [Output]
```

## Environment Variables

- `HF_TOKEN`: HuggingFace API token for model downloads
- `LLM_API_KEY`: API key for LLM prompt enhancement

## Model Requirements

- **T2V Models**: Wan-AI/Wan2.1-T2V-1.3B-Diffusers, Wan-AI/Wan2.2-T2V-A14B-Diffusers
- **FLF2V Models**: Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers
- **BiM-VFI**: Requires separate installation

## GPU Requirements

- Minimum: 8GB VRAM with CPU offload
- Recommended: 24GB+ VRAM for best performance
- Multi-GPU: Use `device_map="balanced"` option

## License

MIT License
