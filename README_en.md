# AnchorFlux

> High-precision subtitle generation engine based on dual-modal spatiotemporal decoupled architecture.

[![Chinese README](https://img.shields.io/badge/README-中文-blue.svg)](README.md)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-v3.1.1-brightgreen.svg)
![Vue](https://img.shields.io/badge/Vue-3.5+-4FC08D?logo=vue.js&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)

<img src="https://cdn.jsdelivr.net/gh/CommentOut64/anchor-flux@main/assets/pic2.png" image-rendering: crisp-edges/>

AnchorFlux employs an innovative dual-anchor architecture: **SenseVoice anchors time boundaries, Whisper anchors semantic content**, coordinated through an asynchronous dual-stream pipeline to achieve a balance between transcription speed and quality.

<a href="https://www.bilibili.com/video/BV1xAqzB5EFN"><img src="https://img.shields.io/static/v1?label=%20&message=Demo%20Video&color=F37697&style=flat&logo=bilibili&logoColor=white&logoWidth=20" height="32"></a>
<a href="https://www.bilibili.com/video/BV1ejvCBPEz9"><img src="https://img.shields.io/static/v1?label=%20&message=Tutorial%20Video&color=F37697&style=flat&logo=bilibili&logoColor=white&logoWidth=20" height="32"></a>

## Features

### Core Features

- **Dual-Modal Alignment**: SenseVoice + Whisper collaborative work
- **Fast-Slow Dual Stream**: Draft displayed quickly, finalized output completed in background with automatic replacement
- **Spatiotemporal Decoupling**: Subtitle text and timestamps processed separately, leveraging strengths of each
- **Intelligent Vocal Separation**: Demucs on-demand processing to improve recognition quality
- **Word-Level Timestamps**: CTC precise boundary detection, word-level time alignment
- **Clean and Efficient Editor**: Designed specifically for video editors, beautiful interface, convenient operation

### Editor Features

- **Real-time Subtitle Overlay Preview**: Built-in video player with real-time subtitle overlay rendering, what you see is what you get
- **High-precision Waveform Visualization**: Audio waveform rendering based on WaveSurfer.js, intuitively displaying speech activity and silent intervals
- **Intuitive Subtitle Range Editing**: Visual "subtitle range boxes" on the waveform, supporting direct mouse drag at edges to quickly adjust subtitle start and end times
- **Interactive Subtitle List**: Complete subtitle editing panel, support clicking subtitle items to jump to corresponding video progress with one click, support freely modifying text content and timestamps, support quickly inserting new subtitles or deleting existing ones
- **Precise Subtitle Splitting**: Quick split functionality based on waveform timeline points or text cursor position
- **One-Click Multi-Format Export**: Export subtitles to SRT, ASS, VTT standard subtitle formats and plain text files with one click
- **History and Auto-Save**: Support unlimited undo and redo, all modifications automatically saved in real-time, no worry about data loss

### User Interface

- **Modern Web UI**: Responsive interface based on Vue.js 3
- **Real-time Dual Stream Preview**: Draft in italic, finalized in regular, status at a glance
- **Drag-and-Drop Upload**: Support drag-and-drop upload of various video formats
- **Real-time Progress Push**: Server-sent events stream, automatic reconnection on disconnect, real-time transcription progress updates

### Technical Features

- **Frontend-Backend Separation**: Vue.js + FastAPI architecture
- **Pipeline Decoupling**: Single responsibility principle, each module focuses on one thing
- **VRAM Adaptive**: Dynamically adjust processing strategy based on GPU memory
- **CPU-Specific Optimization**: Using ONNX quantized models, pure CPU inference speed still impressive
- **Resume from Breakpoint**: Tasks can resume from breakpoint after interruption

### Extended Features (Planned)

- **LLM Proofreading**: Large language model semantic proofreading [Not Implemented]
- **LLM Translation**: Multi-language subtitle translation, directly generate bilingual subtitles [Not Implemented]

## System Requirements

### Basic Requirements
- **OS:** Windows 10/11
- **GPU:** NVIDIA GPU supporting CUDA 11.8+ (at least 4GB VRAM recommended for best performance)
- **Important Note: Even without dedicated graphics card, you can still use express mode (SenseVoice only), but cannot use Whisper model**
- **Memory:** 16GB+ recommended

### Local Deployment Requirements
- **Python:** 3.10+
- **Node.js:** 21+

## Quick Start

### Integrated Package
1. **Download integrated package from release or provided cloud storage**
2. **Extract, double-click One-Click Start.bat**
3. **Wait for automatic dependency download to complete (maintain network connectivity, takes 10-20 minutes)**
4. **Will automatically start and jump to browser page after completion**
5. **Use One-Click Start.bat for next startup**

### Manual Installation
1. **Clone repository**
```bash
git clone https://github.com/CommentOut64/anchor-flux.git
cd anchor-flux-main
```
2. **Install CUDA and cuDNN**
   - Download and install [CUDA 11.8+](https://developer.nvidia.com/cuda-11-8-0-download-archive)
   - Download and install [cuDNN 8](https://developer.nvidia.com/rdp/cudnn-archive)
   - Verify installation: `nvidia-smi` and `nvcc --version`

3. **Run startup script**
```bash
# Run
run.bat
```

### Switching Whisper Models

**Method 1: Auto-download via Configuration (Recommended)**

Edit the `.env` file in the project root, modify the `WHISPER_MODEL` parameter and restart the service. The system will automatically download the model from HuggingFace:
```bash
WHISPER_MODEL=large-v3  # Options: tiny, base, small, medium, large-v3, turbo
```

**Method 2: Manual Download**

Download Faster-Whisper model files from HuggingFace and place them in the specified directory:

1. Visit the model repository (e.g., large-v3): https://huggingface.co/Systran/faster-whisper-large-v3
2. Download the following required files:
   - `model.bin` - Model weights (required)
   - `config.json` - Model configuration (required)
   - `tokenizer.json` - Tokenizer (required)
   - `vocabulary.txt` or `vocabulary.json` - Vocabulary (required)
3. Create the following path in the project directory and place the files:
   ```
   backend/models/huggingface/models--Systran--faster-whisper-large-v3/snapshots/<any-hash-name>/
   ├── model.bin
   ├── config.json
   ├── tokenizer.json
   └── vocabulary.txt
   ```
   > Tip: `<any-hash-name>` can be any string, such as `main` or `v1`

**Available Models:**

| Model | HuggingFace Repository | VRAM Required |
|-------|------------------------|---------------|
| tiny | Systran/faster-whisper-tiny | ~1GB |
| base | Systran/faster-whisper-base | ~1GB |
| small | Systran/faster-whisper-small | ~2GB |
| medium | Systran/faster-whisper-medium | ~5GB |
| large-v3 | Systran/faster-whisper-large-v3 | ~10GB (float16) / ~6GB (int8) |
| turbo | Systran/faster-whisper-large-v3-turbo | ~6GB |

### Runtime Modes

The system supports two runtime modes, switched via the `DEV_MODE` parameter in the `.env` file:

| Mode | DEV_MODE | Frontend | Backend | Access URL |
|------|----------|----------|---------|------------|
| **Production** | `false` (default) | Pre-built static files hosted by backend | Port 8000 | http://localhost:8000 |
| **Development** | `true` | `npm run dev` with hot reload (Port 5173) | Port 8000 | http://localhost:5173 |

- **Production Mode**: Frontend uses pre-built files from `frontend/dist`, served by FastAPI static file hosting, suitable for daily use
- **Development Mode**: Frontend uses Vite dev server with hot reload, suitable for development and debugging

## Technology Stack

### Backend

* **FastAPI** - Async core, providing high-performance streaming interface
* **PyTorch** - Deep learning framework
* **Uvicorn** - ASGI server
* **SSE** - Real-time transmission, streaming push of recognition results

### Frontend

* **Vue 3 / Vite** - Reactive framework
* **WaveSurfer.js** - Waveform visualization
* **Pinia** - State management
* **Element Plus** - Interactive components, dark mode adapted
* **EventSource** - Auto-reconnection, ensuring stable streaming communication

### AI Models

* **Whisper** - Semantic recognition
* **SenseVoice** - Time anchoring and draft, CTC word-level timestamps
* **Silero VAD** - Voice activity detection, implementing intelligent segmentation
* **Demucs** - Vocal separation, eliminating background noise
* **YAMNet** - Audio classification, identifying non-speech environmental sounds

## Architecture Overview

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                 AnchorFlux Dual-Anchor Streaming Transcription Architecture                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Preprocessing Stage                                                            │
│                                                                          │
│    Video Input                                                              │
│       ↓                                                                  │
│    Audio Extraction (FFmpeg 16kHz mono)                                         │
│       ↓                                                                  │
│    Spectral Triage (YAMNet Probe Mode)                                           │
│       ↓                                                                  │
│    Vocal Separation (Demucs On-Demand/Global)                                          │
│       ↓                                                                  │
│    VAD Segmentation (Silero VAD)                                                │
│       ↓                                                                  │
│    Smart Accumulation (Average 12s, Max 30s)                                          │
│       ↓                                                                  │
│    AudioChunk[] (Contains spectral features, separation level)                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 2. Fast-Slow Dual Stream Transcription (Async Pipeline)                                            │
│                                                                          │
│  ┌──────────────────────┐       ┌──────────────────────┐               │
│  │  FastWorker (CPU)    │       │  SlowWorker (GPU)    │               │
│  │  SenseVoice ONNX     │       │  Whisper Large-v3    │               │
│  └──────────┬───────────┘       └──────────┬───────────┘               │
│             │                              │                            │
│             │  Draft Subtitle                    │  Patch Subtitle                  │
│             │  SSE Push                    │  SSE Push                  │
│             │    Italic                      │    Regular                   │
│             │                              │                            │
│             └──────────┬───────────────────┘                            │
│                        ↓                                                │
│              ┌──────────────────┐                                       │
│              │ AlignmentWorker  │                                       │
│              │ Text Timestamp Alignment   │                                       │
│              │ Word-Level Timestamp Calculation   │                                       │
│              └────────┬─────────┘                                       │
│                       ↓                                                 │
│                  Final Subtitles                                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 3. Frontend Real-time Rendering                                                          │
│                                                                          │
│  SSE Event Stream → Dual Stream Progress Bar + Subtitle List + Waveform                            │
│                                                                          │
│  Progress Display:                                                            │
│  ┌────────────────────────────────────────┐                            │
│  │ ████████░░░░ 60%                       │                            │
│  │ ├ SenseVoice: ████████░░ 75%          │                            │
│  │ └ Whisper:    ██████░░░░ 50%          │                            │
│  └────────────────────────────────────────┘                            │
│                                                                          │
│  Real-time Subtitle Update:                                                          │
│         It's still only 7:28 pm         ← Draft                        │
│         There's plenty of time          ← Finalized                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 4. Intelligent Patching and Fuse Breaking                                                        │
│                                                                          │
│  Trigger Conditions: Low Confidence / Short Segment / Single Character / Word-Level Check                        │
│      ↓                                                                  │
│  Whisper Secondary Auscultation                                                       │
│      ↓                                                                  │
│  Fuse Breaker (FuseBreakerV2)                                             │
│      ↓                                                                  │
│  Upgrade Path: NONE → HTDEMUCS                                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

```

## Core Architecture

The system adopts the "spatiotemporal decoupling" philosophy, separating the definition of time boundaries from the generation of semantic content.

### 1. Dual-Anchor Mechanism

* **Time Anchor (SenseVoice):** Utilizes SenseVoice model (ONNX quantized inference) running on CPU. It uses CTC (Connectionist Temporal Classification) decoding to generate high-precision word-level timestamps, defining the absolute time boundaries of subtitles.
* **Content Anchor (Whisper):** Utilizes Whisper Large-v3 model running on GPU. It focuses on semantic coherence and contextual understanding. Through the Needleman-Wunsch algorithm, Whisper's semantic text is aligned with SenseVoice's time anchors.

### 2. Asynchronous Dual-Stream Pipeline

The system adopts an "out-of-order execution, in-order submission" architecture to maximize resource utilization:

* **Fast Stream (CPU Layer):** `FastWorker` executes concurrently on CPU. It handles audio segmentation, SenseVoice inference, and pushes "draft" subtitles via SSE (Server-Sent Events) for instant preview.
* **Slow Stream (GPU Layer):** `SlowWorker` executes sequentially on GPU. It maintains audio context to ensure Whisper's semantic coherence, performs "patching" on draft subtitles, and generates "finalized" output.
* **Sequenced Queue (SequencedQueue):** Acts as a rectifier between dual streams, allowing short audio blocks to be processed out-of-order on CPU, while ensuring GPU receives tasks strictly in chronological order to preserve context.

## Key Technical Features

### Smart Accumulation VAD (Smart Accumulation VAD)

To address the issue of Whisper's attention decay in long audio segments, the system implements a smart accumulation algorithm based on Silero VAD.

* **Logic:** Instead of greedy merging, accumulates speech segments based on semantic pauses.
* **Constraints:** Implements a 12-second "soft limit" (Whisper's optimal processing duration) and a 30-second "hard limit" (physical input window), ensuring optimal segmentation duration.

### Spectral Triage and YAMNet Probe

Before processing, audio blocks undergo spectral analysis to determine if vocal separation is needed.

* **YAMNet Probe:** Samples the beginning, middle, and end of audio blocks to classify audio events (speech vs music).
* **Decision Logic:** Pure speech skips separation; segments with heavy background music (BGM) or high noise levels trigger separation process.

### On-Demand Separation and Fuse Breaker V2 (FuseBreaker V2)

The system supports dynamic resource allocation for vocal separation (Demucs).

* **On-Demand Mode:** Separates only audio blocks marked by spectral triage, saving GPU resources compared to global track separation.
* **Fuse Breaker (FuseBreaker):** If transcription confidence is below threshold or strong interference is detected, the system triggers "fuse breaking", automatically upgrading separation model (e.g., from no separation to HTDemucs) and retrying transcription.

### Three-Level Alignment Strategy

To ensure subtitle stability, `AlignmentWorker` employs a cascade degradation strategy:

1. **Level 1 (Dual-Modal Alignment):** Uses Needleman-Wunsch global sequence alignment algorithm to map Whisper text to SenseVoice timestamps (gold standard).
2. **Level 2 (Pseudo Alignment):** If sequence alignment fails, uses character/word duration ratio to mathematically map Whisper text to SenseVoice time window.
3. **Level 3 (Fallback):** If Whisper completely fails, system falls back to original SenseVoice draft.

### Whisper Arbitration

Secondary verification mechanism prevents common hallucination issues in low-confidence segments. If SenseVoice outputs low-confidence results (typically hallucinations like "SRRCT"), Whisper performs targeted re-transcription to decide whether to keep, replace, or discard the segment.

## Configuration Options

### Preset Modes

| Preset | Use Case | Description |
|------|----------|------|
| **Express** | Meeting recordings, podcasts | SenseVoice only, fastest speed |
| **Balanced** | General videos | Dual stream alignment, balance speed and quality |
| **Precise** | Films, documentaries | Full pipeline, highest quality |

### Port Configuration
```python
# Modify in launcher
backend_port = 8000      # Backend port
frontend_port = 5173     # Frontend port
```

### Environment Variables

Customize system settings by editing the `.env` file in the project root. Restart the service after modifications.

| Variable | Options | Default | Description |
|----------|---------|---------|-------------|
| `DEV_MODE` | `true` / `false` | `false` | Development mode, enables frontend dev server and DEBUG logs |
| `WHISPER_MODEL` | `tiny` / `base` / `small` / `medium` / `large-v3` / `turbo` | `medium` | Whisper model size, affects accuracy and VRAM usage |
| `WHISPER_COMPUTE_TYPE` | `auto` / `int8` / `int8_float16` / `float16` | `auto` | Inference precision, auto selects based on available VRAM |
| `SENSEVOICE_DEVICE` | `cpu` / `cuda` / `auto` | `cpu` | SenseVoice inference device |
| `SENSEVOICE_MODEL_TYPE` | `quantized` / `fp32` | `quantized` | SenseVoice model type, quantized only supports CPU |
| `USE_HF_MIRROR` | `true` / `false` | `true` | Use HuggingFace China mirror |
| `PYPI_MIRROR` | Mirror URL or empty | Tsinghua | Python package download mirror |

## Version History

- **v3.0.0** (In Development) - Dual-modal spatiotemporal decoupled architecture, fast-slow dual stream experience
- **v2.0.0** (2025-08-18) - Complete architecture upgrade, frontend-backend separation
- **v1.1.0** (2025-06-18) - Initial version, command line interface

## Contributing

1. Fork the project
2. Create feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add some AmazingFeature'`
4. Push branch: `git push origin feature/AmazingFeature`
5. Open Pull Request

## License

This project is open source under the MIT License - see [LICENSE](LICENSE) file for details

## Disclaimer

> This tool is for learning and research purposes only, commercial use in any form is prohibited.
> Users must comply with relevant laws and regulations, the author assumes no responsibility for usage consequences.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Semantic recognition core
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - Time anchoring core
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice activity detection
- [Demucs](https://github.com/facebookresearch/demucs) - Vocal separation
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Vue.js](https://vuejs.org/) - Frontend framework
- All contributors of open source libraries

---

**If this project helps you, please give a Star for support!**
