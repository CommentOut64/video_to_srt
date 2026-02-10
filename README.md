# AnchorFlux

> **由双引擎 AI 架构驱动的可视化字幕生产工作站。**

> 告别繁琐的命令行和枯燥的文本校对，让 AI 负责繁重的工作，让你专注于创作的艺术。

[![English README](https://img.shields.io/badge/README-English-blue.svg)](README_en.md)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-v3.2.2-brightgreen.svg)
![Vue](https://img.shields.io/badge/Vue-3.5+-4FC08D?logo=vue.js&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)

<img src="https://cdn.jsdelivr.net/gh/CommentOut64/anchor-flux@main/assets/pic2.png" image-rendering: crisp-edges/>

**AnchorFlux** 不仅仅是一个高精度的语音转写工具，它是一套为视频创作者量身定制的**全流程字幕生产解决方案**。

<a href="https://www.bilibili.com/video/BV1xAqzB5EFN"><img src="https://img.shields.io/static/v1?label=%20&message=%E6%BC%94%E7%A4%BA%E8%A7%86%E9%A2%91&color=F37697&style=flat&logo=bilibili&logoColor=white&logoWidth=20" height="32"></a>
<a href="https://www.bilibili.com/video/BV1ejvCBPEz9"><img src="https://img.shields.io/static/v1?label=%20&message=%E6%95%99%E7%A8%8B%E8%A7%86%E9%A2%91&color=F37697&style=flat&logo=bilibili&logoColor=white&logoWidth=20" height="32"></a>

## 功能特点

### 核心功能

- **双模态对齐**：SenseVoice + Whisper 协同工作
- **快慢双流**：草稿快速上屏，定稿后台完成自动覆盖
- **时空解耦**：字幕文本与时间戳分离处理，各取所长
- **智能人声分离**：Demucs 按需处理，提升识别质量
- **字级时间戳**：CTC 精准边界检测，字级别时间对齐
- **简洁高效的编辑器**：专为视频剪辑者设计，界面美观，操作方便

### 编辑器功能

- **实时字幕叠加预览**：内置视频播放器，支持字幕实时叠加渲染，所见即所得
- **高精度波形可视化**：基于 WaveSurfer.js 渲染音频波形，直观展示语音活动与静音区间
- **直观的字幕范围编辑**：在波形图上提供可视化的“字幕范围框”，支持通过鼠标直接拖拽边缘来快速调整字幕的起始与结束时间
- **交互式字幕列表**：提供完整的字幕编辑面板，支持点击跳转至对应视频进度、修改文本内容与时间戳、快速插入字幕或删除字幕
- **精确的字幕拆分**：提供基于波形图时间点或文本光标位置的快速拆分功能
- **多格式一键导出**：字幕一键导出为 SRT、ASS、VTT 标准字幕格式以及纯文本文件
- **历史记录与自动保存**：支持撤销与重做，所有修改实时自动保存，无需担心数据丢失

### 用户界面

- **现代化 Web UI**：基于 Vue.js 3 的响应式界面
- **实时双流预览**：草稿斜体，定稿正体，状态一目了然
- **拖拽上传**：支持多种视频格式的拖拽上传
- **进度实时推送**：服务端事件流，断线自动重连，实时更新转录进度

### 技术特性

- **前后端分离**：Vue.js + FastAPI 架构
- **流水线解耦**：单一职责原则，每个模块专注一件事
- **显存自适应**：根据 GPU 显存动态调整处理策略
- **CPU专项优化**：使用 ONNX 量化模型，纯 CPU 推理速度依然可观
- **断点续传**：任务中断后可从断点恢复

### 扩展功能 (规划中)

- **LLM 校对**：大语言模型语义校对 [未实现]
- **LLM 翻译**：多语言字幕翻译，直接生成双语字幕 [未实现]

## 系统要求

### 基础要求
- **操作系统:** Windows 10/11
- **GPU:** 支持 CUDA 11.8+ 的 NVIDIA GPU (建议至少 4GB 显存以获得最佳性能)
- **重要提示: 即使没有独立显卡依然可以使用极速配置（仅SenseVoice），但无法使用 Whisper 模型**
- **内存:** 建议 16GB+

### 本地部署要求
- **Python:** 3.10+
- **Node.js:** 21+

## 快速开始

### 整合包
1. **从 release 或提供的网盘下载整合包**
2. **解压，双击 一键启动.bat**
3. **等待依赖自动下载完成（保持网络通畅，需要10-20分钟）**
4. **完成后会自动启动并跳转到浏览器页面**
5. **下次启动同样使用 一键启动.bat**

### 手动安装
1. **克隆仓库**
```bash
git clone https://github.com/CommentOut64/anchor-flux.git
cd anchor-flux-main
```
2. **安装CUDA和cuDNN**
   - 下载并安装 [CUDA 11.8+](https://developer.nvidia.com/cuda-11-8-0-download-archive)
   - 下载并安装 [cuDNN 8](https://developer.nvidia.com/rdp/cudnn-archive)
   - 验证安装: `nvidia-smi` 和 `nvcc --version`

3. **运行启动脚本**
```bash
# 运行
run.bat
```

### 更换 Whisper 模型

**方式一：修改配置自动下载（推荐）**

编辑项目根目录的 `.env` 文件，修改 `WHISPER_MODEL` 参数后重启服务，系统会自动从 HuggingFace 下载对应模型：
```bash
WHISPER_MODEL=large-v3  # 可选: tiny, base, small, medium, large-v3, turbo
```

**方式二：手动下载模型**

从 HuggingFace 下载 Faster-Whisper 模型文件，放置到指定目录：

1. 访问模型仓库（以 large-v3 为例）：https://huggingface.co/Systran/faster-whisper-large-v3
2. 下载以下必需文件：
   - `model.bin` - 模型权重（必需）
   - `config.json` - 模型配置（必需）
   - `tokenizer.json` - 分词器（必需）
   - `vocabulary.txt` 或 `vocabulary.json` - 词汇表（必需）
3. 在项目目录创建以下路径并放入文件：
   ```
   backend/models/huggingface/models--Systran--faster-whisper-large-v3/snapshots/<任意hash名>/
   ├── model.bin
   ├── config.json
   ├── tokenizer.json
   └── vocabulary.txt
   ```
   > 提示：`<任意hash名>` 可以是任意字符串，如 `main` 或 `v1`

**可用模型列表：**

| 模型 | HuggingFace 仓库 | 显存需求 |
|------|------------------|----------|
| tiny | Systran/faster-whisper-tiny | ~1GB |
| base | Systran/faster-whisper-base | ~1GB |
| small | Systran/faster-whisper-small | ~2GB |
| medium | Systran/faster-whisper-medium | ~5GB |
| large-v3 | Systran/faster-whisper-large-v3 | ~10GB (float16) / ~6GB (int8) |
| turbo | Systran/faster-whisper-large-v3-turbo | ~6GB |

### 运行模式说明

系统支持两种运行模式，通过 `.env` 文件中的 `DEV_MODE` 参数切换：

| 模式 | DEV_MODE | 前端 | 后端 | 访问地址 |
|------|----------|------|------|----------|
| **生产模式** | `false`（默认） | 由后端托管已构建的静态文件 | 端口 8000 | http://localhost:8000 |
| **开发模式** | `true` | `npm run dev` 热重载（端口 5173） | 端口 8000 | http://localhost:5173 |

- **生产模式**：前端使用 `frontend/dist` 目录下的预构建文件，由 FastAPI 静态文件托管，适合日常使用
- **开发模式**：前端使用 Vite 开发服务器，支持热重载，适合开发调试

## 技术栈

### 后端

* **FastAPI** - 异步核心，提供高性能流式接口
* **PyTorch** - 深度学习框架
* **Uvicorn** - ASGI 服务器
* **SSE** - 实时传输，流式推送识别结果

### 前端

* **Vue 3 / Vite** - 响应式框架
* **WaveSurfer.js** - 波形可视化
* **Pinia** - 状态管理
* **Element Plus** - 交互组件，适配深色模式
* **EventSource** - 自动重连，确保流式通信稳定

### AI 模型

* **Whisper** - 语义识别，上下文理解
* **SenseVoice** - 时间锚定，CTC 字级时间戳
* **Silero VAD** - 语音活动检测，智能累积切分
* **Demucs** - 人声分离，消除背景噪声
* **YAMNet** - 音频事件分类，频谱分诊决策
* **SpeechBrain** - 语言检测，多语种识别

## 架构概览

```mermaid
flowchart TB
    subgraph 预处理阶段
        A[音频提取] --> B[VAD 切分]
        B --> C[频谱分诊]
        C --> D[人声分离]
        D --> E[语言检测]
    end

    subgraph 双流处理
        E --> F[快流 SenseVoice]
        E --> G[Bridge 语义批次]
        F --> G
        G --> H[慢流 Whisper]
    end

    H --> L0
    F --> L0

    subgraph 后处理
        L0[L0 原始输出层] --> L1[L1 规范化层]
        L1 --> L2[L2 仲裁层]
        L2 --> L3[L3 标点层]
        L3 --> L4[L4 对齐层]
        L4 --> L5[L5 语义注入层]
        L5 --> L6[L6 切分层]
        L6 --> L7[L7 输出层]
    end

    subgraph 推送
        K[草稿推送]
        L[定稿推送]
        M[导出 SRT/ASS]
    end

    L7 --> K
    L7 --> L
    L7 --> M
```

## 核心架构

系统采用"时空解耦"理念，将时间边界的定义与语义内容的生成分离开来，通过七层后处理流水线实现高质量字幕输出。

### 1. 双锚点机制

* **时间锚 (SenseVoice):** 利用在 CPU 上运行的 SenseVoice 模型（ONNX 量化推理），采用 CTC 解码生成高精度的字级时间戳，定义字幕的绝对时间边界。
* **内容锚 (Whisper):** 利用在 GPU 上运行的 Whisper 模型，专注于语义连贯性和上下文理解，通过 L4 对齐层将语义文本映射到时间锚点。

### 2. 异步双流流水线

系统采用"乱序执行、顺序提交"的三级流水线架构：

* **快流 (FastWorker):** 在 CPU 上并发执行 SenseVoice 推理，通过 SSE 推送草稿字幕，实现即时预览。
* **Bridge 语义批次:** 将快流产生的语义句子聚合为 Whisper 批次，按语言切换、说话人切换、停顿超时等条件触发 flush，提升 GPU 效率并强化上下文一致性。
* **慢流 (SlowWorker):** 在 GPU 上顺序执行 Whisper 推理，维护音频上下文以确保语义连贯性。

### 3. 后处理七层架构 (L0-L7)

* **L0 原始输出层:** 接收快流/慢流的原始识别结果
* **L1 规范化层:** 文本清洗、ITN（逆文本规范化）、字符映射
* **L2 仲裁层:** 基于质量信号（置信度、重复检测、幻觉检测）选择快流或慢流文本
* **L3 标点层:** 生成标点位置候选，支持多源标点融合
* **L4 对齐层:** 使用 Needleman-Wunsch 算法将慢流文本对齐到快流时间锚点
* **L5 语义注入层:** 将标点位置注入到对齐后的词流
* **L6 切分层:** 基于多信号边界（停顿、长度、说话人、标点）切分为最终句子
* **L7 输出层:** 格式化与分发，推送定稿字幕

## 关键技术特性

### 智能累积 VAD (Smart Accumulation)

为解决 Whisper 在长音频片段中注意力衰减的问题，系统实现了基于 Silero VAD 的智能累积算法。

* **回溯断点:** 在接近软上限时回溯查找最佳断点，选择 gap 最大且 RMS 最低的位置
* **RMS 验证:** 拒绝在高能量（正在说话）位置切分，避免截断词语
* **双重约束:** 12 秒软上限（SenseVoice 最佳处理时长）+ 30 秒硬上限（物理输入窗口）

### 频谱分诊与智能探针

在处理之前，音频块会经过多层频谱分析以确定是否需要人声分离。

* **SNR+C50 三层决策:** 使用 Brouhaha 模型计算信噪比和清晰度指标，Layer 1 快速筛选（高质量放行/低质量分离），Layer 2 频谱特征补充判断，Layer 3 YAMNet 语义分类兜底。
* **智能探针模式:** 中心扩散探针策略，快速判断视频是否需要全量探测。纯净视频直接标记所有 chunk 无需分离，发现干扰则回退到标准分诊模式。
* **按需分离:** 仅分离被频谱分诊标记的音频块，与全局全轨分离相比节省了 GPU 资源。

### 语言识别与注入

基于 SpeechBrain 的多模式语言检测，检测结果注入 ASR 引擎提升识别准确率。

* **三种检测模式:** fast（快速采样）、balanced（分组探针）、precise（全量检测），根据视频特性自动选择。
* **中心扩散探针:** balanced 模式按时间窗口分组，使用探针策略快速判断语言分布，一致性不足时自动降级。
* **语言注入 ASR:** 检测结果注入 SenseVoice 和 Whisper，消除语言猜测开销，提升多语种场景准确率。

### L2 文本仲裁

基于多维质量信号的智能文本选择机制，防止低质量输出进入最终字幕。

* **质量信号:** 置信度、长度比、重复检测、幻觉检测、ITN 回退标记
* **决策逻辑:** 当慢流出现重复/幻觉时自动回退到快流，确保输出稳定性

### L4 时间对齐

将慢流语义文本精确映射到快流时间锚点的核心机制。

* **Needleman-Wunsch 对齐:** 全局序列对齐算法，处理插入、删除、替换
* **Gap 修复:** 自动检测并修复对齐空隙，支持多种修复策略
* **覆盖率统计:** 实时计算对齐覆盖率，低覆盖时触发降级

## 配置选项

### 预设模式

| 预设 | 适用场景 | 说明 |
|------|----------|------|
| **极速** | 会议录音、播客 | 仅 SenseVoice，最快速度 |
| **均衡** | 一般视频 | 双流对齐，平衡速度与质量 |
| **精准** | 影视、纪录片 | 完整流水线，最高质量 |

### 端口配置
```python
# 在启动器中修改
backend_port = 8000      # 后端端口
frontend_port = 5173     # 前端端口
```

### 环境变量配置

通过编辑项目根目录下的 `.env` 文件可自定义系统配置，修改后需重启服务生效。

| 变量名 | 可选值 | 默认值 | 说明 |
|--------|--------|--------|------|
| `DEV_MODE` | `true` / `false` | `false` | 开发模式，启用后使用前端开发服务器并显示 DEBUG 日志 |
| `WHISPER_MODEL` | `tiny` / `base` / `small` / `medium` / `large-v3` / `turbo` | `medium` | Whisper 模型大小，影响准确率和显存占用 |
| `WHISPER_COMPUTE_TYPE` | `auto` / `int8` / `int8_float16` / `float16` | `auto` | 推理精度，auto 会根据显存自动选择 |
| `SENSEVOICE_DEVICE` | `cpu` / `cuda` / `auto` | `cpu` | SenseVoice 推理设备 |
| `SENSEVOICE_MODEL_TYPE` | `quantized` / `fp32` | `quantized` | SenseVoice 模型类型，量化版仅支持 CPU |
| `USE_HF_MIRROR` | `true` / `false` | `true` | 是否使用 HuggingFace 国内镜像源 |
| `PYPI_MIRROR` | 镜像源 URL 或留空 | 清华源 | Python 包下载镜像源 |

## 版本历史

- **v3.0.0** (开发中) - 双模态时空解耦架构，快慢双流体验
- **v2.0.0** (2025-08-18) - 全面架构升级，前后端分离
- **v1.1.0** (2025-06-18) - 初始版本，命令行界面

## 贡献指南

1. Fork 项目
2. 创建功能分支：`git checkout -b feature/AmazingFeature`
3. 提交更改：`git commit -m 'Add some AmazingFeature'`
4. 推送分支：`git push origin feature/AmazingFeature`
5. 打开 Pull Request

## 开源协议

本项目基于 MIT 协议开源 - 查看 [LICENSE](LICENSE) 文件了解详情

## 免责声明

> 本工具仅供学习与研究使用，禁止任何形式的商业使用。
> 使用者需遵守相关法律法规，作者不对使用后果承担任何责任。

## 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) - 语义识别核心
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - 时间锚定核心
- [Silero VAD](https://github.com/snakers4/silero-vad) - 语音活动检测
- [Demucs](https://github.com/facebookresearch/demucs) - 人声分离
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Web 框架
- [Vue.js](https://vuejs.org/) - 前端框架
- 所有开源库的贡献者们

---

**如果这个项目对你有帮助，请给个 Star 支持！**
