# AnchorFlux

> 双模态时空解耦的高精度字幕生成引擎 - 快慢双流，锚点对齐

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-v3.0.0-brightgreen.svg)
![Vue](https://img.shields.io/badge/Vue-3.5+-4FC08D?logo=vue.js&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)

<img src="https://cdn.jsdelivr.net/gh/CommentOut64/anchor-flux@main/assets/pic1.png" />

AnchorFlux 采用创新的双锚架构：**SenseVoice 锚定时间边界，Whisper 锚定语义内容**，两个模型各司其职，通过锚点对齐算法融合输出高精度字幕。支持快慢双流体验——草稿秒级上屏，定稿延迟覆盖。

## 功能特点

### 核心功能

- **双模态对齐**：SenseVoice + Whisper 协同工作
- **快慢双流**：草稿 0.3s 上屏，定稿 3s 后自动覆盖，用户无需等待
- **时空解耦**：文本/语义与时间/边界分离处理，各取所长
- **智能人声分离**：Demucs 按需处理，提升识别质量
- **字级时间戳**：CTC 精准边界检测，字级别时间对齐

### 用户界面

- **现代化 Web UI**：基于 Vue.js 3 的响应式界面
- **实时双流预览**：草稿灰色斜体，定稿黑色正体，状态一目了然
- **拖拽上传**：支持多种视频格式的拖拽上传
- **SSE 实时推送**：服务端事件流，断线自动重连，实时更新转录进度

### 技术特性

- **前后端分离**：Vue.js + FastAPI 架构
- **流水线解耦**：单一职责原则，每个模块专注一件事
- **显存自适应**：根据 GPU 显存动态调整处理策略
- **断点续传**：任务中断后可从断点恢复

### 扩展功能 (规划中)

- **LLM 校对**：大语言模型语义润色 [未实现]
- **LLM 翻译**：多语言字幕翻译 [未实现]

## 快速开始

### 一键启动（推荐）
```bash
# 双击运行
一键启动.bat
```

### 手动启动
```bash
# 1. 安装依赖
pip install -r requirements.txt
cd frontend && npm install

# 2. 启动应用
python simple_launcher.py
```

## 系统要求

### 基础要求

- **Python**: 3.10+
- **Node.js**: 16+
- **FFmpeg**: 音频处理
- **内存**: 建议 16GB+

### GPU 要求

- **NVIDIA GPU**: 支持 CUDA 11.8+
- **显存**: 建议 6GB+ (支持整轨 Demucs)
- **驱动**: 最新 NVIDIA 驱动，CUDA 11.8+，cuDNN

## 技术栈

### 后端

- **FastAPI** - 高性能异步 Web 框架
- **PyTorch** - 深度学习框架
- **Uvicorn** - ASGI 服务器
- **SSE** - 服务端事件推送

### 前端

- **Vue.js 3** - 前端框架
- **Vite** - 构建工具
- **Pinia** - 状态管理
- **Axios** - HTTP 客户端

### AI 模型

- **Whisper Large-v3** - 语义识别，负责"听清内容"
- **SenseVoice** - 时间锚定，CTC 字级时间戳
- **Silero VAD** - 语音活动检测，智能切分
- **Demucs** - 人声分离，去除背景音乐/噪声

## 架构概览

```
                    AnchorFlux 双锚架构

    [视频输入] --> [音频提取] --> [Demucs 人声分离]
                                        |
                                        v
                              [Silero VAD 智能切分]
                                        |
                    +-------------------+-------------------+
                    |                                       |
                    v                                       v
            [SenseVoice CPU]                        [Whisper GPU]
            时间锚 (0.3s)                           内容锚 (3s)
            字级时间戳                               语义文本
                    |                                       |
                    v                                       v
            [SSE: 草稿上屏]                                 |
            灰色斜体                                        |
                    |                                       |
                    +-------------------+-------------------+
                                        |
                                        v
                              [锚点对齐算法]
                              difflib 匹配
                              VAD 边界校准
                              Gap 线性插值
                                        |
                                        v
                              [SSE: 定稿覆盖]
                              黑色正体
                                        |
                                        v
                              [SRT 字幕输出]
```

## 使用说明

> 待补充

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
frontend_port = 5174     # 前端端口
```

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
