# Video-to-SRT 字幕生成工具

> 仅供学习与研究，禁止任何形式的商业使用。作者不对使用后果承担责任。

这是一个视频字幕自动生成工具，使用 WhisperX 进行语音识别和时间戳对齐，可以快速将视频转换成 SRT 字幕文件。

这是一个半成品，仅作为可行性试验，暂时不能给出必定成功的部署流程。

## 📋 功能特点

* 🎬 支持多种视频格式
* 📊 高精度音频分段与处理，提高长视频的转录质量
* ⏱️ 精确的时间戳对齐
* 🔄 支持断点续传，中断后可从上次进度继续
* 🚀 多线程并行处理，加速转录过程

## 🔧 系统要求

* Python 3.10 或更高版本
* CUDA 支持的 NVIDIA GPU
* 至少 8GB RAM
* 足够的磁盘空间用于临时文件

## 📦 依赖项

* PyTorch
* TorchAudio
* TorchVision
* WhisperX
* FFmpeg
* pydub
* tqdm
* transformers
* 其他辅助库

## 🔍 TODO

* 部分警告无法完全去除
* 未完成依赖检测部分
* 依赖复杂，暂时无法完全给出列表
* 安装方式未给出
* 未能成功打包

## 📄 许可证

MIT License

## 🙏 致谢

* [WhisperX](vscode-file://vscode-app/d:/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) - 提供了核心的转录和对齐功能
* [OpenAI Whisper](vscode-file://vscode-app/d:/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) - 基础的语音识别模型
* [FFmpeg](vscode-file://vscode-app/d:/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) - 视频和音频处理
* 所有开源库的贡献者们
