@echo off
echo 正在创建WhisperX环境 (CUDA 11.8兼容版本)...

:: 创建新的conda环境
call conda create -n whisperx python=3.10 -y
call conda activate whisperx

:: 安装PyTorch (CUDA 11.8版本)
call conda install pytorch==2.0.1 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

:: 安装基础依赖
call conda install -c conda-forge ffmpeg -y
call pip install numpy==1.24.3 scipy==1.10.1 pydub==0.25.1

:: 安装Transformers和Huggingface库
call pip install transformers==4.30.2 huggingface_hub==0.16.4

:: 安装pyannote系列库
call pip install pyannote.audio==2.1.1
call pip install pyannote.core==5.0.0
call pip install pyannote.metrics==3.2.1
call pip install pyannote.pipeline==3.0.1

:: 安装faster-whisper
call pip install faster-whisper==0.9.0

:: 最后安装WhisperX（最新版）
call pip install git+https://github.com/m-bain/whisperx.git

echo.
echo WhisperX环境安装完成！
echo 使用方法: conda activate whisperx
echo.
pause