@echo off
echo ==========================================================
echo 正在创建Conda环境并安装依赖项...
echo ==========================================================

:: 设置清华源
echo 设置Conda清华源...
call conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
call conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
call conda config --set show_channel_urls yes

:: 创建新环境
echo 创建whisper_srt环境 (Python 3.10)...
call conda create -n whisper_srt python=3.10 -y

:: 激活环境
echo 激活whisper_srt环境...
call conda activate whisper_srt

:: 设置pip清华源
echo 设置pip清华源...
call pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

:: 安装PyTorch (与CUDA兼容版本)
echo 安装PyTorch (CUDA)...
call pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

:: 安装基本依赖
echo 安装基本依赖...
call pip install pydub tqdm

:: 安装whisperx
echo 安装whisperx...
call pip install git+https://github.com/m-bain/whisperx.git

:: 检查是否成功安装
echo 验证安装...
call python -c "import sys; print(f'Python 版本: {sys.version}')"
call python -c "import torch; print(f'PyTorch 版本: {torch.__version__}')"
call python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
call python -c "import whisperx; print('WhisperX 已安装')"
call python -c "import pydub; print('Pydub 已安装')"
call python -c "import tqdm; print('TQDM 已安装')"

:: 检查tkinter
echo 检查tkinter...
call python -c "import tkinter; print('Tkinter 已安装')" || echo 警告: Tkinter 未安装，请确保Python环境中包含tkinter

:: 检查ffmpeg
echo 检查ffmpeg...
where ffmpeg >nul 2>&1
if %errorlevel% == 0 (
    echo FFmpeg 已在系统路径中找到。
) else (
    echo 警告: FFmpeg 未在系统路径中找到。
    echo 请从 https://ffmpeg.org/download.html 下载并安装FFmpeg，
    echo 然后将其添加到系统环境变量PATH中。
)

echo.
echo ==========================================================
echo 环境设置完成！可以使用 run.bat 运行程序。
echo 如果遇到任何问题，请重新运行此脚本或检查报错信息。
echo ==========================================================
echo.
pause