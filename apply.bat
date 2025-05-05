@echo off
echo ==========================================================
echo ���ڴ���Conda��������װ������...
echo ==========================================================

:: �����廪Դ
echo ����Conda�廪Դ...
call conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
call conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
call conda config --set show_channel_urls yes

:: �����»���
echo ����whisper_srt���� (Python 3.10)...
call conda create -n whisper_srt python=3.10 -y

:: �����
echo ����whisper_srt����...
call conda activate whisper_srt

:: ����pip�廪Դ
echo ����pip�廪Դ...
call pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

:: ��װPyTorch (��CUDA���ݰ汾)
echo ��װPyTorch (CUDA)...
call pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

:: ��װ��������
echo ��װ��������...
call pip install pydub tqdm

:: ��װwhisperx
echo ��װwhisperx...
call pip install git+https://github.com/m-bain/whisperx.git

:: ����Ƿ�ɹ���װ
echo ��֤��װ...
call python -c "import sys; print(f'Python �汾: {sys.version}')"
call python -c "import torch; print(f'PyTorch �汾: {torch.__version__}')"
call python -c "import torch; print(f'CUDA ����: {torch.cuda.is_available()}')"
call python -c "import whisperx; print('WhisperX �Ѱ�װ')"
call python -c "import pydub; print('Pydub �Ѱ�װ')"
call python -c "import tqdm; print('TQDM �Ѱ�װ')"

:: ���tkinter
echo ���tkinter...
call python -c "import tkinter; print('Tkinter �Ѱ�װ')" || echo ����: Tkinter δ��װ����ȷ��Python�����а���tkinter

:: ���ffmpeg
echo ���ffmpeg...
where ffmpeg >nul 2>&1
if %errorlevel% == 0 (
    echo FFmpeg ����ϵͳ·�����ҵ���
) else (
    echo ����: FFmpeg δ��ϵͳ·�����ҵ���
    echo ��� https://ffmpeg.org/download.html ���ز���װFFmpeg��
    echo Ȼ������ӵ�ϵͳ��������PATH�С�
)

echo.
echo ==========================================================
echo ����������ɣ�����ʹ�� run.bat ���г���
echo ��������κ����⣬���������д˽ű����鱨����Ϣ��
echo ==========================================================
echo.
pause