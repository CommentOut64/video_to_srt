@echo off
echo ���ڴ���WhisperX���� (CUDA 11.8���ݰ汾)...

:: �����µ�conda����
call conda create -n whisperx python=3.10 -y
call conda activate whisperx

:: ��װPyTorch (CUDA 11.8�汾)
call conda install pytorch==2.0.1 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

:: ��װ��������
call conda install -c conda-forge ffmpeg -y
call pip install numpy==1.24.3 scipy==1.10.1 pydub==0.25.1

:: ��װTransformers��Huggingface��
call pip install transformers==4.30.2 huggingface_hub==0.16.4

:: ��װpyannoteϵ�п�
call pip install pyannote.audio==2.1.1
call pip install pyannote.core==5.0.0
call pip install pyannote.metrics==3.2.1
call pip install pyannote.pipeline==3.0.1

:: ��װfaster-whisper
call pip install faster-whisper==0.9.0

:: ���װWhisperX�����°棩
call pip install git+https://github.com/m-bain/whisperx.git

echo.
echo WhisperX������װ��ɣ�
echo ʹ�÷���: conda activate whisperx
echo.
pause