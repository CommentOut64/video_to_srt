@echo off
echo ==========================================================
echo ������Ƶ��Ļ���ɳ���...
echo ==========================================================

:: ����conda����
call conda activate whisper_srt

:: ����Python�ű�
python video_to_srt.py

:: ��ɺ���ͣ
echo.
echo ����ִ����ϣ��밴������˳�...
pause