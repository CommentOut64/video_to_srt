@echo off
echo ==========================================================
echo ������Ƶ��Ļ���ɳ���...
echo ==========================================================

:: ����conda����
call conda activate whisperx

:: ����Python�ű�
python video_to_srt_optimized.py

:: ��ɺ���ͣ
echo.
echo ����ִ����ϣ��밴������˳�...
pause