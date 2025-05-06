@echo off
echo ==========================================================
echo 运行视频字幕生成程序...
echo ==========================================================

:: 激活conda环境
call conda activate whisperx

:: 运行Python脚本
python video_to_srt_optimized.py

:: 完成后暂停
echo.
echo 程序执行完毕，请按任意键退出...
pause