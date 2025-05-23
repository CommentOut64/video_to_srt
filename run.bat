@echo off
echo ==========================================================
echo 运行视频字幕生成程序...
echo ==========================================================

:: 激活conda环境
call conda activate srt_packer

:: 运行Python脚本
python modified_script.py

:: 完成后暂停
pause