"""
音频提取模块 - 极速提取音频流
优先复制流避免重新编码，加速处理
"""
import subprocess
import asyncio
import os
from pathlib import Path
from typing import Optional

from app.core.config import config


class AudioExtractor:
    """音频提取器 - 快速从视频中提取音频"""

    def __init__(self, ffmpeg_cmd: Optional[str] = None):
        self.ffmpeg_cmd = ffmpeg_cmd or config.get_ffmpeg_command()

    async def extract_fast(self, video_path: Path, output_path: Path,
                           audio_codec: str = 'aac') -> Path:
        """
        极速提取音频（优先复制流，避免重新编码）

        Args:
            video_path: 源视频路径
            output_path: 输出音频路径（带扩展名）
            audio_codec: 源音频编码

        Returns:
            实际输出的音频路径
        """
        # 如果音频已经是 AAC/MP3，直接复制（< 1 秒）
        if audio_codec in ['aac', 'mp3']:
            return await self._copy_audio_stream(video_path, output_path)

        # 其他编码，转为 WAV（用于 Whisper）
        return await self._convert_to_wav(video_path, output_path)

    def extract_fast_sync(self, video_path: Path, output_path: Path,
                          audio_codec: str = 'aac') -> Path:
        """同步版本的极速提取"""
        if audio_codec in ['aac', 'mp3']:
            return self._copy_audio_stream_sync(video_path, output_path)
        return self._convert_to_wav_sync(video_path, output_path)

    async def _copy_audio_stream(self, video_path: Path, output_path: Path) -> Path:
        """复制音频流（不重新编码）"""
        # Whisper 需要 WAV，必须转码
        if output_path.suffix == '.wav':
            return await self._convert_to_wav(video_path, output_path)

        cmd = [
            self.ffmpeg_cmd,
            '-i', str(video_path),
            '-vn',                  # 不要视频
            '-acodec', 'copy',      # 复制音频流
            '-y',
            str(output_path)
        ]

        await self._run(cmd)
        return output_path

    def _copy_audio_stream_sync(self, video_path: Path, output_path: Path) -> Path:
        """同步版本复制音频流"""
        if output_path.suffix == '.wav':
            return self._convert_to_wav_sync(video_path, output_path)

        cmd = [
            self.ffmpeg_cmd,
            '-i', str(video_path),
            '-vn',
            '-acodec', 'copy',
            '-y',
            str(output_path)
        ]

        self._run_sync(cmd)
        return output_path

    async def _convert_to_wav(self, video_path: Path, output_path: Path) -> Path:
        """转换为 WAV（用于 Whisper）"""
        wav_path = output_path.with_suffix('.wav')

        cmd = [
            self.ffmpeg_cmd,
            '-i', str(video_path),
            '-vn',
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', '16000',           # Whisper 推荐采样率
            '-ac', '1',               # 单声道（Whisper 会转）
            '-y',
            str(wav_path)
        ]

        await self._run(cmd)
        return wav_path

    def _convert_to_wav_sync(self, video_path: Path, output_path: Path) -> Path:
        """同步版本转换为 WAV"""
        wav_path = output_path.with_suffix('.wav')

        cmd = [
            self.ffmpeg_cmd,
            '-i', str(video_path),
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            str(wav_path)
        ]

        self._run_sync(cmd)
        return wav_path

    async def extract_m4a(self, video_path: Path, output_path: Path) -> Path:
        """
        提取 M4A 格式音频（较小体积，用于波形显示）
        """
        m4a_path = output_path.with_suffix('.m4a')

        cmd = [
            self.ffmpeg_cmd,
            '-i', str(video_path),
            '-vn',
            '-acodec', 'aac',
            '-b:a', '128k',
            '-y',
            str(m4a_path)
        ]

        await self._run(cmd)
        return m4a_path

    async def _run(self, cmd: list):
        """运行 FFmpeg 命令"""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            # V3.1.0+dev.20260104.01: 修复 Windows 编码问题
            raise Exception(f"FFmpeg 失败: {stderr.decode('utf-8', errors='replace')}")

    def _run_sync(self, cmd: list):
        """同步运行 FFmpeg 命令"""
        result = subprocess.run(
            cmd,
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        if result.returncode != 0:
            # V3.1.0+dev.20260104.01: 修复 Windows 编码问题
            raise Exception(f"FFmpeg 失败: {result.stderr.decode('utf-8', errors='replace')}")


# 全局实例
audio_extractor = AudioExtractor()
