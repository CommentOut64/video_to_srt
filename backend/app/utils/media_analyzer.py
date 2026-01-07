"""
媒体分析模块 - 快速提取视频关键信息
支持检测视频编码、分辨率、时长等，判断是否需要转码
"""
import subprocess
import json
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from app.core.config import config

logger = logging.getLogger(__name__)


# 需要转码的编码格式 (浏览器不兼容)
NEED_TRANSCODE_CODECS = {'hevc', 'h265', 'vp9', 'av1', 'mpeg2video'}

# 需要转码的容器格式 (浏览器不兼容)
NEED_TRANSCODE_FORMATS = {'.mkv', '.avi', '.mov', '.wmv', '.flv', '.m4v'}


class MediaAnalyzer:
    """媒体分析器 - 快速分析视频文件信息"""

    def __init__(self, ffprobe_cmd: Optional[str] = None):
        self.ffprobe_cmd = ffprobe_cmd or config.get_ffprobe_command()

    async def analyze(self, video_path: Path) -> Dict:
        """
        分析视频文件，返回完整信息

        Returns:
            dict: {
                'video': { 'codec', 'width', 'height', 'duration', 'bitrate', 'fps' },
                'audio': { 'codec', 'sample_rate', 'channels', 'bitrate' },
                'needs_transcode': bool,
                'transcode_reason': str
            }
        """
        cmd = [
            self.ffprobe_cmd,
            '-v', 'error',
            '-show_streams',
            '-show_format',
            '-of', 'json',
            str(video_path)
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding='utf-8',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        if result.returncode != 0:
            raise Exception(f"FFprobe 失败: {result.stderr}")

        data = json.loads(result.stdout)
        return self._parse_probe_data(data, video_path)

    def analyze_sync(self, video_path: Path) -> Dict:
        """同步版本的分析方法"""
        cmd = [
            self.ffprobe_cmd,
            '-v', 'error',
            '-show_streams',
            '-show_format',
            '-of', 'json',
            str(video_path)
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding='utf-8',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        if result.returncode != 0:
            raise Exception(f"FFprobe 失败: {result.stderr}")

        data = json.loads(result.stdout)
        return self._parse_probe_data(data, video_path)

    def _parse_probe_data(self, data: dict, video_path: Path) -> Dict:
        """解析 FFprobe 输出"""
        video_stream = None
        audio_stream = None

        for stream in data.get('streams', []):
            if stream['codec_type'] == 'video' and not video_stream:
                video_stream = stream
            elif stream['codec_type'] == 'audio' and not audio_stream:
                audio_stream = stream

        if not video_stream:
            raise Exception("未找到视频流")

        # 解析视频信息
        video_info = {
            'codec': video_stream.get('codec_name', '').lower(),
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'duration': float(data['format'].get('duration', 0)),
            'bitrate': int(data['format'].get('bit_rate', 0)),
            'fps': self._parse_fps(video_stream)
        }

        # 解析音频信息
        audio_info = None
        if audio_stream:
            audio_info = {
                'codec': audio_stream.get('codec_name', '').lower(),
                'sample_rate': int(audio_stream.get('sample_rate', 48000)),
                'channels': int(audio_stream.get('channels', 2)),
                'bitrate': int(audio_stream.get('bit_rate', 128000)) if audio_stream.get('bit_rate') else 128000
            }

        # 判断是否需要转码
        needs_transcode, reason = self._check_transcode_needed(
            video_info['codec'], video_path.suffix.lower()
        )

        return {
            'video': video_info,
            'audio': audio_info,
            'needs_transcode': needs_transcode,
            'transcode_reason': reason
        }

    def _parse_fps(self, stream: dict) -> float:
        """解析帧率"""
        fps_str = stream.get('r_frame_rate', '30/1')
        try:
            num, den = fps_str.split('/')
            return float(num) / float(den)
        except:
            return 30.0

    def _check_transcode_needed(self, codec: str, ext: str) -> Tuple[bool, str]:
        """检查是否需要转码"""
        if codec in NEED_TRANSCODE_CODECS:
            return True, f"{codec.upper()} 编码不兼容"

        if ext in NEED_TRANSCODE_FORMATS:
            return True, f"{ext.upper()} 格式不兼容"

        return False, ""

    def get_video_codec(self, video_path: Path) -> Optional[str]:
        """快速获取视频编码格式"""
        cmd = [
            self.ffprobe_cmd, '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10, encoding='utf-8',
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            if result.returncode == 0:
                codec = result.stdout.strip().lower()
                return codec if codec else None
        except Exception as e:
            logger.debug(f"[MediaAnalyzer] 获取视频编码失败: {e}")
        return None

    def get_video_duration(self, video_path: Path) -> float:
        """快速获取视频时长"""
        cmd = [
            self.ffprobe_cmd, '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, encoding='utf-8',
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception as e:
            logger.debug(f"[MediaAnalyzer] 获取视频时长失败: {e}")
        return 0.0


# 全局实例
media_analyzer = MediaAnalyzer()
