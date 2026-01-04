"""
渐进式视频生成器 - 360p -> 720p 两阶段生成
支持 SSE 进度推送，实现视频预览的渐进式加载
"""
import asyncio
import subprocess
import os
from pathlib import Path
from typing import Optional

from app.core.config import config


class ProgressiveVideoGenerator:
    """渐进式视频生成器 - 支持 360p 预览和 720p 高质量两阶段生成"""

    def __init__(self, ffmpeg_cmd: Optional[str] = None, sse_manager=None):
        self.ffmpeg_cmd = ffmpeg_cmd or config.get_ffmpeg_command()
        self.sse = sse_manager

    def set_sse_manager(self, sse_manager):
        """设置 SSE 管理器（延迟注入，避免循环依赖）"""
        self.sse = sse_manager

    async def generate_360p_preview(self, job_id: str, video_path: Path,
                                    output_path: Path):
        """
        生成 360p 预览视频（低优先级，无音频）
        用于快速预览，让用户尽早看到视频内容
        """
        cmd = [
            self.ffmpeg_cmd,
            '-i', str(video_path),
            '-vf', 'scale=-2:360',        # 360p（宽度自适应）
            '-c:v', 'libx264',
            '-preset', 'ultrafast',        # 最快编码（质量较差）
            '-crf', '28',                  # 中等质量
            '-g', '30',                    # 关键帧间隔
            '-an',                         # 去掉音频（减小体积 + 加速）
            '-movflags', '+faststart',     # FastStart
            '-progress', 'pipe:1',
            '-y',
            str(output_path)
        ]

        # 启动进程并监听进度
        await self._run_with_progress(
            cmd, job_id, 'preview_360p', video_path
        )

        # 推送完成事件
        self._push_complete(job_id, 'preview_360p', output_path)

    async def generate_720p_proxy(self, job_id: str, video_path: Path,
                                  output_path: Path):
        """
        生成 720p 高质量视频（转录完成后）
        用于高清预览和导出
        """
        cmd = [
            self.ffmpeg_cmd,
            '-i', str(video_path),
            '-vf', 'scale=-2:720',
            '-c:v', 'libx264',
            '-preset', 'fast',              # 平衡质量和速度
            '-crf', '23',                   # 高质量
            '-g', '30',                     # 关键帧间隔
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-progress', 'pipe:1',
            '-y',
            str(output_path)
        ]

        await self._run_with_progress(
            cmd, job_id, 'proxy_720p', video_path
        )

        # 推送完成事件
        self._push_complete(job_id, 'proxy_720p', output_path)

    async def _run_with_progress(self, cmd: list, job_id: str,
                                 stage: str, video_path: Path):
        """运行 FFmpeg 并推送进度"""
        # 获取视频时长
        duration = await self._get_duration(video_path)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        # 解析进度
        while True:
            line = await process.stdout.readline()
            if not line:
                break

            # V3.1.0+dev.20260104.01: 修复 Windows 编码问题，显式指定 UTF-8
            line_str = line.decode('utf-8', errors='replace').strip()
            if line_str.startswith('out_time_ms='):
                try:
                    out_time_ms = int(line_str.split('=')[1])
                    if duration > 0:
                        progress = min(100, (out_time_ms / 1000000) / duration * 100)

                        # 推送进度
                        self._push_progress(job_id, stage, round(progress, 1))
                except:
                    pass

        await process.wait()

        if process.returncode != 0:
            stderr = await process.stderr.read()
            # V3.1.0+dev.20260104.01: 修复 Windows 编码问题
            raise Exception(f"FFmpeg 失败: {stderr.decode('utf-8', errors='replace')}")

    def _push_progress(self, job_id: str, stage: str, progress: float):
        """推送进度到 SSE"""
        if not self.sse:
            return

        try:
            self.sse.broadcast_sync(
                f"job:{job_id}",
                f"{stage}_progress",
                {
                    "job_id": job_id,
                    "stage": stage,
                    "progress": progress
                }
            )
        except Exception as e:
            # SSE 推送失败不影响视频生成
            print(f"[ProgressiveVideoGenerator] SSE 推送失败: {e}")

    def _push_complete(self, job_id: str, stage: str, output_path: Path):
        """推送完成事件到 SSE"""
        if not self.sse:
            return

        try:
            video_url = f"/api/media/{job_id}/video/preview" if stage == 'preview_360p' else f"/api/media/{job_id}/video"
            resolution = "360p" if stage == 'preview_360p' else "720p"

            self.sse.broadcast_sync(
                f"job:{job_id}",
                f"{stage}_complete",
                {
                    "job_id": job_id,
                    "video_url": video_url,
                    "resolution": resolution,
                    "stage": stage
                }
            )
        except Exception as e:
            print(f"[ProgressiveVideoGenerator] SSE 完成推送失败: {e}")

    async def _get_duration(self, video_path: Path) -> float:
        """获取视频时长"""
        ffprobe_cmd = config.get_ffprobe_command()
        cmd = [
            ffprobe_cmd, '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]

        # V3.1.0+dev.20260104.01: 添加 encoding='utf-8' 避免 Windows GBK 编码问题
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        if result.returncode == 0:
            try:
                return float(result.stdout.strip())
            except:
                pass
        return 0.0


# 全局实例
progressive_video_generator = ProgressiveVideoGenerator()
