"""
媒体资源路由 - 为前端编辑器提供视频、音频、波形数据等资源
支持:
- 视频流（Range请求支持，大视频拖拽跳转）
- 音频文件
- 波形峰值数据（流式计算，避免OOM；支持FFmpeg加速）
- Proxy视频自动生成（解决浏览器格式兼容性，支持SSE进度推送）
- 视频缩略图（支持Sprite图优化）
"""
import os
import io
import json
import wave
import struct
import asyncio
import subprocess
import time
import logging
from pathlib import Path
from typing import Optional, Tuple
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.core.config import config
from app.utils.ass_converter import ASSConverter
from app.utils.text_utils import repair_srt_overlaps, repair_timestamp_overlaps, detect_timestamp_overlaps

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/media", tags=["media"])

# 浏览器兼容的视频格式（扩展名，但需进一步检查编码）
BROWSER_COMPATIBLE_FORMATS = {'.mp4', '.webm'}

# 需要转码的格式（按扩展名）
NEED_TRANSCODE_FORMATS = {'.mkv', '.avi', '.mov', '.wmv', '.flv', '.m4v'}

# 浏览器不兼容的视频编码（需要转码为H.264）
NEED_TRANSCODE_CODECS = {'hevc', 'h265', 'vp9', 'av1'}

# 注意：旧的 _proxy_generation_status 已废弃，改用 MediaPrepService 管理状态


def _find_video_file(job_dir: Path) -> Optional[Path]:
    """在任务目录中查找视频文件"""
    video_exts = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.webm', '.flv', '.m4v']
    for file in job_dir.iterdir():
        if file.is_file() and file.suffix.lower() in video_exts:
            return file
    return None


def _serve_file_with_range(file_path: Path, request: Request, media_type: str):
    """
    支持HTTP Range请求的文件流式传输（允许拖拽进度条）
    """
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")

    if not range_header:
        # 无Range请求，返回完整文件
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=file_path.name
        )

    # 解析Range头：bytes=start-end
    try:
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0]) if byte_range[0] else 0
        end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else file_size - 1

        # 确保范围有效
        if start >= file_size:
            raise HTTPException(status_code=416, detail="请求范围无效")
        end = min(end, file_size - 1)

    except ValueError:
        raise HTTPException(status_code=400, detail="无效的Range头格式")

    # 返回部分内容（状态码206）
    def file_iterator():
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1

            # 动态 chunk_size：根据请求大小调整 (优化大视频拖动性能)
            if remaining < 1024 * 1024:  # < 1MB
                chunk_size = 8192  # 8KB
            elif remaining < 10 * 1024 * 1024:  # < 10MB
                chunk_size = 64 * 1024  # 64KB
            else:
                chunk_size = 256 * 1024  # 256KB

            while remaining > 0:
                read_size = min(chunk_size, remaining)
                data = f.read(read_size)
                if not data:
                    break
                yield data
                remaining -= len(data)

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
    }

    return StreamingResponse(
        file_iterator(),
        status_code=206,
        media_type=media_type,
        headers=headers
    )


async def _get_video_resolution(video_path: Path) -> tuple:
    """使用FFprobe获取视频分辨率"""
    ffprobe_cmd = config.get_ffprobe_command()

    cmd = [
        ffprobe_cmd, '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json',
        str(video_path)
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        data = json.loads(stdout.decode('utf-8'))
        streams = data.get('streams', [])
        if streams:
            width = streams[0].get('width', 0)
            height = streams[0].get('height', 0)
            return (width, height)
    except:
        pass
    return (0, 0)


def _get_video_codec(video_path: Path) -> Optional[str]:
    """
    使用FFprobe获取视频编码格式
    返回编码名称（如 h264, hevc, vp9 等），失败返回 None
    """
    ffprobe_cmd = config.get_ffprobe_command()

    cmd = [
        ffprobe_cmd, '-v', 'error',
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
        print(f"[media] 获取视频编码失败: {e}")
    return None


async def _get_video_duration(video_path: Path) -> float:
    """使用FFprobe获取视频时长"""
    ffprobe_cmd = config.get_ffprobe_command()

    # 检查视频文件是否存在
    if not video_path.exists():
        logger.debug(f"[media] 视频文件不存在: {video_path}")
        return 0.0

    cmd = [
        ffprobe_cmd, '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]

    logger.debug(f"[media] FFprobe 命令: {' '.join(cmd)}")

    try:
        # 使用同步 subprocess 来避免异步事件循环问题
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        if result.returncode != 0:
            logger.debug(f"[media] FFprobe 失败 (返回码 {result.returncode}): {result.stderr}")
            return 0.0

        duration_str = result.stdout.strip()
        if not duration_str:
            logger.debug(f"[media] FFprobe 返回空时长，视频路径: {video_path}")
            return 0.0

        logger.debug(f"FFprobe 成功获取时长: {duration_str}秒")
        return float(duration_str)
    except subprocess.TimeoutExpired:
        logger.debug(f"[media] FFprobe 超时，视频路径: {video_path}")
        return 0.0
    except Exception as e:
        logger.debug(f"[media] 获取视频时长异常: {e}, 路径: {video_path}")
        return 0.0



def _generate_peaks_with_ffmpeg(audio_path: Path, samples: int = 2000) -> Tuple[list, float]:
    """
    使用FFmpeg生成波形峰值数据（比Python wave更高效）
    适用于大文件（>100MB）
    
    【关键修复】基于时间位置计算峰值，确保波形与时间轴严格对齐
    """
    ffmpeg_cmd = config.get_ffmpeg_command()
    ffprobe_cmd = config.get_ffprobe_command()

    # 获取音频时长
    probe_cmd = [
        ffprobe_cmd, '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(audio_path)
    ]

    try:
        result = subprocess.run(
            probe_cmd, capture_output=True, text=True, encoding='utf-8',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        duration = float(result.stdout.strip())
    except:
        duration = 0

    if duration <= 0:
        return [], 0

    # 使用FFmpeg提取原始PCM数据（降采样到8kHz单声道以减少数据量）
    SAMPLE_RATE = 8000  # 采样率常量
    cmd = [
        ffmpeg_cmd,
        '-i', str(audio_path),
        '-ac', '1',           # 单声道
        '-ar', str(SAMPLE_RATE),  # 8kHz采样率
        '-f', 's16le',        # 16-bit PCM
        '-acodec', 'pcm_s16le',
        '-'
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        if result.returncode != 0:
            return [], duration

        # 解析PCM数据
        pcm_data = result.stdout
        actual_sample_count = len(pcm_data) // 2  # 16-bit = 2 bytes

        if actual_sample_count == 0:
            return [], duration

        # 限制峰值数量不超过实际采样数
        if actual_sample_count < samples:
            samples = actual_sample_count

        # 【关键修复】基于时间位置计算每个峰值对应的采样范围
        # 这样确保第 i 个峰值严格对应时间轴上的 [i/samples * duration, (i+1)/samples * duration]
        # 而不是依赖 PCM 数据量（可能与 duration 不精确匹配）
        peaks = []
        time_per_peak = duration / samples  # 每个峰值代表的时间长度

        for i in range(samples):
            # 基于时间计算采样范围
            start_time = i * time_per_peak
            end_time = (i + 1) * time_per_peak
            
            # 转换为采样点索引（基于实际采样率）
            start_sample = int(start_time * SAMPLE_RATE)
            end_sample = int(end_time * SAMPLE_RATE)
            
            # 确保不越界（PCM 数据可能比理论值稍短或稍长）
            start_sample = min(start_sample, actual_sample_count - 1)
            end_sample = min(end_sample, actual_sample_count)
            
            if start_sample >= end_sample:
                peaks.extend([0.0, 0.0])
                continue
            
            start_idx = start_sample * 2  # 16-bit = 2 bytes
            end_idx = end_sample * 2
            chunk_bytes = pcm_data[start_idx:end_idx]

            if len(chunk_bytes) < 2:
                peaks.extend([0.0, 0.0])
                continue

            # 解析为16-bit整数
            fmt = f'<{len(chunk_bytes) // 2}h'
            try:
                samples_chunk = struct.unpack(fmt, chunk_bytes)
                normalized = [s / 32768.0 for s in samples_chunk]
                peaks.append(float(min(normalized)))
                peaks.append(float(max(normalized)))
            except:
                peaks.extend([0.0, 0.0])

        return peaks, duration

    except Exception as e:
        print(f"[media] FFmpeg波形生成失败: {e}")
        return [], duration


def _generate_peaks_with_wave(audio_path: Path, samples: int = 2000) -> Tuple[list, float]:
    """
    使用Python wave模块生成波形峰值数据（流式读取，低内存占用）
    
    【关键修复】基于时间位置计算峰值，确保波形与时间轴严格对齐
    """
    try:
        with wave.open(str(audio_path), 'rb') as wav:
            frame_rate = wav.getframerate()
            n_frames = wav.getnframes()
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            duration = n_frames / frame_rate

            if duration <= 0:
                return [], 0

            if n_frames < samples:
                samples = n_frames
            
            # 【关键修复】基于时间位置计算每个峰值的帧范围
            # 确保第 i 个峰值严格对应时间轴上的 [i/samples * duration, (i+1)/samples * duration]
            time_per_peak = duration / samples
            peaks = []

            for i in range(samples):
                # 基于时间计算帧范围
                start_time = i * time_per_peak
                end_time = (i + 1) * time_per_peak
                
                # 转换为帧索引
                start_frame = int(start_time * frame_rate)
                end_frame = int(end_time * frame_rate)
                
                # 确保不越界
                start_frame = min(start_frame, n_frames - 1)
                end_frame = min(end_frame, n_frames)
                chunk_frames = end_frame - start_frame
                
                if chunk_frames <= 0:
                    peaks.extend([0.0, 0.0])
                    continue
                
                # 定位到精确位置
                wav.setpos(start_frame)
                frames_data = wav.readframes(chunk_frames)

                if not frames_data:
                    peaks.extend([0.0, 0.0])
                    continue

                try:
                    if sample_width == 2:  # 16-bit
                        fmt = f'<{len(frames_data) // 2}h'
                        samples_chunk = struct.unpack(fmt, frames_data)
                        max_val = 32768.0
                    elif sample_width == 4:  # 32-bit
                        fmt = f'<{len(frames_data) // 4}i'
                        samples_chunk = struct.unpack(fmt, frames_data)
                        max_val = 2147483648.0
                    elif sample_width == 1:  # 8-bit
                        samples_chunk = list(frames_data)
                        max_val = 128.0
                    else:
                        samples_chunk = [0]
                        max_val = 1.0

                    if n_channels > 1:
                        samples_chunk = samples_chunk[::n_channels]

                    if samples_chunk:
                        normalized = [s / max_val for s in samples_chunk]
                        peaks.append(float(min(normalized)))
                        peaks.append(float(max(normalized)))
                    else:
                        peaks.extend([0.0, 0.0])
                except:
                    peaks.extend([0.0, 0.0])

            return peaks, duration

    except Exception as e:
        print(f"[media] Wave波形生成失败: {e}")
        return [], 0


async def _generate_sprite_thumbnails(video_path: Path, output_path: Path, count: int = 20, cols: int = 5) -> dict:
    """
    生成Sprite雪碧图（将多个缩略图合并为单张图片，减少HTTP请求）

    Args:
        video_path: 视频文件路径
        output_path: 输出目录
        count: 缩略图数量
        cols: 每行列数

    Returns:
        dict: { sprite_url, thumb_width, thumb_height, cols, rows, timestamps }
    """
    import base64

    ffmpeg_cmd = config.get_ffmpeg_command()
    print(f"[media] 生成Sprite图，视频: {video_path}, FFmpeg: {ffmpeg_cmd}")

    # 获取视频时长
    duration = await _get_video_duration(video_path)
    print(f"[media] Sprite - 视频时长: {duration}秒")
    if duration <= 0:
        print(f"[media] Sprite - 无法获取时长，跳过Sprite生成")
        return None

    # 计算时间点
    interval = duration / count
    timestamps = [i * interval for i in range(count)]

    # 计算行数
    rows = (count + cols - 1) // cols

    # 使用FFmpeg的tile滤镜生成Sprite图
    # 先提取帧，再合并
    sprite_file = output_path / "sprite.jpg"

    # 生成用于提取帧的时间点参数
    select_expr = '+'.join([f'eq(n,{int(t * 25)})' for t in timestamps])  # 假设25fps

    # 使用更可靠的方法：逐帧提取再合并
    temp_frames = []
    thumb_width = 160
    thumb_height = None

    for i, ts in enumerate(timestamps):
        cmd = [
            ffmpeg_cmd,
            '-ss', str(ts),
            '-i', str(video_path),
            '-vframes', '1',
            '-vf', f'scale={thumb_width}:-1',
            '-f', 'image2pipe',
            '-vcodec', 'mjpeg',
            '-q:v', '5',
            '-'
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                timeout=10
            )
            if result.returncode == 0 and result.stdout:
                temp_frames.append(result.stdout)
            else:
                temp_frames.append(None)
        except:
            temp_frames.append(None)

    # 使用PIL合并图片（如果可用）
    try:
        from PIL import Image

        # 获取第一张有效图片的尺寸
        first_valid = None
        for frame_data in temp_frames:
            if frame_data:
                first_valid = Image.open(io.BytesIO(frame_data))
                thumb_width, thumb_height = first_valid.size
                break

        if not first_valid:
            return None

        # 创建Sprite画布
        sprite_width = thumb_width * cols
        sprite_height = thumb_height * rows
        sprite_img = Image.new('RGB', (sprite_width, sprite_height), (0, 0, 0))

        # 粘贴各帧
        for i, frame_data in enumerate(temp_frames):
            if frame_data:
                try:
                    img = Image.open(io.BytesIO(frame_data))
                    col = i % cols
                    row = i // cols
                    x = col * thumb_width
                    y = row * thumb_height
                    sprite_img.paste(img, (x, y))
                except:
                    pass

        # 保存Sprite图
        sprite_img.save(str(sprite_file), 'JPEG', quality=80)

        # 转为base64
        with open(sprite_file, 'rb') as f:
            sprite_base64 = base64.b64encode(f.read()).decode('utf-8')

        return {
            "sprite": f"data:image/jpeg;base64,{sprite_base64}",
            "sprite_url": f"/api/media/{output_path.name}/sprite.jpg",
            "thumb_width": thumb_width,
            "thumb_height": thumb_height,
            "cols": cols,
            "rows": rows,
            "count": count,
            "timestamps": timestamps,
            "duration": duration
        }

    except ImportError:
        # PIL不可用，回退到单独的缩略图
        print("[media] PIL不可用，使用单独缩略图模式")
        return None


@router.get("/{job_id}/video")
async def get_video(job_id: str, request: Request):
    """
    获取视频文件（支持Range请求，自动Proxy转码）

    优先返回Proxy视频（如果存在），否则返回源视频
    对于不兼容的格式或编码（如HEVC/H.265），会触发异步生成Proxy
    """
    job_dir = config.JOBS_DIR / job_id
    logger.debug(f"[media] 收到视频请求: {job_id}")

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    # 1. 按优先级查找可用视频
    # 优先级：720p > remux > 360p > 源视频
    proxy_720p = job_dir / "proxy_720p.mp4"
    remux_video = job_dir / "remux.mp4"
    preview_360p = job_dir / "preview_360p.mp4"

    if proxy_720p.exists():
        logger.debug(f"[media] 返回720p高清视频")
        return _serve_file_with_range(proxy_720p, request, 'video/mp4')

    if remux_video.exists():
        print(f"[media] 返回重封装视频")
        return _serve_file_with_range(remux_video, request, 'video/mp4')

    if preview_360p.exists():
        print(f"[media] 返回360p预览视频")
        return _serve_file_with_range(preview_360p, request, 'video/mp4')

    # 2. 查找源视频
    video_file = _find_video_file(job_dir)
    if not video_file:
        print(f"[media] 视频文件不存在: {job_id}")
        raise HTTPException(status_code=404, detail="视频文件不存在")

    logger.debug(f"找到源视频: {video_file.name}, 扩展名: {video_file.suffix.lower()}")

    # 3. 检查是否需要生成Proxy（先检查扩展名，再检查编码）
    needs_transcode = False
    transcode_reason = ""

    # 3.1 检查文件扩展名
    if video_file.suffix.lower() in NEED_TRANSCODE_FORMATS:
        needs_transcode = True
        transcode_reason = f"格式不兼容 ({video_file.suffix})"
        print(f"[media] 扩展名需要转码: {video_file.suffix.lower()} 属于 {NEED_TRANSCODE_FORMATS}")
    # 3.2 即使是 .mp4/.webm，也需检查实际编码是否兼容
    elif video_file.suffix.lower() in BROWSER_COMPATIBLE_FORMATS:
        codec = _get_video_codec(video_file)
        if codec and codec in NEED_TRANSCODE_CODECS:
            needs_transcode = True
            transcode_reason = f"编码不兼容 ({codec.upper()})"
            logger.debug(f"视频编码检测: {codec} (不兼容，需转码)")
        else:
            # 编码兼容，无需转码
            logger.debug(f"视频编码检测: {codec} (兼容)")

    if needs_transcode:
        print(f"[media] 视频需要转码: {transcode_reason}")
        # 使用 MediaPrepService 管理转码任务（渐进式：360p 优先，720p 后续）
        from app.services.media_prep_service import get_media_prep_service
        media_prep = get_media_prep_service()

        # 360p 预览视频路径
        preview_360p = job_dir / "preview_360p.mp4"

        # 检查 360p 预览状态
        preview_status = media_prep.get_preview_status(job_id)
        preview_in_progress = preview_status and preview_status.get("status") in ["queued", "processing"]
        preview_completed = preview_status and preview_status.get("status") == "completed"
        print(f"[media] 360p预览状态: {preview_status}, 进行中={preview_in_progress}, 已完成={preview_completed}")

        # 检查 720p Proxy 状态
        proxy_status = media_prep.get_proxy_status(job_id)
        proxy_in_progress = proxy_status and proxy_status.get("status") in ["queued", "processing"]
        proxy_completed = proxy_status and proxy_status.get("status") == "completed"
        print(f"[media] 720p Proxy状态: {proxy_status}, 进行中={proxy_in_progress}, 已完成={proxy_completed}")

        # 720p Proxy 视频路径（使用函数开始时定义的变量）
        # proxy_720p 已在 line 509 定义

        # 优先级1: 如果 720p 已完成，返回 720p 视频
        if proxy_completed and proxy_720p.exists():
            print(f"[media] 返回已完成的720p高清视频")
            return _serve_file_with_range(proxy_720p, request, 'video/mp4')

        # 优先级2: 如果 360p 已完成，返回 360p 视频（720p可能正在处理或未启动）
        if preview_completed and preview_360p.exists():
            print(f"[media] 返回已完成的360p预览视频")
            return _serve_file_with_range(preview_360p, request, 'video/mp4')

        # 如果有任务正在处理，返回进度信息
        if preview_in_progress or proxy_in_progress:
            # 优先显示 360p 进度（因为它更快完成）
            current_progress = preview_status.get("progress", 0) if preview_in_progress else proxy_status.get("progress", 0)
            current_stage = "360p预览" if preview_in_progress else "720p高清"
            print(f"[media] 转码进行中: {current_stage}, 进度={current_progress}%")
            raise HTTPException(
                status_code=202,
                detail={
                    "message": f"正在生成{current_stage}版本...",
                    "progress": current_progress,
                    "proxy_generating": True,
                    "stage": "preview_360p" if preview_in_progress else "proxy_720p"
                }
            )

        # 只启动 360p 预览转码（快速让用户看到视频）
        # 720p 高清转码将在转录完成后、队列空闲时自动触发
        print(f"[media] 调用 enqueue_preview() 启动360p转码: {video_file} -> {preview_360p}")
        success = media_prep.enqueue_preview(job_id, video_file, preview_360p, priority=5)
        print(f"[media] enqueue_preview() 返回: {success}")

        raise HTTPException(
            status_code=202,
            detail={
                "message": f"视频{transcode_reason}，正在生成预览版本...",
                "format": video_file.suffix,
                "proxy_generating": True,
                "stage": "preview_360p"
            }
        )

    # 4. 返回兼容格式的源视频
    # 日志已在编码检测时输出，此处不再重复
    return _serve_file_with_range(video_file, request, 'video/mp4')


@router.get("/{job_id}/audio")
async def get_audio(job_id: str, request: Request):
    """获取音频文件（支持Range请求）"""
    job_dir = config.JOBS_DIR / job_id
    audio_file = job_dir / "audio.wav"

    if not audio_file.exists():
        raise HTTPException(status_code=404, detail="音频文件不存在")

    return _serve_file_with_range(audio_file, request, 'audio/wav')


@router.get("/{job_id}/peaks")
async def get_audio_peaks(job_id: str, samples: int = 0, method: str = "auto"):
    """
    获取音频波形峰值数据（优化：动态采样密度）

    Args:
        job_id: 任务ID
        samples: 采样点数（0=自动计算，建议传0）
        method: 生成方法 - "auto"(自动选择), "ffmpeg"(FFmpeg加速), "wave"(Python原生)

    Returns:
        JSON: { peaks: [min, max, min, max, ...], duration: 180.5, method: "ffmpeg" }
    """
    job_dir = config.JOBS_DIR / job_id
    audio_file = job_dir / "audio.wav"

    # 【关键修改】获取音频时长，动态计算采样点
    if samples <= 0:
        # 获取视频/音频时长
        video_file = _find_video_file(job_dir)
        duration = 0.0

        if video_file:
            duration = await _get_video_duration(video_file)

        # 如果从视频获取失败，尝试从音频获取
        if duration <= 0 and audio_file.exists():
            try:
                # 先尝试用 FFmpeg 快速获取时长
                duration = await _get_video_duration(audio_file)
            except:
                # 降级：使用默认值
                duration = 60

        if duration <= 0:
            duration = 60  # 最终降级默认值

        # 每秒20个采样点，保证放大后的连续性
        # 上限100k（约800KB JSON），下限4k
        target_samples = int(duration * 20)
        samples = max(4000, min(target_samples, 100000))
        logger.debug(f"动态采样：时长{duration:.1f}s → {samples}个采样点")

    # 【修复】使用版本号标识缓存，算法更新后自动使旧缓存失效
    # v2: 修复了采样位置计算的累积误差问题
    # v3: 改用基于时间位置的精确计算，确保波形与时间轴严格对齐
    PEAKS_CACHE_VERSION = 3
    peaks_cache_file = job_dir / f"peaks_{samples}_v{PEAKS_CACHE_VERSION}.json"

    # 检查缓存
    if peaks_cache_file.exists():
        try:
            with open(peaks_cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                # 验证缓存版本
                if cached_data.get("cache_version") == PEAKS_CACHE_VERSION:
                    return JSONResponse(cached_data)
        except:
            pass

    if not audio_file.exists():
        raise HTTPException(status_code=404, detail="音频文件不存在")

    # 根据文件大小选择方法
    file_size_mb = audio_file.stat().st_size / (1024 * 1024)

    if method == "auto":
        # 大于50MB使用FFmpeg加速
        use_ffmpeg = file_size_mb > 50
    elif method == "ffmpeg":
        use_ffmpeg = True
    else:
        use_ffmpeg = False

    try:
        if use_ffmpeg:
            peaks, duration = _generate_peaks_with_ffmpeg(audio_file, samples)
            used_method = "ffmpeg"
        else:
            peaks, duration = _generate_peaks_with_wave(audio_file, samples)
            used_method = "wave"

        # 如果FFmpeg失败，回退到wave
        if not peaks and use_ffmpeg:
            peaks, duration = _generate_peaks_with_wave(audio_file, samples)
            used_method = "wave_fallback"

        result = {
            "peaks": peaks,
            "duration": duration,
            "method": used_method,
            "samples": len(peaks) // 2,
            "cache_version": PEAKS_CACHE_VERSION  # 添加版本号到缓存
        }

        # 缓存结果
        try:
            with open(peaks_cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f)
        except:
            pass

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"波形数据生成失败: {str(e)}")


@router.get("/{job_id}/proxy-status")
async def check_proxy_status(job_id: str):
    """
    获取 Proxy 视频完整状态（用于前端刷新后恢复）

    Returns:
        JSON: {
            "state": "transcoding_720",  # 当前状态
            "progress": 45.5,            # 当前进度百分比
            "decision": "transcode_full", # 转码决策
            "urls": {
                "360p": "/api/media/.../video/preview",
                "720p": "/api/media/.../video",
                "source": "/api/media/.../video/source"
            },
            "error": null,
            "started_at": null,
            "estimated_remaining": null
        }
    """
    job_dir = config.JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    # 从 MediaPrepService 获取任务状态
    from app.services.media_prep_service import get_media_prep_service, TranscodeDecision
    media_prep = get_media_prep_service()
    task_status = media_prep.get_full_task_status(job_id)

    # 检查文件存在性，构建 URLs
    preview_360p = job_dir / "preview_360p.mp4"
    proxy_720p = job_dir / "proxy_720p.mp4"  # 修复：使用正确的文件名
    remux_video = job_dir / "remux.mp4"
    source_video = _find_video_file(job_dir)

    urls = {
        "360p": f"/api/media/{job_id}/video/preview" if preview_360p.exists() else None,
        "720p": f"/api/media/{job_id}/video" if (proxy_720p.exists() or remux_video.exists()) else None,
        "source": f"/api/media/{job_id}/video" if source_video else None
    }

    # 如果没有任务状态，根据文件存在性推断状态
    if not task_status:
        if proxy_720p.exists() or remux_video.exists():
            state = "ready_720p"
            progress = 100
        elif preview_360p.exists():
            state = "ready_360p"
            progress = 100

            # 【新增】360p已完成但720p不存在，检查是否需要触发720p转码
            if not proxy_720p.exists() and not remux_video.exists() and source_video:
                print(f"[media] 360p已完成但720p不存在，检查是否触发720p转码: {job_id}")
                try:
                    # 检查720p是否已在队列中
                    proxy_status = media_prep.get_proxy_status(job_id)
                    if not proxy_status or proxy_status.get("status") not in ["queued", "processing"]:
                        # 720p未在队列中，检查转录队列是否空闲
                        from app.services.job_queue_service import get_queue_service

                        queue_idle = True  # 默认认为空闲
                        try:
                            queue_service = get_queue_service()
                            if queue_service:
                                # 检查是否有正在处理或等待的任务
                                active_jobs = [
                                    j for j in queue_service.jobs.values()
                                    if j.status in ['pending', 'processing', 'queued']
                                ]
                                queue_idle = len(active_jobs) == 0
                                print(f"[media] 队列状态检查: 活跃任务数={len(active_jobs)}, 空闲={queue_idle}")
                        except Exception as e:
                            print(f"[media] 检查队列状态失败，默认认为空闲: {e}")

                        if queue_idle:
                            # 队列空闲，自动触发720p转码
                            print(f"[media] 队列空闲，自动触发720p转码: {job_id}")
                            media_prep.enqueue_proxy(job_id, source_video, proxy_720p, priority=10)
                        else:
                            # 队列繁忙，使用低优先级
                            print(f"[media] 队列繁忙，低优先级触发720p转码: {job_id}")
                            media_prep.enqueue_proxy(job_id, source_video, proxy_720p, priority=1)
                except Exception as e:
                    print(f"[media] 自动触发720p转码失败: {e}")
        elif source_video:
            # 分析是否需要转码
            from app.utils.media_analyzer import media_analyzer
            try:
                video_info = media_analyzer.analyze_sync(source_video)
                video_info['container'] = source_video.suffix.lower()
                decision = media_prep.analyze_transcode_decision(video_info)

                if decision == TranscodeDecision.DIRECT_PLAY:
                    state = "direct_play"
                    progress = 100
                else:
                    # 需要转码但任务还没启动，自动启动转码
                    state = "analyzing"
                    progress = 0

                    # 根据决策类型启动相应任务
                    if decision == TranscodeDecision.REMUX_ONLY:
                        # 仅重封装
                        remux_output = job_dir / "remux.mp4"
                        media_prep.enqueue_remux(job_id, source_video, remux_output, priority=3)
                        print(f"[media] 自动启动重封装任务: {job_id}")
                    else:
                        # 需要完整转码，启动360p预览
                        media_prep.enqueue_preview(job_id, source_video, preview_360p, priority=5)
                        print(f"[media] 自动启动360p预览转码: {job_id}")
            except Exception as e:
                print(f"[media] 分析视频失败: {e}")
                state = "idle"
                progress = 0
        else:
            state = "idle"
            progress = 0

        return JSONResponse({
            "state": state,
            "progress": progress,
            "decision": None,
            "urls": urls,
            "error": None,
            "started_at": None,
            "estimated_remaining": None
        })

    # 【新增】检查360p是否需要重试（处理意外关闭的情况）
    # 如果360p不存在且没有正在进行的360p任务，自动启动360p转码
    if not preview_360p.exists() and source_video:
        preview_status = media_prep.get_preview_status(job_id)
        preview_in_progress = preview_status and preview_status.get("status") in ["queued", "processing"]

        if not preview_in_progress:
            print(f"[media] 360p不存在且未在队列中，自动启动360p转码: {job_id}")
            try:
                # 分析视频决策
                from app.utils.media_analyzer import media_analyzer
                video_info = media_analyzer.analyze_sync(source_video)
                video_info['container'] = source_video.suffix.lower()
                decision = media_prep.analyze_transcode_decision(video_info)

                if decision != TranscodeDecision.DIRECT_PLAY:
                    # 需要转码，启动360p预览
                    media_prep.enqueue_preview(job_id, source_video, preview_360p, priority=5)
                    print(f"[media] 已自动启动360p预览转码: {job_id}")
            except Exception as e:
                print(f"[media] 自动启动360p转码失败: {e}")

    # 返回完整状态
    return JSONResponse({
        "state": task_status.get("state", "idle"),
        "progress": task_status.get("progress", 0),
        "decision": task_status.get("decision"),
        "urls": urls,
        "error": task_status.get("error"),
        "started_at": task_status.get("started_at"),
        "estimated_remaining": task_status.get("estimated_remaining")
    })


@router.get("/{job_id}/thumbnail")
async def get_thumbnail(job_id: str):
    """
    获取任务的单个缩略图（第一帧）用于任务卡片展示（第二阶段修复：实时更新）

    Args:
        job_id: 任务ID

    Returns:
        Base64编码的缩略图 or JSON占位符
    """
    job_dir = config.JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    # 检查缓存的缩略图文件
    cached_thumbnail = job_dir / "thumbnail.jpg"
    if cached_thumbnail.exists():
        try:
            import base64
            with open(cached_thumbnail, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
            return JSONResponse({
                "thumbnail": f"data:image/jpeg;base64,{img_base64}",
                "cached": True
            })
        except:
            pass  # 缓存读取失败，重新生成

    try:
        import base64

        # 查找视频文件
        video_file = _find_video_file(job_dir)
        if not video_file:
            proxy = job_dir / "proxy_720p.mp4"
            if proxy.exists():
                video_file = proxy
            else:
                return JSONResponse({
                    "thumbnail": None,
                    "message": "视频文件不存在"
                })

        # 获取视频分辨率（决定缩略图大小）
        width, height = await _get_video_resolution(video_file)

        # 计算缩略图尺寸：只有4K及以上视频才压缩，其他保持原分辨率（但限制最大1920px）
        if width > 3840:  # 4K视频
            thumb_width = 1920
        elif width > 1920:  # 大于1080p但不到4K
            thumb_width = width  # 保持原分辨率
        elif width > 0:
            thumb_width = width  # 保持原分辨率
        else:
            thumb_width = 1280  # 默认值（无法检测分辨率时）

        # 获取第一帧作为缩略图
        ffmpeg_cmd = config.get_ffmpeg_command()

        cmd = [
            ffmpeg_cmd,
            '-ss', '1',  # 从1秒开始（避免纯黑帧）
            '-i', str(video_file),
            '-vframes', '1',  # 只提取1帧
            '-f', 'image2pipe',
            '-vcodec', 'mjpeg',
            '-vf', f'scale={thumb_width}:-1',  # 根据原视频分辨率设置宽度
            '-q:v', '2',  # 高质量（2-5之间，数字越小质量越高）
            '-'
        ]

        result = subprocess.run(
            cmd, capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            timeout=10
        )

        if result.returncode == 0 and result.stdout:
            # 缓存到文件系统
            try:
                with open(cached_thumbnail, 'wb') as f:
                    f.write(result.stdout)
                print(f"[media] 缩略图已缓存: {job_id} (宽度: {thumb_width}px)")
            except Exception as e:
                print(f"[media] 缓存缩略图失败: {e}")

            img_base64 = base64.b64encode(result.stdout).decode('utf-8')
            return JSONResponse({
                "thumbnail": f"data:image/jpeg;base64,{img_base64}",
                "width": thumb_width,
                "cached": False
            })
        else:
            return JSONResponse({
                "thumbnail": None,
                "message": "无法生成缩略图"
            })

    except Exception as e:
        return JSONResponse({
            "thumbnail": None,
            "message": str(e)
        })


@router.get("/{job_id}/thumbnails")
async def get_thumbnails(job_id: str, count: int = 10, sprite: bool = True):
    """
    获取视频缩略图

    Args:
        job_id: 任务ID
        count: 缩略图数量（默认10张）
        sprite: 是否使用Sprite雪碧图（默认True，减少请求数）

    Returns:
        - sprite=True: { sprite, thumb_width, thumb_height, cols, rows, timestamps }
        - sprite=False: { thumbnails: [base64_img1, ...], timestamps: [...] }
    """
    job_dir = config.JOBS_DIR / job_id

    # 检查Sprite缓存
    sprite_cache = job_dir / f"sprite_{count}.json"
    thumbnails_cache = job_dir / f"thumbnails_{count}.json"

    if sprite and sprite_cache.exists():
        try:
            # V3.1.0+dev.20260104.01: 添加 encoding='utf-8'
            with open(sprite_cache, 'r', encoding='utf-8') as f:
                return JSONResponse(json.load(f))
        except:
            pass

    if not sprite and thumbnails_cache.exists():
        try:
            # V3.1.0+dev.20260104.01: 添加 encoding='utf-8'
            with open(thumbnails_cache, 'r', encoding='utf-8') as f:
                return JSONResponse(json.load(f))
        except:
            pass

    # 查找视频文件
    video_file = _find_video_file(job_dir)
    if not video_file:
        proxy = job_dir / "proxy_720p.mp4"
        if proxy.exists():
            video_file = proxy
        else:
            raise HTTPException(status_code=404, detail="视频文件不存在")

    try:
        import base64

        # 尝试生成Sprite图（如果请求且PIL可用）
        if sprite:
            sprite_result = await _generate_sprite_thumbnails(video_file, job_dir, count)
            if sprite_result:
                # 缓存结果
                try:
                    # V3.1.0+dev.20260104.01: 添加 encoding='utf-8'
                    with open(sprite_cache, 'w', encoding='utf-8') as f:
                        json.dump(sprite_result, f)
                except:
                    pass
                return JSONResponse(sprite_result)

        # 回退到单独缩略图模式
        print(f"[media] 回退到单独缩略图模式，视频文件: {video_file}")
        ffmpeg_cmd = config.get_ffmpeg_command()
        duration = await _get_video_duration(video_file)
        print(f"[media] 获取到视频时长: {duration}秒")

        if duration <= 0:
            raise HTTPException(status_code=500, detail=f"无法获取视频时长，视频路径: {video_file}")

        interval = duration / count
        thumbnails = []
        timestamps = []

        for i in range(count):
            timestamp = i * interval
            timestamps.append(timestamp)

            cmd = [
                ffmpeg_cmd,
                '-ss', str(timestamp),
                '-i', str(video_file),
                '-vframes', '1',
                '-f', 'image2pipe',
                '-vcodec', 'mjpeg',
                '-vf', 'scale=160:-1',
                '-q:v', '5',
                '-'
            ]

            result = subprocess.run(
                cmd, capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                timeout=10
            )

            if result.returncode == 0 and result.stdout:
                img_base64 = base64.b64encode(result.stdout).decode('utf-8')
                thumbnails.append(f"data:image/jpeg;base64,{img_base64}")
            else:
                thumbnails.append(None)

        result = {
            "thumbnails": thumbnails,
            "timestamps": timestamps,
            "duration": duration,
            "sprite": False
        }

        # 缓存结果
        try:
            # V3.1.0+dev.20260104.01: 添加 encoding='utf-8'
            with open(thumbnails_cache, 'w', encoding='utf-8') as f:
                json.dump(result, f)
        except:
            pass

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"缩略图生成失败: {str(e)}")


@router.get("/{job_id}/sprite.jpg")
async def get_sprite_image(job_id: str):
    """直接获取Sprite图片文件"""
    job_dir = config.JOBS_DIR / job_id
    sprite_file = job_dir / "sprite.jpg"

    if not sprite_file.exists():
        raise HTTPException(status_code=404, detail="Sprite图不存在")

    return FileResponse(str(sprite_file), media_type="image/jpeg")


@router.get("/{job_id}/video/preview")
async def get_preview_video(job_id: str, request: Request):
    """
    获取 360p 预览视频（渐进式加载第一阶段）

    用于在转码过程中快速预览视频内容
    """
    job_dir = config.JOBS_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    # 查找 360p 预览视频
    preview_360p = job_dir / "preview_360p.mp4"
    if preview_360p.exists():
        return _serve_file_with_range(preview_360p, request, 'video/mp4')

    # 如果没有 360p，尝试返回 720p proxy
    proxy_video = job_dir / "proxy_720p.mp4"
    if proxy_video.exists():
        return _serve_file_with_range(proxy_video, request, 'video/mp4')

    # 都没有，返回 202 表示正在生成
    raise HTTPException(
        status_code=202,
        detail={
            "message": "预览视频正在生成中...",
            "generating": True
        }
    )


@router.get("/{job_id}/status/progressive")
async def get_progressive_status(job_id: str):
    """
    获取渐进式加载状态

    返回当前视频生成的状态信息，包括：
    - 是否需要转码
    - 360p 预览进度/状态
    - 720p 高质量进度/状态
    - 可用的视频 URL
    """
    job_dir = config.JOBS_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    # 检查各种视频文件
    video_file = _find_video_file(job_dir)
    preview_360p = job_dir / "preview_360p.mp4"
    proxy_720p = job_dir / "proxy_720p.mp4"

    # 判断是否需要转码
    needs_transcode = False
    transcode_reason = ""

    if video_file:
        if video_file.suffix.lower() in NEED_TRANSCODE_FORMATS:
            needs_transcode = True
            transcode_reason = f"格式不兼容 ({video_file.suffix})"
        elif video_file.suffix.lower() in BROWSER_COMPATIBLE_FORMATS:
            codec = _get_video_codec(video_file)
            if codec and codec in NEED_TRANSCODE_CODECS:
                needs_transcode = True
                transcode_reason = f"编码不兼容 ({codec.upper()})"

    # 获取生成状态
    preview_status = None
    proxy_status = None

    # 从 MediaPrepService 获取转码状态
    from app.services.media_prep_service import get_media_prep_service
    media_prep = get_media_prep_service()

    # 获取 360p 预览状态
    preview_status_info = media_prep.get_preview_status(job_id)
    if preview_status_info:
        preview_status = {
            "status": preview_status_info.get("status", "not_started"),
            "progress": preview_status_info.get("progress", 0)
        }

    # 获取 720p proxy 状态
    proxy_status_info = media_prep.get_proxy_status(job_id)
    if proxy_status_info:
        proxy_status = {
            "status": proxy_status_info.get("status", "not_started"),
            "progress": proxy_status_info.get("progress", 0)
        }

    # 关键修复：如果需要转码但没有转码任务，自动触发 360p 预览转码
    if needs_transcode and video_file:
        # 检查是否需要启动转码
        preview_exists = preview_360p.exists()
        preview_in_progress = preview_status_info and preview_status_info.get("status") in ["queued", "processing"]

        if not preview_exists and not preview_in_progress:
            # 启动360p预览转码
            print(f"[media] progressive-status 检测到需要转码，自动触发: {job_id}")
            success = media_prep.enqueue_preview(job_id, video_file, preview_360p, priority=5)
            print(f"[media] enqueue_preview() 返回: {success}")

            # 更新状态为已入队
            if success:
                preview_status_info = media_prep.get_preview_status(job_id)
                if preview_status_info:
                    preview_status = {
                        "status": preview_status_info.get("status", "queued"),
                        "progress": preview_status_info.get("progress", 0)
                    }

    # 构建响应
    result = {
        "job_id": job_id,
        "needs_transcode": needs_transcode,
        "transcode_reason": transcode_reason,
        "preview_360p": {
            "exists": preview_360p.exists(),
            "url": f"/api/media/{job_id}/video/preview" if preview_360p.exists() else None,
            "size": preview_360p.stat().st_size if preview_360p.exists() else 0,
            "status": preview_status  # 包含转码状态和进度
        },
        "proxy_720p": {
            "exists": proxy_720p.exists(),
            "url": f"/api/media/{job_id}/video" if proxy_720p.exists() else None,
            "size": proxy_720p.stat().st_size if proxy_720p.exists() else 0,
            "status": proxy_status
        },
        "source": {
            "exists": video_file is not None,
            "filename": video_file.name if video_file else None,
            "compatible": not needs_transcode,
            "url": f"/api/media/{job_id}/video" if video_file and not needs_transcode else None
        },
        "recommended_url": None  # 推荐使用的视频 URL
    }

    # 确定推荐的视频 URL
    if proxy_720p.exists():
        result["recommended_url"] = f"/api/media/{job_id}/video"
        result["current_resolution"] = "720p"
    elif preview_360p.exists():
        result["recommended_url"] = f"/api/media/{job_id}/video/preview"
        result["current_resolution"] = "360p"
    elif video_file and not needs_transcode:
        result["recommended_url"] = f"/api/media/{job_id}/video"
        result["current_resolution"] = "source"
    else:
        result["current_resolution"] = None

    return JSONResponse(result)


@router.post("/{job_id}/generate-preview")
async def trigger_preview_generation(job_id: str):
    """
    手动触发 360p 预览视频生成

    在某些情况下，可能需要手动触发预览生成
    """
    job_dir = config.JOBS_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    video_file = _find_video_file(job_dir)
    if not video_file:
        raise HTTPException(status_code=404, detail="视频文件不存在")

    preview_360p = job_dir / "preview_360p.mp4"

    if preview_360p.exists():
        return JSONResponse({
            "success": True,
            "message": "预览视频已存在",
            "url": f"/api/media/{job_id}/video/preview"
        })

    # 异步生成 360p 预览
    from utils.progressive_video_generator import progressive_video_generator
    try:
        from app.services.sse_service import get_sse_manager
        progressive_video_generator.set_sse_manager(get_sse_manager())
    except:
        pass

    asyncio.create_task(
        progressive_video_generator.generate_360p_preview(job_id, video_file, preview_360p)
    )

    return JSONResponse({
        "success": True,
        "message": "预览视频生成已启动",
        "generating": True
    })


@router.post("/{job_id}/post-process")
async def post_process_transcription(job_id: str):
    """
    转录后处理：预生成编辑器所需的所有数据
    在转录完成后调用，异步生成波形、缩略图、Proxy视频

    Returns:
        JSON: { peaks: bool, thumbnails: bool, proxy: bool, sprite: bool }
    """
    job_dir = config.JOBS_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    results = {
        "peaks": False,
        "thumbnails": False,
        "sprite": False,
        "proxy": False,
        "proxy_needed": False
    }

    # 1. 生成波形峰值（使用动态采样）
    try:
        audio_file = job_dir / "audio.wav"
        if audio_file.exists():
            # 使用 samples=0 让后端自动计算采样点数
            await get_audio_peaks(job_id, 0)
            results["peaks"] = True
    except Exception as e:
        print(f"[media] 波形生成失败: {e}")

    # 2. 生成缩略图（优先Sprite图）
    try:
        sprite_cache = job_dir / "sprite_10.json"
        thumbnails_cache = job_dir / "thumbnails_10.json"
        if not sprite_cache.exists() and not thumbnails_cache.exists():
            await get_thumbnails(job_id, 10, sprite=True)
        results["thumbnails"] = True
        results["sprite"] = sprite_cache.exists()
    except Exception as e:
        print(f"[media] 缩略图生成失败: {e}")

    # 3. 检查是否需要生成Proxy
    video_file = _find_video_file(job_dir)
    if video_file and video_file.suffix.lower() in NEED_TRANSCODE_FORMATS:
        results["proxy_needed"] = True
        proxy_video = job_dir / "proxy_720p.mp4"
        if not proxy_video.exists():
            # 使用 MediaPrepService 管理转码
            from app.services.media_prep_service import get_media_prep_service
            media_prep = get_media_prep_service()
            media_prep.enqueue_proxy(job_id, video_file, proxy_video, priority=20)
            results["proxy"] = True

    return JSONResponse(results)


@router.get("/{job_id}/srt")
async def get_srt_content(job_id: str):
    """
    获取SRT字幕文件内容

    Returns:
        JSON: { job_id, filename, content, encoding }
    """
    job_dir = config.JOBS_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    srt_file = None
    for file in job_dir.iterdir():
        if file.suffix.lower() == '.srt':
            srt_file = file
            break

    if not srt_file:
        raise HTTPException(status_code=404, detail="SRT文件不存在")

    try:
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read()

        return JSONResponse({
            "job_id": job_id,
            "filename": srt_file.name,
            "content": content,
            "encoding": "utf-8"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取SRT文件失败: {str(e)}")


@router.post("/{job_id}/srt")
async def save_srt_content(job_id: str, request: Request):
    """
    保存编辑后的SRT字幕文件
    V3.1.1+dev.20260106.03: 保存前自动修复时间戳重叠

    Body:
        {
            content: "1\n00:00:01,000 --> 00:00:03,000\n...",
            auto_repair: true  // 可选，默认 true，是否自动修复重叠
        }
    """
    job_dir = config.JOBS_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    try:
        body = await request.json()
        content = body.get("content")
        auto_repair = body.get("auto_repair", True)  # 默认启用自动修复

        if not content:
            raise HTTPException(status_code=400, detail="缺少content参数")

        # V3.1.1+dev.20260106.03: 自动修复时间戳重叠
        repaired_count = 0
        if auto_repair:
            content, repaired_count = repair_srt_overlaps(content, gap_ms=1.0)
            if repaired_count > 0:
                logger.info(f"[{job_id}] 自动修复了 {repaired_count} 处时间戳重叠")

        srt_file = None
        for file in job_dir.iterdir():
            if file.suffix.lower() == '.srt':
                srt_file = file
                break

        if not srt_file:
            video_file = _find_video_file(job_dir)
            if video_file:
                srt_file = job_dir / f"{video_file.stem}.srt"
            else:
                srt_file = job_dir / "output.srt"

        # 备份原文件
        if srt_file.exists():
            backup_file = job_dir / f"{srt_file.stem}.srt.bak"
            import shutil
            shutil.copy2(srt_file, backup_file)

        with open(srt_file, 'w', encoding='utf-8') as f:
            f.write(content)

        response_data = {
            "success": True,
            "message": "SRT文件保存成功",
            "filename": srt_file.name
        }

        # V3.1.1+dev.20260106.03: 返回修复信息
        if repaired_count > 0:
            response_data["repaired_overlaps"] = repaired_count
            response_data["message"] = f"SRT文件保存成功，自动修复了 {repaired_count} 处重叠"

        return JSONResponse(response_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存SRT文件失败: {str(e)}")


@router.get("/{job_id}/ass")
async def get_ass_content(job_id: str):
    """
    获取 ASS 字幕文件内容

    Returns:
        JSON: { job_id, filename, content, encoding }
    """
    job_dir = config.JOBS_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    ass_file = None
    for file in job_dir.iterdir():
        if file.suffix.lower() == '.ass':
            ass_file = file
            break

    if not ass_file:
        raise HTTPException(status_code=404, detail="ASS文件不存在")

    try:
        with open(ass_file, 'r', encoding='utf-8-sig') as f:
            content = f.read()

        return JSONResponse({
            "job_id": job_id,
            "filename": ass_file.name,
            "content": content,
            "encoding": "utf-8"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取ASS文件失败: {str(e)}")


@router.post("/{job_id}/ass/generate")
async def generate_ass_from_srt(job_id: str, request: Request):
    """
    从 SRT 文件生成 ASS 字幕文件
    V3.1.1+dev.20260106.03: 生成前自动修复时间戳重叠

    Request Body:
        {
            "style_preset": "default" | "movie" | "news" | "danmaku",
            "title": "字幕标题",
            "video_width": 1920,
            "video_height": 1080,
            "auto_repair": true  // 可选，默认 true，是否自动修复重叠
        }

    Returns:
        JSON: { job_id, filename, message, repaired_overlaps? }
    """
    job_dir = config.JOBS_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    # 查找 SRT 文件
    srt_file = None
    for file in job_dir.iterdir():
        if file.suffix.lower() == '.srt':
            srt_file = file
            break

    if not srt_file:
        raise HTTPException(status_code=404, detail="SRT文件不存在")

    try:
        # 解析请求参数
        body = await request.json()
        style_preset = body.get("style_preset", "default")
        title = body.get("title", srt_file.stem)
        video_width = body.get("video_width", 1920)
        video_height = body.get("video_height", 1080)
        auto_repair = body.get("auto_repair", True)  # V3.1.1+dev.20260106.03: 默认启用自动修复

        # 读取 SRT 文件并解析为字幕数据
        subtitles = []
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 简单的 SRT 解析
        blocks = content.strip().split('\n\n')
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # 解析时间戳行
                time_line = lines[1]
                if ' --> ' in time_line:
                    start_str, end_str = time_line.split(' --> ')
                    start = _parse_srt_timestamp(start_str.strip())
                    end = _parse_srt_timestamp(end_str.strip())
                    text = '\n'.join(lines[2:])
                    subtitles.append({
                        "start": start,
                        "end": end,
                        "text": text
                    })

        # V3.1.1+dev.20260106.03: 自动修复时间戳重叠
        repaired_count = 0
        if auto_repair and subtitles:
            # 检测重叠数量
            overlaps = detect_timestamp_overlaps(subtitles)
            repaired_count = len(overlaps)

            if repaired_count > 0:
                subtitles = repair_timestamp_overlaps(subtitles, gap_ms=1.0)
                logger.info(f"[{job_id}] ASS生成前自动修复了 {repaired_count} 处时间戳重叠")

        # 生成 ASS 文件
        ass_file = job_dir / f"{srt_file.stem}.ass"
        ASSConverter.convert_from_subtitles(
            subtitles=subtitles,
            output_path=ass_file,
            style_preset=style_preset,
            title=title,
            video_width=video_width,
            video_height=video_height
        )

        logger.info(f"ASS 文件已生成: {ass_file}")

        response_data = {
            "job_id": job_id,
            "filename": ass_file.name,
            "message": "ASS文件生成成功"
        }

        # V3.1.1+dev.20260106.03: 返回修复信息
        if repaired_count > 0:
            response_data["repaired_overlaps"] = repaired_count
            response_data["message"] = f"ASS文件生成成功，自动修复了 {repaired_count} 处重叠"

        return JSONResponse(response_data)

    except Exception as e:
        logger.error(f"生成ASS文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成ASS文件失败: {str(e)}")


def _parse_srt_timestamp(timestamp: str) -> float:
    """
    解析 SRT 时间戳为秒数

    Args:
        timestamp: SRT 时间戳 (HH:MM:SS,mmm)

    Returns:
        秒数
    """
    time_part, ms_part = timestamp.replace(',', '.').split('.')
    h, m, s = map(int, time_part.split(':'))
    ms = int(ms_part)
    return h * 3600 + m * 60 + s + ms / 1000


@router.get("/{job_id}/info")
async def get_media_info(job_id: str, retry_missing: bool = True):
    """
    获取任务的媒体信息摘要（支持自动重试生成缺失资源）

    Args:
        job_id: 任务ID
        retry_missing: 是否在资源缺失时尝试重新生成（默认True）

    Returns:
        JSON: 包含视频、音频、SRT等文件的可用状态
    """
    job_dir = config.JOBS_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    video_file = _find_video_file(job_dir)
    audio_file = job_dir / "audio.wav"
    proxy_video = job_dir / "proxy_720p.mp4"
    peaks_cache = job_dir / "peaks_2000.json"
    sprite_cache = job_dir / "sprite_10.json"
    thumbnails_cache = job_dir / "thumbnails_10.json"
    thumbnail_single = job_dir / "thumbnail.jpg"

    srt_file = None
    for file in job_dir.iterdir():
        if file.suffix.lower() == '.srt':
            srt_file = file
            break

    needs_proxy = video_file and video_file.suffix.lower() in NEED_TRANSCODE_FORMATS

    # 检查编码是否需要转码（如 HEVC 等浏览器不兼容的编码）
    video_codec = None
    if video_file and not needs_proxy and video_file.suffix.lower() in BROWSER_COMPATIBLE_FORMATS:
        video_codec = _get_video_codec(video_file)
        if video_codec and video_codec in NEED_TRANSCODE_CODECS:
            needs_proxy = True

    # 获取Proxy生成状态（从 MediaPrepService）
    from app.services.media_prep_service import get_media_prep_service
    media_prep = get_media_prep_service()
    proxy_status = media_prep.get_proxy_status(job_id)

    # 智能重试：如果资源缺失且启用重试，则尝试异步生成
    if retry_missing:
        # 1. 如果波形数据缺失但音频文件存在，触发生成
        if not peaks_cache.exists() and audio_file.exists():
            try:
                print(f"[media] 检测到波形数据缺失，尝试生成: {job_id}")
                asyncio.create_task(_auto_generate_peaks(job_id, audio_file, peaks_cache))
            except Exception as e:
                print(f"[media] 波形数据生成失败: {e}")

        # 2. 如果缩略图缺失但视频文件存在，触发生成
        if not thumbnail_single.exists() and video_file:
            try:
                print(f"[media] 检测到缩略图缺失，尝试生成: {job_id}")
                asyncio.create_task(_auto_generate_thumbnail(job_id, video_file, thumbnail_single))
            except Exception as e:
                print(f"[media] 缩略图生成失败: {e}")

        # 3. 如果需要Proxy但不存在且未在生成中，触发生成
        if needs_proxy and not proxy_video.exists():
            if not (proxy_status and proxy_status.get("status") in ["queued", "processing"]):
                try:
                    print(f"[media] 检测到Proxy缺失，尝试生成: {job_id}")
                    media_prep.enqueue_proxy(job_id, video_file, proxy_video, priority=30)
                except Exception as e:
                    print(f"[media] Proxy视频生成失败: {e}")

    return JSONResponse({
        "job_id": job_id,
        "video": {
            "exists": video_file is not None,
            "filename": video_file.name if video_file else None,
            "format": video_file.suffix if video_file else None,
            "codec": video_codec,  # 视频编码（如 h264, hevc）
            "needs_proxy": needs_proxy,
            "proxy_exists": proxy_video.exists(),
            "proxy_generating": proxy_status and proxy_status.get("status") == "processing",
            "proxy_progress": proxy_status.get("progress", 0) if proxy_status else 0,
            "url": f"/api/media/{job_id}/video" if video_file or proxy_video.exists() else None
        },
        "audio": {
            "exists": audio_file.exists(),
            "url": f"/api/media/{job_id}/audio" if audio_file.exists() else None
        },
        "peaks": {
            "exists": peaks_cache.exists(),
            "generating": not peaks_cache.exists() and audio_file.exists() and retry_missing,
            "url": f"/api/media/{job_id}/peaks" if audio_file.exists() else None
        },
        "thumbnails": {
            "exists": thumbnails_cache.exists() or sprite_cache.exists(),
            "sprite_exists": sprite_cache.exists(),
            "single_exists": thumbnail_single.exists(),
            "generating": not thumbnail_single.exists() and video_file and retry_missing,
            "url": f"/api/media/{job_id}/thumbnails" if video_file else None,
            "sprite_url": f"/api/media/{job_id}/sprite.jpg" if sprite_cache.exists() else None,
            "thumbnail_url": f"/api/media/{job_id}/thumbnail" if video_file else None
        },
        "srt": {
            "exists": srt_file is not None,
            "filename": srt_file.name if srt_file else None,
            "url": f"/api/media/{job_id}/srt" if srt_file else None
        }
    })


async def _auto_generate_peaks(job_id: str, audio_file: Path, peaks_cache: Path):
    """自动生成波形数据（后台任务）- 使用动态采样"""
    try:
        # 获取视频时长以计算动态采样点数
        job_dir = config.JOBS_DIR / job_id
        video_file = _find_video_file(job_dir)
        duration = 0.0

        if video_file:
            duration = await _get_video_duration(video_file)

        if duration <= 0 and audio_file.exists():
            try:
                duration = await _get_video_duration(audio_file)
            except:
                duration = 60

        if duration <= 0:
            duration = 60

        # 动态计算采样点数：每秒20个点
        target_samples = int(duration * 20)
        samples = max(4000, min(target_samples, 100000))
        print(f"[media] 自动生成波形：时长{duration:.1f}s → {samples}个采样点")

        file_size_mb = audio_file.stat().st_size / (1024 * 1024)
        use_ffmpeg = file_size_mb > 50

        if use_ffmpeg:
            peaks, duration = _generate_peaks_with_ffmpeg(audio_file, samples)
        else:
            peaks, duration = _generate_peaks_with_wave(audio_file, samples)

        # 如果FFmpeg失败，回退到wave
        if not peaks and use_ffmpeg:
            peaks, duration = _generate_peaks_with_wave(audio_file, samples)

        if peaks:
            result = {
                "peaks": peaks,
                "duration": duration,
                "method": "auto_ffmpeg" if use_ffmpeg else "auto_wave",
                "samples": len(peaks) // 2
            }
            # V3.1.0+dev.20260104.01: 添加 encoding='utf-8'
            with open(peaks_cache, 'w', encoding='utf-8') as f:
                json.dump(result, f)
            print(f"[media] 波形数据自动生成成功: {job_id}")
    except Exception as e:
        print(f"[media] 波形数据自动生成失败 [{job_id}]: {e}")


async def _auto_generate_thumbnail(job_id: str, video_file: Path, thumbnail_file: Path):
    """自动生成缩略图（后台任务）"""
    try:
        import base64

        # 获取视频分辨率
        width, height = await _get_video_resolution(video_file)

        # 计算缩略图尺寸
        if width > 3840:  # 4K视频
            thumb_width = 1920
        elif width > 1920:
            thumb_width = width
        elif width > 0:
            thumb_width = width
        else:
            thumb_width = 1280

        ffmpeg_cmd = config.get_ffmpeg_command()
        cmd = [
            ffmpeg_cmd,
            '-ss', '1',
            '-i', str(video_file),
            '-vframes', '1',
            '-f', 'image2pipe',
            '-vcodec', 'mjpeg',
            '-vf', f'scale={thumb_width}:-1',
            '-q:v', '2',
            '-'
        ]

        result = subprocess.run(
            cmd, capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            timeout=10
        )

        if result.returncode == 0 and result.stdout:
            with open(thumbnail_file, 'wb') as f:
                f.write(result.stdout)
            print(f"[media] 缩略图自动生成成功: {job_id} (宽度: {thumb_width}px)")
    except Exception as e:
        print(f"[media] 缩略图自动生成失败 [{job_id}]: {e}")


