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
import threading
from pathlib import Path
from typing import Optional, Tuple
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.core.config import config
from app.services.media import get_runtime_media_policy_service
from app.services.project_id_resolver import ProjectIdentity, get_project_id_resolver
from app.utils.ass_converter import ASSConverter
from app.utils.text_utils import (
    repair_srt_overlaps,
    repair_timestamp_overlaps,
    detect_timestamp_overlaps,
    parse_srt_content,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/media", tags=["media"])

# 浏览器兼容的视频格式（扩展名，但需进一步检查编码）
BROWSER_COMPATIBLE_FORMATS = {'.mp4', '.webm'}

# 需要转码的格式（按扩展名）
NEED_TRANSCODE_FORMATS = {'.mkv', '.avi', '.mov', '.wmv', '.flv', '.m4v'}

# 浏览器不兼容的视频编码（需要转码为H.264）
NEED_TRANSCODE_CODECS = {'hevc', 'h265', 'vp9', 'av1'}

# 可作为波形数据来源的音频文件扩展名
AUDIO_SOURCE_EXTS = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.wma', '.opus'}
VIDEO_SOURCE_EXTS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.webm', '.flv', '.m4v'}
NORMALIZED_H264_FILENAME = "normalized_h264.mp4"

# 音频抽取锁（按目录粒度，避免并发请求重复抽取同一文件）
_audio_extract_lock_guard = threading.Lock()
_audio_extract_locks: dict[str, threading.Lock] = {}
_video_request_log_guard = threading.Lock()
_video_request_log_windows: dict[str, Tuple[float, int]] = {}
_transcode_cache_guard = threading.Lock()
_transcode_decision_cache: dict[str, dict] = {}
_VIDEO_REQUEST_LOG_THROTTLE_SECONDS = 1.0
_TRANSCODE_CACHE_TTL_SECONDS = 2.0

# 注意：旧的 _proxy_generation_status 已废弃，改用 MediaPrepService 管理状态


def _find_video_file(job_dir: Path) -> Optional[Path]:
    """在任务目录中查找源视频文件（默认跳过 preview/proxy/remux 产物）。"""
    source_candidates = []
    generated_candidates = []
    for file in sorted(job_dir.iterdir(), key=lambda item: item.name.lower()):
        if not file.is_file() or file.suffix.lower() not in VIDEO_SOURCE_EXTS:
            continue
        normalized_name = file.name.lower()
        if normalized_name.endswith(".tmp"):
            continue
        if normalized_name.startswith(("preview_", "proxy_")) or normalized_name == "remux.mp4":
            generated_candidates.append(file)
            continue
        source_candidates.append(file)
    if source_candidates:
        return source_candidates[0]
    if generated_candidates:
        return generated_candidates[0]
    return None


def _find_audio_source_file(job_dir: Path) -> Optional[Path]:
    """在任务目录中查找可用于生成波形的音频源文件。"""
    for file in job_dir.iterdir():
        if not file.is_file():
            continue
        if file.suffix.lower() in AUDIO_SOURCE_EXTS and file.name.lower() != "audio.wav":
            return file
    return None


def _get_audio_extract_lock(job_dir: Path) -> threading.Lock:
    """获取目录级音频抽取锁，避免并发重复工作。"""
    key = str(job_dir.resolve())
    with _audio_extract_lock_guard:
        lock = _audio_extract_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _audio_extract_locks[key] = lock
    return lock


def _ensure_waveform_audio(job_dir: Path, identifier: str) -> Path:
    """
    确保 `audio.wav` 存在（与 ASR 主流程解耦）。

    场景：
    - 导入项目仅有视频/音频文件时，按需抽取标准 `audio.wav`，供波形与时间轴使用。
    """
    audio_file = job_dir / "audio.wav"
    if audio_file.exists():
        return audio_file

    lock = _get_audio_extract_lock(job_dir)
    with lock:
        if audio_file.exists():
            return audio_file

        source_media = _find_audio_source_file(job_dir) or _find_video_file(job_dir)
        if source_media is None:
            raise HTTPException(status_code=404, detail="音频源不存在")

        # 视频源不含音轨时直接返回 404，避免前端误判为“服务器异常可重试”。
        if source_media.suffix.lower() in VIDEO_SOURCE_EXTS and not _has_audio_stream(source_media):
            raise HTTPException(status_code=404, detail="音频源不存在")

        # 使用带 .wav 后缀的临时文件，避免 FFmpeg 因扩展名无法识别封装格式。
        tmp_audio_file = job_dir / "audio.tmp.wav"
        ffmpeg_cmd = config.get_ffmpeg_command()
        cmd = [
            ffmpeg_cmd,
            "-i",
            str(source_media),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            str(tmp_audio_file),
        ]

        logger.info("[media] 开始按需抽取音频: identifier=%s, source=%s", identifier, source_media.name)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode != 0:
            if tmp_audio_file.exists():
                tmp_audio_file.unlink(missing_ok=True)
            logger.error(
                "[media] 按需抽取音频失败: identifier=%s, source=%s, stderr=%s",
                identifier,
                source_media.name,
                result.stderr.strip(),
            )
            raise HTTPException(status_code=500, detail="音频提取失败")

        os.replace(tmp_audio_file, audio_file)
        logger.info("[media] 按需抽取音频完成: identifier=%s, audio=%s", identifier, audio_file.name)
        return audio_file


def _resolve_media_identity_or_404(identifier: str) -> ProjectIdentity:
    """统一解析媒体身份，输出规范 project 语义。"""
    resolver = get_project_id_resolver()
    try:
        return resolver.resolve_or_fail(identifier)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _resolve_media_dir(identifier: str) -> Path:
    """兼容旧调用：仅返回媒体目录。"""
    return _resolve_media_identity_or_404(identifier).project_dir


def _find_best_h264(job_dir: Path):
    """
    V3.1.2+dev.20260114.21: 查找目录下最高分辨率且视频编码为 h264 的文件
    返回 (path, height)
    """
    try:
        ffprobe_cmd = config.get_ffprobe_command()
    except Exception:
        return None, 0

    best_path = None
    best_height = 0

    for file in job_dir.iterdir():
        if not file.is_file():
            continue
        if file.suffix.lower() not in ['.mp4', '.mov', '.mkv', '.avi', '.flv', '.webm', '.m4v']:
            continue
        if file.name.endswith('.tmp'):
            continue
        # 允许 proxy/remux/用户自带高清视频，但跳过 360p 预览
        if file.name.startswith('preview_'):
            continue
        try:
            cmd = [
                ffprobe_cmd,
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name,height',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(file)
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=8
            )
            if result.returncode != 0 or not result.stdout:
                continue
            lines = result.stdout.strip().splitlines()
            if len(lines) < 2:
                continue
            codec = lines[0].strip()
            try:
                height = int(float(lines[1].strip()))
            except:
                height = 0
            if codec != 'h264':
                continue
            if height > best_height:
                best_height = height
                best_path = file
        except Exception:
            continue

    return best_path, best_height


from app.services.media_stream_tracker import (
    register_stream,
    unregister_stream,
)


def _serve_file_with_range(
    file_path: Path,
    request: Request,
    media_type: str,
    job_id: str = None
):
    """
    支持HTTP Range请求的文件流式传输（允许拖拽进度条）
    """
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")

    # 注册占用，确保删除逻辑可感知活跃流
    registered = False
    if job_id:
        try:
            register_stream(job_id)
            registered = True
        except Exception:
            # 追踪失败不影响读取
            registered = False

    def finalize():
        if registered and job_id:
            unregister_stream(job_id)

    def full_file_iterator():
        try:
            with open(file_path, "rb") as f:
                while True:
                    data = f.read(256 * 1024)
                    if not data:
                        break
                    yield data
        finally:
            finalize()

    if not range_header:
        # 无Range请求，改用流式响应以便在结束时释放占用
        return StreamingResponse(
            full_file_iterator(),
            media_type=media_type,
            headers={
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
            }
        )

    # 解析Range头：bytes=start-end
    try:
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0]) if byte_range[0] else 0
        end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else file_size - 1

        # 确保范围有效
        if start >= file_size:
            finalize()
            raise HTTPException(status_code=416, detail="请求范围无效")
        end = min(end, file_size - 1)

    except ValueError:
        finalize()
        raise HTTPException(status_code=400, detail="无效的Range头格式")

    # 返回部分内容（状态码206）
    def file_iterator():
        try:
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
        finally:
            finalize()

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


def _analyze_transcode_requirement(video_file: Optional[Path]) -> Tuple[bool, str, Optional[str]]:
    """
    统一分析视频是否需要转码，避免多个接口判断口径不一致。

    Returns:
        (needs_transcode, reason, decision)
    """
    if video_file is None or not video_file.exists():
        return False, "", None
    cache_key = str(video_file.resolve())
    file_size = 0
    file_mtime_ns = 0
    try:
        stat = video_file.stat()
        file_size = int(stat.st_size)
        file_mtime_ns = int(stat.st_mtime_ns)
        cached = _get_cached_transcode_decision(
            cache_key=cache_key,
            size=file_size,
            mtime_ns=file_mtime_ns,
        )
        if cached is not None:
            return cached
    except OSError:
        cache_key = ""

    try:
        from app.services.media_prep_service import get_media_prep_service, TranscodeDecision
        from app.utils.media_analyzer import media_analyzer

        media_prep = get_media_prep_service()
        video_info = media_analyzer.analyze_sync(video_file)
        video_info["container"] = video_file.suffix.lower()
        decision = media_prep.analyze_transcode_decision(video_info)
        decision_value = str(getattr(decision, "value", str(decision)))

        if decision == TranscodeDecision.DIRECT_PLAY:
            result = (False, "", decision_value)
            _set_cached_transcode_decision(cache_key, file_size, file_mtime_ns, result)
            return result
        if decision == TranscodeDecision.REMUX_ONLY:
            result = (True, f"容器不兼容 ({video_file.suffix})", decision_value)
            _set_cached_transcode_decision(cache_key, file_size, file_mtime_ns, result)
            return result
        if decision == TranscodeDecision.TRANSCODE_AUDIO:
            result = (True, "音频编码不兼容", decision_value)
            _set_cached_transcode_decision(cache_key, file_size, file_mtime_ns, result)
            return result
        result = (True, "视频编码不兼容", decision_value)
        _set_cached_transcode_decision(cache_key, file_size, file_mtime_ns, result)
        return result
    except Exception as exc:
        logger.warning("[media] 决策分析失败，回退扩展名/编码规则: %s", exc)

    suffix = video_file.suffix.lower()
    if suffix in NEED_TRANSCODE_FORMATS:
        result = (True, f"格式不兼容 ({video_file.suffix})", "fallback_extension")
        _set_cached_transcode_decision(cache_key, file_size, file_mtime_ns, result)
        return result
    if suffix in BROWSER_COMPATIBLE_FORMATS:
        fallback_need_transcode_codecs = set(NEED_TRANSCODE_CODECS)
        if not _is_browser_compat_profile():
            fallback_need_transcode_codecs.difference_update({"hevc", "h265"})
        codec = _get_video_codec(video_file)
        if codec and codec in fallback_need_transcode_codecs:
            result = (True, f"编码不兼容 ({codec.upper()})", "fallback_codec")
            _set_cached_transcode_decision(cache_key, file_size, file_mtime_ns, result)
            return result
    result = (False, "", "fallback_direct")
    _set_cached_transcode_decision(cache_key, file_size, file_mtime_ns, result)
    return result


def _get_cached_transcode_decision(
    cache_key: str,
    size: int,
    mtime_ns: int,
) -> Optional[Tuple[bool, str, Optional[str]]]:
    if not cache_key:
        return None
    now = time.monotonic()
    with _transcode_cache_guard:
        cached = _transcode_decision_cache.get(cache_key)
        if not cached:
            return None
        if now - float(cached.get("cached_at", 0.0)) > _TRANSCODE_CACHE_TTL_SECONDS:
            _transcode_decision_cache.pop(cache_key, None)
            return None
        if int(cached.get("size", -1)) != int(size) or int(cached.get("mtime_ns", -1)) != int(mtime_ns):
            _transcode_decision_cache.pop(cache_key, None)
            return None
        result = cached.get("result")
        if not isinstance(result, tuple) or len(result) != 3:
            _transcode_decision_cache.pop(cache_key, None)
            return None
        return result  # type: ignore[return-value]


def _set_cached_transcode_decision(
    cache_key: str,
    size: int,
    mtime_ns: int,
    result: Tuple[bool, str, Optional[str]],
) -> None:
    if not cache_key:
        return
    with _transcode_cache_guard:
        _transcode_decision_cache[cache_key] = {
            "cached_at": time.monotonic(),
            "size": int(size),
            "mtime_ns": int(mtime_ns),
            "result": result,
        }


def _log_video_request(project_id: str, identifier: str, job_dir: Path) -> None:
    """折叠同项目短时间内的重复视频请求日志，避免高频噪声掩盖异常。"""
    now = time.monotonic()
    suppressed = 0
    with _video_request_log_guard:
        previous = _video_request_log_windows.get(project_id)
        if previous and now - previous[0] < _VIDEO_REQUEST_LOG_THROTTLE_SECONDS:
            _video_request_log_windows[project_id] = (previous[0], previous[1] + 1)
            return
        if previous and previous[1] > 1:
            suppressed = previous[1] - 1
        _video_request_log_windows[project_id] = (now, 1)

    if suppressed > 0:
        logger.debug(
            "[media] 收到视频请求: identifier=%s, project_id=%s, dir=%s (1s内重复请求已折叠: %s)",
            identifier,
            project_id,
            job_dir,
            suppressed,
        )
    else:
        logger.debug(
            "[media] 收到视频请求: identifier=%s, project_id=%s, dir=%s",
            identifier,
            project_id,
            job_dir,
        )


def _is_browser_compat_profile() -> bool:
    return config.is_browser_compat_media_profile()


def _is_browser_preview_profile() -> bool:
    return config.is_browser_like_media_profile()


def _is_electron_native_profile() -> bool:
    return config.is_electron_native_media_profile()


def _normalized_h264_path(job_dir: Path) -> Path:
    return job_dir / NORMALIZED_H264_FILENAME


def _build_ready_video_variants(
    job_dir: Path,
    source_video: Optional[Path] = None,
    *,
    proxy_ready: bool = False,
) -> dict[str, Optional[Path]]:
    """构造当前目录内已就绪的视频候选集合。"""
    proxy_720p = job_dir / "proxy_720p.mp4"
    remux_video = job_dir / "remux.mp4"
    preview_360p = job_dir / "preview_360p.mp4"
    normalized_h264 = _normalized_h264_path(job_dir)
    return {
        "proxy_720p": proxy_720p if (proxy_ready and proxy_720p.exists()) else None,
        "preview_360p": preview_360p if preview_360p.exists() else None,
        "normalized_h264": normalized_h264 if normalized_h264.exists() else None,
        "remux": remux_video if remux_video.exists() else None,
        "source": source_video if (source_video and source_video.exists()) else None,
    }


def _has_audio_stream(video_path: Path) -> bool:
    """检测视频是否包含可提取音轨。"""
    ffprobe_cmd = config.get_ffprobe_command()
    cmd = [
        ffprobe_cmd,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


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


@router.get("/{identifier}/video")
async def get_video(identifier: str, request: Request):
    """
    获取视频文件（支持Range请求，自动Proxy转码）

    优先返回当前 profile 对应的最佳可播放产物；若尚无可播放产物，则按 profile
    触发最合适的后台生成链路。
    """
    job_id = identifier
    media_identity = _resolve_media_identity_or_404(job_id)
    project_id = media_identity.project_id
    job_dir = media_identity.project_dir

    _log_video_request(project_id=project_id, identifier=job_id, job_dir=job_dir)

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    policy_service = get_runtime_media_policy_service()
    policy = policy_service.get_policy(config.MEDIA_PROFILE)
    is_browser_profile = _is_browser_compat_profile()
    is_preview_profile = policy.browser_preview_enabled

    proxy_720p = job_dir / "proxy_720p.mp4"
    remux_video = job_dir / "remux.mp4"
    preview_360p = job_dir / "preview_360p.mp4"
    normalized_h264 = _normalized_h264_path(job_dir)

    proxy_ready = False
    if is_browser_profile:
        from app.services.proxy_720_scheduler import get_proxy_scheduler

        scheduler_state = get_proxy_scheduler().get_state(project_id) or {}
        try:
            from app.services.media_prep_service import get_media_prep_service

            media_prep = get_media_prep_service()
            proxy_status = media_prep.get_proxy_status(project_id)
            proxy_ready = bool(proxy_status and proxy_status.get("status") == "completed")
        except Exception:
            proxy_ready = False
        proxy_ready = proxy_ready or scheduler_state.get("state") == "ready"

    ready_variants = _build_ready_video_variants(job_dir, proxy_ready=proxy_ready)
    best_variant, best_path = policy_service.select_best_variant(ready_variants, policy.profile)
    if best_path is not None:
        logger.debug("[media] 按 profile=%s 返回视频: variant=%s file=%s", policy.profile, best_variant, best_path.name)
        return _serve_file_with_range(best_path, request, 'video/mp4', job_id=project_id)

    video_file = _find_video_file(job_dir)
    if not video_file:
        raise HTTPException(status_code=404, detail="视频文件不存在")

    logger.debug("找到源视频: %s, 扩展名: %s", video_file.name, video_file.suffix.lower())

    needs_transcode, transcode_reason, transcode_decision = _analyze_transcode_requirement(video_file)

    if needs_transcode:
        logger.info(
            "[media] 视频需要转码: project_id=%s, reason=%s, decision=%s, profile=%s",
            project_id,
            transcode_reason,
            transcode_decision,
            policy.profile,
        )
        from app.services.media_prep_service import TranscodeDecision, get_media_prep_service

        media_prep = get_media_prep_service()
        preview_status = media_prep.get_preview_status(project_id)
        preview_in_progress = preview_status and preview_status.get("status") in ["queued", "processing"]
        preview_completed = preview_status and preview_status.get("status") == "completed"

        proxy_status = media_prep.get_proxy_status(project_id)
        proxy_in_progress = proxy_status and proxy_status.get("status") in ["queued", "processing"]
        proxy_completed = proxy_status and proxy_status.get("status") == "completed"

        normalize_status = media_prep.get_normalize_status(project_id)
        normalize_in_progress = normalize_status and normalize_status.get("status") in ["queued", "processing"]
        normalize_completed = normalize_status and normalize_status.get("status") == "completed"

        remux_status = media_prep.get_remux_status(project_id)
        remux_in_progress = remux_status and remux_status.get("status") in ["queued", "processing"]
        remux_completed = remux_status and remux_status.get("status") == "completed"

        completed_variants = _build_ready_video_variants(
            job_dir,
            proxy_ready=proxy_completed or proxy_ready,
        )
        best_variant, best_path = policy_service.select_best_variant(completed_variants, policy.profile)
        if best_path is not None:
            logger.debug("[media] 返回已完成可播放产物: profile=%s variant=%s file=%s", policy.profile, best_variant, best_path.name)
            return _serve_file_with_range(best_path, request, 'video/mp4', job_id=project_id)

        if preview_in_progress or proxy_in_progress or normalize_in_progress or remux_in_progress:
            if is_preview_profile and preview_in_progress:
                current_progress = preview_status.get("progress", 0)
                current_stage = "安全预览" if policy.profile == "lite_safe" else "360p预览"
                current_stage_code = "preview_360p"
            elif normalize_in_progress:
                current_progress = normalize_status.get("progress", 0)
                current_stage = "H264兼容转码"
                current_stage_code = "normalize_h264"
            elif remux_in_progress:
                current_progress = remux_status.get("progress", 0)
                current_stage = "容器重封装"
                current_stage_code = "remux"
            else:
                current_progress = proxy_status.get("progress", 0)
                current_stage = "720p高清"
                current_stage_code = "proxy_720p"
            raise HTTPException(
                status_code=202,
                detail={
                    "message": f"正在生成{current_stage}版本...",
                    "progress": current_progress,
                    "proxy_generating": True,
                    "stage": current_stage_code,
                },
            )

        if transcode_decision == TranscodeDecision.REMUX_ONLY.value:
            media_prep.enqueue_remux(project_id, video_file, remux_video, priority=3)
            raise HTTPException(
                status_code=202,
                detail={
                    "message": "视频容器不兼容，正在重封装...",
                    "format": video_file.suffix,
                    "proxy_generating": True,
                    "stage": "remux",
                },
            )

        if is_browser_profile and transcode_decision == TranscodeDecision.TRANSCODE_AUDIO.value:
            logger.info(
                "[media] browser_compat 下仅音频编码不兼容，直接回源避免阻塞: project_id=%s source=%s",
                project_id,
                video_file.name,
            )
            return _serve_file_with_range(video_file, request, 'video/mp4', job_id=project_id)

        if is_preview_profile:
            media_prep.enqueue_preview(project_id, video_file, preview_360p, priority=5)
            stage = "preview_360p"
            stage_text = "安全预览版本" if policy.profile == "lite_safe" else "预览版本"
        else:
            media_prep.enqueue_normalize_h264(project_id, video_file, normalized_h264, priority=4)
            stage = "normalize_h264"
            stage_text = "H264兼容版本"

        raise HTTPException(
            status_code=202,
            detail={
                "message": f"视频{transcode_reason}，正在生成{stage_text}...",
                "format": video_file.suffix,
                "proxy_generating": True,
                "stage": stage,
            },
        )

    return _serve_file_with_range(video_file, request, 'video/mp4', job_id=project_id)


@router.get("/{identifier}/audio")
async def get_audio(identifier: str, request: Request):
    """获取音频文件（支持Range请求）"""
    job_id = identifier
    media_identity = _resolve_media_identity_or_404(job_id)
    job_dir = media_identity.project_dir
    audio_file = _ensure_waveform_audio(job_dir, media_identity.project_id)

    return _serve_file_with_range(audio_file, request, 'audio/wav', job_id=media_identity.project_id)


@router.get("/{identifier}/peaks")
async def get_audio_peaks(identifier: str, samples: int = 0, method: str = "auto"):
    """
    获取音频波形峰值数据（优化：动态采样密度）

    Args:
        job_id: 任务ID
        samples: 采样点数（0=自动计算，建议传0）
        method: 生成方法 - "auto"(自动选择), "ffmpeg"(FFmpeg加速), "wave"(Python原生)

    Returns:
        JSON: { peaks: [min, max, min, max, ...], duration: 180.5, method: "ffmpeg" }
    """
    job_id = identifier
    media_identity = _resolve_media_identity_or_404(job_id)
    job_dir = media_identity.project_dir
    audio_file = _ensure_waveform_audio(job_dir, media_identity.project_id)

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


@router.get("/{identifier}/proxy-status")
async def check_proxy_status(identifier: str):
    """
    获取 Proxy 视频完整状态（用于前端刷新后恢复）
    """
    job_id = identifier
    media_identity = _resolve_media_identity_or_404(job_id)
    project_id = media_identity.project_id
    job_dir = media_identity.project_dir
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    from app.services.media_prep_service import get_media_prep_service, TranscodeDecision

    media_prep = get_media_prep_service()
    task_status = media_prep.get_full_task_status(project_id)
    policy_service = get_runtime_media_policy_service()
    policy = policy_service.get_policy(config.MEDIA_PROFILE)

    is_browser_profile = _is_browser_compat_profile()
    is_preview_profile = policy.browser_preview_enabled
    preview_360p = job_dir / "preview_360p.mp4"
    proxy_720p = job_dir / "proxy_720p.mp4"
    remux_video = job_dir / "remux.mp4"
    normalized_h264 = _normalized_h264_path(job_dir)
    source_video = _find_video_file(job_dir)

    scheduler_state = {}
    scheduler_version = None
    proxy_ready = False
    if is_browser_profile:
        from app.services.proxy_720_scheduler import get_proxy_scheduler

        proxy_scheduler = get_proxy_scheduler()
        scheduler_state = proxy_scheduler.get_state(project_id) or {}
        scheduler_version = scheduler_state.get("version")
        proxy_ready = scheduler_state.get("state") == "ready"

    source_allowed_for_best = None
    needs_transcode = False
    transcode_decision = None
    if source_video:
        try:
            needs_transcode, _, transcode_decision = _analyze_transcode_requirement(source_video)
            if (not needs_transcode) or (
                is_browser_profile and transcode_decision == TranscodeDecision.TRANSCODE_AUDIO.value
            ):
                source_allowed_for_best = source_video
        except Exception as exc:
            logger.warning("[media] proxy-status 分析转码需求失败: %s", exc)

    ready_variants = _build_ready_video_variants(
        job_dir,
        source_allowed_for_best,
        proxy_ready=proxy_ready or proxy_720p.exists(),
    )
    best_variant, _best_path = policy_service.select_best_variant(ready_variants, policy.profile)
    best_playable_url = policy_service.build_variant_url(project_id, best_variant)
    best_playable_resolution = policy_service.describe_variant_resolution(best_variant)

    has_primary_video_output = any(
        ready_variants.get(key) is not None for key in ("proxy_720p", "normalized_h264", "remux")
    )
    urls = {
        "360p": f"/api/media/{project_id}/video/preview" if preview_360p.exists() else None,
        "720p": f"/api/media/{project_id}/video" if has_primary_video_output else None,
        "source": f"/api/media/{project_id}/video/source" if source_video else None,
        "normalize": f"/api/media/{project_id}/video" if normalized_h264.exists() else None,
    }

    if not task_status:
        state = "idle"
        progress = 0
        if best_variant in {"proxy_720p", "normalized_h264", "remux"}:
            state = "ready_720p"
            progress = 100
        elif best_variant == "preview_360p":
            state = scheduler_state.get("state") or "ready_360p"
            progress = 100
            if source_video and is_browser_profile:
                try:
                    from app.services.proxy_720_scheduler import get_proxy_scheduler

                    get_proxy_scheduler().ensure_tracked(project_id, source_video, trigger_type="editor_check")
                except Exception:
                    pass
        elif source_video:
            try:
                if not needs_transcode:
                    state = "direct_play"
                    progress = 100
                elif transcode_decision == TranscodeDecision.REMUX_ONLY.value:
                    state = "analyzing"
                    media_prep.enqueue_remux(project_id, source_video, remux_video, priority=3)
                elif is_browser_profile and transcode_decision == TranscodeDecision.TRANSCODE_AUDIO.value:
                    state = "direct_play"
                    progress = 100
                elif is_preview_profile:
                    state = "analyzing"
                    media_prep.enqueue_preview(project_id, source_video, preview_360p, priority=5)
                else:
                    state = "analyzing"
                    media_prep.enqueue_normalize_h264(project_id, source_video, normalized_h264, priority=4)
            except Exception as exc:
                logger.warning("[media] proxy-status 自动分析失败: %s", exc)
                state = "idle"
                progress = 0

        return JSONResponse({
            "state": state,
            "progress": progress,
            "decision": None,
            "urls": urls,
            "error": None,
            "started_at": None,
            "estimated_remaining": None,
            "auto_trigger_720p": policy.auto_trigger_720p,
            "version": scheduler_version,
            "best_playable_url": best_playable_url,
            "best_playable_resolution": best_playable_resolution,
            "project_id": project_id,
            "legacy_job_id": None,
            "media_profile": policy.profile,
            "normalize_h264": media_prep.get_normalize_status(project_id),
        })

    if source_video and needs_transcode:
        try:
            remux_status = media_prep.get_remux_status(project_id)
            remux_in_progress = remux_status and remux_status.get("status") in ["queued", "processing"]
            normalize_status = media_prep.get_normalize_status(project_id)
            normalize_in_progress = normalize_status and normalize_status.get("status") in ["queued", "processing"]
            preview_status = media_prep.get_preview_status(project_id)
            preview_in_progress = preview_status and preview_status.get("status") in ["queued", "processing"]

            if transcode_decision == TranscodeDecision.REMUX_ONLY.value:
                if not remux_video.exists() and not remux_in_progress:
                    media_prep.enqueue_remux(project_id, source_video, remux_video, priority=3)
            elif is_browser_profile and transcode_decision == TranscodeDecision.TRANSCODE_AUDIO.value:
                pass
            elif is_preview_profile and transcode_decision in {
                TranscodeDecision.TRANSCODE_AUDIO.value,
                TranscodeDecision.TRANSCODE_VIDEO.value,
                TranscodeDecision.TRANSCODE_FULL.value,
            }:
                if not preview_360p.exists() and not preview_in_progress:
                    media_prep.enqueue_preview(project_id, source_video, preview_360p, priority=5)
            elif transcode_decision in {
                TranscodeDecision.TRANSCODE_AUDIO.value,
                TranscodeDecision.TRANSCODE_VIDEO.value,
                TranscodeDecision.TRANSCODE_FULL.value,
            }:
                if not normalized_h264.exists() and not normalize_in_progress:
                    media_prep.enqueue_normalize_h264(project_id, source_video, normalized_h264, priority=4)
        except Exception as exc:
            logger.warning("[media] proxy-status 补偿触发失败: %s", exc)

    if is_browser_profile:
        proxy_state = task_status.get("proxy_720p") if isinstance(task_status, dict) else None
        if proxy_state and proxy_state.get("status") in ["waiting_check", "queued"] and proxy_state.get("error") == "paused_for_new_job":
            proxy_state = dict(proxy_state)
            proxy_state["error"] = None
            task_status = dict(task_status)
            task_status["proxy_720p"] = proxy_state

    return JSONResponse({
        "state": task_status.get("state", "idle"),
        "progress": task_status.get("progress", 0),
        "decision": task_status.get("decision"),
        "urls": urls,
        "error": task_status.get("error"),
        "started_at": task_status.get("started_at"),
        "estimated_remaining": task_status.get("estimated_remaining"),
        "auto_trigger_720p": policy.auto_trigger_720p,
        "version": scheduler_version,
        "best_playable_url": best_playable_url,
        "best_playable_resolution": best_playable_resolution,
        "project_id": project_id,
        "legacy_job_id": None,
        "media_profile": policy.profile,
        "normalize_h264": task_status.get("normalize_h264"),
    })


@router.get("/{identifier}/thumbnail")
async def get_thumbnail(identifier: str):
    """
    获取任务的单个缩略图（第一帧）用于任务卡片展示（第二阶段修复：实时更新）

    Args:
        job_id: 任务ID

    Returns:
        Base64编码的缩略图 or JSON占位符
    """
    job_id = identifier
    try:
        job_dir = _resolve_media_dir(job_id)
    except HTTPException as exc:
        if exc.status_code == 404:
            return JSONResponse(
                {
                    "thumbnail": None,
                    "message": "任务不存在",
                }
            )
        raise
    if not job_dir.exists():
        return JSONResponse(
            {
                "thumbnail": None,
                "message": "任务不存在",
            }
        )

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


@router.get("/{identifier}/thumbnails")
async def get_thumbnails(identifier: str, count: int = 10, sprite: bool = True):
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
    job_id = identifier
    job_dir = _resolve_media_dir(job_id)

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


@router.get("/{identifier}/sprite.jpg")
async def get_sprite_image(identifier: str):
    """直接获取Sprite图片文件"""
    job_id = identifier
    job_dir = _resolve_media_dir(job_id)
    sprite_file = job_dir / "sprite.jpg"

    if not sprite_file.exists():
        raise HTTPException(status_code=404, detail="Sprite图不存在")

    return FileResponse(str(sprite_file), media_type="image/jpeg")


@router.get("/{identifier}/video/preview")
async def get_preview_video(identifier: str, request: Request):
    """
    获取 360p 预览视频（渐进式加载第一阶段）

    用于在转码过程中快速预览视频内容
    """
    job_id = identifier
    media_identity = _resolve_media_identity_or_404(job_id)
    project_id = media_identity.project_id
    job_dir = media_identity.project_dir

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    # 查找 360p 预览视频
    preview_360p = job_dir / "preview_360p.mp4"
    if preview_360p.exists():
        return _serve_file_with_range(preview_360p, request, 'video/mp4', job_id=project_id)

    # 如果没有 360p，按 profile 尝试返回可播放替代
    normalized_h264 = _normalized_h264_path(job_dir)
    if normalized_h264.exists():
        return _serve_file_with_range(normalized_h264, request, 'video/mp4', job_id=project_id)

    remux_video = job_dir / "remux.mp4"
    if remux_video.exists():
        return _serve_file_with_range(remux_video, request, 'video/mp4', job_id=project_id)

    proxy_video = job_dir / "proxy_720p.mp4"
    if proxy_video.exists():
        return _serve_file_with_range(proxy_video, request, 'video/mp4', job_id=project_id)

    # 都没有，返回 202 表示正在生成
    raise HTTPException(
        status_code=202,
        detail={
            "message": "预览视频正在生成中...",
            "generating": True
        }
    )


@router.get("/{identifier}/status/progressive")
async def get_progressive_status(identifier: str):
    """
    获取渐进式加载状态

    返回当前视频生成的状态信息，包括：
    - 是否需要转码
    - 360p 预览进度/状态
    - 720p 高质量进度/状态
    - 可用的视频 URL
    """
    job_id = identifier
    media_identity = _resolve_media_identity_or_404(job_id)
    project_id = media_identity.project_id
    job_dir = media_identity.project_dir

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    # 检查各种视频文件
    video_file = _find_video_file(job_dir)
    is_browser_profile = _is_browser_compat_profile()
    preview_360p = job_dir / "preview_360p.mp4"
    proxy_720p = job_dir / "proxy_720p.mp4"
    remux_video = job_dir / "remux.mp4"
    normalized_h264 = _normalized_h264_path(job_dir)

    # 判断是否需要转码（统一决策）
    needs_transcode, transcode_reason, transcode_decision = _analyze_transcode_requirement(video_file)

    # 获取生成状态
    preview_status = None
    proxy_status = None
    normalize_status = None

    # 从 MediaPrepService 获取转码状态
    from app.services.media_prep_service import get_media_prep_service
    media_prep = get_media_prep_service()

    # 获取 360p 预览状态
    preview_status_info = media_prep.get_preview_status(project_id)
    if preview_status_info:
        preview_status = {
            "status": preview_status_info.get("status", "not_started"),
            "progress": preview_status_info.get("progress", 0)
        }

    # 获取 720p proxy 状态
    proxy_status_info = media_prep.get_proxy_status(project_id)
    if proxy_status_info:
        proxy_status = {
            "status": proxy_status_info.get("status", "not_started"),
            "progress": proxy_status_info.get("progress", 0)
        }

    normalize_status_info = media_prep.get_normalize_status(project_id)
    if normalize_status_info:
        normalize_status = {
            "status": normalize_status_info.get("status", "not_started"),
            "progress": normalize_status_info.get("progress", 0)
        }

    # 关键修复：如果需要转码但没有转码任务，自动触发 360p 预览转码
    if needs_transcode and video_file:
        # 检查是否需要启动转码
        preview_exists = preview_360p.exists()
        preview_in_progress = preview_status_info and preview_status_info.get("status") in ["queued", "processing"]
        remux_status_info = media_prep.get_remux_status(project_id)
        remux_in_progress = remux_status_info and remux_status_info.get("status") in ["queued", "processing"]
        normalize_in_progress = normalize_status_info and normalize_status_info.get("status") in ["queued", "processing"]

        if transcode_decision == "remux_only":
            if not remux_video.exists() and not remux_in_progress:
                print(f"[media] progressive-status 检测到需重封装，自动触发: {project_id}")
                remux_success = media_prep.enqueue_remux(project_id, video_file, remux_video, priority=3)
                print(f"[media] enqueue_remux() 返回: {remux_success}")
        elif is_browser_profile and transcode_decision == "transcode_full":
            if not preview_exists and not preview_in_progress:
                # 仅完整转码决策才补触发 360p 预览
                print(f"[media] progressive-status 检测到需要转码，自动触发: {project_id}")
                success = media_prep.enqueue_preview(project_id, video_file, preview_360p, priority=5)
                print(f"[media] enqueue_preview() 返回: {success}")

                # 更新状态为已入队
                if success:
                    preview_status_info = media_prep.get_preview_status(project_id)
                    if preview_status_info:
                        preview_status = {
                            "status": preview_status_info.get("status", "queued"),
                            "progress": preview_status_info.get("progress", 0)
                        }
        elif (not is_browser_profile) and transcode_decision in {"transcode_full", "transcode_audio", "transcode_video"}:
            if not normalized_h264.exists() and not normalize_in_progress:
                print(f"[media] progressive-status 检测到需要 normalize_h264，自动触发: {project_id}")
                normalize_success = media_prep.enqueue_normalize_h264(project_id, video_file, normalized_h264, priority=4)
                print(f"[media] enqueue_normalize_h264() 返回: {normalize_success}")

                if normalize_success:
                    normalize_status_info = media_prep.get_normalize_status(project_id)
                    if normalize_status_info:
                        normalize_status = {
                            "status": normalize_status_info.get("status", "queued"),
                            "progress": normalize_status_info.get("progress", 0)
                        }

    # 构建响应
    result = {
        "job_id": project_id,
        "project_id": project_id,
        "needs_transcode": needs_transcode,
        "transcode_reason": transcode_reason,
        "preview_360p": {
            "exists": preview_360p.exists(),
            "url": f"/api/media/{project_id}/video/preview" if preview_360p.exists() else None,
            "size": preview_360p.stat().st_size if preview_360p.exists() else 0,
            "status": preview_status  # 包含转码状态和进度
        },
        "proxy_720p": {
            "exists": proxy_720p.exists(),
            "url": f"/api/media/{project_id}/video" if proxy_720p.exists() else None,
            "size": proxy_720p.stat().st_size if proxy_720p.exists() else 0,
            "status": proxy_status
        },
        "normalize_h264": {
            "exists": normalized_h264.exists(),
            "url": f"/api/media/{project_id}/video" if normalized_h264.exists() else None,
            "size": normalized_h264.stat().st_size if normalized_h264.exists() else 0,
            "status": normalize_status
        },
        "source": {
            "exists": video_file is not None,
            "filename": video_file.name if video_file else None,
            "compatible": not needs_transcode,
            "url": f"/api/media/{project_id}/video" if video_file and not needs_transcode else None
        },
        "recommended_url": None,  # 推荐使用的视频 URL
        "media_profile": config.MEDIA_PROFILE,
    }

    # 确定推荐的视频 URL
    if is_browser_profile and proxy_720p.exists():
        result["recommended_url"] = f"/api/media/{project_id}/video"
        result["current_resolution"] = "720p"
    elif (not is_browser_profile) and normalized_h264.exists():
        result["recommended_url"] = f"/api/media/{project_id}/video"
        result["current_resolution"] = "normalized_h264"
    elif remux_video.exists():
        result["recommended_url"] = f"/api/media/{project_id}/video"
        result["current_resolution"] = "remux"
    elif preview_360p.exists():
        result["recommended_url"] = f"/api/media/{project_id}/video/preview"
        result["current_resolution"] = "360p"
    elif video_file and not needs_transcode:
        result["recommended_url"] = f"/api/media/{project_id}/video"
        result["current_resolution"] = "source"
    else:
        result["current_resolution"] = None

    return JSONResponse(result)


@router.post("/{identifier}/generate-preview")
async def trigger_preview_generation(identifier: str):
    """
    手动触发 360p 预览视频生成

    在某些情况下，可能需要手动触发预览生成
    """
    job_id = identifier
    media_identity = _resolve_media_identity_or_404(job_id)
    project_id = media_identity.project_id
    job_dir = media_identity.project_dir

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
            "url": f"/api/media/{project_id}/video/preview"
        })

    # 异步生成 360p 预览
    from utils.progressive_video_generator import progressive_video_generator
    try:
        from app.services.sse_service import get_sse_manager
        progressive_video_generator.set_sse_manager(get_sse_manager())
    except:
        pass

    asyncio.create_task(
        progressive_video_generator.generate_360p_preview(project_id, video_file, preview_360p)
    )

    return JSONResponse({
        "success": True,
        "message": "预览视频生成已启动",
        "generating": True
    })


@router.post("/{identifier}/post-process")
async def post_process_transcription(identifier: str):
    """
    转录后处理：预生成编辑器所需的所有数据
    在转录完成后调用，异步生成波形、缩略图、Proxy视频

    Returns:
        JSON: { peaks: bool, thumbnails: bool, proxy: bool, sprite: bool }
    """
    job_id = identifier
    media_identity = _resolve_media_identity_or_404(job_id)
    project_id = media_identity.project_id
    job_dir = media_identity.project_dir

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
            await get_audio_peaks(project_id, 0)
            results["peaks"] = True
    except Exception as e:
        print(f"[media] 波形生成失败: {e}")

    # 2. 生成缩略图（优先Sprite图）
    try:
        sprite_cache = job_dir / "sprite_10.json"
        thumbnails_cache = job_dir / "thumbnails_10.json"
        if not sprite_cache.exists() and not thumbnails_cache.exists():
            await get_thumbnails(project_id, 10, sprite=True)
        results["thumbnails"] = True
        results["sprite"] = sprite_cache.exists()
    except Exception as e:
        print(f"[media] 缩略图生成失败: {e}")

    # 3. 检查是否需要生成Proxy/重封装（统一转码决策）
    video_file = _find_video_file(job_dir)
    if video_file:
        needs_proxy, _, decision = _analyze_transcode_requirement(video_file)
    else:
        needs_proxy, decision = False, None

    if needs_proxy:
        results["proxy_needed"] = True
        from app.services.media_prep_service import get_media_prep_service

        media_prep = get_media_prep_service()
        if decision == "remux_only":
            remux_video = job_dir / "remux.mp4"
            if not remux_video.exists():
                results["proxy"] = bool(
                    media_prep.enqueue_remux(project_id, video_file, remux_video, priority=20)
                )
        else:
            if _is_browser_compat_profile():
                proxy_video = job_dir / "proxy_720p.mp4"
                if not proxy_video.exists():
                    results["proxy"] = bool(
                        media_prep.enqueue_proxy(project_id, video_file, proxy_video, priority=20)
                    )
            else:
                normalized_h264 = _normalized_h264_path(job_dir)
                if not normalized_h264.exists():
                    results["proxy"] = bool(
                        media_prep.enqueue_normalize_h264(project_id, video_file, normalized_h264, priority=20)
                    )

    return JSONResponse(results)


@router.get("/{identifier}/srt")
async def get_srt_content(identifier: str):
    """
    获取SRT字幕文件内容

    Returns:
        JSON: { job_id, filename, content, encoding }
    """
    job_id = identifier
    media_identity = _resolve_media_identity_or_404(job_id)
    project_id = media_identity.project_id
    job_dir = media_identity.project_dir

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
            "job_id": project_id,
            "project_id": project_id,
            "filename": srt_file.name,
            "content": content,
            "encoding": "utf-8"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取SRT文件失败: {str(e)}")


@router.post("/{identifier}/srt")
async def save_srt_content(identifier: str, request: Request):
    """
    保存编辑后的SRT字幕文件
    V3.1.1+dev.20260106.03: 保存前自动修复时间戳重叠

    Body:
        {
            content: "1\n00:00:01,000 --> 00:00:03,000\n...",
            auto_repair: true  // 可选，默认 true，是否自动修复重叠
        }
    """
    job_id = identifier
    media_identity = _resolve_media_identity_or_404(job_id)
    project_id = media_identity.project_id
    job_dir = media_identity.project_dir

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
                logger.info(f"[{project_id}] 自动修复了 {repaired_count} 处时间戳重叠")

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
            "filename": srt_file.name,
            "project_id": project_id,
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


@router.get("/{identifier}/ass")
async def get_ass_content(identifier: str):
    """
    获取 ASS 字幕文件内容

    Returns:
        JSON: { job_id, filename, content, encoding }
    """
    job_id = identifier
    media_identity = _resolve_media_identity_or_404(job_id)
    project_id = media_identity.project_id
    job_dir = media_identity.project_dir

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
            "job_id": project_id,
            "project_id": project_id,
            "filename": ass_file.name,
            "content": content,
            "encoding": "utf-8"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取ASS文件失败: {str(e)}")


@router.post("/{identifier}/ass/generate")
async def generate_ass_from_srt(identifier: str, request: Request):
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
    job_id = identifier
    media_identity = _resolve_media_identity_or_404(job_id)
    project_id = media_identity.project_id
    job_dir = media_identity.project_dir

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
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        subtitles = parse_srt_content(content)

        # V3.1.1+dev.20260106.03: 自动修复时间戳重叠
        repaired_count = 0
        if auto_repair and subtitles:
            # 检测重叠数量
            overlaps = detect_timestamp_overlaps(subtitles)
            repaired_count = len(overlaps)

            if repaired_count > 0:
                subtitles = repair_timestamp_overlaps(subtitles, gap_ms=1.0)
                logger.info(f"[{project_id}] ASS生成前自动修复了 {repaired_count} 处时间戳重叠")

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
            "job_id": project_id,
            "project_id": project_id,
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


@router.get("/{identifier}/info")
async def get_media_info(identifier: str, retry_missing: bool = True):
    """
    获取任务的媒体信息摘要（支持自动重试生成缺失资源）

    Args:
        job_id: 任务ID
        retry_missing: 是否在资源缺失时尝试重新生成（默认True）

    Returns:
        JSON: 包含视频、音频、SRT等文件的可用状态
    """
    job_id = identifier
    media_identity = _resolve_media_identity_or_404(job_id)
    project_id = media_identity.project_id
    job_dir = media_identity.project_dir

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    video_file = _find_video_file(job_dir)
    audio_file = job_dir / "audio.wav"
    audio_source_file = _find_audio_source_file(job_dir)
    proxy_video = job_dir / "proxy_720p.mp4"
    normalized_h264 = _normalized_h264_path(job_dir)
    peaks_cache = job_dir / "peaks_2000.json"
    peaks_caches = list(job_dir.glob("peaks_*_v*.json"))
    sprite_cache = job_dir / "sprite_10.json"
    thumbnails_cache = job_dir / "thumbnails_10.json"
    thumbnail_single = job_dir / "thumbnail.jpg"

    srt_file = None
    for file in job_dir.iterdir():
        if file.suffix.lower() == '.srt':
            srt_file = file
            break

    # 统一决策 Proxy 需求，避免接口间口径不一致导致重复触发
    needs_proxy, _, transcode_decision = _analyze_transcode_requirement(video_file)
    video_codec = _get_video_codec(video_file) if video_file else None

    # 获取Proxy生成状态（从 MediaPrepService）
    from app.services.media_prep_service import get_media_prep_service
    media_prep = get_media_prep_service()
    is_browser_profile = _is_browser_compat_profile()
    proxy_status = media_prep.get_proxy_status(project_id)
    normalize_status = media_prep.get_normalize_status(project_id)

    # 智能重试：如果资源缺失且启用重试，则尝试异步生成
    if retry_missing:
        # 1. 如果波形数据缺失但音频文件存在，触发生成
        if not peaks_cache.exists() and audio_file.exists():
            try:
                print(f"[media] 检测到波形数据缺失，尝试生成: {job_id}")
                asyncio.create_task(_auto_generate_peaks(project_id, audio_file, peaks_cache))
            except Exception as e:
                print(f"[media] 波形数据生成失败: {e}")

        # 2. 如果缩略图缺失但视频文件存在，触发生成
        if not thumbnail_single.exists() and video_file:
            try:
                print(f"[media] 检测到缩略图缺失，尝试生成: {job_id}")
                asyncio.create_task(_auto_generate_thumbnail(project_id, video_file, thumbnail_single))
            except Exception as e:
                print(f"[media] 缩略图生成失败: {e}")

        # 3. 如果需要Proxy但不存在且未在生成中，触发生成
        if needs_proxy and not proxy_video.exists():
            try:
                queue_busy = bool(media_prep._is_transcription_queue_busy())  # noqa: SLF001
            except Exception:
                queue_busy = False

            if queue_busy:
                print(f"[media] 转录队列繁忙，跳过自动Proxy生成: {project_id}")
            elif transcode_decision == "remux_only":
                remux_video = job_dir / "remux.mp4"
                remux_status = media_prep.get_remux_status(project_id)
                remux_in_progress = remux_status and remux_status.get("status") in ["queued", "processing"]
                if not remux_video.exists() and not remux_in_progress:
                    try:
                        print(f"[media] 检测到仅需重封装，尝试生成 remux: {project_id}")
                        media_prep.enqueue_remux(project_id, video_file, remux_video, priority=30)
                    except Exception as e:
                        print(f"[media] remux 生成失败: {e}")
            elif is_browser_profile:
                if not (proxy_status and proxy_status.get("status") in ["queued", "processing"]):
                    try:
                        print(f"[media] 检测到Proxy缺失，尝试生成: {project_id}")
                        media_prep.enqueue_proxy(project_id, video_file, proxy_video, priority=30)
                    except Exception as e:
                        print(f"[media] Proxy视频生成失败: {e}")
            else:
                normalize_in_progress = normalize_status and normalize_status.get("status") in ["queued", "processing"]
                if not normalized_h264.exists() and not normalize_in_progress:
                    try:
                        print(f"[media] 检测到normalize_h264缺失，尝试生成: {project_id}")
                        media_prep.enqueue_normalize_h264(project_id, video_file, normalized_h264, priority=30)
                    except Exception as e:
                        print(f"[media] normalize_h264生成失败: {e}")

    video_has_audio_stream = _has_audio_stream(video_file) if video_file else False
    audio_extractable = bool(audio_source_file or video_has_audio_stream)
    audio_state = "ready" if (audio_file.exists() or audio_source_file) else ("derivable" if video_has_audio_stream else "absent")
    media_mode = "with_audio" if audio_state in {"ready", "derivable"} else "no_media"

    return JSONResponse({
        "job_id": project_id,
        "project_id": project_id,
        "identifier": str(job_id or "").strip(),
        "media_mode": media_mode,
        "video": {
            "exists": video_file is not None,
            "filename": video_file.name if video_file else None,
            "format": video_file.suffix if video_file else None,
            "codec": video_codec,  # 视频编码（如 h264, hevc）
            "needs_proxy": needs_proxy,
            "proxy_exists": proxy_video.exists() or normalized_h264.exists(),
            "proxy_generating": (
                (proxy_status and proxy_status.get("status") == "processing")
                or (normalize_status and normalize_status.get("status") == "processing")
            ),
            "proxy_progress": (
                normalize_status.get("progress", 0)
                if (normalize_status and normalize_status.get("status") in {"queued", "processing"})
                else (proxy_status.get("progress", 0) if proxy_status else 0)
            ),
            "normalize_exists": normalized_h264.exists(),
            "normalize_generating": normalize_status and normalize_status.get("status") == "processing",
            "normalize_progress": normalize_status.get("progress", 0) if normalize_status else 0,
            "media_profile": config.MEDIA_PROFILE,
            "url": f"/api/media/{project_id}/video" if video_file or proxy_video.exists() or normalized_h264.exists() else None
        },
        "audio": {
            "exists": audio_file.exists(),
            "source_exists": audio_source_file is not None,
            "extractable": audio_extractable,
            "state": audio_state,
            "url": f"/api/media/{project_id}/audio" if (audio_file.exists() or audio_extractable) else None
        },
        "peaks": {
            "exists": peaks_cache.exists() or len(peaks_caches) > 0,
            "cache_files": len(peaks_caches),
            "generating": not (peaks_cache.exists() or len(peaks_caches) > 0) and audio_extractable and retry_missing,
            "url": f"/api/media/{project_id}/peaks" if audio_extractable else None
        },
        "thumbnails": {
            "exists": thumbnails_cache.exists() or sprite_cache.exists(),
            "sprite_exists": sprite_cache.exists(),
            "single_exists": thumbnail_single.exists(),
            "generating": not thumbnail_single.exists() and video_file and retry_missing,
            "url": f"/api/media/{project_id}/thumbnails" if video_file else None,
            "sprite_url": f"/api/media/{project_id}/sprite.jpg" if sprite_cache.exists() else None,
            "thumbnail_url": f"/api/media/{project_id}/thumbnail" if video_file else None
        },
        "srt": {
            "exists": srt_file is not None,
            "filename": srt_file.name if srt_file else None,
            "url": f"/api/media/{project_id}/srt" if srt_file else None
        }
    })


async def _auto_generate_peaks(job_id: str, audio_file: Path, peaks_cache: Path):
    """自动生成波形数据（后台任务）- 使用动态采样"""
    try:
        # 获取视频时长以计算动态采样点数
        job_dir = _resolve_media_dir(job_id)
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


@router.post("/{identifier}/upgrade-720p")
async def upgrade_to_720p(identifier: str):
    """
    V3.1.2+dev.20260114.07: 手动触发720p转码（自动启用时禁止手动）

    检查条件：
    1. 任务存在且360p已完成
    2. 720p未完成
    3. 队列空闲
    4. 无转码进行

    Returns:
        JSONResponse: {
            "success": bool,
            "message": str,
            "reason": str (失败时)
        }
    """
    try:
        from app.services.proxy_720_scheduler import get_proxy_scheduler

        scheduler = get_proxy_scheduler()
        job_id = identifier
        media_identity = _resolve_media_identity_or_404(job_id)
        project_id = media_identity.project_id
        job_dir = media_identity.project_dir

        if _is_electron_native_profile():
            return JSONResponse(
                content={
                    "success": False,
                    "message": f"当前运行 profile 为 {config.MEDIA_PROFILE}，已禁用720p升级接口",
                    "reason": "disabled_by_profile",
                    "media_profile": config.MEDIA_PROFILE,
                },
                status_code=400,
            )

        # 自动模式下禁止手动
        auto_enabled = config.PROXY_CONFIG.get('auto_trigger_720p', False)
        if auto_enabled:
            return JSONResponse(
                content={
                    "success": False,
                    "message": "已开启自动720p，不能手动触发",
                    "reason": "auto_enabled"
                },
                status_code=400
            )

        if not job_dir.exists():
            return JSONResponse(
                content={
                    "success": False,
                    "message": "任务目录不存在",
                    "reason": "job_not_found"
                },
                status_code=404
            )

        # 查找视频文件
        video_file = None
        video_exts = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.webm', '.flv', '.m4v']
        for f in job_dir.iterdir():
            if f.is_file() and f.suffix.lower() in video_exts and not f.name.startswith(('preview_', 'proxy_')):
                video_file = f
                break

        if not video_file:
            return JSONResponse(
                content={
                    "success": False,
                    "message": "未找到视频文件",
                    "reason": "video_not_found"
                },
                status_code=400
            )

        # 调度器统一检查：360p、队列、进行中等
        result = scheduler.request(
            project_id,
            video_file,
            trigger_type="manual",
            auto_enabled=auto_enabled,
            force=False,
            priority=50  # 手动优先级高于自动
        )

        if result.get("accepted"):
            return JSONResponse(content={
                "success": True,
                "message": "后台已排队，队列空闲后自动生成720p"
            }, status_code=200)

        reason = result.get("reason", "unknown")
        message = {
            "auto_disabled": "未开启自动模式，但请求被拒绝",
            "disabled_by_profile": "当前运行 profile 已禁用720p升级",
            "preview_not_ready": "360p未完成，无法升级720p",
            "already_exists": "720p已存在",
            "already_processing": "720p正在处理中",
            "queue_busy": "有任务正在运行，请稍后再试",
            "job_not_found": "任务不存在"
        }.get(reason, "请求被拒绝")

        status_code = 400
        if reason in ["job_not_found"]:
            status_code = 404
        elif reason == "queue_busy":
            status_code = 503

        return JSONResponse(
            content={"success": False, "message": message, "reason": reason},
            status_code=status_code
        )

    except Exception as e:
        logger.error(f"[media] 手动触发720p异常: {job_id}, {e}", exc_info=True)
        return JSONResponse(
            content={
                "success": False,
                "message": f"服务器错误: {str(e)}",
                "reason": "server_error"
            },
            status_code=500
        )
