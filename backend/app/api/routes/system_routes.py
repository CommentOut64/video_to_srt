"""
系统管理API路由
包含客户端心跳、系统关闭等功能
"""

import asyncio
import logging
import subprocess
import os
import gc
import signal
import json
from datetime import datetime
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional, List

logger = logging.getLogger(__name__)

router = APIRouter()


# ========== 请求/响应模型 ==========

class HeartbeatRequest(BaseModel):
    """心跳请求"""
    client_id: str


class RegisterRequest(BaseModel):
    """客户端注册请求"""
    client_id: str
    user_agent: Optional[str] = None


class UnregisterRequest(BaseModel):
    """客户端注销请求"""
    client_id: str


class ShutdownRequest(BaseModel):
    """系统关闭请求"""
    cleanup_temp: bool = False
    force: bool = False


class LogLevelRequest(BaseModel):
    """日志级别设置请求"""
    level: str


# ========== 客户端心跳管理 ==========

@router.post("/api/system/heartbeat")
async def heartbeat(req: HeartbeatRequest):
    """客户端心跳"""
    from app.services.client_registry import get_client_registry

    client_registry = get_client_registry()
    success = client_registry.heartbeat(req.client_id)

    if not success:
        # 客户端未注册，自动注册
        client_registry.register(req.client_id)
        logger.debug(f"客户端自动注册: {req.client_id}")

    return {
        "success": True,
        "active_clients": client_registry.get_active_count()
    }


@router.post("/api/system/register")
async def register_client(req: RegisterRequest):
    """注册新客户端"""
    from app.services.client_registry import get_client_registry

    client_registry = get_client_registry()
    client_registry.register(req.client_id, req.user_agent)

    return {
        "success": True,
        "client_id": req.client_id
    }


@router.post("/api/system/unregister")
async def unregister_client(req: UnregisterRequest):
    """注销客户端（页面关闭时调用）"""
    from app.services.client_registry import get_client_registry

    client_registry = get_client_registry()
    client_registry.unregister(req.client_id)

    return {"success": True}


@router.get("/api/system/has-active-clients")
async def has_active_clients():
    """检查是否有活跃的浏览器标签页"""
    from app.services.client_registry import get_client_registry

    client_registry = get_client_registry()
    return {
        "has_active": client_registry.has_active_clients(),
        "count": client_registry.get_active_count()
    }


# ========== 系统关闭 ==========

@router.post("/api/system/shutdown")
async def shutdown_system(req: ShutdownRequest):
    """
    安全关闭系统
    
    执行顺序:
    1. 保存所有运行中任务的断点
    2. 停止任务队列服务
    3. 终止所有 FFmpeg 子进程
    4. 卸载 GPU 模型并清理显存
    5. 清理临时文件（可选，保留断点数据）
    6. 终止所有相关进程和命令行窗口
    """
    cleanup_report = {}

    try:
        logger.info("=" * 60)
        logger.info("收到系统关闭请求")
        logger.info("=" * 60)

        # ========== Phase 1: 保存断点数据 ==========
        logger.info("Phase 1: 保存断点数据...")
        
        # 1.1 保存所有运行中任务的状态
        try:
            from app.services.job_queue_service import get_queue_service
            queue_service = get_queue_service()
            
            # 获取当前运行的任务
            running_job_id = queue_service.running_job_id
            if running_job_id:
                job = queue_service.get_job(running_job_id)
                if job:
                    # 设置暂停标志，让流水线保存 checkpoint
                    job.paused = True
                    job.message = "系统关闭，自动保存进度"
                    # 保存任务元信息
                    queue_service.transcription_service.save_job_meta(job)
                    logger.info(f"已保存运行中任务状态: {running_job_id}")
            
            # 保存队列状态
            queue_service._save_state()
            cleanup_report["checkpoint_saved"] = True
            logger.info("断点数据已保存")
        except Exception as e:
            logger.warning(f"保存断点数据失败: {e}")
            cleanup_report["checkpoint_saved"] = False

        # ========== Phase 2: 停止服务 ==========
        logger.info("Phase 2: 停止后台服务...")
        
        # 2.1 停止媒体准备服务（包括终止其 FFmpeg 子进程）
        try:
            from app.services.media_prep_service import get_media_prep_service
            media_prep = get_media_prep_service()
            killed_media = media_prep.kill_all_subprocesses()
            media_prep.shutdown()
            cleanup_report["media_prep_stopped"] = True
            cleanup_report["media_prep_killed"] = killed_media
            logger.info(f"媒体准备服务已停止 (终止 {killed_media} 个进程)")
        except Exception as e:
            logger.warning(f"停止媒体准备服务失败: {e}")
            cleanup_report["media_prep_stopped"] = False

        # 2.2 停止任务队列服务
        try:
            from app.services.job_queue_service import get_queue_service
            queue_service = get_queue_service()
            queue_service.shutdown()
            cleanup_report["queue_service_stopped"] = True
            logger.info("任务队列服务已停止")
        except Exception as e:
            logger.warning(f"停止任务队列服务失败: {e}")
            cleanup_report["queue_service_stopped"] = False

        # ========== Phase 3: 终止所有 FFmpeg 进程 ==========
        logger.info("Phase 3: 终止所有FFmpeg进程...")
        
        killed_ffmpeg = _kill_all_ffmpeg_processes()
        cleanup_report["ffmpeg_killed"] = killed_ffmpeg
        if killed_ffmpeg > 0:
            logger.info(f"已终止 {killed_ffmpeg} 个 FFmpeg 进程")

        # ========== Phase 4: 清理 GPU 资源 ==========
        logger.info("Phase 4: 清理GPU资源...")

        # 4.1 卸载所有模型
        try:
            from app.services.model_preload_manager import get_model_manager
            model_manager = get_model_manager()
            if model_manager:
                model_manager.clear_cache()
                cleanup_report["models_unloaded"] = True
                logger.info("GPU模型已卸载")
        except Exception as e:
            logger.warning(f"卸载模型失败: {e}")
            cleanup_report["models_unloaded"] = False

        # 4.2 清理 GPU 缓存
        try:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info("GPU缓存已清理")
                    cleanup_report["gpu_cache_cleared"] = True
            except ImportError:
                cleanup_report["gpu_cache_cleared"] = False
        except Exception as e:
            logger.warning(f"清理GPU缓存失败: {e}")
            cleanup_report["gpu_cache_cleared"] = False

        # ========== Phase 5: 清理临时文件（可选）和日志 ==========
        # V3.1.1+dev.20260106.02: 每次关闭都清理超过7天的日志
        logger.info("Phase 5: 清理过期日志...")
        logs_cleaned = _cleanup_old_logs(days=7)
        cleanup_report["logs_cleaned"] = logs_cleaned

        if req.cleanup_temp:
            logger.info("Phase 5: 清理临时文件...")
            cleaned = _cleanup_temp_files_safely()
            cleanup_report["temp_files_cleaned"] = cleaned
        else:
            cleanup_report["temp_files_cleaned"] = False

        # 发送成功响应
        response = {
            "success": True,
            "message": "系统正在关闭...",
            "cleanup_report": cleanup_report
        }

        logger.info("资源清理完成，准备关闭进程")
        logger.info(f"清理报告: {cleanup_report}")

    except Exception as e:
        logger.error(f"关闭系统失败: {str(e)}", exc_info=True)
        response = {
            "success": False,
            "message": f"关闭系统失败: {str(e)}"
        }

    # Phase 6: 异步执行进程终止（响应发送后执行）
    asyncio.create_task(_terminate_processes())

    return response


def _kill_all_ffmpeg_processes() -> int:
    """
    终止所有 FFmpeg 相关进程
    
    Returns:
        int: 被终止的进程数
    """
    killed_count = 0
    
    try:
        import psutil
        
        current_pid = os.getpid()
        ffmpeg_names = {'ffmpeg.exe', 'ffprobe.exe', 'ffmpeg', 'ffprobe'}
        
        for proc in psutil.process_iter(['pid', 'name', 'ppid']):
            try:
                proc_name = proc.info['name'].lower() if proc.info['name'] else ''
                
                # 检查是否是 FFmpeg 相关进程
                if proc_name in ffmpeg_names:
                    # 终止进程
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    killed_count += 1
                    logger.debug(f"已终止 FFmpeg 进程: PID={proc.info['pid']}")
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
                
    except ImportError:
        logger.warning("psutil 未安装，使用 taskkill 回退方案")
        # 回退方案：使用 taskkill
        try:
            result = subprocess.run(
                ['taskkill', '/F', '/IM', 'ffmpeg.exe'],
                capture_output=True,
                timeout=10
            )
            if result.returncode == 0:
                killed_count += 1
        except:
            pass
            
    except Exception as e:
        logger.warning(f"终止 FFmpeg 进程失败: {e}")
        
    return killed_count


def _cleanup_temp_files_safely() -> bool:
    """
    安全清理临时文件
    
    注意：保留以下关键数据：
    - jobs/ 目录下的所有数据（断点、元信息、输出文件）- 永不清理
    - models/ 目录下的模型文件
    - output/ 目录下的输出文件
    
    只清理：
    - temp/ 目录下的临时文件（排除包含断点数据的目录）
    
    Returns:
        bool: 是否清理成功
    """
    try:
        from app.core.config import config
        import shutil
        
        cleaned = False
        
        # 只清理 temp 目录，绝对不清理 jobs 目录
        if config.TEMP_DIR.exists():
            # 遍历 temp 目录，只删除确定可以清理的临时文件
            for item in config.TEMP_DIR.iterdir():
                try:
                    if item.is_file():
                        # 只删除临时文件（如 .tmp, .part 等）
                        safe_extensions = {'.tmp', '.part', '.temp', '.log'}
                        if item.suffix.lower() in safe_extensions:
                            item.unlink()
                            cleaned = True
                    elif item.is_dir():
                        # 检查是否是需要保留的目录
                        # 保留包含任何重要数据的目录
                        should_keep = False
                        important_files = {
                            'checkpoint.json', 'meta.json', 'progress.json',
                            'job_meta.json', 'subtitles.json', '.srt', '.vtt'
                        }
                        
                        for sub_item in item.rglob('*'):
                            if sub_item.name in important_files or sub_item.suffix in {'.srt', '.vtt', '.json'}:
                                should_keep = True
                                break
                        
                        if not should_keep:
                            shutil.rmtree(item, ignore_errors=True)
                            cleaned = True
                except Exception as e:
                    logger.debug(f"清理临时文件失败: {item} - {e}")
        
        if cleaned:
            logger.info("临时文件已清理（保留所有任务数据）")
        
        return cleaned

    except Exception as e:
        logger.warning(f"清理临时文件失败: {e}")
        return False


def _cleanup_old_logs(days: int = 7) -> int:
    """
    V3.1.1+dev.20260106.02: 清理超过指定天数的日志文件

    Args:
        days: 保留天数，默认7天

    Returns:
        int: 清理的日志文件数量
    """
    try:
        from app.core.config import config
        from datetime import datetime, timedelta

        cleaned_count = 0
        cutoff_time = datetime.now() - timedelta(days=days)

        if not config.LOG_DIR.exists():
            return 0

        # 遍历日志目录
        for log_file in config.LOG_DIR.iterdir():
            try:
                # 只处理 .log 文件
                if not log_file.is_file() or log_file.suffix.lower() != '.log':
                    continue

                # 获取文件修改时间
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)

                # 如果文件超过指定天数，删除
                if mtime < cutoff_time:
                    log_file.unlink()
                    cleaned_count += 1
                    logger.debug(f"已删除过期日志: {log_file.name} (修改于 {mtime.strftime('%Y-%m-%d')})")

            except Exception as e:
                logger.debug(f"清理日志文件失败: {log_file} - {e}")

        if cleaned_count > 0:
            logger.info(f"已清理 {cleaned_count} 个超过 {days} 天的日志文件")

        return cleaned_count

    except Exception as e:
        logger.warning(f"清理日志失败: {e}")
        return 0


async def _terminate_processes():
    """
    终止所有相关进程和命令行窗口
    
    使用多种策略确保进程被正确终止:
    1. 使用 psutil 查找并终止子进程
    2. 使用 taskkill 按进程名终止
    3. 使用 taskkill 按窗口标题终止
    """
    await asyncio.sleep(0.5)  # 等待响应发送完成

    logger.info("=" * 60)
    logger.info("Phase 6: 终止所有进程")
    logger.info("=" * 60)

    try:
        # 方法1: 使用 psutil 终止当前进程的所有子进程
        try:
            import psutil
            current_process = psutil.Process(os.getpid())
            children = current_process.children(recursive=True)
            
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # 等待子进程结束
            gone, alive = psutil.wait_procs(children, timeout=3)
            
            # 强制终止仍存活的进程
            for proc in alive:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
                    
            logger.info(f"已终止 {len(children)} 个子进程")
        except ImportError:
            logger.debug("psutil 不可用，使用备选方案")
        except Exception as e:
            logger.debug(f"psutil 终止子进程失败: {e}")

        # 方法2: 使用 psutil 精确终止占用端口 5173 的进程（前端）
        if os.name == 'nt':
            try:
                import psutil
                for conn in psutil.net_connections(kind='inet'):
                    if conn.laddr.port == 5173 and conn.status == 'LISTEN':
                        try:
                            proc = psutil.Process(conn.pid)
                            logger.info(f"终止前端进程: PID={conn.pid}")
                            proc.terminate()
                            try:
                                proc.wait(timeout=3)
                            except psutil.TimeoutExpired:
                                proc.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
            except Exception as e:
                logger.debug(f"psutil 终止前端进程失败: {e}")
                # 备选方案：使用 netstat + taskkill
                try:
                    result = subprocess.run(
                        'netstat -ano | findstr ":5173" | findstr "LISTENING"',
                        shell=True,
                        capture_output=True,
                        timeout=5
                    )
                    if result.returncode == 0 and result.stdout:
                        # V3.1.0+dev.20260104.01: 修复 Windows 编码问题
                        for line in result.stdout.decode('utf-8', errors='replace').strip().split('\n'):
                            parts = line.split()
                            if len(parts) >= 5:
                                pid = parts[-1]
                                subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True, timeout=3)
                                logger.info(f"已终止前端进程 (PID: {pid})")
                except Exception:
                    pass

        # 终止主窗口（run.bat 的 cmd 窗口）和子窗口
        # 使用 taskkill 终止所有相关的 cmd.exe 进程
        # 注意：taskkill 使用 /FI 过滤器时必须同时指定 /IM 参数
        if os.name == 'nt':
            try:
                # 终止标题包含 Video2SRT 的 cmd 窗口（Backend/Frontend 子窗口）
                subprocess.run(
                    'taskkill /F /IM cmd.exe /FI "WINDOWTITLE eq Video2SRT*"',
                    shell=True,
                    capture_output=True,
                    timeout=3
                )
                # 终止主窗口（标题是 "Video to SRT GPU"，精确匹配）
                subprocess.run(
                    'taskkill /F /IM cmd.exe /FI "WINDOWTITLE eq Video to SRT GPU"',
                    shell=True,
                    capture_output=True,
                    timeout=3
                )
                # 备用：匹配 "Video to SRT*" 模式
                subprocess.run(
                    'taskkill /F /IM cmd.exe /FI "WINDOWTITLE eq Video to SRT*"',
                    shell=True,
                    capture_output=True,
                    timeout=3
                )
                logger.info("主窗口终止命令已发送")
            except Exception as e:
                logger.debug(f"终止主窗口失败: {e}")

        logger.info("相关进程终止命令已发送")

    except Exception as e:
        logger.error(f"终止进程失败: {e}")

    # 自我终止
    logger.info("后端进程即将退出...")
    logger.info("=" * 60)

    # 使用 os._exit 确保立即退出，不执行 cleanup handlers
    os._exit(0)


# ========== 日志级别管理 ==========

@router.get("/api/system/log-level")
async def get_log_level():
    """获取当前日志级别"""
    try:
        # 根logger设置为DEBUG，实际级别由处理器控制
        root_logger = logging.getLogger()
        level_name = "INFO"  # 默认值

        # 从控制台处理器获取实际的日志级别
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                level_name = logging.getLevelName(handler.level)
                break

        return {
            "success": True,
            "level": level_name,
            "message": "日志级别获取成功"
        }
    except Exception as e:
        logger.error(f"获取日志级别失败: {e}")
        return {
            "success": False,
            "level": "INFO",
            "message": f"获取日志级别失败: {str(e)}"
        }


@router.post("/api/system/log-level")
async def set_log_level(req: LogLevelRequest):
    """设置日志级别"""
    try:
        # 验证日志级别
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        level = req.level.upper()

        if level not in valid_levels:
            return {
                "success": False,
                "message": f"无效的日志级别: {req.level}，有效值: {', '.join(valid_levels)}"
            }

        # 设置根日志记录器的级别
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level))

        # 同时更新所有已存在的日志记录器
        for name in logging.Logger.manager.loggerDict:
            logger_obj = logging.getLogger(name)
            if isinstance(logger_obj, logging.Logger):
                logger_obj.setLevel(getattr(logging, level))

        logger.info(f"日志级别已更新为: {level}")

        return {
            "success": True,
            "level": level,
            "message": f"日志级别已更新为 {level}，重启系统后永久生效"
        }
    except Exception as e:
        logger.error(f"设置日志级别失败: {e}")
        return {
            "success": False,
            "message": f"设置日志级别失败: {str(e)}"
        }


# ========== 更新系统 API (V3.1.0+dev.20260104.01) ==========

class CheckUpdateResponse(BaseModel):
    """检查更新响应"""
    has_update: bool
    current_version: str
    latest_version: Optional[str] = None
    changelog: Optional[str] = None
    download_url: Optional[str] = None
    release_date: Optional[str] = None
    force_update: bool = False  # V3.1.1+dev.20260105.01: 强制更新标志


class TriggerUpdateRequest(BaseModel):
    """触发更新请求"""
    download_url: str
    version: str
    changelog: Optional[str] = None
    delay_mode: bool = False  # V3.1.1+dev.20260105.01: 延迟更新模式（重启时更新）


# 当前版本号
CURRENT_VERSION = "3.1.1"

# V3.1.1+dev.20260105.01: 版本检查配置（使用 Gitee 镜像的 version.json）
VERSION_CHECK_URL = "https://gitee.com/comment_out/anchor-flux-update/raw/master/version.json"


@router.get("/api/system/version")
async def get_current_version():
    """获取当前系统版本"""
    return {
        "success": True,
        "version": CURRENT_VERSION,
        "build_date": "2026-01-04"  # TODO: 从构建信息读取
    }


@router.get("/api/system/check-update")
async def check_update():
    """
    检查是否有新版本可用

    V3.1.1+dev.20260105.01: 从 Gitee 镜像的 version.json 获取版本信息
    version.json 格式:
    {
        "latest_version": "3.1.3",
        "force_update": false,
        "changelog": "更新内容",
        "download_url": "https://github.com/.../update_v3.1.3.zip"
    }
    """
    import urllib.request
    import urllib.error

    try:
        logger.info(f"Checking for updates: {VERSION_CHECK_URL}")

        # 发送请求获取 version.json
        req = urllib.request.Request(
            VERSION_CHECK_URL,
            headers={
                'User-Agent': 'AnchorFlux-Updater/1.0',
                'Accept': 'application/json',
                'Cache-Control': 'no-cache'  # 避免缓存
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.info("version.json not found")
                return CheckUpdateResponse(
                    has_update=False,
                    current_version=CURRENT_VERSION
                ).dict()
            raise

        # 解析 version.json 格式
        latest_version = data.get('latest_version', '').lstrip('v')
        changelog = data.get('changelog', '')
        download_url = data.get('download_url', '')
        force_update = data.get('force_update', False)

        # 比较版本号（使用元组比较以支持语义化版本）
        def parse_version(v):
            """将版本字符串解析为可比较的元组"""
            try:
                parts = v.split('.')
                return tuple(int(p) for p in parts[:3])
            except (ValueError, AttributeError):
                return (0, 0, 0)

        current_tuple = parse_version(CURRENT_VERSION)
        latest_tuple = parse_version(latest_version)
        has_update = latest_tuple > current_tuple

        logger.info(f"Current: {CURRENT_VERSION} ({current_tuple}), Latest: {latest_version} ({latest_tuple}), Has update: {has_update}")

        return CheckUpdateResponse(
            has_update=has_update,
            current_version=CURRENT_VERSION,
            latest_version=latest_version,
            changelog=changelog,
            download_url=download_url,
            force_update=force_update
        ).dict()

    except Exception as e:
        logger.error(f"检查更新失败: {e}")
        return {
            "success": False,
            "has_update": False,
            "current_version": CURRENT_VERSION,
            "message": f"检查更新失败: {str(e)}"
        }


@router.post("/api/system/trigger-update")
async def trigger_update(req: TriggerUpdateRequest):
    """
    触发更新流程

    V3.1.1+dev.20260105.01: 支持延迟更新模式
    - delay_mode=False: 立即更新（写入信号文件后关闭后端）
    - delay_mode=True: 重启时更新（仅写入信号文件，不关闭后端）

    此 API 会:
    1. 在项目根目录写入 update_signal.json
    2. 如果不是延迟模式，关闭后端服务
    3. Bootloader 检测到信号文件后执行更新
    """
    from app.core.config import config

    try:
        # 获取项目根目录
        project_root = config.PROJECT_ROOT
        signal_file = project_root / "update_signal.json"

        # 写入更新信号文件
        signal_data = {
            "version": req.version,
            "download_url": req.download_url,
            "changelog": req.changelog,
            "triggered_at": datetime.now().isoformat(),
            "current_version": CURRENT_VERSION,
            "delay_mode": req.delay_mode  # V3.1.1+dev.20260105.01: 记录延迟模式
        }

        with open(signal_file, 'w', encoding='utf-8') as f:
            json.dump(signal_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Update signal written: {signal_file}")
        logger.info(f"Target version: {req.version}, Delay mode: {req.delay_mode}")

        # 根据模式返回不同响应
        if req.delay_mode:
            # 延迟模式：仅写入信号文件，不关闭后端
            response = {
                "success": True,
                "message": "更新已安排，将在下次重启时执行",
                "signal_file": str(signal_file),
                "delay_mode": True
            }
        else:
            # 立即模式：写入信号文件后关闭后端
            response = {
                "success": True,
                "message": "更新信号已写入，系统即将重启进行更新",
                "signal_file": str(signal_file),
                "delay_mode": False
            }
            # 异步关闭后端（等待响应发送后）
            asyncio.create_task(_shutdown_for_update())

        return response

    except Exception as e:
        logger.error(f"触发更新失败: {e}")
        return {
            "success": False,
            "message": f"触发更新失败: {str(e)}"
        }


async def _shutdown_for_update():
    """为更新关闭后端服务"""
    await asyncio.sleep(1)  # 等待响应发送完成

    logger.info("=" * 60)
    logger.info("Shutting down for update...")
    logger.info("=" * 60)

    # 保存必要状态
    try:
        from app.services.job_queue_service import get_queue_service
        queue_service = get_queue_service()
        queue_service._save_state()
    except Exception as e:
        logger.warning(f"保存队列状态失败: {e}")

    # 退出进程（Bootloader 会检测到并执行更新）
    logger.info("Backend exiting for update...")
    os._exit(0)
