import os
import sys
import uuid
import shutil
import logging
import asyncio
import time
import subprocess
import warnings

# 抑制 libpng iCCP 警告（必须在导入图像库之前设置）
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'
warnings.filterwarnings('ignore', message='.*iCCP.*')
warnings.filterwarnings('ignore', message='.*sRGB.*')

# V3.2.0+dev.20260218.01: 抑制第三方库的弃用/未来警告（来自依赖库内部，不影响功能）
# --- SpeechBrain 相关 ---
warnings.filterwarnings('ignore', message=r'.*torchaudio\._backend\.list_audio_backends has been deprecated.*')
warnings.filterwarnings('ignore', message=r'.*torch\.cuda\.amp\.custom_fwd.*is deprecated.*')
warnings.filterwarnings('ignore', message=r".*Module 'speechbrain\.pretrained' was deprecated.*")
warnings.filterwarnings('ignore', message=r'.*Requested Pretrainer collection using symlinks on Windows.*')
# --- pyannote / lightning 相关 ---
# torchcodec 警告消息以 \n 开头，需要 (?s) 使 . 匹配换行符
warnings.filterwarnings('ignore', message=r'(?s).*torchcodec is not installed correctly.*')
warnings.filterwarnings('ignore', message=r'(?s).*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*')
warnings.filterwarnings('ignore', message=r'.*TensorFloat-32 \(TF32\) has been disabled.*')
warnings.filterwarnings('ignore', message=r'.*std\(\): degrees of freedom is <= 0.*')

# 尝试抑制 C 库级别的 libpng 警告
try:
    import ctypes
    # Windows: 抑制 C 运行时警告
    if sys.platform == 'win32':
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
except Exception:
    pass

from fastapi import FastAPI, Form, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入核心配置和日志
from app.core.config import FLAVOR, IS_LITE, config
from app.core.logging import setup_logging

from app.config.model_config import ModelPreloadConfig

# 导入API路由
from app.api.routes import media_routes  # 新增：媒体资源路由
from app.api.routes.project_routes import router as project_router
from app.api.routes.legacy_compat_routes import router as legacy_router
from app.api.routes.stream_routes import router as stream_router
from app.api.routes.file_routes import create_file_router  # 新增：文件管理路由
from app.api.routes import system_routes  # 新增：系统管理路由
from app.api.routes import config_routes  # 新增：用户配置路由
from app.api.routes import debug_routes  # 新增：调试路由
from app.services.file_service import FileManagementService

# 导入FFmpeg管理器
from app.services.ffmpeg_manager import get_ffmpeg_manager

# 配置日志（在其他初始化之前）
logger = setup_logging()


def cleanup_old_processes():
    """
    启动时清理可能残留的旧进程
    
    清理对象：
    1. 占用端口 8000 的旧后端进程（排除自身）
    2. 占用端口 5173 的旧前端进程
    3. 旧的 FFmpeg 进程
    
    使用 psutil 进行更可靠的进程管理
    """
    logger.info("=" * 40)
    logger.info("检查并清理残留的旧进程...")
    logger.info("=" * 40)
    cleaned_count = 0
    current_pid = os.getpid()
    
    if os.name != 'nt':
        logger.info("非 Windows 系统，跳过旧进程清理")
        return cleaned_count
    
    try:
        import psutil
        
        # 1. 查找并清理占用端口 8000 的进程（排除自身）
        logger.info("检查端口 8000...")
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == 8000 and conn.status == 'LISTEN':
                try:
                    proc = psutil.Process(conn.pid)
                    if proc.pid != current_pid:
                        logger.info(f"发现占用端口 8000 的旧进程: PID={proc.pid}, Name={proc.name()}")
                        proc.terminate()
                        try:
                            proc.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            proc.kill()
                        cleaned_count += 1
                        logger.info(f"已终止旧后端进程 (PID: {proc.pid})")
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.debug(f"无法处理进程: {e}")
        
        # 2. 查找并清理占用端口 5173 的进程
        logger.info("检查端口 5173...")
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == 5173 and conn.status == 'LISTEN':
                try:
                    proc = psutil.Process(conn.pid)
                    logger.info(f"发现占用端口 5173 的旧进程: PID={proc.pid}, Name={proc.name()}")
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    cleaned_count += 1
                    logger.info(f"已终止旧前端进程 (PID: {proc.pid})")
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.debug(f"无法处理进程: {e}")
        
        # 3. 清理所有 FFmpeg 进程
        logger.info("检查 FFmpeg 进程...")
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] and proc.info['name'].lower() in ['ffmpeg.exe', 'ffprobe.exe']:
                    logger.info(f"发现 FFmpeg 进程: PID={proc.info['pid']}")
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    cleaned_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # 4. 清理可能残留的 Python uvicorn 进程（按命令行参数匹配）
        logger.info("检查残留的 uvicorn 进程...")
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['pid'] == current_pid:
                    continue  # 跳过自身
                    
                cmdline = proc.info.get('cmdline') or []
                cmdline_str = ' '.join(cmdline).lower()
                
                # 匹配 uvicorn 且运行 app.main:app
                if 'uvicorn' in cmdline_str and 'app.main:app' in cmdline_str:
                    logger.info(f"发现残留的 uvicorn 进程: PID={proc.info['pid']}")
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    cleaned_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
    except ImportError:
        logger.warning("psutil 未安装，使用备选方案清理进程")
        # 回退到 taskkill 方案
        try:
            # 使用 netstat 和 taskkill
            result = subprocess.run(
                'netstat -ano | findstr ":8000" | findstr "LISTENING"',
                shell=True,
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout:
                # V3.1.0+dev.20260104.01: 修复 Windows 编码问题
                lines = result.stdout.decode('utf-8', errors='replace').strip().split('\n')
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        if pid != str(current_pid):
                            subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True, timeout=3)
                            cleaned_count += 1
                            logger.info(f"已清理端口 8000 上的旧进程 (PID: {pid})")
        except Exception as e:
            logger.warning(f"备选清理方案失败: {e}")
    
    except Exception as e:
        logger.warning(f"清理旧进程时发生错误: {e}")
    
    if cleaned_count > 0:
        logger.info(f"共清理 {cleaned_count} 个残留进程")
        # 等待进程完全退出
        time.sleep(2)
    else:
        logger.info("未发现需要清理的残留进程")
    
    logger.info("=" * 40)
    return cleaned_count


# 注意：不再在模块加载时自动执行清理
# run.bat 会在启动前进行清理，避免这里阻塞启动
# 如需手动清理，可以调用 cleanup_old_processes()


app = FastAPI(title="Video To SRT API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册API路由
app.include_router(media_routes.router)  # 新增：媒体资源路由
app.include_router(system_routes.router)  # 新增：系统管理路由
app.include_router(config_routes.router)  # 新增：用户配置路由
app.include_router(debug_routes.router)  # 新增：调试路由
app.include_router(project_router)  # Task6: 项目路由
app.include_router(legacy_router)  # Task6: legacy 兼容路由
app.include_router(stream_router)  # Task6: 项目级 SSE 路由

# 注册仅 Full 模式可用的路由
if not IS_LITE:
    from app.api.routes import model_routes
    from app.api.routes import model_runtime_routes
    from app.api.routes.demucs_routes import create_demucs_router

    app.include_router(model_routes.router)
    app.include_router(model_runtime_routes.router)

    # 注册Demucs配置路由（需要在转录路由之前注册）
    demucs_router = create_demucs_router()
    app.include_router(demucs_router)

@app.on_event("startup")
async def startup_event():
    """应用启动事件 - 初始化模型管理器和FFmpeg检测"""
    try:
        logger.info("="  * 60)
        logger.info("服务启动中...")
        logger.info("=" * 60)

        # 1. 设置SSE事件循环引用（必须在模型管理器初始化之前！）
        logger.info("步骤 1/4: 设置SSE事件循环...")
        try:
            import asyncio
            current_loop = asyncio.get_running_loop()

            # 设置统一SSE管理器的事件循环（用于转录进度和模型下载）
            from app.services.sse_service import get_sse_manager
            sse_manager = get_sse_manager()
            sse_manager.set_event_loop(current_loop)
            logger.info("统一SSE管理器事件循环已设置")

        except Exception as e:
            logger.warning(f"设置SSE事件循环异常: {e}")

        # 2. FFmpeg检测和自动下载
        logger.info("步骤 2/4: 检测FFmpeg...")
        ffmpeg_mgr = get_ffmpeg_manager()
        try:
            ffmpeg_path = ffmpeg_mgr.ensure_ffmpeg()
            logger.info(f"FFmpeg检测完成: {ffmpeg_path}")
        except RuntimeError as e:
            # FFmpeg不可用但不阻止启动，只是记录警告
            logger.warning(f"FFmpeg检测失败: {e}")
            logger.warning("转录功能可能无法使用，请手动安装FFmpeg")

        if not IS_LITE:
            # 3. 初始化模型管理器（此时事件循环已设置，后台验证可以正常推送SSE）
            logger.info("步骤 3/4: 初始化模型管理器...")
            from app.services.model_manager_v2 import get_model_manager_v2
            model_manager = get_model_manager_v2()
            logger.info("模型管理器初始化成功 (V2)")

            # 4. 初始化队列服务（新增）
            logger.info("步骤 4/4: 初始化任务队列服务...")
            from app.services.job_queue_service import get_queue_service
            from app.services.transcription_service import get_transcription_service
            from app.core.config import config
            transcription_service = get_transcription_service(str(config.JOBS_DIR))
            queue_service = get_queue_service(transcription_service)
            logger.info("任务队列服务已启动")
        else:
            logger.info("Lite 模式启动: 跳过模型管理与转录队列初始化")

        # 4.5. 初始化媒体准备服务（独立线程执行转码，不阻塞主流程）
        from app.services.media_prep_service import get_media_prep_service
        media_prep_service = get_media_prep_service()
        logger.info("媒体准备服务已启动")

        # 5. 清理无效的任务索引映射（第一阶段修复：数据同步）
        logger.info("清理无效的任务索引映射...")
        try:
            from app.services.job_index_service import get_job_index_service
            job_index = get_job_index_service(str(config.JOBS_DIR))
            job_index.cleanup_invalid_mappings()
            logger.info("任务索引映射清理完成")
        except Exception as e:
            logger.warning(f"清理任务索引映射失败: {e}")

        # 不在启动时预加载模型，等待前端就绪后通过API调用
        logger.info("后端服务已就绪，等待前端启动后进行模型预加载")

        # 6. 延迟检查并打开浏览器（如果没有活跃客户端）
        asyncio.create_task(open_browser_if_needed())

        logger.info("=" * 60)
        logger.info("服务启动完成")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"启动初始化失败: {str(e)}", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件 - 清理资源"""
    try:
        # 停止媒体准备服务
        from app.services.media_prep_service import shutdown_media_prep_service
        try:
            shutdown_media_prep_service()
            logger.info("媒体准备服务已停止")
        except:
            pass

        if not IS_LITE:
            # 停止队列服务（新增）
            from app.services.job_queue_service import get_queue_service
            try:
                queue_service = get_queue_service()
                queue_service.shutdown()
                logger.info("任务队列服务已停止")
            except:
                pass

            # 清理模型缓存
            from app.services.model_manager_v2 import get_model_manager_v2
            model_manager = get_model_manager_v2()
            if model_manager:
                model_manager.clear_cache()
                logger.info("已清理模型缓存")
    except Exception as e:
        logger.error(f"清理资源失败: {str(e)}")

# 使用统一配置中的目录
INPUT_DIR = str(config.INPUT_DIR)
OUTPUT_DIR = str(config.OUTPUT_DIR)
JOBS_DIR = str(config.JOBS_DIR)
TEMP_DIR = str(config.TEMP_DIR)

# 初始化转录服务（仅 Full 模式）
transcription_service = None
if not IS_LITE:
    from app.services.transcription_service import get_transcription_service
    transcription_service = get_transcription_service(JOBS_DIR)

# 初始化文件管理服务
file_service = FileManagementService(INPUT_DIR, OUTPUT_DIR)

# 注册文件管理路由
file_router = create_file_router(file_service)
app.include_router(file_router)

# 注册转录路由（包含暂停、恢复等新功能）
if not IS_LITE:
    from app.api.routes.transcription_routes import create_transcription_router
    from app.api.routes.speaker_routes import create_speaker_router

    transcription_router = create_transcription_router(transcription_service, file_service, OUTPUT_DIR)
    app.include_router(transcription_router)

    # 注册 speaker 路由（说话人 profile/改绑/合并）
    speaker_router = create_speaker_router(transcription_service)
    app.include_router(speaker_router)
else:
    logger.info("Lite 模式启动: 跳过 transcription/speaker 路由注册")

ModelPreloadConfig.print_config()

class UploadResponse(BaseModel):
    job_id: str
    filename: str
    original_name: str
    message: str

class FileInfo(BaseModel):
    name: str
    size: int
    modified: str
    path: str

def get_file_size_str(size_bytes):
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"

def is_video_or_audio_file(filename):
    """检查是否为支持的视频或音频文件"""
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    ext = os.path.splitext(filename.lower())[1]
    return ext in video_extensions or ext in audio_extensions

@app.get("/api/files")
async def list_files():
    """获取输入目录中的所有媒体文件（含断点信息）"""
    try:
        files = []
        if os.path.exists(INPUT_DIR):
            for filename in os.listdir(INPUT_DIR):
                file_path = os.path.join(INPUT_DIR, filename)
                if os.path.isfile(file_path) and is_video_or_audio_file(filename):
                    stat = os.stat(file_path)

                    # 检查是否有断点
                    checkpoint_info = None
                    if transcription_service is not None:
                        checkpoint_info = transcription_service.check_file_checkpoint(file_path)

                    file_info = FileInfo(
                        name=filename,
                        size=stat.st_size,
                        modified=datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        path=file_path
                    )

                    # 添加断点信息
                    file_dict = file_info.dict()
                    if checkpoint_info:
                        file_dict['checkpoint'] = checkpoint_info
                    else:
                        file_dict['checkpoint'] = None

                    files.append(file_dict)

        # 按修改时间倒序排列
        files.sort(key=lambda x: x['modified'], reverse=True)
        return {"files": files, "input_dir": INPUT_DIR}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")

@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """删除input目录中的文件"""
    try:
        file_path = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        os.remove(file_path)
        return {"success": True, "message": f"文件 {filename} 已删除"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}")

# 所有转录相关的路由已经在transcription_routes.py中定义，这里的重复定义已被删除
# 原有的/api/upload, /api/create-job, /api/start(已注释), /api/status/{job_id},
# /api/download/{job_id}, /api/copy-result/{job_id}等端点现在都由transcription_routes.py处理

@app.get("/api/ping")
async def ping():
    return {"pong": True}

@app.get("/api/test/stream")
async def test_sse_stream(request: Request):
    """SSE 测试端点 - 用于前端测试 SSE 连接"""
    from app.services.sse_service import get_sse_manager
    sse_manager = get_sse_manager()

    # 使用固定的测试频道
    channel_id = "test:sse"

    # 定义初始状态
    def get_initial_state():
        return {
            "type": "test",
            "message": "SSE 测试连接已建立",
            "timestamp": time.time()
        }

    return StreamingResponse(
        sse_manager.subscribe(channel_id, request, initial_state_callback=get_initial_state),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/api/cpu-info")
async def get_cpu_info():
    """获取系统CPU信息和亲和性支持状态"""
    try:
        from app.services.hardware_profile_service import get_hardware_profile_provider
        provider = get_hardware_profile_provider()
        cpu_info = provider.get_cpu_system_info()
        return {
            "success": True,
            "cpu_info": cpu_info,
            "available_strategies": ["auto", "half", "custom"]
        }
    except Exception as e:
        return {
            "success": False, 
            "error": str(e),
            "cpu_info": {"supported": False}
        }

@app.get("/api/hardware/basic")
async def get_hardware_basic():
    """获取核心硬件信息"""
    try:
        from app.services.hardware_profile_service import get_hardware_profile_provider
        provider = get_hardware_profile_provider()
        hardware_info = provider.get_hardware_info()
        
        return {
            "success": True,
            "hardware": hardware_info.to_dict(),
            "message": "硬件信息获取成功"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"获取硬件信息失败: {str(e)}"
        }

@app.get("/api/hardware/optimize")
async def get_hardware_optimization():
    """获取基于硬件的优化配置"""
    try:
        from app.services.hardware_profile_service import get_hardware_profile_provider
        provider = get_hardware_profile_provider()
        hardware_info = provider.get_hardware_info()
        optimization_config = provider.get_optimization_config(hardware_info)
        
        return {
            "success": True,
            "optimization": optimization_config.to_dict(),
            "message": "优化配置获取成功"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"获取优化配置失败: {str(e)}"
        }

@app.get("/api/hardware/status")
async def get_hardware_status():
    """获取完整的硬件状态和优化信息"""
    try:
        from app.services.hardware_profile_service import get_hardware_profile_provider
        provider = get_hardware_profile_provider()
        hardware_info = provider.get_hardware_info()
        optimization_config = provider.get_optimization_config(hardware_info)
        
        return {
            "success": True,
            "hardware": hardware_info.to_dict(),
            "optimization": optimization_config.to_dict(),
            "message": "硬件状态获取成功"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"获取硬件状态失败: {str(e)}"
        }

# 模型管理API端点
@app.post("/api/models/cache/clear")
async def clear_models_cache():
    """清空模型缓存（ModelManager V2）。"""
    if IS_LITE:
        raise HTTPException(status_code=422, detail="Lite 模式不支持模型缓存管理")
    try:
        from app.services.model_manager_v2 import get_model_manager_v2
        model_manager = get_model_manager_v2()
        model_manager.unload_all()
        logger.info("手动清空模型缓存成功")
        return {
            "success": True,
            "message": "模型缓存已清空"
        }
    except Exception as e:
        logger.error(f"清空模型缓存失败: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"清空缓存失败: {str(e)}"
        }
# ========== 默认预加载模型配置API ==========

@app.post("/api/models/cache/unload")
async def unload_model(request: dict):
    """卸载指定模型（使用 ModelManager V2）。"""
    if IS_LITE:
        raise HTTPException(status_code=422, detail="Lite 模式不支持模型缓存管理")
    try:
        model_id = request.get("model_id")
        if not model_id:
            return {"success": False, "message": "缺少model_id参数"}
        from app.services.model_manager_v2 import get_model_manager_v2
        model_manager = get_model_manager_v2()
        model_manager.unload_all()
        logger.info(f"卸载模型: {model_id} (清空缓存)")
        return {"success": True, "message": f"模型 {model_id} 已从缓存卸载"}
    except Exception as e:
        logger.error(f"卸载模型失败: {str(e)}", exc_info=True)
        return {"success": False, "message": f"卸载失败: {str(e)}"}

@app.post("/api/shutdown")
async def shutdown_server():
    """优雅关闭服务器"""
    try:
        logger.info("收到关闭服务器请求")

        # 清理资源
        if not IS_LITE:
            from app.services.model_manager_v2 import get_model_manager_v2
            model_manager = get_model_manager_v2()
            if model_manager:
                model_manager.unload_all()
                logger.info("已清理模型缓存")
        
        # 返回成功响应
        response = {
            "success": True,
            "message": "服务器正在优雅关闭"
        }
        
        # 异步关闭服务器
        import asyncio
        import os
        asyncio.create_task(delayed_shutdown())
        
        return response
        
    except Exception as e:
        logger.error(f"关闭服务器失败: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"关闭服务器失败: {str(e)}"
        }

async def delayed_shutdown():
    """延迟关闭服务器，给响应时间返回"""
    await asyncio.sleep(1)  # 等待1秒让响应返回
    logger.info("服务器即将关闭...")
    import os
    os._exit(0)


async def open_browser_if_needed():
    """
    启动后智能打开浏览器

    策略：
    1. 等待前端服务启动完成
    2. 多次检查是否有活跃的客户端（旧页面需要时间重新连接）
    3. 如果有活跃客户端，跳过打开新页面（旧页面会自动刷新）
    4. 如果没有活跃客户端，打开新的浏览器标签页

    V3.1.1+dev.20260105.05: 开发模式下打开 localhost:5173，生产模式打开 localhost:8000
    """
    from app.services.client_registry import get_client_registry

    # V3.1.1+dev.20260105.05: 检测开发模式，决定打开哪个端口
    dev_mode = os.environ.get('DEV_MODE', '').lower() in ('true', '1', 'yes')
    target_url = "http://localhost:5173" if dev_mode else "http://localhost:8000"

    # 等待前端服务启动（run.bat 中前端在后端之后启动）
    logger.info("等待前端服务启动...")
    await asyncio.sleep(5)

    # 多次检查活跃客户端，给旧页面足够时间重新连接
    # 旧页面在检测到后端恢复后会立即发送心跳
    registry = get_client_registry()

    for i in range(3):  # 最多检查3次，每次间隔2秒
        if registry.has_active_clients():
            active_count = registry.get_active_count()
            logger.info(f"检测到 {active_count} 个活跃客户端，跳过打开新浏览器（旧页面将自动刷新）")
            return
        if i < 2:  # 前两次检查后等待
            logger.info(f"第 {i+1} 次检查无活跃客户端，等待2秒后重试...")
            await asyncio.sleep(2)

    # 没有活跃客户端，打开浏览器
    logger.info(f"无活跃客户端，正在打开浏览器: {target_url}")
    try:
        if os.name == 'nt':
            # Windows: 使用 start 命令强制打开新窗口/标签页
            subprocess.Popen(
                ['cmd', '/c', 'start', '', target_url],
                shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        else:
            import webbrowser
            webbrowser.open(target_url, new=2)
        logger.info(f"浏览器标签页已打开: {target_url}")
    except Exception as e:
        logger.error(f"打开浏览器失败: {e}")


# ========== 静态文件托管 (生产模式) ==========
# V3.1.1+dev.20260105.05: 检测 DEV_MODE 环境变量，开发模式下不托管静态文件
# 开发模式应使用 npm run dev (localhost:5173)，生产模式使用后端托管
DEV_MODE = os.environ.get('DEV_MODE', '').lower() in ('true', '1', 'yes')
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend", "dist")

if DEV_MODE:
    logger.info("开发模式: 静态文件托管已禁用，请使用 http://localhost:5173 访问前端")
elif os.path.exists(FRONTEND_DIST):
    # 托管静态资源 (js, css, images等)
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="static-assets")

    # 根路径返回 index.html
    @app.get("/", response_class=HTMLResponse)
    async def serve_index():
        index_path = os.path.join(FRONTEND_DIST, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="Frontend not found", status_code=404)

    # SPA 路由支持: 所有非 API/media 路由都返回 index.html
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # 排除 API 和 media 路由
        if full_path.startswith(("api/", "media/", "docs", "openapi.json", "redoc")):
            raise HTTPException(status_code=404, detail="Not Found")

        # 检查是否是静态资源文件
        static_file = os.path.join(FRONTEND_DIST, full_path)
        if os.path.exists(static_file) and os.path.isfile(static_file):
            return FileResponse(static_file)

        # SPA 路由回退到 index.html
        index_path = os.path.join(FRONTEND_DIST, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="Frontend not found", status_code=404)

    logger.info(f"前端静态文件托管已启用: {FRONTEND_DIST}")
else:
    logger.info("未检测到前端 dist 目录，静态文件托管未启用 (开发模式请使用 npm run dev)")


if __name__ == "__main__":
    import uvicorn
    # 直接传入 app，关闭 reload，确保使用当前文件内定义的应用实例
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False,
                limit_max_requests=1000, limit_concurrency=50)
