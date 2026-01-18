"""
统一配置管理
严格遵守独立打包原则：
1. 杜绝硬编码绝对路径
2. 杜绝依赖系统环境变量
3. 强制接管模型下载路径
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_env_file(env_path: Path) -> None:
    """加载.env文件到环境变量 (简化版,无需第三方库)"""
    if not env_path.exists():
        return

    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue
                # 解析KEY=VALUE格式
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # 移除引号
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    # 只在环境变量不存在时设置
                    if key and key not in os.environ:
                        os.environ[key] = value
        logger.info(f"[Config] Loaded environment variables from {env_path}")
    except Exception as e:
        logger.warning(f"[Config] Failed to load .env file: {e}")


class ProjectConfig:
    """项目配置类"""

    def __init__(self):
        # ========== 路径配置（基于项目根目录） ==========
        # 获取项目根目录（从当前文件位置向上三级）
        # backend/app/core/config.py -> backend/app/core -> backend/app -> backend -> project_root
        self.BASE_DIR = Path(__file__).parent.parent.parent.parent.resolve()

        # V3.1.0+dev.20260104.01: 添加 PROJECT_ROOT 别名（供更新系统使用）
        self.PROJECT_ROOT = self.BASE_DIR

        # 加载.env文件 (在设置其他配置之前)
        env_file = self.BASE_DIR / ".env"
        load_env_file(env_file)

        # 输入输出目录
        self.INPUT_DIR = self.BASE_DIR / "input"
        self.OUTPUT_DIR = self.BASE_DIR / "output"
        self.JOBS_DIR = self.BASE_DIR / "jobs"
        self.TEMP_DIR = self.BASE_DIR / "temp"

        # FFmpeg路径（优先使用项目内的tools目录，支持独立打包）
        self.FFMPEG_DIR = self.BASE_DIR / "tools"
        self.FFMPEG_EXE = self.FFMPEG_DIR / "ffmpeg.exe"

        # 模型缓存目录（强制接管，不使用默认的用户目录）
        self.MODELS_DIR = self.BASE_DIR / "models"
        self.HF_CACHE_DIR = self.MODELS_DIR / "huggingface"
        self.TORCH_CACHE_DIR = self.MODELS_DIR / "torch"

        # 设置环境变量，强制模型下载到项目目录
        os.environ['HF_HOME'] = str(self.HF_CACHE_DIR)
        os.environ['TORCH_HOME'] = str(self.TORCH_CACHE_DIR)
        os.environ['TRANSFORMERS_CACHE'] = str(self.HF_CACHE_DIR / "transformers")
        os.environ['HF_HUB_CACHE'] = str(self.HF_CACHE_DIR / "hub")
        
        # HuggingFace 镜像源配置（解决国内访问问题）
        # 默认启用镜像源，可通过环境变量 USE_HF_MIRROR=false 禁用
        use_mirror = os.getenv('USE_HF_MIRROR', 'true').lower() == 'true'

        if use_mirror:
            # 使用国内镜像源（HF-Mirror 公益镜像）
            self.HF_ENDPOINT = 'https://hf-mirror.com'
            os.environ['HF_ENDPOINT'] = self.HF_ENDPOINT
            print(f"HuggingFace 镜像源: {self.HF_ENDPOINT}")
            print("提示：如需使用官方源，请设置环境变量 USE_HF_MIRROR=false")
        else:
            # 使用官方源
            self.HF_ENDPOINT = 'https://huggingface.co'
            if 'HF_ENDPOINT' in os.environ:
                del os.environ['HF_ENDPOINT']
            print(f"使用 HuggingFace 官方源: {self.HF_ENDPOINT}")
            print("提示：如遇访问问题，可设置环境变量 USE_HF_MIRROR=true 使用镜像源")

        # 确保目录存在（tools目录不自动创建，需用户手动准备）
        for dir_path in [
            self.INPUT_DIR,
            self.OUTPUT_DIR,
            self.JOBS_DIR,
            self.TEMP_DIR,
            self.MODELS_DIR,
            self.HF_CACHE_DIR,
            self.TORCH_CACHE_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # ========== 音频处理配置 ==========
        self.SEGMENT_LENGTH_MS = 60_000      # 60秒
        self.SILENCE_SEARCH_MS = 2_000       # 2秒
        self.MIN_SILENCE_LEN_MS = 300        # 300毫秒
        self.SILENCE_THRESHOLD_DBFS = -40    # -40dB

        # ========== Demucs配置 ==========
        self.DEMUCS_ENABLED = True           # 全局开关
        self.DEMUCS_MODEL = "htdemucs"       # 模型名称
        self.DEMUCS_DEVICE = "cuda"          # 默认设备

        # ========== 进度权重配置 ==========
        # 核心流程权重（转录+对齐 = 100%）
        self.PHASE_WEIGHTS = {
            "pending": 0,        # 等待开始
            "extract": 5,        # 音频提取占5%
            "bgm_detect": 2,     # BGM检测占2%
            "demucs_global": 8,  # 全局人声分离占8%
            "split": 5,          # 音频分段占5%
            "transcribe": 50,    # 转录处理占50%（从60%降低，为Demucs让出空间）
            "align": 20,         # 对齐处理占20%（增加，因为是批次对齐）
            "translate": 0,      # 翻译（预留，当前不启用）
            "proofread": 0,      # 校对（预留，当前不启用）
            "srt": 10,           # SRT生成占10%
            "complete": 0        # 完成
        }
        self.TOTAL_WEIGHT = sum(self.PHASE_WEIGHTS.values())  # 计算总和，保证为100

        # ========== 模型配置 ==========
        self.DEFAULT_MODEL = "medium"
        self.DEFAULT_DEVICE = "cuda"  # 自动检测会覆盖
        self.DEFAULT_COMPUTE_TYPE = "float16"
        self.DEFAULT_BATCH_SIZE = 16
        self.MAX_CACHE_SIZE = 3              # 最多缓存3个模型
        self.MEMORY_THRESHOLD = 0.8          # 内存使用阈值

        # ========== 服务器配置 ==========
        self.API_HOST = "127.0.0.1"
        self.API_PORT = 8000
        self.API_RELOAD = False

        # ========== CPU亲和性配置 ==========
        self.CPU_AFFINITY_ENABLED = True
        self.CPU_AFFINITY_STRATEGY = "auto"  # auto/half/custom

        # ========== 日志配置 ==========
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_DIR = self.BASE_DIR / "logs"
        # 每次启动创建新的日志文件（带时间戳）
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.LOG_FILE = self.LOG_DIR / f"app_{timestamp}.log"
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

        # ========== SSE配置 ==========
        self.SSE_HEARTBEAT_INTERVAL = 10  # 心跳间隔（秒）
        self.SSE_MAX_QUEUE_SIZE = 1000     # 每个连接的消息队列大小
        self.SSE_MAX_CONNECTIONS_PER_CHANNEL = 10  # 每个频道最大连接数

        # ========== 渐进式加载配置 ==========
        self.ENABLE_PROGRESSIVE_LOADING = True   # 是否启用渐进式加载
        self.PREVIEW_RESOLUTION = 360            # 预览视频分辨率（高度）
        self.PROXY_RESOLUTION = 720              # 高质量代理视频分辨率
        self.PREVIEW_QUALITY = 28                # 预览视频 CRF 质量（1-51，越低越好）
        self.PROXY_QUALITY = 23                  # 高质量代理视频 CRF 质量
        self.PREVIEW_PRESET = "ultrafast"        # 预览视频编码预设
        self.PROXY_PRESET = "fast"               # 高质量视频编码预设

        # ========== Proxy 视频配置（重构新增）==========
        # 统一管理所有 Proxy 转码相关参数
        # 注意: ffmpeg_cpu_threads 使用延迟计算，避免循环导入
        self._ffmpeg_cpu_threads_cache = None  # 延迟计算缓存
        self.PROXY_CONFIG = {
            # V3.1.2+dev.20260113.02: 720p自动触发配置（默认启用）
            "auto_trigger_720p": True,  # 默认自动触发720p转码
            # FFmpeg CPU 线程配置（避免使用全部核心导致降频）
            # 注意: 实际值通过 get_ffmpeg_cpu_threads() 方法获取（延迟计算）
            "ffmpeg_cpu_threads": None,  # 占位符，实际使用 get_ffmpeg_cpu_threads()
            # 360p 预览参数（极速模式）
            "preview_360p": {
                "scale": 360,
                "preset": "ultrafast",
                "crf": 28,
                "gop": 30,              # 关键帧间隔
                "keyint_min": 15,       # 最小关键帧间隔
                "audio": True,          # 包含音频（字幕编辑必需）
                "audio_bitrate": "64k", # 低码率音频（加速生成）
                "tune": "fastdecode",   # 优化解码速度
            },
            # 720p 高清参数（平衡质量和速度）
            "proxy_720p": {
                "scale": 720,
                "preset": "fast",              # CPU 编码预设
                "preset_nvenc": "p4",          # NVENC 编码预设 (p1-p7, p4 为质量和速度平衡)
                "crf": 23,
                "gop": 30,
                "keyint_min": 15,
                "audio_bitrate": "128k",
                "audio_sample_rate": 44100,
            },
            # 容器重封装配置（零转码）
            "remux": {
                "enabled": True,                          # 是否启用重封装优化
                "compatible_codecs": {"h264", "aac", "mp3"},  # 可直接复制的编解码器
                "target_container": "mp4",                # 目标容器格式
            },
            # SSE 推送配置
            "sse": {
                "progress_interval": 0.5,   # 进度推送间隔（秒）
                "retry_count": 3,           # 推送失败重试次数
                "retry_delay": 0.1,         # 重试延迟（秒）
            }
        }

        # ========== 浏览器兼容性配置 ==========
        # 用于智能转码决策
        self.BROWSER_COMPATIBILITY = {
            # 浏览器原生支持的容器格式
            "compatible_containers": {".mp4", ".webm"},
            # 浏览器原生支持的视频编解码器
            "compatible_video_codecs": {"h264", "vp8"},
            # 浏览器原生支持的音频编解码器
            "compatible_audio_codecs": {"aac", "mp3", "opus", "vorbis"},
            # 需要强制转码的视频编解码器
            "need_transcode_codecs": {"hevc", "h265", "vp9", "av1", "mpeg2video"},
            # 需要转码的容器格式
            "need_transcode_formats": {".mkv", ".avi", ".mov", ".wmv", ".flv", ".m4v"},
        }

        # ========== 流水线配置（V3.1.0 新增）==========
        # 双流对齐流水线模式
        # - True: 三级异步流水线（错位并行，性能提升 30-50%）
        # - False: 串行流水线（稳定版，V3.0 兼容）
        self.USE_ASYNC_PIPELINE = os.getenv('USE_ASYNC_PIPELINE', 'true').lower() == 'true'

        # 队列背压大小（控制并发度，防止内存溢出）
        # 支持三种配置方式：
        # 1. 'auto': 根据系统内存自动计算（推荐）
        # 2. 数字: 手动指定队列大小（如 '15'）
        # 3. 未设置: 默认 'auto'
        queue_size_config = os.getenv('PIPELINE_QUEUE_MAXSIZE', 'auto')

        if queue_size_config.lower() == 'auto':
            self.PIPELINE_QUEUE_MAXSIZE = self._calculate_adaptive_queue_size()
        else:
            self.PIPELINE_QUEUE_MAXSIZE = int(queue_size_config)

    def _calculate_ffmpeg_threads(self) -> int:
        """
        计算 FFmpeg 转码使用的 CPU 线程数

        使用硬件能力提供者的智能计算，避免全核心占用导致 CPU 降频

        策略：
        - Intel 混合架构：仅使用 P-Core 的 60%
        - Intel 传统架构/AMD：使用物理核心的 60%
        - 未知架构：使用物理核心的 50%

        Returns:
            int: 推荐的 FFmpeg 线程数
        """
        try:
            from app.services.hardware_profile_service import get_hardware_profile_provider

            provider = get_hardware_profile_provider()
            optimal_threads = provider.get_onnx_thread_budget(usage_ratio=0.6)
            info = provider.get_last_onnx_info()

            logger.info(
                f"FFmpeg CPU 线程配置: {optimal_threads} 线程 "
                f"({info.get('strategy', 'unknown')})"
            )
            return optimal_threads

        except Exception as e:
            # 回退：使用 psutil 获取物理核心数的 60%
            try:
                import psutil
                physical_cores = psutil.cpu_count(logical=False) or 4
                threads = max(1, int(physical_cores * 0.6))
                logger.warning(
                    f"硬件能力提供者不可用，回退计算: {threads} 线程 "
                    f"(物理核心 {physical_cores} × 60%): {e}"
                )
                return threads
            except:
                # 最终回退
                logger.warning(f"无法检测 CPU 核心数，使用默认值 4 线程: {e}")
                return 4

    def _calculate_adaptive_queue_size(self) -> int:
        """
        根据系统可用内存自适应计算队列大小

        策略：
        - 每个 chunk 平均占用 ~300MB（音频数据 + 对齐中间结果）
        - 预留 30% 内存给系统和模型
        - 队列大小 = (可用内存 × 0.7) / 300MB
        - 限制范围：3 ~ 30

        Returns:
            int: 自适应队列大小
        """
        try:
            import psutil

            # 获取系统总内存（GB）
            total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)

            # 根据内存档位计算队列大小
            if total_memory_gb < 8:
                # <8GB: 保守策略（老旧设备）
                queue_size = 3
                reason = "低内存(<8GB)"
            elif total_memory_gb < 16:
                # 8-16GB: 适中策略
                queue_size = 8
                reason = "中等内存(8-16GB)"
            elif total_memory_gb < 32:
                # 16-32GB: 平衡策略
                queue_size = 15
                reason = "较大内存(16-32GB)"
            else:
                # 32GB+: 激进策略（工作站）
                queue_size = 25
                reason = "大内存(32GB+)"

            logger.info(
                f"自适应队列大小: {queue_size} "
                f"(系统内存: {total_memory_gb:.1f}GB, {reason})"
            )
            return queue_size

        except Exception as e:
            # psutil 不可用或出错，回退到保守默认值
            logger.warning(f"无法检测系统内存，使用默认队列大小 10: {e}")
            return 10

    def get_ffmpeg_cpu_threads(self) -> int:
        """
        获取 FFmpeg CPU 线程数（延迟计算，避免循环导入）

        Returns:
            int: FFmpeg 线程数
        """
        if self._ffmpeg_cpu_threads_cache is None:
            self._ffmpeg_cpu_threads_cache = self._calculate_ffmpeg_threads()
        return self._ffmpeg_cpu_threads_cache

    def get_ffmpeg_command(self) -> str:
        """
        获取FFmpeg命令
        优先使用项目内的FFmpeg，支持独立打包

        Returns:
            str: FFmpeg可执行文件路径
        """
        if self.FFMPEG_EXE.exists():
            # 使用项目内的FFmpeg
            return str(self.FFMPEG_EXE)
        else:
            # 回退到系统命令
            return "ffmpeg"

    def get_ffprobe_command(self) -> str:
        """
        获取FFprobe命令
        优先使用项目内的FFprobe，支持独立打包

        Returns:
            str: FFprobe可执行文件路径
        """
        ffprobe_exe = self.FFMPEG_DIR / "ffprobe.exe"
        if ffprobe_exe.exists():
            # 使用项目内的FFprobe
            return str(ffprobe_exe)
        else:
            # 回退到系统命令
            return "ffprobe"

    def get_audio_config(self) -> dict:
        """获取音频处理配置"""
        return {
            "segment_length_ms": self.SEGMENT_LENGTH_MS,
            "silence_search_ms": self.SILENCE_SEARCH_MS,
            "min_silence_len_ms": self.MIN_SILENCE_LEN_MS,
            "silence_threshold_dbfs": self.SILENCE_THRESHOLD_DBFS
        }

    def get_phase_weights(self) -> dict:
        """获取进度权重配置"""
        return {
            "weights": self.PHASE_WEIGHTS,
            "total": self.TOTAL_WEIGHT
        }

    def get_model_config(self) -> dict:
        """获取模型配置"""
        return {
            "default_model": self.DEFAULT_MODEL,
            "default_device": self.DEFAULT_DEVICE,
            "default_compute_type": self.DEFAULT_COMPUTE_TYPE,
            "default_batch_size": self.DEFAULT_BATCH_SIZE,
            "max_cache_size": self.MAX_CACHE_SIZE,
            "memory_threshold": self.MEMORY_THRESHOLD
        }

    def calculate_dynamic_weights(
        self,
        engine: str,
        total_segments: int,
        segments_to_separate: int,
        segments_to_retry: int
    ) -> dict:
        """
        根据引擎和实际场景动态计算权重

        Args:
            engine: 'faster_whisper' | 'sensevoice'
            total_segments: 总片段数
            segments_to_separate: 需要分离的片段数
            segments_to_retry: 需要补刀的片段数

        Returns:
            动态权重字典
        """
        if engine == 'faster_whisper':
            return self.PHASE_WEIGHTS.copy()

        base_weights = self.PHASE_WEIGHTS.copy()

        if total_segments > 0:
            sep_ratio = segments_to_separate / total_segments
            retry_ratio = segments_to_retry / total_segments

            base_weights['demucs_global'] = int(15 * sep_ratio)
            base_weights['transcribe'] = 70
            base_weights['align'] = 0

            if 'retry' not in base_weights:
                base_weights['retry'] = 0
            base_weights['retry'] = int(15 * retry_ratio)

            used = sum(base_weights.values())
            if used < 100:
                base_weights['transcribe'] += (100 - used)

        return base_weights


# 全局配置实例
config = ProjectConfig()

# 打印配置信息（启动时显示）
# 注意：避免在模块导入时使用emoji，以防编码问题
try:
    import sys
    # 在Windows上设置UTF-8输出
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass

    print(f"""
项目配置已加载
项目根目录: {config.BASE_DIR}
输入目录: {config.INPUT_DIR}
输出目录: {config.OUTPUT_DIR}
FFmpeg: {config.get_ffmpeg_command()}
模型缓存: {config.MODELS_DIR}
日志目录: {config.LOG_DIR}
""")
except Exception:
    # 如果打印失败，静默忽略
    pass
