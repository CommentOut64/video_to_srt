"""
Faster-Whisper 转录服务

职责：
- Whisper 补刀（后处理增强阶段）
- 仅提供文本，时间戳由 SenseVoice 确定，使用伪对齐
- 自动检测并下载缺失的 Whisper 模型（默认 medium）
- 自动使用 HuggingFace 镜像源（hf-mirror.com）
"""
# 延迟导入 faster_whisper，避免启动时加载 ctranslate2 导致首次启动卡死
# from faster_whisper import WhisperModel  # 已移至 load_model() 内部延迟导入
from typing import Optional, Dict, Any, Union, Tuple, TYPE_CHECKING
import numpy as np
import logging
import gc
import os
import re
from pathlib import Path

# 用于类型检查的条件导入（不会在运行时触发导入）
if TYPE_CHECKING:
    from faster_whisper import WhisperModel

from app.core import config

logger = logging.getLogger(__name__)

# ========== 模型配置常量 ==========
# 默认模型名称 (可通过.env文件中的 WHISPER_MODEL 环境变量覆盖)
DEFAULT_WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")

# 默认计算类型 (可通过.env文件中的 WHISPER_COMPUTE_TYPE 环境变量覆盖)
# 支持: float16, int8, int8_float16, auto
# auto: 根据显存自动选择 (>=8GB用int8_float16, <8GB用int8)
DEFAULT_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "auto")

# 支持的模型列表及其 HuggingFace 仓库 ID
WHISPER_MODELS = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "turbo": "Systran/faster-whisper-large-v3-turbo",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}

# CTranslate2 格式模型必需的文件
REQUIRED_MODEL_FILES = ["model.bin", "config.json", "vocabulary.txt"]

# HuggingFace 镜像源
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"


def get_auto_compute_type(device: str = "cuda") -> str:
    """
    根据显存大小自动选择计算类型

    Args:
        device: 设备类型 (cuda/cpu)

    Returns:
        str: 自动选择的计算类型

    规则:
        - CPU模式: 返回 int8 (最快)
        - GPU >= 8GB: 返回 int8_float16 (精度高，显存优化)
        - GPU < 8GB: 返回 int8 (最省显存)
        - 检测失败: 返回 int8_float16 (保守选择)
    """
    # CPU模式直接用int8
    if device == "cpu":
        logger.info("CPU模式，自动选择 compute_type=int8")
        return "int8"

    # GPU模式，检测显存
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，回退到 int8")
            return "int8"

        # 获取第一个GPU的显存 (MB)
        device_props = torch.cuda.get_device_properties(0)
        gpu_memory_mb = device_props.total_memory // (1024 * 1024)
        gpu_memory_gb = gpu_memory_mb / 1024
        gpu_name = device_props.name

        # 根据显存选择计算类型
        if gpu_memory_mb >= 8 * 1024:  # >= 8GB
            compute_type = "int8_float16"
            logger.info(f"检测到GPU: {gpu_name} ({gpu_memory_gb:.1f}GB), 自动选择 compute_type={compute_type}")
        else:  # < 8GB
            compute_type = "int8"
            logger.info(f"检测到GPU: {gpu_name} ({gpu_memory_gb:.1f}GB), 显存较小，自动选择 compute_type={compute_type}")

        return compute_type

    except Exception as e:
        logger.warning(f"显存检测失败: {e}, 使用保守策略 int8_float16")
        return "int8_float16"


class WhisperService:
    """Faster-Whisper 转录服务（自动下载模型，自动使用镜像源）"""

    def __init__(self):
        self.model: Optional[WhisperModel] = None
        self._model_name: str = ""
        self._device: str = "cuda"
        self._compute_type: str = DEFAULT_COMPUTE_TYPE  # 使用环境变量配置的默认值

        # 强制设置镜像源环境变量
        self._setup_hf_mirror()

    def _setup_hf_mirror(self):
        """配置 HuggingFace 镜像源（解决国内访问问题）"""
        # 检查是否禁用镜像
        use_mirror = os.getenv('USE_HF_MIRROR', 'true').lower() == 'true'
        
        if use_mirror:
            os.environ['HF_ENDPOINT'] = HF_MIRROR_ENDPOINT
            logger.info(f"WhisperService: 使用 HuggingFace 镜像源: {HF_MIRROR_ENDPOINT}")
        else:
            logger.info("WhisperService: 使用 HuggingFace 官方源")

    @staticmethod
    def get_model_repo_id(model_name: str) -> str:
        """
        获取模型对应的 HuggingFace 仓库 ID
        
        Args:
            model_name: 模型名称 (tiny, base, small, medium, large-v2, large-v3 等)
            
        Returns:
            str: 完整的仓库 ID (如 Systran/faster-whisper-medium)
        """
        # 如果已经是完整的仓库 ID，直接返回
        if "/" in model_name:
            return model_name
        
        # 查找预定义的模型
        if model_name in WHISPER_MODELS:
            return WHISPER_MODELS[model_name]
        
        # 尝试构造默认格式
        return f"Systran/faster-whisper-{model_name}"

    @property
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

    @property
    def model_name(self) -> str:
        """获取当前加载的模型名称"""
        return self._model_name

    @property
    def device(self) -> str:
        """获取当前设备"""
        return self._device

    @property
    def compute_type(self) -> str:
        """获取当前计算类型"""
        return self._compute_type

    def load_model(
        self,
        model_name: str = DEFAULT_WHISPER_MODEL,
        device: str = "cuda",
        compute_type: str = None,  # None表示使用DEFAULT_COMPUTE_TYPE
        download_root: str = None,
        local_files_only: bool = False,
        auto_download: bool = True
    ) -> "WhisperService":
        """
        加载 Faster-Whisper 模型（自动下载缺失的模型）

        Args:
            model_name: 模型名称 (tiny, base, small, medium, large-v2, large-v3)
                       默认值: medium（平衡速度与精度）
            device: 设备 (cuda, cpu)
            compute_type: 计算类型 (float16, int8, int8_float16, auto)
                         None 表示使用 DEFAULT_COMPUTE_TYPE（从环境变量读取）
                         auto 表示根据显存自动选择
            download_root: 模型下载目录（默认使用 config.HF_CACHE_DIR）
            local_files_only: 是否仅使用本地文件
            auto_download: 是否自动下载缺失的模型（默认 True）

        Returns:
            self: 支持链式调用
        """
        # 如果未指定compute_type，使用默认值（从环境变量读取）
        if compute_type is None:
            compute_type = DEFAULT_COMPUTE_TYPE

        # 处理 auto 模式：根据显存自动选择
        if compute_type == "auto":
            compute_type = get_auto_compute_type(device)
            logger.info(f"auto模式已解析为: {compute_type}")

        # 如果已加载相同模型和计算类型，跳过
        if self.model and self._model_name == model_name and self._compute_type == compute_type:
            logger.debug(f"模型 {model_name} (compute_type={compute_type}) 已加载，跳过")
            return self

        # 卸载旧模型
        self.unload_model()

        # 设置下载目录
        if download_root is None:
            download_root = str(config.HF_CACHE_DIR)
        
        # 确保下载目录存在
        Path(download_root).mkdir(parents=True, exist_ok=True)

        # 确保镜像源配置
        self._setup_hf_mirror()

        # 获取完整的仓库 ID
        model_repo_id = self.get_model_repo_id(model_name)

        logger.info(f"=" * 50)
        logger.info(f"加载 Faster-Whisper 模型")
        logger.info(f"  模型名称: {model_name}")
        logger.info(f"  仓库 ID: {model_repo_id}")
        logger.info(f"  设备: {device}, 计算类型: {compute_type}")
        logger.info(f"  缓存目录: {download_root}")
        logger.info(f"  镜像源: {os.environ.get('HF_ENDPOINT', '官方源')}")
        logger.info(f"=" * 50)

        try:
            # 检查模型是否存在本地，不存在则下载
            if auto_download:
                model_path = self._ensure_model_downloaded(
                    model_repo_id, 
                    download_root,
                    force_download=False
                )
                if model_path:
                    # 使用本地路径加载
                    logger.info(f"使用本地模型路径: {model_path}")
                    model_repo_id = model_path

            # 加载模型
            # 延迟导入 faster_whisper，避免启动时加载 ctranslate2 导致首次启动卡死
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                model_repo_id,
                device=device,
                compute_type=compute_type,
                download_root=download_root,
                local_files_only=local_files_only
            )

            self._model_name = model_name
            self._device = device
            self._compute_type = compute_type

            logger.info(f"✓ Faster-Whisper 模型加载完成: {model_name}")

        except Exception as e:
            logger.error(f"✗ 加载 Faster-Whisper 模型失败: {e}")
            raise

        return self

    def _ensure_model_downloaded(
        self, 
        model_repo_id: str, 
        cache_dir: str,
        force_download: bool = False
    ) -> Optional[str]:
        """
        确保模型已下载到本地（使用镜像源）

        Args:
            model_repo_id: 模型仓库 ID (如 Systran/faster-whisper-medium)
            cache_dir: 缓存目录
            force_download: 是否强制重新下载

        Returns:
            str: 模型本地路径，如果模型已存在或下载成功；None 表示需要让 Faster-Whisper 自行处理
        """
        from huggingface_hub import snapshot_download, HfApi
        
        logger.info(f"检查模型是否存在本地: {model_repo_id}")

        # 计算模型缓存路径
        model_cache_name = f"models--{model_repo_id.replace('/', '--')}"
        model_dir = Path(cache_dir) / model_cache_name

        # 检查模型是否已存在且完整
        if not force_download:
            local_path = self._check_local_model(model_dir)
            if local_path:
                logger.info(f"✓ 模型已存在本地: {local_path}")
                return local_path

        # 模型不存在或不完整，开始下载
        logger.info(f"=" * 40)
        logger.info(f"模型不存在本地，开始下载...")
        logger.info(f"  仓库: {model_repo_id}")
        logger.info(f"  镜像源: {os.environ.get('HF_ENDPOINT', '官方源')}")
        logger.info(f"  目标目录: {cache_dir}")
        logger.info(f"  这可能需要几分钟，请耐心等待...")
        logger.info(f"=" * 40)

        try:
            # 使用 snapshot_download 下载完整模型
            model_path = snapshot_download(
                repo_id=model_repo_id,
                cache_dir=cache_dir,
                resume_download=True,  # 支持断点续传
                local_files_only=False,
                # 不指定 revision，使用默认的 main 分支
            )

            logger.info(f"✓ 模型下载完成: {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"✗ 模型下载失败: {e}")
            logger.warning(f"将尝试让 Faster-Whisper 自行处理下载...")
            
            # 提供更详细的错误提示
            if "Connection" in str(e) or "timeout" in str(e).lower():
                logger.error("网络连接问题，请检查网络或尝试使用 VPN")
                logger.info("提示: 可以手动下载模型到以下目录:")
                logger.info(f"  {cache_dir}")
            
            return None

    def _check_local_model(self, model_dir: Path) -> Optional[str]:
        """
        检查本地模型是否存在且完整

        Args:
            model_dir: 模型缓存目录 (如 .../models--Systran--faster-whisper-medium)

        Returns:
            str: 模型 snapshot 路径，如果模型完整；否则返回 None
        """
        if not model_dir.exists():
            logger.debug(f"模型目录不存在: {model_dir}")
            return None

        # HuggingFace Hub 的缓存结构: models--xxx/snapshots/hash/
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            logger.debug(f"snapshots 目录不存在: {snapshots_dir}")
            return None

        # 找到最新的 snapshot
        snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if not snapshots:
            logger.debug("没有找到任何 snapshot")
            return None

        # 按修改时间排序，取最新的
        latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)

        # 检查必需文件是否存在
        missing_files = []
        for required_file in REQUIRED_MODEL_FILES:
            if not (latest_snapshot / required_file).exists():
                missing_files.append(required_file)

        if missing_files:
            logger.warning(f"模型文件不完整，缺少: {missing_files}")
            return None

        return str(latest_snapshot)

    def check_model_exists(self, model_name: str = DEFAULT_WHISPER_MODEL) -> Dict[str, Any]:
        """
        检查指定模型是否存在本地（供 API 调用）

        Args:
            model_name: 模型名称

        Returns:
            dict: {
                "exists": bool,
                "model_name": str,
                "repo_id": str,
                "local_path": str or None,
                "cache_dir": str
            }
        """
        cache_dir = str(config.HF_CACHE_DIR)
        repo_id = self.get_model_repo_id(model_name)
        model_cache_name = f"models--{repo_id.replace('/', '--')}"
        model_dir = Path(cache_dir) / model_cache_name
        
        local_path = self._check_local_model(model_dir)
        
        return {
            "exists": local_path is not None,
            "model_name": model_name,
            "repo_id": repo_id,
            "local_path": local_path,
            "cache_dir": cache_dir
        }

    def list_available_models(self) -> Dict[str, str]:
        """
        列出所有支持的模型

        Returns:
            dict: {model_name: repo_id, ...}
        """
        return WHISPER_MODELS.copy()

    def list_local_models(self) -> list:
        """
        列出本地已下载的模型

        Returns:
            list: 已下载模型的列表
        """
        cache_dir = Path(config.HF_CACHE_DIR)
        local_models = []

        for model_name, repo_id in WHISPER_MODELS.items():
            model_cache_name = f"models--{repo_id.replace('/', '--')}"
            model_dir = cache_dir / model_cache_name
            local_path = self._check_local_model(model_dir)
            
            if local_path:
                local_models.append({
                    "model_name": model_name,
                    "repo_id": repo_id,
                    "local_path": local_path
                })

        return local_models

    def unload_model(self):
        """卸载模型释放显存"""
        if self.model:
            logger.info(f"卸载 Faster-Whisper 模型: {self._model_name}")
            del self.model
            self.model = None
            self._model_name = ""

            # 清理内存
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        language: str = None,
        initial_prompt: str = None,
        word_timestamps: bool = False,
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_parameters: dict = None,
        temperature: float = 0.0,
        condition_on_previous_text: bool = True,
        suppress_tokens: list = None,  # 幻觉抑制 Token ID 列表
        repetition_penalty: float = 1.0,  # 重复惩罚系数（>1 抑制重复）
        no_repeat_ngram_size: int = 0  # 禁止重复的 N-gram 大小（0=禁用）
    ) -> Dict[str, Any]:
        """
        转录音频

        Args:
            audio: 音频文件路径或 numpy 数组 (16kHz, mono)
            language: 语言代码 (zh, en, ja 等)，None 表示自动检测
            initial_prompt: 上下文提示（提高准确性）
            word_timestamps: 是否生成词级时间戳
            beam_size: beam search 大小
            vad_filter: 是否启用内置 VAD 过滤
            vad_parameters: VAD 参数
            temperature: 采样温度
            condition_on_previous_text: 是否基于前文条件生成
            suppress_tokens: 幻觉抑制 Token ID 列表（None 则自动从配置获取）
            repetition_penalty: 重复惩罚系数，>1.0 抑制重复（推荐 1.1-1.3），默认 1.0 无惩罚
            no_repeat_ngram_size: 禁止重复的 N-gram 大小，>0 时禁止相同 N-gram 连续出现（推荐 3），默认 0 禁用

        Returns:
            dict: {
                "text": str,              # 完整文本
                "segments": [...],        # 分段结果
                "language": str,          # 检测到的语言
                "language_probability": float  # 语言检测置信度
            }
        """
        if not self.model:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        # 处理语言代码：'auto' 或空字符串应转换为 None（自动检测）
        if language is None or language == 'auto' or language == '':
            language = None

        # 获取幻觉抑制 Token ID（如果未指定）
        if suppress_tokens is None:
            from app.config.model_config import get_whisper_suppress_tokens
            suppress_tokens = get_whisper_suppress_tokens(self._model_name)
            if suppress_tokens:
                logger.debug(f"启用幻觉抑制: {len(suppress_tokens)} 个 Token ID")

        # 执行转录
        segments_generator, info = self.model.transcribe(
            audio,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            temperature=temperature,
            condition_on_previous_text=condition_on_previous_text,
            suppress_tokens=suppress_tokens if suppress_tokens else None,  # 幻觉抑制
            repetition_penalty=repetition_penalty,  # 重复惩罚
            no_repeat_ngram_size=no_repeat_ngram_size  # N-gram 重复抑制
        )

        # 转换生成器为列表
        segment_list = list(segments_generator)

        # V3.1.0+dev.20260104.02: 过滤 Whisper 可能输出的 prompt 前缀
        # Whisper 有时会将 initial_prompt 内容当作转录结果输出
        def clean_prompt_leak(text: str) -> str:
            """清理 Whisper 输出中泄漏的 prompt 内容"""
            if not text:
                return text
            # 移除 "Glossary: xxx." 开头的内容
            # 匹配 "Glossary:" 开头，到第一个句号或换行结束
            cleaned = re.sub(r'^Glossary:\s*[^.]*\.\s*', '', text, flags=re.IGNORECASE)
            # 如果整个文本就是 Glossary 格式，返回空
            if cleaned == text and text.lower().startswith('glossary:'):
                return ''
            return cleaned.strip()

        # 清理每个 segment 的文本
        for seg in segment_list:
            seg.text = clean_prompt_leak(seg.text)

        # 构建统一格式的返回结果
        result = {
            "text": " ".join(seg.text.strip() for seg in segment_list),
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "avg_logprob": seg.avg_logprob,
                    "no_speech_prob": seg.no_speech_prob,
                    "words": [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability
                        }
                        for w in (seg.words or [])
                    ] if word_timestamps and seg.words else []
                }
                for seg in segment_list
            ],
            "language": info.language,
            "language_probability": info.language_probability
        }

        return result

    def transcribe_segment(
        self,
        audio: Union[str, np.ndarray],
        start_time: float,
        end_time: float,
        language: str = None,
        initial_prompt: str = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0
    ) -> Dict[str, Any]:
        """
        转录指定时间段的音频（用于补刀场景）

        Args:
            audio: 完整音频数组 (16kHz)
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            language: 语言代码
            initial_prompt: 上下文提示
            repetition_penalty: 重复惩罚系数（推荐 1.1-1.3）
            no_repeat_ngram_size: N-gram 重复抑制大小（推荐 3）

        Returns:
            dict: 转录结果
        """
        if isinstance(audio, np.ndarray):
            # 切片音频
            sr = 16000
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            audio_segment = audio[start_sample:end_sample]
        else:
            # 如果是文件路径，需要先加载再切片
            import librosa
            full_audio, _ = librosa.load(audio, sr=16000, mono=True)
            sr = 16000
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            audio_segment = full_audio[start_sample:end_sample]

        return self.transcribe(
            audio=audio_segment,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=False,  # 补刀场景使用伪对齐，不需要词级时间戳
            beam_size=5,
            vad_filter=False,  # 已经是切片，不需要 VAD
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size
        )

    def warmup(self):
        """预热模型（空跑一次确保完全加载到显存）"""
        if not self.model:
            logger.warning("模型未加载，无法预热")
            return

        logger.debug("开始 Faster-Whisper 模型预热")

        # 创建 1 秒静音音频
        dummy_audio = np.zeros(16000, dtype=np.float32)

        try:
            segments, _ = self.model.transcribe(dummy_audio)
            _ = list(segments)  # 触发生成器执行
            logger.debug("Faster-Whisper 模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")

    def estimate_confidence(self, result: Dict[str, Any]) -> float:
        """
        估算转录结果的置信度

        Args:
            result: transcribe() 返回的结果

        Returns:
            float: 0-1 之间的置信度分数
        """
        segments = result.get("segments", [])
        text = result.get("text", "")

        # V3.8 修复: 空输出返回极低置信度，触发回退机制
        if not segments or not text or not text.strip():
            # 检查是否有 no_speech_prob 指示（可能是真正的静音段）
            # 如果连 segments 都没有，说明 Whisper 完全没有输出，置信度应该极低
            logger.warning(f"Whisper 输出为空: segments={len(segments)}, text_len={len(text)}, 返回低置信度 0.1")
            return 0.1  # 极低置信度，确保触发回退

        # 基于 avg_logprob 和 no_speech_prob 计算
        total_logprob = sum(s.get("avg_logprob", -0.5) for s in segments)
        avg_logprob = total_logprob / len(segments)

        avg_no_speech = sum(s.get("no_speech_prob", 0.1) for s in segments) / len(segments)

        # 转换为 0-1 置信度
        # logprob 范围大约 -1 到 0，越接近 0 越好
        confidence = min(1.0, max(0.0, 1.0 + avg_logprob))
        # no_speech_prob 越低越好
        confidence *= (1.0 - avg_no_speech)

        return round(confidence, 3)


# ========== 音频加载工具函数 ==========

def load_audio(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    加载音频文件为 numpy 数组

    Args:
        audio_path: 音频文件路径
        sr: 采样率（默认 16000）

    Returns:
        np.ndarray: 音频数组
    """
    import librosa
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    return audio.astype(np.float32)


# ========== 单例访问 ==========

_whisper_service_instance: Optional[WhisperService] = None


def get_whisper_service() -> WhisperService:
    """获取 Whisper 服务单例"""
    global _whisper_service_instance
    if _whisper_service_instance is None:
        _whisper_service_instance = WhisperService()
    return _whisper_service_instance


def reset_whisper_service():
    """重置 Whisper 服务（用于测试）"""
    global _whisper_service_instance
    if _whisper_service_instance:
        _whisper_service_instance.unload_model()
    _whisper_service_instance = None
