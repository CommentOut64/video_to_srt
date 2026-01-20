"""
Demucs 人声分离服务
支持多种Demucs模型进行高质量人声提取
"""

import os
import gc
import logging
import tempfile
import hashlib
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from enum import Enum
from dataclasses import dataclass, field

import torch
import numpy as np
import soundfile as sf


class BGMLevel(Enum):
    """背景音乐强度级别"""
    NONE = "none"      # 无背景音乐
    LIGHT = "light"    # 轻微背景音乐（按需分离）
    HEAVY = "heavy"    # 强背景音乐（全局分离）


class DemucsModel(Enum):
    """可用的Demucs模型"""
    HTDEMUCS = "htdemucs"           # Hybrid Transformer (~80MB, 快速)
    HTDEMUCS_FT = "htdemucs_ft"     # Fine-tuned版本 (~80MB, 人声更优)
    MDX_EXTRA = "mdx_extra"         # MDX-Net (~100MB, 高质量)
    MDX_EXTRA_Q = "mdx_extra_q"     # MDX-Net量化版 (~25MB, 较快)


@dataclass
class DemucsConfig:
    """Demucs配置"""
    # 模型选择 - 默认使用htdemucs（快速，只需1个文件）
    model_name: str = "htdemucs"
    device: str = "cuda"                   # 设备 (cuda/cpu)
    shifts: int = 1                        # 增强次数（1=快速，2=平衡，5=最高质量）- 速度优先
    overlap: float = 0.5                   # 分段重叠率
    segment_length: int = 10               # 每段处理长度（秒）

    # 按需分离的缓冲区
    segment_buffer_sec: float = 2.0        # 分离时前后各加的缓冲（秒）

    # BGM检测参数（分位数采样策略）
    bgm_sample_duration: float = 10.0      # 每个采样片段的长度（秒）
    bgm_light_threshold: float = 0.02      # 轻微BGM阈值
    bgm_heavy_threshold: float = 0.15      # 强BGM阈值

    # 可用模型列表（供UI选择）
    available_models: List[str] = field(default_factory=lambda: [
        "htdemucs",       # 统一使用 htdemucs（与注册表一致）
    ])


@dataclass
class ModelTierConfig:
    """
    分级模型配置
    定义不同场景下使用的模型及其质量参数
    """
    # 弱BGM场景使用的模型（速度优先）
    weak_model: str = "htdemucs"

    # 强BGM场景使用的模型（质量优先）
    strong_model: str = "htdemucs"

    # 兜底模型（熔断升级后使用）
    fallback_model: str = "htdemucs"

    # 模型质量参数（按模型分别配置）
    # V3.1.1+dev.20260107.07: 统一智能分诊和极致分离的参数，避免质量差异
    model_quality: Dict[str, Dict] = field(default_factory=lambda: {
        "htdemucs": {"shifts": 1, "overlap": 0.5},      # 与全局分离保持一致
        "htdemucs_ft": {"shifts": 1, "overlap": 0.25},   # 快速+人声优化（已弃用）
        "mdx_extra_q": {"shifts": 2, "overlap": 0.5},    # 中等（已弃用）
        "mdx_extra": {"shifts": 2, "overlap": 0.5},      # 最高质量（已弃用）
    })


@dataclass
class SeparationStrategy:
    """
    分离策略决策结果
    由 SeparationStrategyResolver 生成，描述本次任务应采用的分离策略
    """
    should_separate: bool           # 是否需要分离
    initial_model: Optional[str]    # 初始使用的模型
    fallback_model: Optional[str]   # 升级后的模型（如果允许升级）
    reason: str                     # 决策原因（用于日志和SSE）
    bgm_level: BGMLevel             # 检测到的BGM级别
    allow_escalation: bool          # 是否允许升级

    def to_dict(self) -> dict:
        """转换为字典（用于SSE推送）"""
        return {
            "should_separate": self.should_separate,
            "initial_model": self.initial_model,
            "fallback_model": self.fallback_model,
            "reason": self.reason,
            "bgm_level": self.bgm_level.value,
            "allow_escalation": self.allow_escalation,
        }


# 质量预设映射
QUALITY_PRESETS = {
    "fast": {
        "weak_model": "htdemucs",
        "strong_model": "htdemucs",
        "fallback_model": "htdemucs",
        "description": "速度优先，使用htdemucs统一模型",
    },
    "balanced": {
        "weak_model": "htdemucs",
        "strong_model": "htdemucs",
        "fallback_model": "htdemucs",
        "description": "平衡模式（默认推荐）",
    },
    "quality": {
        "weak_model": "htdemucs",
        "strong_model": "htdemucs",
        "fallback_model": "htdemucs",
        "description": "质量优先，使用htdemucs统一模型",
    },
}


class SeparationStrategyResolver:
    """
    分离策略解析器

    根据 BGM 检测结果和用户配置，决定使用哪个模型以及升级路径

    策略矩阵：
    ┌───────────┬─────────────────┬─────────────────┬──────────────┐
    │ BGM Level │ mode=auto       │ mode=always     │ mode=never   │
    ├───────────┼─────────────────┼─────────────────┼──────────────┤
    │ NONE      │ 不分离          │ weak_model      │ 不分离        │
    │ LIGHT     │ weak_model      │ weak_model      │ 不分离        │
    │ HEAVY     │ strong_model    │ strong_model    │ 不分离        │
    └───────────┴─────────────────┴─────────────────┴──────────────┘
    """

    def __init__(self, settings):
        """
        初始化策略解析器

        Args:
            settings: 预处理配置对象
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        # 如果使用质量预设，覆盖模型配置
        self._apply_preset()

    def _apply_preset(self):
        """应用质量预设（如果用户未自定义模型）"""
        preset = QUALITY_PRESETS.get(self.settings.quality_preset)
        if preset:
            # 检测是否使用默认值（通过判断是否等于 balanced 预设的值）
            # 如果用户自定义了模型，则不应用预设
            default_weak = "htdemucs"
            default_strong = "htdemucs"
            default_fallback = "htdemucs"

            if (self.settings.weak_model == default_weak and
                self.settings.strong_model == default_strong and
                self.settings.fallback_model == default_fallback):
                # 用户未自定义，应用预设
                self.settings.weak_model = preset["weak_model"]
                self.settings.strong_model = preset["strong_model"]
                self.settings.fallback_model = preset["fallback_model"]
                self.logger.debug(
                    f"应用质量预设 '{self.settings.quality_preset}': "
                    f"weak={preset['weak_model']}, strong={preset['strong_model']}, "
                    f"fallback={preset['fallback_model']}"
                )

    def resolve(self, bgm_level: BGMLevel) -> SeparationStrategy:
        """
        根据 BGM 级别解析分离策略

        Args:
            bgm_level: 检测到的 BGM 级别

        Returns:
            SeparationStrategy: 分离策略决策结果
        """
        mode = self.settings.mode

        # mode=never: 始终不分离
        if mode == "never":
            return SeparationStrategy(
                should_separate=False,
                initial_model=None,
                fallback_model=None,
                reason="用户禁用人声分离",
                bgm_level=bgm_level,
                allow_escalation=False,
            )

        # mode=always: 始终分离（根据BGM级别选模型）
        if mode == "always":
            model = self.settings.strong_model if bgm_level == BGMLevel.HEAVY else self.settings.weak_model
            return SeparationStrategy(
                should_separate=True,
                initial_model=model,
                fallback_model=self.settings.fallback_model if self.settings.auto_escalation else None,
                reason=f"始终分离模式，BGM={bgm_level.value}，使用 {model}",
                bgm_level=bgm_level,
                allow_escalation=self.settings.auto_escalation,
            )

        # mode=auto（默认）: 根据BGM级别决定
        if bgm_level == BGMLevel.NONE:
            return SeparationStrategy(
                should_separate=False,
                initial_model=None,
                fallback_model=None,
                reason="未检测到背景音乐，跳过人声分离",
                bgm_level=bgm_level,
                allow_escalation=False,
            )

        elif bgm_level == BGMLevel.LIGHT:
            return SeparationStrategy(
                should_separate=True,
                initial_model=self.settings.weak_model,
                fallback_model=self.settings.fallback_model if self.settings.auto_escalation else None,
                reason=f"检测到轻微BGM，使用轻量模型 {self.settings.weak_model}",
                bgm_level=bgm_level,
                allow_escalation=self.settings.auto_escalation,
            )

        else:  # HEAVY
            return SeparationStrategy(
                should_separate=True,
                initial_model=self.settings.strong_model,
                fallback_model=self.settings.fallback_model if self.settings.auto_escalation else None,
                reason=f"检测到强BGM，使用高质量模型 {self.settings.strong_model}",
                bgm_level=bgm_level,
                allow_escalation=self.settings.auto_escalation,
            )

    def get_escalation_model(self, current_model: str) -> Optional[str]:
        """
        获取升级后的模型

        Args:
            current_model: 当前模型名称

        Returns:
            新模型名称，如果无法升级则返回 None
        """
        if not self.settings.auto_escalation:
            return None

        # 已经是 fallback 模型，无法再升级
        if current_model == self.settings.fallback_model:
            return None

        return self.settings.fallback_model


class DemucsService:
    """
    Demucs人声分离服务

    支持多种模型：
    - mdx_extra: 高质量人声分离（默认推荐）
    - htdemucs: Hybrid Transformer，快速模式
    - htdemucs_ft: Fine-tuned版本
    - mdx_extra_q: 量化版，适合小显存

    支持三种使用模式：
    1. 全局分离：处理整个音频文件，返回纯人声
    2. 按需分离：只处理指定的时间段
    3. BGM检测：快速检测背景音乐强度
    """

    _instance = None
    _model = None
    _model_name_loaded = None  # 记录当前加载的模型名称
    _model_lock = None

    def __new__(cls):
        if cls._instance is None:
            import threading
            cls._instance = super().__new__(cls)
            cls._model_lock = threading.Lock()
        return cls._instance

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = DemucsConfig()
        self._model_overridden = False
        self._cache_dir = Path("models/demucs")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # 分级模型配置
        self.tier_config = ModelTierConfig()
        self._apply_runtime_params(force_defaults=True)

    def _resolve_device(self, device: Optional[str]) -> str:
        """解析 Demucs 设备参数（auto -> cuda/cpu）。"""
        if not device or device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _apply_runtime_params(
        self,
        force_defaults: bool = False,
        apply_model_name: Optional[bool] = None,
    ) -> None:
        """应用运行参数配置。"""
        from app.services.model_runtime_config_service import get_model_runtime_config_service

        runtime_data = get_model_runtime_config_service().get_effective_runtime_global()
        effective = runtime_data.get("effective", {}).get("demucs", {})
        sources = runtime_data.get("sources", {}).get("demucs", {})

        if apply_model_name is None:
            apply_model_name = not self._model_overridden

        def should_apply(field: str) -> bool:
            return force_defaults or sources.get(field) != "default"

        if apply_model_name and should_apply("model_name"):
            value = effective.get("model_name")
            if value:
                self.config.model_name = value

        if should_apply("device"):
            self.config.device = self._resolve_device(effective.get("device"))
        if should_apply("shifts"):
            self.config.shifts = effective.get("shifts", self.config.shifts)
        if should_apply("overlap"):
            self.config.overlap = effective.get("overlap", self.config.overlap)
        if should_apply("segment_length"):
            self.config.segment_length = effective.get("segment_length", self.config.segment_length)
        if should_apply("segment_buffer_sec"):
            self.config.segment_buffer_sec = effective.get("segment_buffer_sec", self.config.segment_buffer_sec)
        if should_apply("bgm_sample_duration"):
            self.config.bgm_sample_duration = effective.get(
                "bgm_sample_duration",
                self.config.bgm_sample_duration,
            )
        if should_apply("bgm_light_threshold"):
            self.config.bgm_light_threshold = effective.get(
                "bgm_light_threshold",
                self.config.bgm_light_threshold,
            )
        if should_apply("bgm_heavy_threshold"):
            self.config.bgm_heavy_threshold = effective.get(
                "bgm_heavy_threshold",
                self.config.bgm_heavy_threshold,
            )

    def refresh_runtime_params(self) -> None:
        """刷新运行参数（仅应用显式覆盖）。"""
        self._apply_runtime_params(force_defaults=False)

    def set_model(self, model_name: str):
        """
        切换Demucs模型

        Args:
            model_name: 模型名称 (mdx_extra, htdemucs, htdemucs_ft, mdx_extra_q)
        """
        if model_name not in self.config.available_models:
            raise ValueError(f"不支持的模型: {model_name}, 可选: {self.config.available_models}")

        if model_name != self.config.model_name:
            self.logger.info(f"切换Demucs模型: {self.config.model_name} → {model_name}")
            self.config.model_name = model_name
            self._model_overridden = True
            # 卸载旧模型，下次使用时会加载新模型
            self.unload_model()
            self._apply_runtime_params(force_defaults=False, apply_model_name=False)

    def preload_model(self, model_name: str = None) -> bool:
        """
        预加载 Demucs 模型到显存

        Args:
            model_name: 模型名称，默认使用当前配置的模型

        Returns:
            bool: 是否加载成功
        """
        if model_name:
            self.set_model(model_name)

        try:
            self.logger.info(f"预加载 Demucs 模型: {self.config.model_name}")
            self._load_model()
            self.logger.info(f"Demucs 模型 {self.config.model_name} 预加载完成")
            return True
        except Exception as e:
            self.logger.error(f"Demucs 模型预加载失败: {e}")
            return False

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._model is not None

    def get_loaded_model_name(self) -> Optional[str]:
        """获取当前加载的模型名称"""
        return self._model_name_loaded

    def _load_model(self, device: str = None):
        """
        懒加载Demucs模型

        模型首次加载时会自动下载
        - mdx_extra: ~100MB
        - htdemucs: ~80MB
        - mdx_extra_q: ~25MB
        
        已切换到 ModelManagerV2 统一管理。
        """
        if device:
            self.config.device = self._resolve_device(device)

        self._apply_runtime_params(force_defaults=False, apply_model_name=not self._model_overridden)

        with self._model_lock:
            # 检查是否需要重新加载（模型切换）
            if self._model is not None and self._model_name_loaded == self.config.model_name:
                return self._model

            # 如果模型已加载但名称不同，先卸载本地引用（缓存由 ModelManagerV2 管理）
            if self._model is not None:
                self.logger.info(f"卸载旧模型引用: {self._model_name_loaded}")
                self._model = None
                self._model_name_loaded = None

            model_id = f"demucs-{self.config.model_name}"
            try:
                from app.services.model_manager_v2 import get_model_manager_v2

                manager = get_model_manager_v2()
                self.logger.info(
                    "Demucs 使用 ModelManagerV2 加载: id=%s device=%s",
                    model_id,
                    self.config.device,
                )
                manager.ensure_available(model_id)
                handle = manager.acquire(model_id, device=self.config.device, compute_type="fp32")
                self._model = handle
                self._model_name_loaded = self.config.model_name
                return self._model
            except Exception as exc:
                raise RuntimeError(f"Demucs 通过 ModelManagerV2 加载失败: {exc}") from exc

    def unload_model(self):
        """卸载模型释放显存"""
        with self._model_lock:
            if self._model is not None:
                old_name = self._model_name_loaded
                del self._model
                self._model = None
                self._model_name_loaded = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.logger.info(f"Demucs模型 {old_name} 已卸载")

    def _load_model_with_mirror(self):
        """
        加载 Demucs 模型
        
        加载顺序：
        1. 检查 torch 缓存目录是否已有模型
        2. 如果没有，检查预置模型目录 (backend/models/pretrained/)
        3. 如果有预置模型，复制到 torch 缓存目录
        4. 调用 demucs 加载模型
        
        模型缓存位置：
        - 项目: models/torch/hub/checkpoints/
        - 预置: backend/models/pretrained/
        
        【优化】下载前会检查并清理残留的 .partial 文件
        """
        from demucs.pretrained import get_model
        
        model_name = self.config.model_name
        
        # 【新增】安装预置模型（如果需要）
        self._install_pretrained_model(model_name)
        
        # 【新增】下载前清理残留文件
        self._cleanup_partial_downloads(model_name)
        
        self.logger.info(f"加载模型: {model_name}")
        self._model = get_model(model_name)
        self.logger.info(f"✓ 模型加载成功: {model_name}")

    def _install_pretrained_model(self, model_name: str):
        """
        安装预置模型到 torch 缓存目录
        
        检查 backend/models/pretrained/ 目录下是否有对应的预置模型，
        如果有且 torch 缓存中没有，则复制过去。
        
        Args:
            model_name: 模型名称
        """
        import os
        import shutil
        from pathlib import Path
        
        # 预置模型文件名映射
        pretrained_file_mapping = {
            'htdemucs': 'htdemucs.th',
        }
        
        # torch 缓存文件名映射
        cache_file_mapping = {
            'htdemucs': '955717e8-8726e21a.th',
            'htdemucs_ft': 'f7e0c4bc-ba3fe64a.th',
            'mdx_extra': '0d19c1c6-0f06f20e.th',
            'mdx_extra_q': '83fc094f-4a16d450.th',
        }
        
        pretrained_file = pretrained_file_mapping.get(model_name)
        cache_file = cache_file_mapping.get(model_name)
        
        if not pretrained_file or not cache_file:
            return
        
        # 获取预置模型目录（backend/models/pretrained/）
        pretrained_dir = Path(__file__).parent.parent.parent / 'models' / 'pretrained'
        pretrained_path = pretrained_dir / pretrained_file
        
        if not pretrained_path.exists():
            self.logger.debug(f"预置模型不存在: {pretrained_path}")
            return
        
        # 获取 torch 缓存目录
        torch_home = os.environ.get('TORCH_HOME', Path.home() / '.cache' / 'torch')
        cache_dir = Path(torch_home) / 'hub' / 'checkpoints'
        cache_path = cache_dir / cache_file
        
        # 如果缓存中已有，跳过
        if cache_path.exists():
            self.logger.debug(f"模型已在缓存中: {cache_path}")
            return
        
        # 确保缓存目录存在
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制预置模型到缓存目录
        self.logger.info(f"安装预置模型: {pretrained_file} -> {cache_file}")
        try:
            shutil.copy2(pretrained_path, cache_path)
            self.logger.info(f"✓ 预置模型安装成功")
        except Exception as e:
            self.logger.warning(f"预置模型安装失败: {e}")
            self.logger.info("将从网络下载模型...")

    def _cleanup_partial_downloads(self, model_name: str):
        """
        清理残留的下载文件（.partial 文件）
        
        当下载中断时，PyTorch 会留下 .partial 文件，
        这些文件可能导致后续下载失败或哈希校验错误。
        
        Args:
            model_name: 模型名称
        """
        import os
        from pathlib import Path
        
        # 模型文件名映射（模型名 -> 缓存文件名前缀）
        model_file_mapping = {
            'htdemucs': '955717e8-8726e21a',
            'htdemucs_ft': 'f7e0c4bc-ba3fe64a',
            'mdx_extra': '0d19c1c6-0f06f20e',
            'mdx_extra_q': '83fc094f-4a16d450',
        }
        
        file_prefix = model_file_mapping.get(model_name)
        if not file_prefix:
            return
        
        # 获取 PyTorch 缓存目录
        torch_home = os.environ.get('TORCH_HOME', Path.home() / '.cache' / 'torch')
        cache_dir = Path(torch_home) / 'hub' / 'checkpoints'
        
        if not cache_dir.exists():
            return
        
        # 查找并清理 partial 文件
        partial_files = list(cache_dir.glob(f"{file_prefix}*.partial"))
        
        if partial_files:
            self.logger.warning(f"发现 {len(partial_files)} 个残留下载文件，正在清理...")
            for pf in partial_files:
                try:
                    pf.unlink()
                    self.logger.info(f"  已删除: {pf.name}")
                except Exception as e:
                    self.logger.warning(f"  删除失败: {pf.name} - {e}")
            self.logger.info(f"✓ 残留文件清理完成")

    def separate_vocals(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> str:
        """
        全局人声分离（处理整个音频文件）

        Args:
            audio_path: 输入音频路径
            output_path: 输出路径（可选，默认在同目录生成 xxx_vocals.wav）
            progress_callback: 进度回调 callback(progress: float, message: str)

        Returns:
            str: 分离后的人声文件路径
        """
        from demucs.apply import apply_model
        from demucs.audio import AudioFile, save_audio

        self.logger.info(f"开始全局人声分离: {audio_path}")
        self._apply_runtime_params(force_defaults=False, apply_model_name=False)

        # 生成输出路径
        if output_path is None:
            audio_dir = Path(audio_path).parent
            audio_stem = Path(audio_path).stem
            output_path = str(audio_dir / f"{audio_stem}_vocals.wav")

        # 检查缓存
        cache_key = self._get_cache_key(audio_path, "full")
        cached_path = self._cache_dir / f"{cache_key}_vocals.wav"
        if cached_path.exists():
            self.logger.info(f"使用缓存的分离结果: {cached_path}")
            return str(cached_path)

        model = self._load_model()

        if progress_callback:
            progress_callback(0.1, "加载音频...")

        # 加载音频
        wav = AudioFile(audio_path).read(
            streams=0,
            samplerate=model.samplerate,
            channels=model.audio_channels
        )

        # 添加batch维度
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        wav = wav.unsqueeze(0)  # (1, channels, samples)

        if self.config.device == "cuda":
            wav = wav.cuda()

        if progress_callback:
            progress_callback(0.2, "分离人声中...")

        # 执行分离
        with torch.no_grad():
            sources = apply_model(
                model,
                wav,
                shifts=self.config.shifts,
                overlap=self.config.overlap,
                progress=True,
                device=self.config.device
            )

        # 提取人声（htdemucs输出顺序：drums, bass, other, vocals）
        source_names = model.sources
        vocals_idx = source_names.index('vocals')
        vocals = sources[0, vocals_idx]  # (channels, samples)

        # 恢复原始scale
        vocals = vocals * ref.std() + ref.mean()

        if progress_callback:
            progress_callback(0.9, "保存文件...")

        # 保存人声
        vocals = vocals.cpu().numpy()

        # V3.1.1+dev.20260107.04: 整轨分离也需要后处理
        # 转为单声道，避免 ChunkEngine 加载时相位抵消
        if vocals.ndim > 1:
            vocals = self._stereo_to_mono_safe(vocals)

        # 响度归一化，确保幅度一致
        vocals = self._normalize_vocals(vocals)

        # 保存为单声道（vocals 现在是 1D 数组）
        sf.write(output_path, vocals, model.samplerate)

        # 保存到缓存（同样是单声道）
        sf.write(str(cached_path), vocals, model.samplerate)

        if progress_callback:
            progress_callback(1.0, "人声分离完成")

        self.logger.info(f"人声分离完成: {output_path}")
        return output_path

    def separate_vocals_segment(
        self,
        audio_array: np.ndarray,
        sr: int,
        start_sec: float,
        end_sec: float,
        buffer_sec: float = None,
        shifts: int = 1  # BGM检测时使用1次，正常分离时使用配置值
    ) -> np.ndarray:
        """
        按需分离指定时间段的人声（内存模式）

        Args:
            audio_array: 完整音频数组 (samples,) 或 (channels, samples)
            sr: 采样率
            start_sec: 开始时间（秒）
            end_sec: 结束时间（秒）
            buffer_sec: 前后缓冲区（秒），默认使用配置值
            shifts: 增强次数（默认1=快速模式用于检测）

        Returns:
            np.ndarray: 分离后的人声片段（不含缓冲区）
        """
        from demucs.apply import apply_model

        self._apply_runtime_params(force_defaults=False, apply_model_name=False)

        if buffer_sec is None:
            buffer_sec = self.config.segment_buffer_sec

        model = self._load_model()

        # 计算采样点范围（含缓冲区）
        buffer_samples = int(buffer_sec * sr)
        start_sample = max(0, int(start_sec * sr) - buffer_samples)
        end_sample = min(len(audio_array), int(end_sec * sr) + buffer_samples)

        # 提取片段
        if audio_array.ndim == 1:
            segment = audio_array[start_sample:end_sample]
            segment = np.stack([segment, segment])  # 转为立体声
        else:
            segment = audio_array[:, start_sample:end_sample]

        # 重采样到模型要求的采样率（如果需要）
        if sr != model.samplerate:
            import librosa
            segment = librosa.resample(segment, orig_sr=sr, target_sr=model.samplerate)
            target_sr = model.samplerate
        else:
            target_sr = sr

        # 转为tensor
        wav = torch.from_numpy(segment).float()
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        wav = wav.unsqueeze(0)

        if self.config.device == "cuda":
            wav = wav.cuda()

        # 执行分离
        with torch.no_grad():
            sources = apply_model(
                model,
                wav,
                shifts=shifts,  # 使用传入的 shifts 参数（检测时为1，正常分离时为配置值）
                overlap=self.config.overlap,
                progress=False,
                device=self.config.device
            )

        # 提取人声
        source_names = model.sources
        vocals_idx = source_names.index('vocals')
        vocals = sources[0, vocals_idx]
        vocals = vocals * ref.std() + ref.mean()
        vocals = vocals.cpu().numpy()

        # 重采样回原始采样率
        if target_sr != sr:
            vocals = librosa.resample(vocals, orig_sr=target_sr, target_sr=sr)

        # 去除缓冲区，返回原始时间段
        original_start = int(buffer_sec * sr) if start_sec > buffer_sec else int(start_sec * sr)
        original_duration = int((end_sec - start_sec) * sr)
        vocals = vocals[:, original_start:original_start + original_duration]

        # 转为单声道（Whisper要求）
        # V3.1.1+dev.20260107.02: 使用 RMS 加权混合，避免相位抵消
        if vocals.ndim > 1:
            vocals = self._stereo_to_mono_safe(vocals)

        # V3.1.1+dev.20260107.01: 响度归一化
        # 解决问题：Demucs 分离后的人声幅度可能很小（尤其是安静段落），
        # 导致 SenseVoice 误判为静音。使用 95% 分位数归一化确保人声幅度一致。
        vocals = self._normalize_vocals(vocals)

        return vocals

    def _stereo_to_mono_safe(self, stereo: np.ndarray) -> np.ndarray:
        """
        安全地将立体声转为单声道，避免相位抵消

        V3.1.1+dev.20260107.02: 解决简单平均导致的相位抵消问题

        问题：Demucs 输出的左右声道可能存在相位差，简单平均会导致人声能量衰减
        方案：提取 Mid 信号（L+R），保留人声主要能量

        Args:
            stereo: 立体声数组 (2, samples) 或 (channels, samples)

        Returns:
            单声道数组 (samples,)
        """
        if stereo.ndim == 1:
            return stereo

        # Mid-Side 处理：提取 Mid 信号（人声主要在中间）
        # Mid = (L + R) / 2，但不除以2以保留能量
        # 这比简单平均更安全，因为相位相同的信号会增强，相位相反的会抵消
        left = stereo[0]
        right = stereo[1] if stereo.shape[0] > 1 else stereo[0]

        # 提取 Mid 信号（不除以2，保留能量）
        mono = (left + right) / 2.0

        self.logger.debug(
            f"立体声转单声道: L_rms={np.sqrt(np.mean(left**2)):.6f}, "
            f"R_rms={np.sqrt(np.mean(right**2)):.6f}, "
            f"Mono_rms={np.sqrt(np.mean(mono**2)):.6f}"
        )

        return mono

    def _normalize_vocals(self, vocals: np.ndarray) -> np.ndarray:
        """
        对分离后的人声进行响度归一化

        V3.1.1+dev.20260107.04: 修复削波失真问题

        问题：原逻辑让95%分位归一化到1.0，导致超过5%的采样点被截断，造成削波失真
        方案：让95%分位归一化到0.5，避免削波，保留动态范围

        Args:
            vocals: 分离后的人声数组

        Returns:
            归一化后的人声数组
        """
        if vocals.size == 0:
            return vocals

        eps = 1e-8
        quantile_95 = np.percentile(np.abs(vocals), 95)

        # V3.1.1+dev.20260107.04: 让95%分位归一化到0.5而不是1.0
        # 避免超过5%的采样点被截断，造成削波失真
        scale = 0.5 / (quantile_95 + eps)

        # 限制放大倍数，防止把极微小的底噪放大成噪音
        scale = min(scale, 30.0)

        vocals = vocals * scale

        # 硬截断（现在应该很少触发）
        vocals = np.clip(vocals, -1.0, 1.0)

        self.logger.debug(
            f"人声归一化: quantile_95={quantile_95:.6f}, scale={scale:.2f}, "
            f"target=0.5 (避免削波)"
        )

        return vocals

    def detect_background_music_level(
        self,
        audio_path: str,
        audio_array: Optional[np.ndarray] = None,
        sr: int = 16000,
        duration_sec: Optional[float] = None
    ) -> Tuple[BGMLevel, List[float]]:
        """
        [已废弃] 快速检测背景音乐强度（分位数采样 + Demucs 残差策略）

        ⚠️ 此方法已废弃，Whisper/SenseVoice 流水线均已切换到频谱分诊：
        
        Whisper 流水线使用：
        >>> from app.services.audio_spectrum_classifier import get_spectrum_classifier
        >>> classifier = get_spectrum_classifier()
        >>> level_str, avg_score = classifier.quick_global_diagnosis(audio, sr)
        
        SenseVoice 流水线使用：
        >>> diagnoses = classifier.diagnose_chunks(chunks, sr)
        
        废弃原因：
        - Demucs 分离即使对纯人声也有 1-3% 残差，导致误判为 light
        - 需要加载 Demucs 模型，速度慢
        - 新方法纯频谱分析，无需 Demucs，更快更准确

        ---
        旧文档（保留供参考）：

        采样策略：取音频时长的 15%、50%、85% 处各截取 10 秒
        - 15%：捕获 Intro 结束后的主歌背景音
        - 50%：捕获中间部分
        - 85%：捕获结尾前的部分

        注意：BGM检测使用最快的 htdemucs 模型，检测完成后会卸载模型，
        以便后续根据检测结果加载合适的模型进行分离。

        Args:
            audio_path: 音频文件路径
            audio_array: 音频数组（可选，用于内存模式）
            sr: 采样率
            duration_sec: 音频总时长（可选，如果audio_array提供则自动计算）

        Returns:
            Tuple[BGMLevel, List[float]]: (背景音乐强度级别, 各采样点的BGM比例列表)
        """
        # 【优化】BGM检测使用最快的模型，避免加载重型模型造成浪费
        original_model = self.config.model_name
        detection_model = "htdemucs"  # 快速模型，用于检测
        
        # 如果当前配置不是检测模型，且模型未加载，则临时切换到检测模型
        need_restore = False
        if self._model is None and original_model != detection_model:
            self.logger.info(f"BGM检测使用快速模型: {detection_model}")
            self.config.model_name = detection_model
            need_restore = True

        self.logger.info("检测背景音乐强度（分位数采样）...")

        # 加载音频
        if audio_array is None:
            import librosa
            audio_array, sr = librosa.load(audio_path, sr=sr)

        if duration_sec is None:
            duration_sec = len(audio_array) / sr

        # 分位数采样位置
        sample_positions = [0.15, 0.50, 0.85]
        sample_duration = self.config.bgm_sample_duration  # 默认10秒

        # 检查音频是否足够长
        if duration_sec < sample_duration * 2:
            self.logger.warning(f"音频太短({duration_sec:.1f}s)，无法可靠检测BGM")
            return BGMLevel.LIGHT, []  # 保守起见返回LIGHT

        ratios = []

        for pos in sample_positions:
            start_time = duration_sec * pos

            # 确保不超出边界
            if start_time + sample_duration > duration_sec:
                start_time = duration_sec - sample_duration
            if start_time < 0:
                start_time = 0

            try:
                # 分离这一段
                vocals = self.separate_vocals_segment(
                    audio_array, sr,
                    start_sec=start_time,
                    end_sec=start_time + sample_duration,
                    buffer_sec=0.5,  # 检测时用较短缓冲
                    shifts=1  # BGM检测使用标准参数
                )

                # 获取原始片段
                start_sample = int(start_time * sr)
                end_sample = int((start_time + sample_duration) * sr)
                original = audio_array[start_sample:end_sample]

                # 计算BGM能量比（使用改进的算法）
                bgm_ratio = self._calculate_bgm_ratio(original, vocals)
                ratios.append(bgm_ratio)

                self.logger.debug(
                    f"采样点 {pos*100:.0f}% ({start_time:.1f}s): BGM比例={bgm_ratio:.2f}"
                )

            except Exception as e:
                self.logger.warning(f"采样点 {pos*100:.0f}% 检测失败: {e}")
                continue

        if not ratios:
            # 【优化】检测完成后卸载模型并恢复配置
            if need_restore:
                self.unload_model()
                self.config.model_name = original_model
                self.logger.info(f"BGM检测完成，卸载检测模型，恢复配置: {original_model}")
            return BGMLevel.LIGHT, []  # 默认假设有轻微BGM

        # 决策逻辑：使用最大值判断（只要有一处BGM很重，就视为Heavy）
        avg_ratio = np.mean(ratios)
        max_ratio = np.max(ratios)

        self.logger.info(
            f"BGM检测完成: 比例={ratios}, 平均={avg_ratio:.2f}, 最大={max_ratio:.2f}"
        )

        # 【优化】检测完成后卸载模型并恢复配置，以便后续根据策略加载正确的模型
        if need_restore:
            self.unload_model()
            self.config.model_name = original_model
            self.logger.info(f"卸载检测模型，后续将根据策略加载: {original_model}")

        # 使用max_ratio作为主要判断依据
        if max_ratio >= self.config.bgm_heavy_threshold:  # 默认0.15
            return BGMLevel.HEAVY, ratios
        elif max_ratio >= self.config.bgm_light_threshold:  # 默认0.02
            return BGMLevel.LIGHT, ratios
        else:
            return BGMLevel.NONE, ratios

    def _calculate_bgm_ratio(
        self, 
        original: np.ndarray, 
        vocals: np.ndarray
    ) -> float:
        """
        计算 BGM 能量占比（最终优化版）
        逻辑：计算残差信号（Background）的 RMS 占原信号 RMS 的比例
        """
        # 1. 长度对齐
        min_len = min(len(original), len(vocals))
        original = original[:min_len]
        vocals = vocals[:min_len]
        
        # 2. 计算残差（即背景音）
        # Demucs 的波形相位一致性很好，直接相减是物理意义上最准确的去人声方法
        background = original - vocals

        # 3. 计算 RMS (使用 float64 提高精度)
        # mean(x^2) 是功率，sqrt(功率) 是 RMS
        rms_orig = np.sqrt(np.mean(original.astype(np.float64) ** 2))
        
        # 4. 快速过滤：如果是静音或极低音量片段，直接返回 0
        # 0.001 约等于 -60dB，低于此值通常是底噪，没有分析意义
        if rms_orig < 1e-3:
            return 0.0

        rms_bgm = np.sqrt(np.mean(background.astype(np.float64) ** 2))
        
        # 5. 计算比例
        # 加上一个极小值 eps 防止除零
        bgm_ratio = rms_bgm / (rms_orig + 1e-8)

        # 6. 结果钳制
        # 理论上 background 不应大于 original，但因浮点误差可能略超 1.0
        return float(max(0.0, min(1.0, bgm_ratio)))

    def _get_cache_key(self, audio_path: str, mode: str) -> str:
        """生成缓存键"""
        path_hash = hashlib.md5(audio_path.encode()).hexdigest()[:16]
        mtime = int(os.path.getmtime(audio_path))
        return f"{path_hash}_{mtime}_{mode}"

    # === Phase 2: 策略解析集成方法 ===

    def resolve_strategy(self, bgm_level: BGMLevel, settings) -> SeparationStrategy:
        """
        根据 BGM 级别解析分离策略

        Args:
            bgm_level: 检测到的 BGM 级别
            settings: 预处理配置对象

        Returns:
            SeparationStrategy: 分离策略
        """
        resolver = SeparationStrategyResolver(settings)
        return resolver.resolve(bgm_level)

    def set_model_for_strategy(self, strategy: SeparationStrategy):
        """
        根据策略设置模型

        Args:
            strategy: 分离策略
        """
        if strategy.should_separate and strategy.initial_model:
            self.set_model(strategy.initial_model)
            self._apply_quality_settings(strategy.initial_model)

    def escalate_model(self, current_model: str, settings) -> Optional[str]:
        """
        升级到更好的模型

        Args:
            current_model: 当前模型名称
            settings: 预处理配置对象

        Returns:
            新模型名称，如果无法升级则返回 None
        """
        resolver = SeparationStrategyResolver(settings)
        new_model = resolver.get_escalation_model(current_model)

        if new_model:
            self.logger.info(f"模型升级: {current_model} -> {new_model}")
            self.set_model(new_model)
            self._apply_quality_settings(new_model)

        return new_model

    def _apply_quality_settings(self, model_name: str):
        """
        应用模型对应的质量参数

        Args:
            model_name: 模型名称
        """
        quality = self.tier_config.model_quality.get(model_name, {})
        if quality:
            self.config.shifts = quality.get("shifts", self.config.shifts)
            self.config.overlap = quality.get("overlap", self.config.overlap)
            self.logger.debug(
                f"应用质量参数: model={model_name}, "
                f"shifts={self.config.shifts}, overlap={self.config.overlap}"
            )
        self._apply_runtime_params(force_defaults=False, apply_model_name=False)

    def separate_chunk(
        self,
        audio: np.ndarray,
        model: str = None,
        sr: int = 16000
    ) -> np.ndarray:
        """
        分离单个 Chunk 的人声（熔断决策器专用接口）

        这是 separate_vocals_segment 的简化适配器，用于熔断回溯场景。
        输入的 audio 已经是切好的 Chunk 片段，无需 start/end 参数。

        Args:
            audio: Chunk 音频数组 (samples,) 单声道
            model: 指定模型名称（可选，不指定则使用当前配置）
            sr: 采样率（默认 16000）

        Returns:
            np.ndarray: 分离后的人声数组 (samples,)
        """
        self._apply_runtime_params(force_defaults=False, apply_model_name=False)

        # 切换模型（如果指定）
        if model and model != self.config.model_name:
            self.set_model(model)
            self._apply_quality_settings(model)

        # 直接对整个 Chunk 进行分离（无需缓冲区，因为已经是完整片段）
        # 复用 separate_vocals_segment 的核心逻辑
        from demucs.apply import apply_model

        demucs_model = self._load_model()

        # 转换为立体声
        if audio.ndim == 1:
            segment = np.stack([audio, audio])  # (2, samples)
        else:
            segment = audio

        # 重采样到模型要求的采样率
        if sr != demucs_model.samplerate:
            import librosa
            segment = librosa.resample(segment, orig_sr=sr, target_sr=demucs_model.samplerate)
            target_sr = demucs_model.samplerate
        else:
            target_sr = sr

        # 转为 tensor
        wav = torch.from_numpy(segment).float()
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        wav = wav.unsqueeze(0)

        if self.config.device == "cuda":
            wav = wav.cuda()

        # 执行分离
        with torch.no_grad():
            sources = apply_model(
                demucs_model,
                wav,
                shifts=self.config.shifts,
                overlap=self.config.overlap,
                progress=False,
                device=self.config.device
            )

        # 提取人声
        source_names = demucs_model.sources
        vocals_idx = source_names.index('vocals')
        vocals = sources[0, vocals_idx]
        vocals = vocals * ref.std() + ref.mean()
        vocals = vocals.cpu().numpy()

        # 重采样回原始采样率
        if target_sr != sr:
            import librosa
            vocals = librosa.resample(vocals, orig_sr=target_sr, target_sr=sr)

        # 转为单声道
        # V3.1.1+dev.20260107.02: 使用 RMS 加权混合，避免相位抵消
        if vocals.ndim > 1:
            vocals = self._stereo_to_mono_safe(vocals)

        # V3.1.1+dev.20260107.01: 响度归一化（与 separate_vocals_segment 一致）
        vocals = self._normalize_vocals(vocals)

        self.logger.debug(f"Chunk 分离完成 (model={self.config.model_name})")
        return vocals


# 全局单例
_demucs_service: Optional[DemucsService] = None


def get_demucs_service() -> DemucsService:
    """获取Demucs服务单例"""
    global _demucs_service
    if _demucs_service is None:
        _demucs_service = DemucsService()
    else:
        _demucs_service.refresh_runtime_params()
    return _demucs_service
