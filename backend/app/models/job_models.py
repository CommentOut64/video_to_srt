"""
任务相关的数据模型定义 - v3.5 重构版

与 preset_models.py 中的 1+3 预设模式保持一致
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any


# ========== 分组一: 预处理与音频设置 ==========

@dataclass
class PreprocessingConfig:
    """
    预处理与音频配置 (Demucs 人声分离 + 频谱分诊 + 熔断回溯)
    对应文档分组一
    """
    # ========== 人声分离配置 ==========
    # 人声分离策略: off/auto/force_on
    demucs_strategy: str = "auto"

    # 分离模型: htdemucs/htdemucs_ft/mdx_q/mdx_extra
    demucs_model: str = "htdemucs"

    # 分离预测次数: 1-5
    demucs_shifts: int = 1

    # 分离模式: global/on_demand (新增)
    separation_mode: str = "on_demand"

    # ========== 频谱分诊配置 ==========
    # 是否启用频谱分诊 (新增)
    enable_spectral_triage: bool = True

    # 分诊灵敏度: 0.0-1.0 (默认从 spectrum_thresholds.py: 0.35)
    spectrum_threshold: float = 0.35

    # V3.1.1+dev.20260108.02: 是否启用 SNR+C50 三层决策策略（默认启用）
    use_snr_triage: bool = True

    # ========== 熔断回溯配置 (新增) ==========
    # 是否启用熔断回溯
    enable_fuse_breaker: bool = True

    # 最大重试次数（默认1，只升级到 HTDEMUCS）
    fuse_max_retry: int = 1

    # 置信度阈值
    fuse_confidence_threshold: float = 0.5

    # 是否启用第二次自动升级到 MDX_EXTRA（默认False，可选配置）
    fuse_auto_upgrade: bool = False

    # ========== VAD配置 ==========
    # VAD 静音过滤开关
    vad_filter: bool = True

    # ========== LangID 语言检测配置 ==========
    # V3.2.0+dev.20260127.02: 语言检测模式与设备选择
    language_detection_mode: str = "balanced"  # fast/balanced/precise
    language_detection_device: str = "auto"    # auto/cpu/cuda
    langid_confidence_threshold: float = 0.7
    langid_whitelist: List[str] = field(default_factory=lambda: ["zh", "ja", "en"])
    langid_logit_bias_score: float = 2.5
    enable_speaker_embedding: bool = False

    # ========== 预处理缓存配置 ==========
    # 是否启用预处理缓存 GC（默认关闭）
    is_preprocess_cache_gc_enabled: bool = False

    # 缓存预算（GB），0 表示不限制
    cache_budget_gb: float = 0.0

    # 缓存 TTL（小时），0 表示不限制
    ttl_hours: float = 0.0

    # 缓存任务数上限，0 表示不限制
    max_tasks: int = 0


# ========== 分组二: 转录核心设置 ==========

@dataclass
class TranscriptionConfig:
    """
    转录核心配置 (ASR 引擎)
    对应文档分组二
    """
    # 转录流水线模式: sensevoice_only/sv_whisper_patch/sv_whisper_dual
    transcription_profile: str = "sensevoice_only"

    # 主引擎运行设备: auto/cpu
    sensevoice_device: str = "auto"

    # 辅助/复核模型: tiny/small/medium/large-v3
    whisper_model: str = "medium"

    # 复核触发阈值: 0.0-1.0
    patching_threshold: float = 0.60


# ========== 分组三: 增强与润色设置 ==========

@dataclass
class RefinementConfig:
    """
    增强与润色配置 (LLM)
    对应文档分组三
    """
    # LLM 任务目标: off/proofread/translate
    llm_task: str = "off"

    # 介入范围: sparse/global
    llm_scope: str = "sparse"

    # 稀疏校对阈值: 0.0-1.0
    sparse_threshold: float = 0.70

    # 目标语言
    target_language: str = "zh"

    # 模型提供商: openai_compatible/local_ollama
    llm_provider: str = "openai_compatible"

    # 模型名称
    llm_model_name: str = "gpt-4o-mini"

    # API Key (运行时注入，不持久化)
    api_key: Optional[str] = None

    # 自定义 API 地址
    base_url: Optional[str] = None


# ========== 分组四: 计算与系统设置 ==========

@dataclass
class ComputeConfig:
    """
    计算与系统配置
    对应文档分组四
    """
    # 并发调度策略: auto/parallel/serial
    concurrency_strategy: str = "auto"

    # GPU 选择
    gpu_id: int = 0

    # 输出格式列表
    output_formats: List[str] = field(default_factory=lambda: ["srt"])

    # 临时文件策略: delete_on_complete/keep
    temp_file_policy: str = "delete_on_complete"


# ========== 调试配置 ==========

@dataclass
class DebugConfig:
    """
    调试配置（按需启用）。
    """
    punctuation_output: bool = False  # 标点调试输出（SSE + 文件）


# ========== 任务设置 ==========

@dataclass
class JobSettings:
    """
    转录任务设置 - v3.5 重构版
    仅保留新版 task_config（预设 + 四分组）
    """

    # === 新版 1+3 预设配置 ===
    # 选择的宏预设 ID (fast/balanced/quality/custom)
    preset_id: str = "balanced"

    # 四个设置分组
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            # 新版配置
            "preset_id": self.preset_id,
            "preprocessing": {
                "demucs_strategy": self.preprocessing.demucs_strategy,
                "demucs_model": self.preprocessing.demucs_model,
                "demucs_shifts": self.preprocessing.demucs_shifts,
                "separation_mode": self.preprocessing.separation_mode,
                "enable_spectral_triage": self.preprocessing.enable_spectral_triage,
                "spectrum_threshold": self.preprocessing.spectrum_threshold,
                "use_snr_triage": self.preprocessing.use_snr_triage,
                "enable_fuse_breaker": self.preprocessing.enable_fuse_breaker,
                "fuse_max_retry": self.preprocessing.fuse_max_retry,
                "fuse_confidence_threshold": self.preprocessing.fuse_confidence_threshold,
                "fuse_auto_upgrade": self.preprocessing.fuse_auto_upgrade,
                "vad_filter": self.preprocessing.vad_filter,
                "language_detection_mode": self.preprocessing.language_detection_mode,
                "language_detection_device": self.preprocessing.language_detection_device,
                "langid_confidence_threshold": self.preprocessing.langid_confidence_threshold,
                "langid_whitelist": self.preprocessing.langid_whitelist,
                "langid_logit_bias_score": self.preprocessing.langid_logit_bias_score,
                "enable_speaker_embedding": self.preprocessing.enable_speaker_embedding,
                "enable_preprocess_cache_gc": self.preprocessing.is_preprocess_cache_gc_enabled,
                "cache_budget_gb": self.preprocessing.cache_budget_gb,
                "ttl_hours": self.preprocessing.ttl_hours,
                "max_tasks": self.preprocessing.max_tasks,
            },
            "transcription": {
                "transcription_profile": self.transcription.transcription_profile,
                "sensevoice_device": self.transcription.sensevoice_device,
                "whisper_model": self.transcription.whisper_model,
                "patching_threshold": self.transcription.patching_threshold,
            },
            "refinement": {
                "llm_task": self.refinement.llm_task,
                "llm_scope": self.refinement.llm_scope,
                "sparse_threshold": self.refinement.sparse_threshold,
                "target_language": self.refinement.target_language,
                "llm_provider": self.refinement.llm_provider,
                "llm_model_name": self.refinement.llm_model_name,
            },
            "compute": {
                "concurrency_strategy": self.compute.concurrency_strategy,
                "gpu_id": self.compute.gpu_id,
                "output_formats": self.compute.output_formats,
                "temp_file_policy": self.compute.temp_file_policy,
            },
            "debug": {
                "punctuation_output": self.debug.punctuation_output,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobSettings":
        """从字典创建设置"""
        raw_data = data
        # 允许直接传入 task_config
        if "task_config" in data and isinstance(data["task_config"], dict):
            data = data["task_config"]

        legacy_keys = {
            "engine",
            "sensevoice",
            "demucs",
            "model",
            "device",
            "compute_type",
            "word_timestamps",
            "batch_size",
            "cpu_affinity",
            "cpu_affinity_enabled",
            "cpu_affinity_strategy",
            "cpu_affinity_custom_cores",
            "cpu_affinity_exclude_cores",
        }
        if any(key in raw_data for key in legacy_keys) or any(key in data for key in legacy_keys):
            raise ValueError("检测到旧版任务配置字段，已停止兼容，请使用 task_config 发送新版配置。")

        # 解析新版配置
        preprocessing_data = data.get("preprocessing", {})
        transcription_data = data.get("transcription", {})
        refinement_data = data.get("refinement", {})
        compute_data = data.get("compute", {})
        debug_data = data.get("debug", {})

        raw_whitelist = preprocessing_data.get("langid_whitelist", ["zh", "ja", "en"])
        if isinstance(raw_whitelist, str):
            langid_whitelist = [raw_whitelist]
        elif isinstance(raw_whitelist, list):
            langid_whitelist = raw_whitelist
        else:
            langid_whitelist = ["zh", "ja", "en"]

        return cls(
            # 新版配置
            preset_id=data.get("preset_id", "balanced"),
            preprocessing=PreprocessingConfig(
                demucs_strategy=preprocessing_data.get("demucs_strategy", "auto"),
                demucs_model=preprocessing_data.get("demucs_model", "htdemucs"),
                demucs_shifts=preprocessing_data.get("demucs_shifts", 1),
                separation_mode=preprocessing_data.get("separation_mode", "on_demand"),
                enable_spectral_triage=preprocessing_data.get("enable_spectral_triage", True),
                spectrum_threshold=preprocessing_data.get("spectrum_threshold", 0.35),
                use_snr_triage=preprocessing_data.get("use_snr_triage", True),
                enable_fuse_breaker=preprocessing_data.get("enable_fuse_breaker", True),
                fuse_max_retry=preprocessing_data.get("fuse_max_retry", 2),
                fuse_confidence_threshold=preprocessing_data.get("fuse_confidence_threshold", 0.5),
                fuse_auto_upgrade=preprocessing_data.get("fuse_auto_upgrade", True),
                vad_filter=preprocessing_data.get("vad_filter", True),
                language_detection_mode=preprocessing_data.get("language_detection_mode", "balanced"),
                language_detection_device=preprocessing_data.get("language_detection_device", "auto"),
                langid_confidence_threshold=preprocessing_data.get("langid_confidence_threshold", 0.7),
                langid_whitelist=langid_whitelist,
                langid_logit_bias_score=preprocessing_data.get("langid_logit_bias_score", 2.5),
                enable_speaker_embedding=preprocessing_data.get("enable_speaker_embedding", False),
                is_preprocess_cache_gc_enabled=preprocessing_data.get("enable_preprocess_cache_gc", False),
                cache_budget_gb=preprocessing_data.get("cache_budget_gb", 0.0),
                ttl_hours=preprocessing_data.get("ttl_hours", 0.0),
                max_tasks=preprocessing_data.get("max_tasks", 0),
            ),
            transcription=TranscriptionConfig(
                transcription_profile=transcription_data.get("transcription_profile", "sensevoice_only"),
                sensevoice_device=transcription_data.get("sensevoice_device", "auto"),
                whisper_model=transcription_data.get("whisper_model", "medium"),
                patching_threshold=transcription_data.get("patching_threshold", 0.60),
            ),
            refinement=RefinementConfig(
                llm_task=refinement_data.get("llm_task", "off"),
                llm_scope=refinement_data.get("llm_scope", "sparse"),
                sparse_threshold=refinement_data.get("sparse_threshold", 0.70),
                target_language=refinement_data.get("target_language", "zh"),
                llm_provider=refinement_data.get("llm_provider", "openai_compatible"),
                llm_model_name=refinement_data.get("llm_model_name", "gpt-4o-mini"),
            ),
            compute=ComputeConfig(
                concurrency_strategy=compute_data.get("concurrency_strategy", "auto"),
                gpu_id=compute_data.get("gpu_id", 0),
                output_formats=compute_data.get("output_formats", ["srt"]),
                temp_file_policy=compute_data.get("temp_file_policy", "delete_on_complete"),
            ),
            debug=DebugConfig(
                punctuation_output=bool(debug_data.get("punctuation_output", False)),
            ),
        )

    @classmethod
    def from_preset(cls, preset_id: str) -> "JobSettings":
        """从预设创建设置"""
        from app.models.preset_models import MACRO_PRESETS, PRESET_BALANCED

        preset = MACRO_PRESETS.get(preset_id)
        if not preset:
            preset = PRESET_BALANCED
            preset_id = "balanced"

        return cls(
            preset_id=preset_id,
            preprocessing=PreprocessingConfig(
                demucs_strategy=preset.preprocessing.demucs_strategy,
                demucs_model=preset.preprocessing.demucs_model,
                demucs_shifts=preset.preprocessing.demucs_shifts,
                separation_mode=preset.preprocessing.separation_mode,
                enable_spectral_triage=preset.preprocessing.enable_spectral_triage,
                spectrum_threshold=preset.preprocessing.spectrum_threshold,
                use_snr_triage=True,  # V3.1.1+dev.20260108.02: 默认启用 SNR+C50 策略
                vad_filter=preset.preprocessing.vad_filter,
                enable_fuse_breaker=True,
                fuse_max_retry=1,
                fuse_confidence_threshold=0.5,
                fuse_auto_upgrade=False,
                language_detection_mode="balanced",
                language_detection_device="auto",
                langid_confidence_threshold=0.7,
                langid_whitelist=["zh", "ja", "en"],
                langid_logit_bias_score=2.5,
                is_preprocess_cache_gc_enabled=False,
                cache_budget_gb=0.0,
                ttl_hours=0.0,
                max_tasks=0,
            ),
            transcription=TranscriptionConfig(
                transcription_profile=preset.transcription.transcription_profile,
                sensevoice_device=preset.transcription.sensevoice_device,
                whisper_model=preset.transcription.whisper_model,
                patching_threshold=preset.transcription.patching_threshold,
            ),
            refinement=RefinementConfig(
                llm_task=preset.refinement.llm_task,
                llm_scope=preset.refinement.llm_scope,
                sparse_threshold=preset.refinement.sparse_threshold,
                target_language=preset.refinement.target_language,
                llm_provider=preset.refinement.llm_provider,
                llm_model_name=preset.refinement.llm_model_name,
            ),
            compute=ComputeConfig(
                concurrency_strategy=preset.compute.concurrency_strategy,
                gpu_id=preset.compute.gpu_id,
                output_formats=preset.compute.output_formats,
                temp_file_policy=preset.compute.temp_file_policy,
            ),
        )


@dataclass
class MediaStatus:
    """媒体资源状态（用于编辑器）"""
    video_exists: bool = False          # 视频文件是否存在
    video_format: Optional[str] = None  # 视频格式（.mp4, .mkv等）
    needs_proxy: bool = False           # 是否需要Proxy转码
    proxy_exists: bool = False          # Proxy视频是否已生成
    audio_exists: bool = False          # 音频文件是否存在
    peaks_ready: bool = False           # 波形峰值数据是否就绪
    thumbnails_ready: bool = False      # 缩略图是否就绪
    srt_exists: bool = False            # SRT文件是否存在


@dataclass
class JobState:
    """转录任务状态"""
    job_id: str
    filename: str
    dir: str
    input_path: str = ""  # 添加原始输入路径记录
    settings: JobSettings = field(default_factory=JobSettings)
    status: str = "queued"  # queued, processing, finished, failed, canceled, paused
    phase: str = "pending"  # extract, split, transcribe, srt
    progress: float = 0.0
    phase_percent: float = 0.0  # 当前阶段内进度 (0-100)
    message: str = "等待开始"
    error: Optional[str] = None
    segments: List[Dict] = field(default_factory=list)
    processed: int = 0
    total: int = 0
    language: Optional[str] = None
    srt_path: Optional[str] = None
    canceled: bool = False
    paused: bool = False  # 暂停标志
    title: str = ""  # 用户自定义的任务名称，为空时使用 filename
    createdAt: Optional[int] = None  # 创建时间戳
    updatedAt: Optional[int] = None  # 更新时间戳（毫秒）
    # V3.2.0+dev.20260130.10: 任务级字幕时间偏移（秒，None 表示使用全局默认）
    subtitle_time_offset: Optional[float] = None

    # 媒体状态（用于编辑器，转录完成后更新）
    media_status: Optional[MediaStatus] = None

    def to_dict(self):
        """转换为字典格式，用于API响应"""
        d = asdict(self)
        d.pop('segments', None)  # 不透出内部详情
        return d

    def to_meta_dict(self) -> dict:
        """
        转换为元信息字典格式，用于持久化到 job_meta.json
        只保存恢复任务所需的核心信息，不包含 segments 等大数据
        """
        import time
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "title": self.title,
            "dir": self.dir,
            "input_path": self.input_path,
            "status": self.status,
            "phase": self.phase,
            "progress": self.progress,
            "phase_percent": self.phase_percent,
            "message": self.message,
            "error": self.error,
            "processed": self.processed,
            "total": self.total,
            "language": self.language,
            "srt_path": self.srt_path,
            "canceled": self.canceled,
            "paused": self.paused,
            "settings": self.settings.to_dict(),
            "subtitle_time_offset": self.subtitle_time_offset,
            "updated_at": time.time()
        }

    @classmethod
    def from_meta_dict(cls, data: dict) -> "JobState":
        """
        从元信息字典恢复 JobState 对象

        Args:
            data: job_meta.json 中的数据

        Returns:
            JobState: 恢复的任务状态对象
        """
        settings_data = data.get("settings", {})
        settings = JobSettings.from_dict(settings_data)
        updated_at = data.get("updated_at", data.get("updatedAt"))
        if updated_at is not None and updated_at < 1_000_000_000_000:
            updated_at = int(updated_at * 1000)

        return cls(
            job_id=data["job_id"],
            filename=data.get("filename", "unknown"),
            title=data.get("title", ""),
            dir=data.get("dir", ""),
            input_path=data.get("input_path", ""),
            settings=settings,
            status=data.get("status", "queued"),
            phase=data.get("phase", "pending"),
            progress=data.get("progress", 0.0),
            phase_percent=data.get("phase_percent", 0.0),
            message=data.get("message", ""),
            error=data.get("error"),
            processed=data.get("processed", 0),
            total=data.get("total", 0),
            language=data.get("language"),
            srt_path=data.get("srt_path"),
            canceled=data.get("canceled", False),
            paused=data.get("paused", False),
            updatedAt=updated_at,
            subtitle_time_offset=data.get("subtitle_time_offset"),
        )

    def update_media_status(self, job_dir: str):
        """
        更新媒体状态（检查各类资源文件是否就绪）

        Args:
            job_dir: 任务目录路径
        """
        from pathlib import Path

        job_path = Path(job_dir)
        if not job_path.exists():
            return

        # 需要转码的格式
        need_transcode_formats = {'.mkv', '.avi', '.mov', '.wmv', '.flv', '.m4v'}

        # 查找视频文件
        video_file = None
        video_exts = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.webm', '.flv', '.m4v']
        for file in job_path.iterdir():
            if file.is_file() and file.suffix.lower() in video_exts:
                video_file = file
                break

        # 检查各项资源
        audio_file = job_path / "audio.wav"
        proxy_file = job_path / "proxy.mp4"
        peaks_file = job_path / "peaks_2000.json"
        thumbnails_file = job_path / "thumbnails_10.json"

        # 查找SRT文件
        srt_exists = False
        for file in job_path.iterdir():
            if file.suffix.lower() == '.srt':
                srt_exists = True
                break

        # 更新媒体状态
        self.media_status = MediaStatus(
            video_exists=video_file is not None,
            video_format=video_file.suffix if video_file else None,
            needs_proxy=video_file is not None and video_file.suffix.lower() in need_transcode_formats,
            proxy_exists=proxy_file.exists(),
            audio_exists=audio_file.exists(),
            peaks_ready=peaks_file.exists(),
            thumbnails_ready=thumbnails_file.exists(),
            srt_exists=srt_exists
        )
