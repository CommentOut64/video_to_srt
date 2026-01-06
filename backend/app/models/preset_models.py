"""
预设配置数据模型 - v3.5 重构版

实现 1+3 预设模式:
- 顶层: 3个快捷场景宏 (Macro Presets) - 极速预览/智能均衡/影视精修
- 底层: 3个独立模块 (Component Modules) - 人声分离/转录核心/增强

高级设置分组:
- 分组一: 预处理与音频 (Preprocessing)
- 分组二: 转录核心 (Transcription)
- 分组三: 增强与润色 (Refinement/LLM)
- 分组四: 计算与系统 (Compute & System)
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


# ========== 模块一: 人声分离 (Demucs) ==========

class DemucsStrategy(Enum):
    """人声分离策略"""
    OFF = "off"                 # 禁止分离 - 不进行人声分离
    AUTO = "auto"               # 智能分诊 - 依赖频谱分诊建议 (默认)
    FORCE_ON = "force_on"       # 极致分离 - 强制使用 mdx_extra


class DemucsModel(Enum):
    """Demucs 模型选择"""
    HTDEMUCS = "htdemucs"           # Hybrid Transformer (默认, 推荐)
    HTDEMUCS_FT = "htdemucs_ft"     # Fine-tuned 版本
    MDX_Q = "mdx_q"                 # MDX-Net 量化版
    MDX_EXTRA = "mdx_extra"         # MDX-Net 完整版 (效果最好但较慢)


# ========== 模块二: 转录核心 (ASR) ==========

class TranscriptionProfile(Enum):
    """转录流水线模式"""
    SENSEVOICE_ONLY = "sensevoice_only"     # 仅 SenseVoice (极速, 默认)
    SV_WHISPER_PATCH = "sv_whisper_patch"   # SV + Whisper 补刀
    SV_WHISPER_DUAL = "sv_whisper_dual"     # SV + Whisper 双流并行


class SenseVoiceDevice(Enum):
    """主引擎运行设备"""
    AUTO = "auto"       # 自动选择 (优先GPU)
    CPU = "cpu"         # 强制CPU


class WhisperModel(Enum):
    """Whisper 模型选择"""
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"       # 默认
    LARGE_V3 = "large-v3"


# ========== 模块三: 增强与润色 (LLM) ==========

class LLMTask(Enum):
    """LLM 任务目标"""
    OFF = "off"             # 关闭 (默认)
    PROOFREAD = "proofread" # 仅校对 (修正错字)
    TRANSLATE = "translate" # 翻译 (含校对)


class LLMScope(Enum):
    """LLM 介入范围"""
    SPARSE = "sparse"   # 稀疏模式 - 仅低置信度句子 (默认)
    GLOBAL = "global"   # 全局模式 - 全文处理


class LLMProvider(Enum):
    """LLM 提供商"""
    OPENAI_COMPATIBLE = "openai_compatible"  # OpenAI Compatible API
    LOCAL_OLLAMA = "local_ollama"            # Local Ollama


# ========== 计算与系统 ==========

class ConcurrencyStrategy(Enum):
    """并发调度策略"""
    AUTO = "auto"           # 自动 - 根据显存余量决定 (默认)
    PARALLEL = "parallel"   # 并行 - 速度最快
    SERIAL = "serial"       # 串行 - 节省显存


class OutputFormat(Enum):
    """输出格式"""
    SRT = "srt"
    VTT = "vtt"
    TXT = "txt"
    JSON = "json"


class TempFilePolicy(Enum):
    """临时文件策略"""
    DELETE_ON_COMPLETE = "delete_on_complete"   # 任务完成后删除 (默认)
    KEEP = "keep"                               # 保留 (用于Debug)


# ========== 高级设置数据类 ==========

@dataclass
class PreprocessingSettings:
    """
    分组一: 预处理与音频设置
    控制 Demucs 人声分离行为
    """
    # 人声分离策略: off/auto/force_on
    demucs_strategy: str = "auto"

    # 分离模型: htdemucs/htdemucs_ft/mdx_q/mdx_extra
    demucs_model: str = "htdemucs"

    # 分离预测次数: 1-5, 数值越高效果越好但速度越慢
    demucs_shifts: int = 1

    # 分离模式: global/on_demand
    separation_mode: str = "on_demand"

    # 是否启用频谱分诊（直通模式应设为 false）
    enable_spectral_triage: bool = True

    # 分诊灵敏度: 0.0-1.0, 值越低越容易触发分离
    # 默认值从 spectrum_thresholds.py 获取: 0.35
    spectrum_threshold: float = 0.35

    # VAD 静音过滤开关
    vad_filter: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "demucs_strategy": self.demucs_strategy,
            "demucs_model": self.demucs_model,
            "demucs_shifts": self.demucs_shifts,
            "separation_mode": self.separation_mode,
            "enable_spectral_triage": self.enable_spectral_triage,
            "spectrum_threshold": self.spectrum_threshold,
            "vad_filter": self.vad_filter,
        }


@dataclass
class TranscriptionSettings:
    """
    分组二: 转录核心设置
    控制 ASR 引擎组合与策略
    """
    # 转录流水线模式: sensevoice_only/sv_whisper_patch/sv_whisper_dual
    transcription_profile: str = "sensevoice_only"

    # 主引擎运行设备: auto/cpu
    sensevoice_device: str = "auto"

    # 辅助/补刀模型: tiny/small/medium/large-v3
    whisper_model: str = "medium"

    # 补刀触发阈值: 0.0-1.0, 低于此置信度的句子送给 Whisper 重跑
    patching_threshold: float = 0.60

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transcription_profile": self.transcription_profile,
            "sensevoice_device": self.sensevoice_device,
            "whisper_model": self.whisper_model,
            "patching_threshold": self.patching_threshold,
        }


@dataclass
class RefinementSettings:
    """
    分组三: 增强与润色设置 (LLM)
    控制大语言模型的后期处理
    """
    # LLM 任务目标: off/proofread/translate
    llm_task: str = "off"

    # 介入范围: sparse/global
    llm_scope: str = "sparse"

    # 稀疏校对阈值: 0.0-1.0, 只有置信度低于此值的句子会被送去校对
    sparse_threshold: float = 0.70

    # 目标语言 (翻译时使用)
    target_language: str = "zh"

    # 模型提供商: openai_compatible/local_ollama
    llm_provider: str = "openai_compatible"

    # 模型名称
    llm_model_name: str = "gpt-4o-mini"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm_task": self.llm_task,
            "llm_scope": self.llm_scope,
            "sparse_threshold": self.sparse_threshold,
            "target_language": self.target_language,
            "llm_provider": self.llm_provider,
            "llm_model_name": self.llm_model_name,
        }


@dataclass
class ComputeSettings:
    """
    分组四: 计算与系统设置
    资源调度与硬件保护
    """
    # 并发调度策略: auto/parallel/serial
    concurrency_strategy: str = "auto"

    # GPU 选择 (多卡系统)
    gpu_id: int = 0

    # 输出格式列表
    output_formats: List[str] = field(default_factory=lambda: ["srt"])

    # 临时文件策略: delete_on_complete/keep
    temp_file_policy: str = "delete_on_complete"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concurrency_strategy": self.concurrency_strategy,
            "gpu_id": self.gpu_id,
            "output_formats": self.output_formats,
            "temp_file_policy": self.temp_file_policy,
        }


# ========== 快捷场景宏 (Macro Presets) ==========

@dataclass
class MacroPreset:
    """
    快捷场景宏配置
    定义三个预设: 极速预览/智能均衡/影视精修
    """
    id: str                             # 预设 ID
    name: str                           # 显示名称
    description: str                    # 描述
    icon: str                           # 图标 (用于前端显示)

    # 三个模块的配置
    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)
    transcription: TranscriptionSettings = field(default_factory=TranscriptionSettings)
    refinement: RefinementSettings = field(default_factory=RefinementSettings)
    compute: ComputeSettings = field(default_factory=ComputeSettings)

    # 硬件要求
    min_vram_mb: int = 2000             # 最低显存要求 (MB)
    recommended_vram_mb: int = 4000     # 推荐显存 (MB)
    requires_gpu: bool = False          # 是否必须有 GPU

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "preprocessing": self.preprocessing.to_dict(),
            "transcription": self.transcription.to_dict(),
            "refinement": self.refinement.to_dict(),
            "compute": self.compute.to_dict(),
            "min_vram_mb": self.min_vram_mb,
            "recommended_vram_mb": self.recommended_vram_mb,
            "requires_gpu": self.requires_gpu,
        }


# ========== 预定义快捷场景宏 ==========

# 预设 A: 极速预览
PRESET_FAST = MacroPreset(
    id="fast",
    name="极速预览",
    description="会议记录、快速浏览内容",
    icon="bolt",  # 闪电图标
    preprocessing=PreprocessingSettings(
        demucs_strategy="off",          # 强制关闭人声分离
        demucs_model="htdemucs",
        demucs_shifts=1,
        separation_mode="on_demand",
        enable_spectral_triage=False,   # 直通模式: 跳过频谱分诊
        spectrum_threshold=0.35,
        vad_filter=True,
    ),
    transcription=TranscriptionSettings(
        transcription_profile="sensevoice_only",    # 仅 SenseVoice
        sensevoice_device="auto",
        whisper_model="medium",
        patching_threshold=0.60,
    ),
    refinement=RefinementSettings(
        llm_task="off",                 # 关闭 LLM
        llm_scope="sparse",
        sparse_threshold=0.70,
        target_language="zh",
        llm_provider="openai_compatible",
        llm_model_name="gpt-4o-mini",
    ),
    compute=ComputeSettings(
        concurrency_strategy="auto",
        gpu_id=0,
        output_formats=["srt"],
        temp_file_policy="delete_on_complete",
    ),
    min_vram_mb=1500,
    recommended_vram_mb=2000,
    requires_gpu=False,
)

# 预设 B: 智能均衡 (默认推荐)
PRESET_BALANCED = MacroPreset(
    id="balanced",
    name="智能均衡",
    description="短视频、Vlog、一般内容字幕 (推荐)",
    icon="scale",  # 天平图标
    preprocessing=PreprocessingSettings(
        demucs_strategy="auto",         # 智能分诊
        demucs_model="htdemucs",
        demucs_shifts=1,
        separation_mode="on_demand",
        enable_spectral_triage=True,    # 智能模式: 启用频谱分诊
        spectrum_threshold=0.35,
        vad_filter=True,
    ),
    transcription=TranscriptionSettings(
        transcription_profile="sv_whisper_patch",   # SV + Whisper 补刀
        sensevoice_device="auto",
        whisper_model="medium",         # Whisper Medium 用于补刀
        patching_threshold=0.60,        # 置信度 < 60% 触发补刀
    ),
    refinement=RefinementSettings(
        llm_task="proofread",           # 稀疏校对
        llm_scope="sparse",             # 只处理低置信度段落
        sparse_threshold=0.70,
        target_language="zh",
        llm_provider="openai_compatible",
        llm_model_name="gpt-4o-mini",
    ),
    compute=ComputeSettings(
        concurrency_strategy="auto",
        gpu_id=0,
        output_formats=["srt"],
        temp_file_policy="delete_on_complete",
    ),
    min_vram_mb=4000,
    recommended_vram_mb=6000,
    requires_gpu=True,
)

# 预设 C: 影视精修
PRESET_QUALITY = MacroPreset(
    id="quality",
    name="影视精修",
    description="电影压制、复杂背景音、高精度要求",
    icon="film",  # 电影图标
    preprocessing=PreprocessingSettings(
        demucs_strategy="force_on",     # 强制开启人声分离
        demucs_model="mdx_extra",       # 使用 mdx_extra 高质量模型
        demucs_shifts=1,                # 增加预测次数
        separation_mode="global",
        enable_spectral_triage=False,   # 强制分离模式: 跳过频谱分诊
        spectrum_threshold=0.35,
        vad_filter=True,
    ),
    transcription=TranscriptionSettings(
        transcription_profile="sv_whisper_dual",    # 双流并行
        sensevoice_device="auto",
        whisper_model="large-v3",       # Whisper Large 全文填词
        patching_threshold=0.60,
    ),
    refinement=RefinementSettings(
        llm_task="proofread",           # 全局校对
        llm_scope="global",             # 全文处理
        sparse_threshold=0.70,
        target_language="zh",
        llm_provider="openai_compatible",
        llm_model_name="gpt-4o-mini",
    ),
    compute=ComputeSettings(
        concurrency_strategy="auto",
        gpu_id=0,
        output_formats=["srt"],
        temp_file_policy="delete_on_complete",
    ),
    min_vram_mb=8000,
    recommended_vram_mb=12000,
    requires_gpu=True,
)

# 预设定义字典
MACRO_PRESETS: Dict[str, MacroPreset] = {
    "fast": PRESET_FAST,
    "balanced": PRESET_BALANCED,
    "quality": PRESET_QUALITY,
}


# ========== 完整任务配置 ==========

@dataclass
class TaskConfig:
    """
    完整任务配置
    整合所有设置分组，用于提交转录任务
    """
    # 选择的宏预设 ID (fast/balanced/quality/custom)
    preset_id: str = "balanced"

    # 四个设置分组
    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)
    transcription: TranscriptionSettings = field(default_factory=TranscriptionSettings)
    refinement: RefinementSettings = field(default_factory=RefinementSettings)
    compute: ComputeSettings = field(default_factory=ComputeSettings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preset_id": self.preset_id,
            "preprocessing": self.preprocessing.to_dict(),
            "transcription": self.transcription.to_dict(),
            "refinement": self.refinement.to_dict(),
            "compute": self.compute.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConfig":
        """从字典创建配置"""
        preprocessing_data = data.get("preprocessing", {})
        transcription_data = data.get("transcription", {})
        refinement_data = data.get("refinement", {})
        compute_data = data.get("compute", {})

        return cls(
            preset_id=data.get("preset_id", "balanced"),
            preprocessing=PreprocessingSettings(
                demucs_strategy=preprocessing_data.get("demucs_strategy", "auto"),
                demucs_model=preprocessing_data.get("demucs_model", "htdemucs"),
                demucs_shifts=preprocessing_data.get("demucs_shifts", 1),
                separation_mode=preprocessing_data.get("separation_mode", "on_demand"),
                spectrum_threshold=preprocessing_data.get("spectrum_threshold", 0.35),
                vad_filter=preprocessing_data.get("vad_filter", True),
            ),
            transcription=TranscriptionSettings(
                transcription_profile=transcription_data.get("transcription_profile", "sensevoice_only"),
                sensevoice_device=transcription_data.get("sensevoice_device", "auto"),
                whisper_model=transcription_data.get("whisper_model", "medium"),
                patching_threshold=transcription_data.get("patching_threshold", 0.60),
            ),
            refinement=RefinementSettings(
                llm_task=refinement_data.get("llm_task", "off"),
                llm_scope=refinement_data.get("llm_scope", "sparse"),
                sparse_threshold=refinement_data.get("sparse_threshold", 0.70),
                target_language=refinement_data.get("target_language", "zh"),
                llm_provider=refinement_data.get("llm_provider", "openai_compatible"),
                llm_model_name=refinement_data.get("llm_model_name", "gpt-4o-mini"),
            ),
            compute=ComputeSettings(
                concurrency_strategy=compute_data.get("concurrency_strategy", "auto"),
                gpu_id=compute_data.get("gpu_id", 0),
                output_formats=compute_data.get("output_formats", ["srt"]),
                temp_file_policy=compute_data.get("temp_file_policy", "delete_on_complete"),
            ),
        )

    @classmethod
    def from_preset(cls, preset_id: str) -> "TaskConfig":
        """从预设创建配置"""
        preset = MACRO_PRESETS.get(preset_id)
        if not preset:
            # 默认使用智能均衡
            preset = PRESET_BALANCED
            preset_id = "balanced"

        return cls(
            preset_id=preset_id,
            preprocessing=PreprocessingSettings(
                demucs_strategy=preset.preprocessing.demucs_strategy,
                demucs_model=preset.preprocessing.demucs_model,
                demucs_shifts=preset.preprocessing.demucs_shifts,
                separation_mode=preset.preprocessing.separation_mode,
                spectrum_threshold=preset.preprocessing.spectrum_threshold,
                vad_filter=preset.preprocessing.vad_filter,
            ),
            transcription=TranscriptionSettings(
                transcription_profile=preset.transcription.transcription_profile,
                sensevoice_device=preset.transcription.sensevoice_device,
                whisper_model=preset.transcription.whisper_model,
                patching_threshold=preset.transcription.patching_threshold,
            ),
            refinement=RefinementSettings(
                llm_task=preset.refinement.llm_task,
                llm_scope=preset.refinement.llm_scope,
                sparse_threshold=preset.refinement.sparse_threshold,
                target_language=preset.refinement.target_language,
                llm_provider=preset.refinement.llm_provider,
                llm_model_name=preset.refinement.llm_model_name,
            ),
            compute=ComputeSettings(
                concurrency_strategy=preset.compute.concurrency_strategy,
                gpu_id=preset.compute.gpu_id,
                output_formats=preset.compute.output_formats,
                temp_file_policy=preset.compute.temp_file_policy,
            ),
        )


# ========== 辅助函数 ==========

def get_preset(preset_id: str) -> Optional[MacroPreset]:
    """获取指定预设配置"""
    return MACRO_PRESETS.get(preset_id)


def get_all_presets() -> List[MacroPreset]:
    """获取所有预设配置"""
    return list(MACRO_PRESETS.values())


def get_recommended_preset(vram_mb: int, has_gpu: bool = True) -> MacroPreset:
    """
    根据硬件条件推荐预设

    Args:
        vram_mb: 可用显存 (MB)
        has_gpu: 是否有 GPU

    Returns:
        推荐的预设配置
    """
    if not has_gpu:
        return PRESET_FAST

    if vram_mb >= 8000:
        return PRESET_QUALITY
    elif vram_mb >= 4000:
        return PRESET_BALANCED
    else:
        return PRESET_FAST


def check_preset_compatibility(preset: MacroPreset, vram_mb: int, has_gpu: bool = True) -> Dict[str, Any]:
    """
    检查预设与硬件的兼容性

    Args:
        preset: 预设配置
        vram_mb: 可用显存 (MB)
        has_gpu: 是否有 GPU

    Returns:
        兼容性检查结果
    """
    warnings = []
    errors = []

    # 检查 GPU 要求
    if preset.requires_gpu and not has_gpu:
        errors.append({
            "type": "no_gpu",
            "message": f"预设 '{preset.name}' 需要 GPU，但未检测到可用 GPU"
        })

    # 检查显存要求
    if has_gpu:
        if vram_mb < preset.min_vram_mb:
            errors.append({
                "type": "insufficient_vram",
                "message": f"预设 '{preset.name}' 需要至少 {preset.min_vram_mb}MB 显存，当前只有 {vram_mb}MB"
            })
        elif vram_mb < preset.recommended_vram_mb:
            warnings.append({
                "type": "low_vram",
                "message": f"当前显存 ({vram_mb}MB) 低于推荐值 ({preset.recommended_vram_mb}MB)，可能影响性能"
            })

    # 检查特定配置的显存需求
    if preset.preprocessing.demucs_model == "mdx_extra" and vram_mb < 6000:
        warnings.append({
            "type": "demucs_vram",
            "message": "mdx_extra 模型建议 6GB+ 显存"
        })

    if preset.transcription.whisper_model == "large-v3" and vram_mb < 8000:
        warnings.append({
            "type": "whisper_vram",
            "message": "Whisper large-v3 模型建议 8GB+ 显存"
        })

    return {
        "compatible": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
    }


def match_preset_from_settings(
    preprocessing: PreprocessingSettings,
    transcription: TranscriptionSettings,
    refinement: RefinementSettings
) -> Optional[str]:
    """
    根据当前设置匹配对应的预设 ID

    如果设置与某个预设完全匹配，返回该预设 ID；否则返回 None (表示自定义)

    Args:
        preprocessing: 预处理设置
        transcription: 转录设置
        refinement: 增强设置

    Returns:
        匹配的预设 ID 或 None
    """
    for preset_id, preset in MACRO_PRESETS.items():
        if (
            preprocessing.demucs_strategy == preset.preprocessing.demucs_strategy and
            transcription.transcription_profile == preset.transcription.transcription_profile and
            refinement.llm_task == preset.refinement.llm_task and
            refinement.llm_scope == preset.refinement.llm_scope
        ):
            return preset_id

    return None
