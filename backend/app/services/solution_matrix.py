"""
组合方案矩阵 (Solution Matrix)

提供高度模块化的配置，允许在"速度、成本、质量"之间自由组合。
前端预设对应后端具体配置。

v3.5 更新: 支持从新版 transcription_profile 和 refinement 配置创建 SolutionConfig
v3.1.0 更新: 使用 ConfigAdapter 统一新旧配置访问
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EnhancementMode(Enum):
    """增强模式"""
    OFF = "off"                      # A1: SenseVoice Only
    SMART_PATCH = "smart_patch"      # B1: SenseVoice + Whisper Partial（推荐）
    DEEP_LISTEN = "deep_listen"      # C1: SenseVoice + Whisper Full


class ProofreadMode(Enum):
    """校对模式"""
    OFF = "off"
    SPARSE = "sparse"                # P1: 按需修复（推荐）
    FULL = "full"                    # P2: 全文精修


class TranslateMode(Enum):
    """翻译模式"""
    OFF = "off"
    FULL = "full"                    # T1: 全量翻译
    PARTIAL = "partial"              # T2: 部分翻译


@dataclass
class SolutionConfig:
    """方案配置"""
    preset_id: str = "default"
    enhancement: EnhancementMode = EnhancementMode.OFF
    proofread: ProofreadMode = ProofreadMode.OFF
    translate: TranslateMode = TranslateMode.OFF
    target_language: Optional[str] = None

    # 高级选项
    confidence_threshold: float = 0.6

    @classmethod
    def from_preset(cls, preset_id: str) -> 'SolutionConfig':
        """根据预设ID创建配置 (兼容旧版)"""
        presets = {
            'default': cls(
                preset_id='default',
                enhancement=EnhancementMode.OFF,
                proofread=ProofreadMode.OFF,
                translate=TranslateMode.OFF
            ),
            'preset1': cls(
                preset_id='preset1',
                enhancement=EnhancementMode.SMART_PATCH,
                proofread=ProofreadMode.OFF,
                translate=TranslateMode.OFF
            ),
            'preset2': cls(
                preset_id='preset2',
                enhancement=EnhancementMode.SMART_PATCH,
                proofread=ProofreadMode.SPARSE,
                translate=TranslateMode.OFF
            ),
            'preset3': cls(
                preset_id='preset3',
                enhancement=EnhancementMode.SMART_PATCH,
                proofread=ProofreadMode.FULL,
                translate=TranslateMode.OFF
            ),
            'preset4': cls(
                preset_id='preset4',
                enhancement=EnhancementMode.SMART_PATCH,
                proofread=ProofreadMode.FULL,
                translate=TranslateMode.FULL,
                target_language='en'
            ),
            'preset5': cls(
                preset_id='preset5',
                enhancement=EnhancementMode.SMART_PATCH,
                proofread=ProofreadMode.FULL,
                translate=TranslateMode.PARTIAL
            ),
        }
        return presets.get(preset_id, cls())

    @classmethod
    def from_job_settings(cls, job_settings) -> 'SolutionConfig':
        """
        从 JobSettings 创建配置 (使用 ConfigAdapter 统一新旧配置)

        v3.1.0 更新: 使用 ConfigAdapter 自动兼容新旧配置格式

        映射关系:
        - transcription_profile -> enhancement
          - sensevoice_only -> OFF
          - sv_whisper_patch -> SMART_PATCH
          - sv_whisper_dual -> DEEP_LISTEN
        - llm_task + llm_scope -> proofread/translate
          - off -> proofread=OFF, translate=OFF
          - proofread + sparse -> proofread=SPARSE
          - proofread + global -> proofread=FULL
          - translate -> translate=FULL (含校对)
        """
        from app.services.config_adapter import ConfigAdapter

        # 使用 ConfigAdapter 统一获取配置 (自动兼容新旧格式)
        transcription_profile = ConfigAdapter.get_transcription_profile(job_settings)
        preset_id = ConfigAdapter.get_preset_id(job_settings)
        llm_task = ConfigAdapter.get_llm_task(job_settings)
        llm_scope = ConfigAdapter.get_llm_scope(job_settings)
        target_language = ConfigAdapter.get_target_language(job_settings)
        confidence_threshold = ConfigAdapter.get_patching_threshold(job_settings)

        # 映射 transcription_profile -> enhancement
        profile_to_enhancement = {
            'sensevoice_only': EnhancementMode.OFF,
            'sv_whisper_patch': EnhancementMode.SMART_PATCH,
            'sv_whisper_dual': EnhancementMode.DEEP_LISTEN,
        }
        enhancement = profile_to_enhancement.get(transcription_profile, EnhancementMode.OFF)

        # 映射 llm_task + llm_scope -> proofread/translate
        proofread = ProofreadMode.OFF
        translate = TranslateMode.OFF

        if llm_task == 'translate':
            # 翻译任务，同时包含校对
            translate = TranslateMode.FULL
            proofread = ProofreadMode.FULL
        elif llm_task == 'proofread':
            if llm_scope == 'global':
                proofread = ProofreadMode.FULL
            else:  # sparse
                proofread = ProofreadMode.SPARSE

        return cls(
            preset_id=preset_id,
            enhancement=enhancement,
            proofread=proofread,
            translate=translate,
            target_language=target_language if translate != TranslateMode.OFF else None,
            confidence_threshold=confidence_threshold
        )


# ========== 方案矩阵描述 ==========

SOLUTION_MATRIX = {
    # ========== 基础层 ==========
    "A1": {
        "name": "SenseVoice Only",
        "flow": ["sensevoice"],
        "scenario": "实时预览、即时日志",
        "note": "极速。原汁原味"
    },
    "B1": {
        "name": "SenseVoice + Whisper Partial",
        "flow": ["sensevoice", "whisper_partial", "pseudo_align"],
        "scenario": "嘈杂、多BGM环境",
        "note": "默认推荐。仅对低置信度片段进行Whisper重听，伪对齐"
    },
    # ========== 语义层（可叠加） ==========
    "P1": {
        "name": "LLM Partial Proof",
        "flow": ["llm_sparse_proof"],
        "scenario": "个人笔记、日常Vlog",
        "note": "性价比之王。按需稀疏校对，节省90%+Token"
    },
    "P2": {
        "name": "LLM Full Proof",
        "flow": ["llm_full_proof"],
        "scenario": "正式出版、文稿整理",
        "note": "高质量。全量滑动窗口，润色口语、修正逻辑"
    },

    # ========== 翻译层 ==========
    "T1": {
        "name": "LLM Full Trans",
        "flow": ["llm_full_translate"],
        "scenario": "跨语言内容",
        "note": "传统的全量翻译"
    },
    "T2": {
        "name": "LLM Partial Trans",
        "flow": ["llm_partial_translate"],
        "scenario": "教学重点标注",
        "note": "仅翻译用户指定的重点段落"
    }
}


# ========== 前端预设方案 ==========

FRONTEND_PRESETS = [
    {
        "id": "default",
        "name": "SenseVoice Only",
        "description": "极速模式，仅使用 SenseVoice 转录",
        "timeMultiplier": 0.1
    },
    {
        "id": "preset1",
        "name": "智能补刀",
        "description": "SV + Whisper 局部补刀，平衡速度与质量",
        "timeMultiplier": 0.15
    },
    {
        "id": "preset2",
        "name": "轻度校对",
        "description": "智能补刀 + LLM 按需校对问题片段",
        "timeMultiplier": 0.2
    },
    {
        "id": "preset3",
        "name": "深度校对",
        "description": "智能补刀 + LLM 全文精修润色",
        "timeMultiplier": 0.3
    },
    {
        "id": "preset4",
        "name": "校对+翻译",
        "description": "深度校对 + 全文翻译（同步处理）",
        "timeMultiplier": 0.5
    },
    {
        "id": "preset5",
        "name": "校对+重点翻译",
        "description": "深度校对 + 仅翻译标记的重点段落",
        "timeMultiplier": 0.35
    }
]
