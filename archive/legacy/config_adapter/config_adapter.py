"""
配置适配器 (Config Adapter)

统一新旧配置的访问接口，解决 v3.5 重构中新旧配置混用的问题。

映射关系:
- 旧版 sensevoice.enhancement -> 新版 transcription.transcription_profile
  - off -> sensevoice_only
  - smart_patch -> sv_whisper_patch
  - deep_listen -> sv_whisper_dual

- 旧版 demucs.mode -> 新版 preprocessing.demucs_strategy
  - auto -> auto
  - always -> force_on
  - never -> off

- 旧版 sensevoice.preset_id -> 新版 preset_id (顶层)

使用方式:
    from app.services.config_adapter import ConfigAdapter

    # 获取转录流水线模式
    profile = ConfigAdapter.get_transcription_profile(job.settings)

    # 判断是否需要双流对齐
    use_dual = ConfigAdapter.needs_dual_alignment(job.settings)
"""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.models.job_models import JobSettings


class ConfigAdapter:
    """
    新旧配置适配器 - 统一配置访问接口

    所有配置读取都应通过此适配器，确保新旧配置的兼容性。
    优先读取新版配置，回退到旧版配置。
    """

    # ========== 转录流水线相关 ==========

    @staticmethod
    def get_transcription_profile(settings: "JobSettings") -> str:
        """
        获取转录流水线模式

        Returns:
            str: sensevoice_only / sv_whisper_patch / sv_whisper_dual
        """
        # 优先新版: transcription.transcription_profile
        if hasattr(settings, 'transcription') and settings.transcription:
            profile = getattr(settings.transcription, 'transcription_profile', None)
            if profile and profile != 'sensevoice_only':
                # 新版配置有效且非默认值，直接使用
                return profile

        # 回退旧版: sensevoice.enhancement -> profile 映射
        if hasattr(settings, 'sensevoice') and settings.sensevoice:
            enhancement = getattr(settings.sensevoice, 'enhancement', 'off')
            mapping = {
                'off': 'sensevoice_only',
                'smart_patch': 'sv_whisper_patch',
                'deep_listen': 'sv_whisper_dual'
            }
            return mapping.get(enhancement, 'sensevoice_only')

        return 'sensevoice_only'

    @staticmethod
    def needs_dual_alignment(settings: "JobSettings") -> bool:
        """
        判断是否需要双流对齐流水线（新架构）

        V3.1.0: 所有 SenseVoice 模式都走新架构（支持 ProgressEmitter）
        - sensevoice_only (极速模式)
        - sv_whisper_patch (智能补刀)
        - sv_whisper_dual (双流精校)

        Returns:
            bool: 是否使用新架构流水线
        """
        engine = getattr(settings, 'engine', 'sensevoice')
        if engine != 'sensevoice':
            return False

        profile = ConfigAdapter.get_transcription_profile(settings)
        # V3.1.0: 所有 SenseVoice 模式都走新架构
        return profile in ['sensevoice_only', 'sv_whisper_patch', 'sv_whisper_dual']

    @staticmethod
    def get_preset_id(settings: "JobSettings") -> str:
        """
        获取预设 ID

        Returns:
            str: 预设 ID (default/preset1-5/fast/balanced/quality/custom)
        """
        # 优先新版顶层 preset_id
        new_preset = getattr(settings, 'preset_id', None)
        if new_preset and new_preset != 'balanced':
            # 新版配置有效且非默认值
            return new_preset

        # 回退旧版: sensevoice.preset_id
        if hasattr(settings, 'sensevoice') and settings.sensevoice:
            old_preset = getattr(settings.sensevoice, 'preset_id', 'default')
            if old_preset != 'default':
                return old_preset

        # 返回新版默认值
        return new_preset or 'balanced'

    # ========== Demucs 预处理相关 ==========

    @staticmethod
    def get_demucs_strategy(settings: "JobSettings") -> str:
        """
        获取 Demucs 人声分离策略

        Returns:
            str: off / auto / force_on
        """
        # 优先新版: preprocessing.demucs_strategy
        if hasattr(settings, 'preprocessing') and settings.preprocessing:
            strategy = getattr(settings.preprocessing, 'demucs_strategy', None)
            if strategy:
                return strategy

        # 回退旧版: demucs.enabled + demucs.mode
        if hasattr(settings, 'demucs') and settings.demucs:
            if not getattr(settings.demucs, 'enabled', True):
                return 'off'
            mode = getattr(settings.demucs, 'mode', 'auto')
            # 旧版 mode 映射到新版 strategy
            mode_mapping = {
                'auto': 'auto',
                'always': 'force_on',
                'never': 'off',
                'on_demand': 'auto'
            }
            return mode_mapping.get(mode, 'auto')

        return 'auto'

    @staticmethod
    def is_demucs_enabled(settings: "JobSettings") -> bool:
        """
        判断 Demucs 是否启用

        Returns:
            bool: 是否启用 Demucs
        """
        strategy = ConfigAdapter.get_demucs_strategy(settings)
        return strategy != 'off'

    @staticmethod
    def get_demucs_model(settings: "JobSettings") -> str:
        """
        获取 Demucs 模型名称

        Returns:
            str: htdemucs / htdemucs_ft / mdx_q / mdx_extra
        """
        # 优先新版
        if hasattr(settings, 'preprocessing') and settings.preprocessing:
            model = getattr(settings.preprocessing, 'demucs_model', None)
            if model:
                return model

        # 回退旧版 (使用 weak_model 作为默认)
        if hasattr(settings, 'demucs') and settings.demucs:
            return getattr(settings.demucs, 'weak_model', 'htdemucs')

        return 'htdemucs'

    @staticmethod
    def get_max_escalations(settings: "JobSettings") -> int:
        """
        获取 Demucs 最大升级次数

        Returns:
            int: 最大升级次数
        """
        # 新版没有这个字段，直接读旧版
        if hasattr(settings, 'demucs') and settings.demucs:
            return getattr(settings.demucs, 'max_escalations', 1)

        # 新版默认值: 基于 demucs_shifts
        if hasattr(settings, 'preprocessing') and settings.preprocessing:
            shifts = getattr(settings.preprocessing, 'demucs_shifts', 1)
            return max(1, shifts)

        return 1

    # ========== 阈值相关 ==========

    @staticmethod
    def get_patching_threshold(settings: "JobSettings") -> float:
        """
        获取 Whisper 补刀触发阈值

        Returns:
            float: 阈值 (0.0-1.0)
        """
        # 优先新版
        if hasattr(settings, 'transcription') and settings.transcription:
            threshold = getattr(settings.transcription, 'patching_threshold', None)
            if threshold is not None:
                return threshold

        # 回退旧版
        if hasattr(settings, 'sensevoice') and settings.sensevoice:
            return getattr(settings.sensevoice, 'whisper_patch_threshold', 0.5)

        return 0.6

    @staticmethod
    def get_confidence_threshold(settings: "JobSettings") -> float:
        """
        获取置信度阈值

        Returns:
            float: 阈值 (0.0-1.0)
        """
        # 优先新版 (使用 patching_threshold)
        if hasattr(settings, 'transcription') and settings.transcription:
            threshold = getattr(settings.transcription, 'patching_threshold', None)
            if threshold is not None:
                return threshold

        # 回退旧版
        if hasattr(settings, 'sensevoice') and settings.sensevoice:
            return getattr(settings.sensevoice, 'confidence_threshold', 0.6)

        return 0.6

    # ========== Whisper 模型相关 ==========

    @staticmethod
    def get_whisper_model(settings: "JobSettings") -> str:
        """
        获取 Whisper 模型名称

        Returns:
            str: tiny / small / medium / large-v3
        """
        # 优先新版
        if hasattr(settings, 'transcription') and settings.transcription:
            model = getattr(settings.transcription, 'whisper_model', None)
            if model:
                return model

        # 回退旧版顶层 model 字段
        return getattr(settings, 'model', 'medium')

    # ========== LLM 润色相关 ==========

    @staticmethod
    def get_llm_task(settings: "JobSettings") -> str:
        """
        获取 LLM 任务类型

        Returns:
            str: off / proofread / translate
        """
        # 优先新版
        if hasattr(settings, 'refinement') and settings.refinement:
            task = getattr(settings.refinement, 'llm_task', None)
            if task:
                return task

        # 回退旧版: 根据 sensevoice.proofread 和 translate 推断
        if hasattr(settings, 'sensevoice') and settings.sensevoice:
            translate = getattr(settings.sensevoice, 'translate', 'off')
            if translate != 'off':
                return 'translate'
            proofread = getattr(settings.sensevoice, 'proofread', 'off')
            if proofread != 'off':
                return 'proofread'

        return 'off'

    @staticmethod
    def get_llm_scope(settings: "JobSettings") -> str:
        """
        获取 LLM 介入范围

        Returns:
            str: sparse / global
        """
        # 优先新版
        if hasattr(settings, 'refinement') and settings.refinement:
            scope = getattr(settings.refinement, 'llm_scope', None)
            if scope:
                return scope

        # 回退旧版: 根据 sensevoice.proofread 推断
        if hasattr(settings, 'sensevoice') and settings.sensevoice:
            proofread = getattr(settings.sensevoice, 'proofread', 'off')
            if proofread == 'full':
                return 'global'
            elif proofread == 'sparse':
                return 'sparse'

        return 'sparse'

    # ========== 语言相关 ==========

    @staticmethod
    def get_target_language(settings: "JobSettings") -> str:
        """
        获取目标语言

        Returns:
            str: 语言代码 (zh/en/ja 等)
        """
        # 优先新版
        if hasattr(settings, 'refinement') and settings.refinement:
            lang = getattr(settings.refinement, 'target_language', None)
            if lang:
                return lang

        # 回退旧版
        if hasattr(settings, 'sensevoice') and settings.sensevoice:
            return getattr(settings.sensevoice, 'target_language', 'zh')

        return 'zh'

    # ========== 调试辅助 ==========

    @staticmethod
    def get_config_source(settings: "JobSettings") -> str:
        """
        判断配置来源 (用于调试)

        Returns:
            str: 'new' / 'legacy' / 'mixed'
        """
        has_new = False
        has_legacy = False

        # 检查新版配置
        if hasattr(settings, 'transcription') and settings.transcription:
            profile = getattr(settings.transcription, 'transcription_profile', 'sensevoice_only')
            if profile != 'sensevoice_only':
                has_new = True

        if hasattr(settings, 'preprocessing') and settings.preprocessing:
            strategy = getattr(settings.preprocessing, 'demucs_strategy', 'auto')
            if strategy != 'auto':
                has_new = True

        # 检查旧版配置
        if hasattr(settings, 'sensevoice') and settings.sensevoice:
            enhancement = getattr(settings.sensevoice, 'enhancement', 'off')
            preset_id = getattr(settings.sensevoice, 'preset_id', 'default')
            if enhancement != 'off' or preset_id != 'default':
                has_legacy = True

        if has_new and has_legacy:
            return 'mixed'
        elif has_new:
            return 'new'
        elif has_legacy:
            return 'legacy'
        else:
            return 'default'

    @staticmethod
    def to_unified_dict(settings: "JobSettings") -> dict:
        """
        将配置转换为统一格式的字典 (用于调试和日志)

        Returns:
            dict: 统一格式的配置字典
        """
        return {
            'source': ConfigAdapter.get_config_source(settings),
            'transcription_profile': ConfigAdapter.get_transcription_profile(settings),
            'needs_dual_alignment': ConfigAdapter.needs_dual_alignment(settings),
            'preset_id': ConfigAdapter.get_preset_id(settings),
            'demucs_strategy': ConfigAdapter.get_demucs_strategy(settings),
            'demucs_model': ConfigAdapter.get_demucs_model(settings),
            'whisper_model': ConfigAdapter.get_whisper_model(settings),
            'patching_threshold': ConfigAdapter.get_patching_threshold(settings),
            'llm_task': ConfigAdapter.get_llm_task(settings),
            'llm_scope': ConfigAdapter.get_llm_scope(settings),
            'target_language': ConfigAdapter.get_target_language(settings),
        }
