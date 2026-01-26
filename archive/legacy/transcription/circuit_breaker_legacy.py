"""
Legacy CircuitBreaker Classes - 已被 FuseBreakerV2 替代

归档日期：V3.2.0+dev.20260125.09
原位置：backend/app/services/transcription_service.py (lines 32-311)
替代方案：backend/app/services/fuse_breaker.py (FuseBreakerV2)

这些类在旧架构中用于监控转录质量并触发人声分离升级。
新架构使用 FuseBreakerV2，在预处理阶段通过频谱分诊和事件标签权重进行决策。
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging


class BreakToGlobalSeparation(Exception):
    """熔断异常：触发时需要升级为全局人声分离模式"""
    pass


@dataclass
class CircuitBreakerState:
    """
    熔断器状态（支持模型升级）

    用于监控转录质量，当大量段落需要重试时：
    1. 优先尝试升级模型（如果允许且未达上限）
    2. 无法升级时才触发熔断
    """
    consecutive_retries: int = 0        # 连续重试计数
    total_retries: int = 0              # 总重试次数
    total_segments: int = 0             # 总段落数
    processed_segments: int = 0         # 已处理段落数

    # === Phase 3: 升级跟踪 ===
    escalation_count: int = 0                           # 已升级次数
    current_model: Optional[str] = None                 # 当前使用的模型
    escalation_history: List[str] = field(default_factory=list)  # 升级历史

    def record_retry(self):
        """记录一次重试"""
        self.consecutive_retries += 1
        self.total_retries += 1

    def record_success(self):
        """记录一次成功（重置连续计数）"""
        self.consecutive_retries = 0
        self.processed_segments += 1

    def record_escalation(self, new_model: str):
        """记录一次模型升级"""
        if self.current_model:
            self.escalation_history.append(f"{self.current_model} -> {new_model}")
        self.current_model = new_model
        self.escalation_count += 1
        self.consecutive_retries = 0

    def should_escalate(self, demucs_settings) -> bool:
        """判断是否应该升级模型（优先于熔断）"""
        if not demucs_settings.auto_escalation:
            return False
        if self.escalation_count >= demucs_settings.max_escalations:
            return False
        return self._check_break_condition(demucs_settings)

    def should_break(self, demucs_settings) -> bool:
        """判断是否应该触发熔断"""
        if not demucs_settings.circuit_breaker_enabled:
            return False
        if self.should_escalate(demucs_settings):
            return False
        return self._check_break_condition(demucs_settings)

    def _check_break_condition(self, demucs_settings) -> bool:
        """检查是否满足熔断/升级条件"""
        if self.consecutive_retries >= demucs_settings.consecutive_threshold:
            return True
        if self.processed_segments >= 5:
            retry_ratio = self.total_retries / self.processed_segments
            if retry_ratio >= demucs_settings.ratio_threshold:
                return True
        return False

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "consecutive_retries": self.consecutive_retries,
            "total_retries": self.total_retries,
            "total_segments": self.total_segments,
            "processed_segments": self.processed_segments,
            "retry_ratio": self.total_retries / max(1, self.processed_segments),
            "escalation_count": self.escalation_count,
            "current_model": self.current_model,
            "escalation_history": self.escalation_history,
        }


class CircuitBreakAction(Enum):
    """熔断后的处理动作"""
    CONTINUE = "continue"           # 继续处理，标记问题段落
    FALLBACK_ORIGINAL = "fallback"  # 降级使用原始音频
    FAIL = "fail"                   # 任务失败
    PAUSE = "pause"                 # 暂停等待人工介入


class CircuitBreakHandler:
    """熔断异常处理器 - 负责在熔断触发时执行用户配置的处理策略"""

    def __init__(self, job, settings):
        self.job = job
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.problem_segments: List[int] = []

    def handle(
        self,
        breaker_state: CircuitBreakerState,
        current_segment_idx: int,
        sse_manager=None
    ) -> CircuitBreakAction:
        """处理熔断异常"""
        action_str = self.settings.on_break
        try:
            action = CircuitBreakAction(action_str)
        except ValueError:
            self.logger.warning(f"无效的熔断处理策略: {action_str}，使用默认值 continue")
            action = CircuitBreakAction.CONTINUE

        self.problem_segments.append(current_segment_idx)

        if sse_manager:
            self._push_circuit_break_event(breaker_state, action, sse_manager)

        if action == CircuitBreakAction.FAIL:
            self.logger.error(f"熔断触发，任务终止。问题段落: {self.problem_segments}")
            raise BreakToGlobalSeparation(f"熔断触发，任务终止。问题段落: {self.problem_segments}")
        elif action == CircuitBreakAction.PAUSE:
            self.logger.warning(f"熔断触发，等待人工介入。问题段落: {self.problem_segments}")
            self.job.paused = True
            self.job.status = "paused"
            self.job.message = f"熔断触发，等待人工介入。问题段落: {self.problem_segments}"
            raise BreakToGlobalSeparation(self.job.message)
        else:
            self.logger.warning(f"熔断触发，采用 {action.value} 策略继续处理。问题段落: {self.problem_segments}")

        return action

    def get_problem_report(self) -> Dict:
        """获取问题报告"""
        return {
            "total_problem_segments": len(self.problem_segments),
            "problem_indices": self.problem_segments,
            "suggestion": self._get_suggestion()
        }

    def _get_suggestion(self) -> str:
        """根据问题段落数量给出建议"""
        count = len(self.problem_segments)
        if count == 0:
            return "所有段落处理正常"
        elif count <= 3:
            return "少量段落可能需要手动调整时间轴"
        elif count <= 10:
            return "建议检查这些段落的字幕准确性"
        else:
            return "大量段落有问题，建议使用更高质量的模型重新处理"

    def _push_circuit_break_event(self, state, action, sse_manager):
        """推送熔断处理事件"""
        try:
            sse_manager.push_event(
                self.job.job_id,
                "circuit_breaker_handled",
                {
                    "action": action.value,
                    "problem_segments": self.problem_segments,
                    "stats": state.get_stats(),
                    "suggestion": self._get_suggestion()
                }
            )
        except Exception as e:
            self.logger.debug(f"SSE推送失败（非致命）: {e}")
