"""
Segmentation Diagnostic Service - 断句诊断服务

V3.2.0+dev.20260131: 用于追踪和诊断断句错误的全链路日志服务

当发现错误断句时，可以通过此服务快速判断问题出在哪个阶段：
1. VAD 切分 - 在语音中间切断
2. ASR 模型 - 模型自己截断或产生幻觉
3. 标点模型 - 错误添加句末标点
4. 标点后处理 - 后处理逻辑错误

使用方法：
1. 诊断模式默认启用，如需禁用：设置环境变量 SEGMENTATION_DIAGNOSTIC=0
2. 处理完成后，在 jobs/<job_id>/diagnostic/ 目录下查看诊断文件
3. 根据时间戳定位问题区间，对比各阶段输出
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class VADSegmentDiagnostic:
    """VAD 切分诊断信息"""
    index: int
    start: float  # 秒
    end: float    # 秒
    duration: float
    gap_before: Optional[float] = None  # 与前一段的间隔
    gap_after: Optional[float] = None   # 与后一段的间隔
    split_reason: Optional[str] = None  # 切分原因（硬上限/软上限/回溯断点等）
    rms_at_start: Optional[float] = None  # 起始点 RMS（越低越好）
    rms_at_end: Optional[float] = None    # 结束点 RMS（越低越好）
    is_force_split: bool = False  # 是否为强制拆分的结果


@dataclass
class ASROutputDiagnostic:
    """ASR 输出诊断信息"""
    chunk_index: int
    chunk_start: float  # Chunk 起始时间（秒）
    chunk_end: float    # Chunk 结束时间（秒）
    raw_text: str       # ASR 原始输出文本
    language: Optional[str] = None
    # 词级时间戳（如果有）
    word_timestamps: Optional[List[Dict[str, Any]]] = None
    # 模型信息
    model_name: Optional[str] = None
    # 是否检测到截断（文本末尾无标点但被切断）
    appears_truncated: bool = False


@dataclass
class PunctuationDiagnostic:
    """标点处理诊断信息"""
    chunk_index: int
    text_before_punctuation: str   # 标点模型处理前
    text_after_punctuation: str    # 标点模型处理后
    text_after_postprocess: str    # 后处理后
    # 标点变化详情
    punctuation_changes: List[Dict[str, Any]] = field(default_factory=list)
    # 后处理变化详情
    postprocess_changes: List[Dict[str, Any]] = field(default_factory=list)
    # 分句结果
    split_points: List[float] = field(default_factory=list)  # 分句点位置


@dataclass
class ChunkDiagnosticRecord:
    """单个 Chunk 的完整诊断记录"""
    chunk_index: int
    vad: Optional[VADSegmentDiagnostic] = None
    asr: Optional[ASROutputDiagnostic] = None
    punctuation: Optional[PunctuationDiagnostic] = None
    
    # 最终输出
    final_sentences: List[Dict[str, Any]] = field(default_factory=list)
    
    # 问题标记
    potential_issues: List[str] = field(default_factory=list)


class SegmentationDiagnosticService:
    """
    断句诊断服务
    
    提供全链路的断句诊断信息收集和分析功能。
    """
    
    def __init__(
        self,
        job_id: str,
        job_dir: Optional[Path] = None,
        enabled: Optional[bool] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化诊断服务
        
        Args:
            job_id: 任务 ID
            job_dir: 任务目录（用于保存诊断文件）
            enabled: 是否启用诊断。如果为 None，从环境变量读取
            logger: 日志记录器
        """
        self.job_id = job_id
        self.job_dir = job_dir
        self.logger = logger or logging.getLogger(__name__)
        
        # 从环境变量判断是否启用（默认启用）
        if enabled is None:
            self.enabled = os.environ.get("SEGMENTATION_DIAGNOSTIC", "1") != "0"
        else:
            self.enabled = enabled
        
        # 诊断记录
        self._records: Dict[int, ChunkDiagnosticRecord] = {}
        self._vad_segments: List[VADSegmentDiagnostic] = []
        
        if self.enabled:
            self.logger.info(f"[诊断] 断句诊断已启用 job_id={job_id}")
    
    def _get_or_create_record(self, chunk_index: int) -> ChunkDiagnosticRecord:
        """获取或创建 Chunk 诊断记录"""
        if chunk_index not in self._records:
            self._records[chunk_index] = ChunkDiagnosticRecord(chunk_index=chunk_index)
        return self._records[chunk_index]
    
    # ========== VAD 诊断 ==========
    
    def record_vad_segments(
        self,
        segments: List[Dict],
        split_reasons: Optional[Dict[int, str]] = None
    ):
        """
        记录 VAD 切分结果
        
        Args:
            segments: VAD 切分后的段列表
            split_reasons: 每段的切分原因（可选）
        """
        if not self.enabled:
            return
        
        split_reasons = split_reasons or {}
        
        for i, seg in enumerate(segments):
            gap_before = None
            gap_after = None
            
            if i > 0:
                gap_before = seg['start'] - segments[i - 1]['end']
            if i < len(segments) - 1:
                gap_after = segments[i + 1]['start'] - seg['end']
            
            diag = VADSegmentDiagnostic(
                index=i,
                start=seg['start'],
                end=seg['end'],
                duration=seg['end'] - seg['start'],
                gap_before=gap_before,
                gap_after=gap_after,
                split_reason=split_reasons.get(i),
                is_force_split=seg.get('_force_split', False)
            )
            self._vad_segments.append(diag)
            
            # 同时更新 Chunk 记录
            record = self._get_or_create_record(i)
            record.vad = diag
        
        self.logger.debug(f"[诊断] 记录了 {len(segments)} 个 VAD 段")
    
    def record_vad_rms(self, chunk_index: int, rms_at_start: float, rms_at_end: float):
        """记录 VAD 段边界的 RMS 值"""
        if not self.enabled:
            return
        
        record = self._get_or_create_record(chunk_index)
        if record.vad:
            record.vad.rms_at_start = rms_at_start
            record.vad.rms_at_end = rms_at_end
    
    # ========== ASR 诊断 ==========
    
    def record_asr_output(
        self,
        chunk_index: int,
        chunk_start: float,
        chunk_end: float,
        raw_text: str,
        language: Optional[str] = None,
        word_timestamps: Optional[List[Dict]] = None,
        model_name: Optional[str] = None
    ):
        """
        记录 ASR 原始输出
        
        Args:
            chunk_index: Chunk 索引
            chunk_start: Chunk 起始时间
            chunk_end: Chunk 结束时间
            raw_text: ASR 原始文本
            language: 检测到的语言
            word_timestamps: 词级时间戳
            model_name: 模型名称
        """
        if not self.enabled:
            return
        
        # 检测是否看起来被截断
        text_stripped = raw_text.strip()
        appears_truncated = False
        if text_stripped and not text_stripped[-1] in '.!?。！？':
            # 文本末尾无标点，可能被截断
            appears_truncated = True
        
        diag = ASROutputDiagnostic(
            chunk_index=chunk_index,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            raw_text=raw_text,
            language=language,
            word_timestamps=word_timestamps,
            model_name=model_name,
            appears_truncated=appears_truncated
        )
        
        record = self._get_or_create_record(chunk_index)
        record.asr = diag
        
        self.logger.debug(
            f"[诊断] Chunk {chunk_index} ASR: {len(raw_text)} chars, "
            f"truncated={appears_truncated}"
        )
    
    # ========== 标点诊断 ==========
    
    def record_punctuation(
        self,
        chunk_index: int,
        text_before: str,
        text_after: str,
        text_postprocessed: str,
        split_points: Optional[List[float]] = None
    ):
        """
        记录标点处理结果
        
        Args:
            chunk_index: Chunk 索引
            text_before: 标点处理前的文本
            text_after: 标点处理后的文本
            text_postprocessed: 后处理后的文本
            split_points: 分句点位置列表
        """
        if not self.enabled:
            return
        
        # 分析标点变化
        punctuation_changes = self._analyze_punctuation_changes(text_before, text_after)
        postprocess_changes = self._analyze_punctuation_changes(text_after, text_postprocessed)
        
        diag = PunctuationDiagnostic(
            chunk_index=chunk_index,
            text_before_punctuation=text_before,
            text_after_punctuation=text_after,
            text_after_postprocess=text_postprocessed,
            punctuation_changes=punctuation_changes,
            postprocess_changes=postprocess_changes,
            split_points=split_points or []
        )
        
        record = self._get_or_create_record(chunk_index)
        record.punctuation = diag
        
        self.logger.debug(
            f"[诊断] Chunk {chunk_index} 标点: "
            f"{len(punctuation_changes)} 处标点变化, "
            f"{len(postprocess_changes)} 处后处理变化"
        )
    
    def _analyze_punctuation_changes(
        self,
        text_before: str,
        text_after: str
    ) -> List[Dict[str, Any]]:
        """分析两个文本之间的标点差异"""
        changes = []
        
        # 简单的差异分析：找出标点变化的位置
        sentence_end_marks = set('.!?。！？')
        
        # 提取句末标点位置
        def get_punctuation_positions(text: str) -> List[int]:
            return [i for i, c in enumerate(text) if c in sentence_end_marks]
        
        before_positions = get_punctuation_positions(text_before)
        after_positions = get_punctuation_positions(text_after)
        
        # 找出新增的标点位置
        before_set = set(before_positions)
        after_set = set(after_positions)
        
        for pos in after_set - before_set:
            if pos < len(text_after):
                context_start = max(0, pos - 10)
                context_end = min(len(text_after), pos + 10)
                changes.append({
                    "type": "added",
                    "position": pos,
                    "char": text_after[pos],
                    "context": text_after[context_start:context_end]
                })
        
        for pos in before_set - after_set:
            if pos < len(text_before):
                context_start = max(0, pos - 10)
                context_end = min(len(text_before), pos + 10)
                changes.append({
                    "type": "removed",
                    "position": pos,
                    "char": text_before[pos],
                    "context": text_before[context_start:context_end]
                })
        
        return changes
    
    # ========== 最终结果诊断 ==========
    
    def record_final_sentences(
        self,
        chunk_index: int,
        sentences: List[Dict[str, Any]]
    ):
        """记录最终的分句结果"""
        if not self.enabled:
            return
        
        record = self._get_or_create_record(chunk_index)
        record.final_sentences = sentences
    
    # ========== 问题检测 ==========
    
    def analyze_potential_issues(self):
        """分析潜在问题并标记"""
        if not self.enabled:
            return
        
        for chunk_index, record in self._records.items():
            issues = []
            
            # 检查 VAD 问题
            if record.vad:
                # 边界 RMS 过高
                if record.vad.rms_at_end and record.vad.rms_at_end > 0.1:
                    issues.append(f"VAD: 结束点 RMS 过高 ({record.vad.rms_at_end:.4f})，可能切在语音中间")
                if record.vad.rms_at_start and record.vad.rms_at_start > 0.1:
                    issues.append(f"VAD: 起始点 RMS 过高 ({record.vad.rms_at_start:.4f})，可能切在语音中间")
                # 时长过短
                if record.vad.duration < 1.0:
                    issues.append(f"VAD: 时长过短 ({record.vad.duration:.2f}s)，可能是碎片")
            
            # 检查 ASR 问题
            if record.asr:
                if record.asr.appears_truncated:
                    issues.append("ASR: 文本看起来被截断（末尾无标点）")
                if record.asr.raw_text and len(record.asr.raw_text.strip()) < 5:
                    issues.append(f"ASR: 输出过短 ({len(record.asr.raw_text)} chars)")
            
            # 检查标点问题
            if record.punctuation:
                # 标点模型添加了句末标点但后处理移除了
                for change in record.punctuation.postprocess_changes:
                    if change['type'] == 'removed' and change['char'] in '.!?。！？':
                        issues.append(f"后处理: 移除了标点模型添加的句末标点 at {change['position']}")
                
                # 标点模型没有添加句末标点但后处理添加了
                for change in record.punctuation.postprocess_changes:
                    if change['type'] == 'added' and change['char'] in '.!?。！？':
                        issues.append(f"后处理: 添加了句末标点 at {change['position']}")
            
            record.potential_issues = issues
    
    # ========== 导出 ==========
    
    def export_to_file(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        导出诊断信息到文件
        
        Args:
            output_path: 输出路径。如果为 None，使用默认路径
        
        Returns:
            导出的文件路径
        """
        if not self.enabled:
            return None
        
        # 分析潜在问题
        self.analyze_potential_issues()
        
        # 确定输出路径
        if output_path is None:
            if self.job_dir:
                diag_dir = self.job_dir / "diagnostic"
                diag_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = diag_dir / f"segmentation_diagnostic_{timestamp}.json"
            else:
                self.logger.warning("[诊断] 无法导出：未指定 job_dir 或 output_path")
                return None
        
        # 构建导出数据
        export_data = {
            "job_id": self.job_id,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_chunks": len(self._records),
                "chunks_with_issues": sum(
                    1 for r in self._records.values() if r.potential_issues
                ),
                "total_issues": sum(
                    len(r.potential_issues) for r in self._records.values()
                )
            },
            "vad_segments": [asdict(seg) for seg in self._vad_segments],
            "chunk_records": {
                str(k): self._record_to_dict(v) for k, v in self._records.items()
            }
        }
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"[诊断] 诊断信息已导出到: {output_path}")
        return output_path
    
    def _record_to_dict(self, record: ChunkDiagnosticRecord) -> Dict:
        """将 ChunkDiagnosticRecord 转换为字典"""
        return {
            "chunk_index": record.chunk_index,
            "vad": asdict(record.vad) if record.vad else None,
            "asr": asdict(record.asr) if record.asr else None,
            "punctuation": asdict(record.punctuation) if record.punctuation else None,
            "final_sentences": record.final_sentences,
            "potential_issues": record.potential_issues
        }
    
    def get_quick_summary(self) -> str:
        """获取快速摘要（用于日志输出）"""
        if not self.enabled:
            return "诊断未启用"
        
        self.analyze_potential_issues()
        
        lines = [f"=== 断句诊断摘要 (job_id={self.job_id}) ==="]
        
        # 统计各类问题
        issue_counts = {}
        for record in self._records.values():
            for issue in record.potential_issues:
                category = issue.split(":")[0]
                issue_counts[category] = issue_counts.get(category, 0) + 1
        
        if issue_counts:
            lines.append("问题分布:")
            for category, count in sorted(issue_counts.items()):
                lines.append(f"  - {category}: {count} 处")
        else:
            lines.append("未检测到明显问题")
        
        # 列出有问题的 Chunk
        problem_chunks = [
            (k, v) for k, v in self._records.items() if v.potential_issues
        ]
        if problem_chunks:
            lines.append(f"\n有问题的 Chunk ({len(problem_chunks)} 个):")
            for chunk_index, record in problem_chunks[:5]:  # 最多显示5个
                lines.append(f"  Chunk {chunk_index}:")
                for issue in record.potential_issues[:3]:  # 每个最多3个问题
                    lines.append(f"    - {issue}")
            if len(problem_chunks) > 5:
                lines.append(f"  ... 还有 {len(problem_chunks) - 5} 个有问题的 Chunk")
        
        return "\n".join(lines)


# ========== 便捷函数 ==========

_diagnostic_instances: Dict[str, SegmentationDiagnosticService] = {}


def get_diagnostic_service(
    job_id: str,
    job_dir: Optional[Path] = None,
    enabled: Optional[bool] = None
) -> SegmentationDiagnosticService:
    """
    获取或创建诊断服务实例
    
    Args:
        job_id: 任务 ID
        job_dir: 任务目录
        enabled: 是否启用
    
    Returns:
        SegmentationDiagnosticService 实例
    """
    if job_id not in _diagnostic_instances:
        _diagnostic_instances[job_id] = SegmentationDiagnosticService(
            job_id=job_id,
            job_dir=job_dir,
            enabled=enabled
        )
    return _diagnostic_instances[job_id]


def cleanup_diagnostic_service(job_id: str):
    """清理诊断服务实例"""
    if job_id in _diagnostic_instances:
        del _diagnostic_instances[job_id]
