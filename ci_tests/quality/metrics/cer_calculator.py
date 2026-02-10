# -*- coding: utf-8 -*-
"""
CER (Character Error Rate) 计算器。

基于 Wagner-Fischer 编辑距离算法，纯 Python 实现。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class CERResult:
    """CER 计算结果。"""

    cer: float  # 字符错误率 (0.0 - 1.0+)
    edit_distance: int  # 编辑距离
    reference_length: int  # 参考文本长度
    hypothesis_length: int  # 假设文本长度
    insertions: int = 0
    deletions: int = 0
    substitutions: int = 0


class CERCalculator:
    """CER 计算器，基于编辑距离。"""

    @staticmethod
    def calculate(reference: str, hypothesis: str) -> CERResult:
        """计算两个字符串之间的 CER。

        Args:
            reference: 参考文本（标准答案）
            hypothesis: 假设文本（待评估）

        Returns:
            CERResult
        """
        ref_chars = list(reference.replace(" ", ""))
        hyp_chars = list(hypothesis.replace(" ", ""))

        ref_len = len(ref_chars)
        hyp_len = len(hyp_chars)

        if ref_len == 0:
            return CERResult(
                cer=1.0 if hyp_len > 0 else 0.0,
                edit_distance=hyp_len,
                reference_length=0,
                hypothesis_length=hyp_len,
                insertions=hyp_len,
            )

        # Wagner-Fischer 算法
        d, ops = CERCalculator._edit_distance_with_ops(ref_chars, hyp_chars)

        cer = d / ref_len if ref_len > 0 else 0.0

        return CERResult(
            cer=cer,
            edit_distance=d,
            reference_length=ref_len,
            hypothesis_length=hyp_len,
            insertions=ops["ins"],
            deletions=ops["del"],
            substitutions=ops["sub"],
        )

    @staticmethod
    def _edit_distance_with_ops(
        ref: List[str], hyp: List[str]
    ) -> tuple[int, dict]:
        """计算编辑距离并统计操作类型。"""
        m, n = len(ref), len(hyp)

        # DP 表
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i - 1] == hyp[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # 删除
                        dp[i][j - 1],      # 插入
                        dp[i - 1][j - 1],  # 替换
                    )

        # 回溯统计操作
        ops = {"ins": 0, "del": 0, "sub": 0}
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                ops["sub"] += 1
                i -= 1
                j -= 1
            elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
                ops["ins"] += 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                ops["del"] += 1
                i -= 1
            else:
                break

        return dp[m][n], ops

    @staticmethod
    def calculate_per_segment(
        ref_segments: List[str],
        hyp_segments: List[str],
    ) -> List[CERResult]:
        """逐段计算 CER。"""
        results = []
        count = min(len(ref_segments), len(hyp_segments))
        for i in range(count):
            results.append(
                CERCalculator.calculate(ref_segments[i], hyp_segments[i])
            )
        return results
