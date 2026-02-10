# -*- coding: utf-8 -*-
"""
WER (Word Error Rate) 计算器。

中文按字分词，英文按空格分词。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class WERResult:
    """WER 计算结果。"""

    wer: float  # 词错误率 (0.0 - 1.0+)
    edit_distance: int
    reference_word_count: int
    hypothesis_word_count: int
    insertions: int = 0
    deletions: int = 0
    substitutions: int = 0


class WERCalculator:
    """WER 计算器。"""

    @staticmethod
    def tokenize(text: str, language: str = "zh") -> List[str]:
        """分词。中文按字，英文按空格。"""
        if language in ("zh", "ja", "ko", "yue"):
            # 中日韩：按字符分词，保留英文单词
            tokens = []
            current_word = ""
            for char in text:
                if re.match(r"[a-zA-Z0-9]", char):
                    current_word += char
                else:
                    if current_word:
                        tokens.append(current_word)
                        current_word = ""
                    if not char.isspace():
                        tokens.append(char)
            if current_word:
                tokens.append(current_word)
            return tokens
        else:
            # 英文等：按空格分词
            return text.split()

    @staticmethod
    def calculate(
        reference: str,
        hypothesis: str,
        language: str = "zh",
    ) -> WERResult:
        """计算 WER。"""
        ref_words = WERCalculator.tokenize(reference, language)
        hyp_words = WERCalculator.tokenize(hypothesis, language)

        ref_len = len(ref_words)
        hyp_len = len(hyp_words)

        if ref_len == 0:
            return WERResult(
                wer=1.0 if hyp_len > 0 else 0.0,
                edit_distance=hyp_len,
                reference_word_count=0,
                hypothesis_word_count=hyp_len,
                insertions=hyp_len,
            )

        d, ops = WERCalculator._edit_distance(ref_words, hyp_words)
        wer = d / ref_len if ref_len > 0 else 0.0

        return WERResult(
            wer=wer,
            edit_distance=d,
            reference_word_count=ref_len,
            hypothesis_word_count=hyp_len,
            insertions=ops["ins"],
            deletions=ops["del"],
            substitutions=ops["sub"],
        )

    @staticmethod
    def _edit_distance(ref: List[str], hyp: List[str]) -> tuple[int, dict]:
        """编辑距离计算。"""
        m, n = len(ref), len(hyp)
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
                        dp[i - 1][j],
                        dp[i][j - 1],
                        dp[i - 1][j - 1],
                    )

        # 回溯
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
