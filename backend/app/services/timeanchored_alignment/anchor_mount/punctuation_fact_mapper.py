"""把 char-space 标点证据映射到 token-unit space。"""

from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.contracts import (
    PunctuationFact,
    PunctuationMappingDiagnostics,
    PunctuationPairState,
)
from app.services.timeanchored_alignment.preparation.contracts import (
    PreparedTokenUnit,
    PunctuationEvidence,
    SlowWindowTextPackage,
)


_SENTENCE_END_MARKS = {"。", "！", "？", ".", "!", "?"}
_WEAK_MARKS = {"，", "、", ",", ";", "；", ":", "："}
_OPEN_QUOTES = {"“": ("quote", "”"), "\"": ("quote", "\""), "‘": ("quote", "’")}
_OPEN_BRACKETS = {"(": ("bracket", ")"), "（": ("bracket", "）"), "《": ("bracket", "》"), "【": ("bracket", "】")}
_CLOSE_QUOTES = {"”", "\"", "’"}
_CLOSE_BRACKETS = {")", "）", "》", "】"}


def _classify_mark(mark: str) -> str:
    if mark in _SENTENCE_END_MARKS:
        return "sentence_end"
    if mark in _WEAK_MARKS:
        return "weak"
    if mark in _OPEN_QUOTES or mark in _CLOSE_QUOTES:
        return "quote"
    if mark in _OPEN_BRACKETS or mark in _CLOSE_BRACKETS:
        return "bracket"
    return "other"


def _token_for_char(token_units: tuple[PreparedTokenUnit, ...], char_index: int) -> int | None:
    for index, token_unit in enumerate(token_units):
        if token_unit.char_start <= char_index < token_unit.char_end:
            return index
    return None


def _next_token(token_units: tuple[PreparedTokenUnit, ...], char_index: int) -> int | None:
    for index, token_unit in enumerate(token_units):
        if char_index < token_unit.char_start:
            return index
    return None


def _previous_token(token_units: tuple[PreparedTokenUnit, ...], char_index: int) -> int | None:
    candidate: int | None = None
    for index, token_unit in enumerate(token_units):
        if token_unit.char_end <= char_index:
            candidate = index
            continue
        break
    return candidate


class PunctuationFactMapper:
    """把 Preparation 标点证据映射到 token-unit 域。"""

    def map(
        self,
        *,
        window_text: SlowWindowTextPackage,
        token_units: tuple[PreparedTokenUnit, ...],
        punctuation_evidences: tuple[PunctuationEvidence, ...],
    ) -> tuple[
        tuple[PunctuationFact, ...],
        tuple[PunctuationPairState, ...],
        PunctuationMappingDiagnostics,
    ]:
        facts: list[PunctuationFact] = []
        pair_states: list[PunctuationPairState] = []
        open_stack: dict[str, list[tuple[str, str]]] = {"quote": [], "bracket": []}
        unmapped: list[dict[str, object]] = []
        dropped: list[dict[str, object]] = []

        for evidence_index, evidence in enumerate(punctuation_evidences):
            mark = str(evidence.mark or "")
            if not mark:
                dropped.append(
                    {
                        "evidence_index": evidence_index,
                        "reason": "empty_mark",
                    }
                )
                continue
            char_index = int(evidence.source_char_index)
            if char_index < 0:
                dropped.append(
                    {
                        "evidence_index": evidence_index,
                        "mark": mark,
                        "reason": "negative_char_index",
                    }
                )
                continue
            if char_index > len(str(window_text.text or "")):
                unmapped.append(
                    {
                        "evidence_index": evidence_index,
                        "mark": mark,
                        "char_index": char_index,
                        "attach_side": str(evidence.attach_side or ""),
                        "reason": "char_index_out_of_window_text",
                    }
                )
                continue
            token_index = _token_for_char(token_units, char_index)
            next_token_index = _next_token(token_units, char_index)
            previous_token_index = _previous_token(token_units, char_index)
            attach_mode = "standalone"
            left_token_index: int | None = None
            right_token_index: int | None = None
            if evidence.attach_side == "after":
                if token_index is not None:
                    left_token_index = token_index
                    right_token_index = (
                        token_index + 1 if token_index + 1 < len(token_units) else None
                    )
                    attach_mode = "between" if right_token_index is not None else "trailing"
                elif next_token_index is not None:
                    right_token_index = next_token_index
                    attach_mode = "leading"
                elif previous_token_index is not None:
                    left_token_index = previous_token_index
                    attach_mode = "trailing"
            else:
                if next_token_index is not None:
                    right_token_index = next_token_index
                    left_token_index = next_token_index - 1 if next_token_index > 0 else None
                    attach_mode = "between" if left_token_index is not None else "leading"
                elif token_index is not None:
                    right_token_index = token_index
                    attach_mode = "leading"
                elif previous_token_index is not None:
                    left_token_index = previous_token_index
                    attach_mode = "trailing"

            if left_token_index is None and right_token_index is None:
                unmapped.append(
                    {
                        "evidence_index": evidence_index,
                        "mark": mark,
                        "char_index": char_index,
                        "attach_side": str(evidence.attach_side or ""),
                        "reason": "no_token_mapping",
                    }
                )
                continue

            punct_class = _classify_mark(mark)
            fact_id = f"punct-{evidence_index}"
            group_id: str | None = None
            if mark in _OPEN_QUOTES:
                group_id = f"quote-{len(pair_states)}"
                open_stack["quote"].append((group_id, fact_id))
            elif mark in _OPEN_BRACKETS:
                group_id = f"bracket-{len(pair_states)}"
                open_stack["bracket"].append((group_id, fact_id))
            elif punct_class == "quote" and open_stack["quote"]:
                group_id, open_fact_id = open_stack["quote"].pop()
                pair_states.append(
                    PunctuationPairState(
                        group_id=group_id,
                        pair_kind="quote",
                        open_fact_id=open_fact_id,
                        close_fact_id=fact_id,
                        state="closed",
                    )
                )
            elif punct_class == "bracket" and open_stack["bracket"]:
                group_id, open_fact_id = open_stack["bracket"].pop()
                pair_states.append(
                    PunctuationPairState(
                        group_id=group_id,
                        pair_kind="bracket",
                        open_fact_id=open_fact_id,
                        close_fact_id=fact_id,
                        state="closed",
                    )
                )

            facts.append(
                PunctuationFact(
                    fact_id=fact_id,
                    left_token_index=left_token_index,
                    right_token_index=right_token_index,
                    attach_mode=attach_mode,
                    normalized_text=mark,
                    punct_class=punct_class,
                    confidence=1.0,
                    source=str(evidence.evidence_source or "slow"),
                    group_id=group_id,
                    boundary_weight=1.0 if punct_class == "sentence_end" else 0.6,
                    render_default=True,
                    metadata={
                        "source_char_index": char_index,
                        "window_text": window_text.text,
                    },
                )
            )

        for pair_kind, stack in open_stack.items():
            for group_id, open_fact_id in stack:
                pair_states.append(
                    PunctuationPairState(
                        group_id=group_id,
                        pair_kind=pair_kind,
                        open_fact_id=open_fact_id,
                        close_fact_id=None,
                        state="open",
                    )
                )
        return (
            tuple(facts),
            tuple(pair_states),
            PunctuationMappingDiagnostics(
                unmapped=tuple(unmapped),
                dropped_with_reason=tuple(dropped),
            ),
        )
