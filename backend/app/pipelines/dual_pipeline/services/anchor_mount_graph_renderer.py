"""
AnchorMount 挂载图渲染器。
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, Iterable, List


class AnchorMountGraphRenderer:
    """将 anchor_mount 结果渲染为图结构与 SVG。"""

    SUCCESS_STATUSES = {"anchored", "merged"}
    FALLBACK_STATUSES = {"inferred"}

    def build_graph_payload(
        self,
        *,
        window_id: str,
        items: Iterable[Any],
        envelopes: Iterable[Any],
        boundary_evidences: Iterable[Any],
        hooks: Iterable[Any],
        metrics: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        hook_index = {
            str(getattr(hook, "hook_id", "")): {
                "hook_id": str(getattr(hook, "hook_id", "")),
                "text": str(getattr(hook, "hook_text", "") or ""),
            }
            for hook in hooks or ()
        }
        envelope_rows = list(envelopes or ())
        boundary_after_by_split_idx: Dict[int, List[Dict[str, Any]]] = {}
        for item in boundary_evidences or ():
            raw_split_idx = getattr(item, "split_idx", -1)
            split_idx = int(raw_split_idx) if raw_split_idx is not None else -1
            if split_idx < 0:
                continue
            boundary_after_by_split_idx.setdefault(split_idx, []).append(
                {
                    "split_idx": split_idx,
                    "event_time": float(getattr(item, "event_time", 0.0) or 0.0),
                    "left_end": float(getattr(item, "left_end", 0.0) or 0.0),
                    "right_start": float(getattr(item, "right_start", 0.0) or 0.0),
                    "reason": str(getattr(item, "reason", "") or ""),
                    "score": float(getattr(item, "score", 0.0) or 0.0),
                    "hard_flag": bool(getattr(item, "hard_flag", False)),
                    "metadata": dict(getattr(item, "metadata", {}) or {}),
                }
            )
        mounts: List[Dict[str, Any]] = []
        for index, item in enumerate(items or ()):
            mount_status = str(getattr(item, "mount_status", "") or "")
            if mount_status in self.SUCCESS_STATUSES:
                classification = "success"
            elif mount_status in self.FALLBACK_STATUSES:
                classification = "fallback"
            else:
                classification = "unresolved"
            hook_ids = [str(v) for v in (getattr(item, "source_hook_ids", ()) or ())]
            envelope = envelope_rows[index] if index < len(envelope_rows) else None
            mounts.append(
                {
                    "unit_id": str(getattr(item, "unit_id", "") or ""),
                    "unit_index": int(getattr(item, "unit_index", 0) or 0),
                    "text": str(getattr(item, "display_text", "") or ""),
                    "mount_status": mount_status,
                    "classification": classification,
                    "anchor_kind": str(getattr(item, "anchor_kind", "") or ""),
                    "hook_ids": hook_ids,
                    "hooks": [hook_index.get(hook_id, {"hook_id": hook_id, "text": ""}) for hook_id in hook_ids],
                    "source_chunk_ids": [
                        str(chunk_id)
                        for chunk_id in (getattr(item, "source_chunk_ids", ()) or ())
                    ],
                    "source_chunk_indices": [
                        int(chunk_index)
                        for chunk_index in (getattr(item, "source_chunk_indices", ()) or ())
                    ],
                    "alignment_block_id": str(getattr(item, "alignment_block_id", "") or "") or None,
                    "match_confidence": float(getattr(item, "match_confidence", 0.0) or 0.0),
                    "envelope": None
                    if envelope is None
                    else {
                        "kind": str(getattr(envelope, "envelope_kind", "") or ""),
                        "provisional_start": float(
                            getattr(envelope, "provisional_start", 0.0) or 0.0
                        ),
                        "provisional_end": float(
                            getattr(envelope, "provisional_end", 0.0) or 0.0
                        ),
                        "left_bound": float(getattr(envelope, "left_bound", 0.0) or 0.0),
                        "right_bound": float(getattr(envelope, "right_bound", 0.0) or 0.0),
                        "source_chunk_ids": [
                            str(chunk_id)
                            for chunk_id in (getattr(envelope, "source_chunk_ids", ()) or ())
                        ],
                        "source_chunk_indices": [
                            int(chunk_index)
                            for chunk_index in (getattr(envelope, "source_chunk_indices", ()) or ())
                        ],
                    },
                    "boundary_after": list(
                        boundary_after_by_split_idx.get(int(getattr(item, "unit_index", 0)), [])
                    ),
                }
            )
        mounts.sort(key=lambda row: int(row["unit_index"]))
        summary = {
            "total": len(mounts),
            "success": sum(1 for row in mounts if row["classification"] == "success"),
            "fallback": sum(1 for row in mounts if row["classification"] == "fallback"),
            "unresolved": sum(1 for row in mounts if row["classification"] == "unresolved"),
        }
        return {
            "window_id": str(window_id or ""),
            "summary": summary,
            "metrics": dict(metrics or {}),
            "mounts": mounts,
        }

    def render_svg(self, graph: Dict[str, Any]) -> str:
        mounts = list(graph.get("mounts") or [])
        width = 1260
        header_height = 92
        row_gap = 8
        line_height = 14
        row_top_padding = 18
        row_bottom_padding = 10

        row_blocks: List[Dict[str, Any]] = []
        for row in mounts:
            lines = self._build_row_lines(row)
            row_height = row_top_padding + len(lines) * line_height + row_bottom_padding
            row_blocks.append(
                {
                    "row": row,
                    "lines": lines,
                    "row_height": max(44, row_height),
                }
            )
        body_height = sum(int(item["row_height"]) + row_gap for item in row_blocks) or 52
        height = header_height + body_height + 16
        svg_lines: List[str] = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
            "<style>",
            ".title{font:700 18px 'Segoe UI';fill:#1f2937;}",
            ".meta{font:12px 'Segoe UI';fill:#4b5563;}",
            ".text{font:12px 'Consolas';fill:#111827;}",
            ".chip{font:11px 'Segoe UI';fill:#ffffff;}",
            "</style>",
            "<rect x='0' y='0' width='100%' height='100%' fill='#f8fafc'/>",
            f"<text x='20' y='28' class='title'>Anchor Mount Graph - window={self._escape(graph.get('window_id', ''))}</text>",
        ]
        summary = graph.get("summary", {}) or {}
        summary_text = (
            f"success={int(summary.get('success', 0))}  "
            f"fallback={int(summary.get('fallback', 0))}  "
            f"unresolved={int(summary.get('unresolved', 0))}"
        )
        svg_lines.append(f"<text x='20' y='50' class='meta'>{summary_text}</text>")
        svg_lines.append("<line x1='20' y1='66' x2='1230' y2='66' stroke='#e5e7eb'/>")

        y = 78
        for block in row_blocks:
            row = block["row"]
            lines = block["lines"]
            row_height = int(block["row_height"])
            classification = str(row.get("classification", "unresolved"))
            if classification == "success":
                color = "#16a34a"
            elif classification == "fallback":
                color = "#f59e0b"
            else:
                color = "#ef4444"
            status_text = self._escape(str(row.get("mount_status", "")))
            svg_lines.append(
                f"<rect x='20' y='{y}' width='1210' height='{row_height}' fill='#ffffff' stroke='#e5e7eb' rx='6'/>"
            )
            svg_lines.extend(
                [
                    f"<rect x='28' y='{y+8}' width='6' height='{max(20, row_height - 16)}' fill='{color}' rx='2'/>",
                    f"<rect x='1090' y='{y+10}' width='128' height='20' fill='{color}' rx='10'/>",
                    f"<text x='1102' y='{y+24}' class='chip'>{status_text}</text>",
                ]
            )
            text_y = y + 24
            for line in lines:
                svg_lines.append(
                    f"<text x='44' y='{text_y}' class='text'>{self._escape(line)}</text>"
                )
                text_y += line_height
            y += row_height + row_gap

        svg_lines.append("</svg>")
        return "\n".join(svg_lines)

    def _build_row_lines(self, row: Dict[str, Any]) -> List[str]:
        unit_index = int(row.get("unit_index", 0) or 0)
        unit_id = str(row.get("unit_id", "") or "")
        unit_text = str(row.get("text", "") or "")
        confidence = float(row.get("match_confidence", 0.0) or 0.0)
        envelope = dict(row.get("envelope") or {})
        chunk_ids = ",".join(str(item) for item in (row.get("source_chunk_ids") or [])) or "(none)"
        hook_ids = ",".join(str(item) for item in (row.get("hook_ids") or [])) or "(none)"
        block_id = str(row.get("alignment_block_id", "") or "") or "(none)"
        anchor_kind = str(row.get("anchor_kind", "") or "") or "(none)"
        lines: List[str] = []
        lines.extend(
            self._wrap_text(
                f"unit[{unit_index}] id: {unit_id} text: {unit_text}",
                max_chars=116,
            )
        )
        lines.append(f"match_confidence: {confidence:.3f}")
        if envelope:
            lines.append(
                "time: "
                f"{self._format_float(envelope.get('provisional_start'))} -> "
                f"{self._format_float(envelope.get('provisional_end'))} "
                "bounds: "
                f"{self._format_float(envelope.get('left_bound'))} -> "
                f"{self._format_float(envelope.get('right_bound'))}"
            )
        lines.extend(
            self._wrap_text(
                f"chunks: {chunk_ids} hooks: {hook_ids} block: {block_id} anchor_kind: {anchor_kind}",
                max_chars=110,
            )
        )
        for index, boundary in enumerate(list(row.get("boundary_after") or [])):
            metadata = dict(boundary.get("metadata") or {})
            metadata_text = ""
            if metadata:
                metadata_text = f" metadata={metadata}"
            lines.extend(
                self._wrap_text(
                    "boundary_after"
                    f"[{index}] {str(boundary.get('reason', '') or '')} "
                    f"score={self._format_float(boundary.get('score'))} "
                    f"gap={self._format_float(boundary.get('left_end'))}->{self._format_float(boundary.get('right_start'))}"
                    f"{metadata_text}",
                    max_chars=110,
                )
            )
        hooks = list(row.get("hooks") or [])
        if not hooks:
            lines.append("hook: (none)")
            return lines
        for hook in hooks:
            hook_id = str(hook.get("hook_id", "") or "")
            hook_text = str(hook.get("text", "") or "")
            lines.extend(
                self._wrap_text(
                    f"hook[{hook_id}] text: {hook_text}",
                    max_chars=110,
                )
            )
        return lines

    @staticmethod
    def _wrap_text(text: str, *, max_chars: int) -> List[str]:
        if not text:
            return [""]
        lines = textwrap.wrap(
            text,
            width=max(20, int(max_chars)),
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        return lines or [text]

    @staticmethod
    def _escape(value: Any) -> str:
        text = str(value or "")
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\"", "&quot;")
            .replace("'", "&apos;")
        )

    @staticmethod
    def _format_float(value: Any) -> str:
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "0.000"
