"""构建不能同时进入主链的候选冲突簇。"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorCandidate,
    AnchorMountInputView,
)


def _normalize(text: str) -> str:
    return "".join(
        char.lower()
        for char in str(text or "")
        if char.isalnum() or ("\u4e00" <= char <= "\u9fff")
    )


@dataclass(frozen=True)
class AmbiguityCluster:
    cluster_id: str
    hook_span: tuple[int, ...]
    unit_span: tuple[int, ...]
    candidate_ids: tuple[str, ...]
    candidate_unit_spans: tuple[tuple[int, ...], ...]
    candidate_hook_spans: tuple[tuple[int, ...], ...]
    cluster_tags: tuple[str, ...]
    conflict_basis: tuple[str, ...]


class AmbiguityClusterBuilder:
    """把互斥候选收束成稳定 cluster，供 trust/disambiguation 使用。"""

    def build(
        self,
        *,
        input_view: AnchorMountInputView,
        candidates: tuple[AnchorCandidate, ...],
    ) -> tuple[AmbiguityCluster, ...]:
        ordered = tuple(candidates)
        if not ordered:
            return tuple()

        adjacency: dict[int, set[int]] = {index: set() for index in range(len(ordered))}
        pair_basis: dict[tuple[int, int], set[str]] = {}
        pair_tags: dict[tuple[int, int], set[str]] = {}
        for left_index in range(len(ordered)):
            for right_index in range(left_index + 1, len(ordered)):
                basis = self._conflict_basis(
                    input_view=input_view,
                    left=ordered[left_index],
                    right=ordered[right_index],
                )
                if not basis:
                    continue
                tags = self._cluster_tags(
                    input_view=input_view,
                    left=ordered[left_index],
                    right=ordered[right_index],
                )
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
                pair_basis[(left_index, right_index)] = basis
                pair_tags[(left_index, right_index)] = tags

        visited: set[int] = set()
        clusters: list[AmbiguityCluster] = []
        for start_index in range(len(ordered)):
            if start_index in visited:
                continue
            queue = [start_index]
            component: list[int] = []
            while queue:
                current = queue.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                queue.extend(sorted(adjacency[current] - visited))
            if len(component) <= 1:
                continue
            component.sort()
            basis_tags: set[str] = set()
            cluster_tags: set[str] = set()
            for left_pos, left_index in enumerate(component):
                for right_index in component[left_pos + 1 :]:
                    key = (min(left_index, right_index), max(left_index, right_index))
                    basis_tags.update(pair_basis.get(key, set()))
                    cluster_tags.update(pair_tags.get(key, set()))
            cluster_candidates = [ordered[index] for index in component]
            hook_span = tuple(
                sorted({index for candidate in cluster_candidates for index in candidate.hook_indices})
            )
            unit_span = tuple(
                sorted({index for candidate in cluster_candidates for index in candidate.unit_indices})
            )
            clusters.append(
                AmbiguityCluster(
                    cluster_id=f"cluster-{len(clusters)}",
                    hook_span=hook_span,
                    unit_span=unit_span,
                    candidate_ids=tuple(candidate.candidate_id for candidate in cluster_candidates),
                    candidate_unit_spans=tuple(candidate.unit_indices for candidate in cluster_candidates),
                    candidate_hook_spans=tuple(candidate.hook_indices for candidate in cluster_candidates),
                    cluster_tags=tuple(sorted(cluster_tags)),
                    conflict_basis=tuple(sorted(basis_tags)),
                )
            )
        return tuple(clusters)

    def _conflict_basis(
        self,
        *,
        input_view: AnchorMountInputView,
        left: AnchorCandidate,
        right: AnchorCandidate,
    ) -> set[str]:
        basis: set[str] = set()
        if set(left.hook_indices) & set(right.hook_indices):
            basis.add("same_hook_span")
        if set(left.unit_indices) & set(right.unit_indices):
            basis.add("same_unit_span")
        if self._is_merge_width_competition(left=left, right=right):
            basis.add("merge_width_competition")
        if self._surface_key(input_view=input_view, candidate=left) == self._surface_key(
            input_view=input_view,
            candidate=right,
        ):
            basis.add("duplicate_surface")
        if not basis:
            return basis
        if self._crosses_boundary(input_view=input_view, candidate=left) or self._crosses_boundary(
            input_view=input_view,
            candidate=right,
        ):
            basis.add("boundary_cross_risk")
        if self._is_low_information(input_view=input_view, candidate=left) or self._is_low_information(
            input_view=input_view,
            candidate=right,
        ):
            basis.add("low_information_risk")
        return basis

    def _cluster_tags(
        self,
        *,
        input_view: AnchorMountInputView,
        left: AnchorCandidate,
        right: AnchorCandidate,
    ) -> set[str]:
        tags: set[str] = set()
        if len(left.unit_indices) != len(right.unit_indices):
            tags.add("span_width_conflict")
        if self._surface_key(input_view=input_view, candidate=left) == self._surface_key(
            input_view=input_view,
            candidate=right,
        ):
            tags.add("duplicate_surface")
        if self._is_low_information(input_view=input_view, candidate=left) or self._is_low_information(
            input_view=input_view,
            candidate=right,
        ):
            tags.add("low_information")
        return tags

    @staticmethod
    def _is_merge_width_competition(*, left: AnchorCandidate, right: AnchorCandidate) -> bool:
        left_units = set(left.unit_indices)
        right_units = set(right.unit_indices)
        if not (left_units & right_units):
            return False
        return len(left.unit_indices) != len(right.unit_indices)

    @staticmethod
    def _crosses_boundary(
        *,
        input_view: AnchorMountInputView,
        candidate: AnchorCandidate,
    ) -> bool:
        if len(candidate.unit_indices) <= 1:
            return False
        units = [input_view.token_units[index] for index in candidate.unit_indices]
        return any(
            left.speaker_id != right.speaker_id or left.turn_id != right.turn_id
            for left, right in zip(units, units[1:], strict=False)
        )

    def _surface_key(
        self,
        *,
        input_view: AnchorMountInputView,
        candidate: AnchorCandidate,
    ) -> str:
        unit_surface = "".join(
            _normalize(input_view.token_units[index].token_text)
            for index in candidate.unit_indices
        )
        hook_surface = "".join(
            _normalize(input_view.fast_hooks[index].hook_text)
            for index in candidate.hook_indices
        )
        return unit_surface or hook_surface

    def _is_low_information(
        self,
        *,
        input_view: AnchorMountInputView,
        candidate: AnchorCandidate,
    ) -> bool:
        return len(self._surface_key(input_view=input_view, candidate=candidate)) <= 1
