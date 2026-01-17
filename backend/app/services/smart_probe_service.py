"""
SmartProbeService - 智能探针分诊服务

V3.1.2+dev.20260109.01: 新增智能探针分诊策略

使用中心扩散探针策略，快速判断视频是否需要全量分离：
- 从中心点开始探测
- 使用斐波那契数列扩散（1, 1, 2, 3, 5, 8...）
- 达到最大步长后转为线性扫描
- 一旦发现"脏"chunk（SNR < 阈值），立即触发全量分离
- 如果全部通过，则判定为纯净视频

核心优势：
- 快速筛选：平均只需检测 30-40% 的 chunks
- 智能扩散：优先检测中心区域，逐步扩散到边缘
- 保守策略：一旦发现干扰，立即触发全量分离
"""
import logging
from typing import List, Dict, Tuple, Optional

from app.services.runtime_param_resolver import get_smart_probe_runtime_params

logger = logging.getLogger(__name__)


class SmartProbeService:
    """
    智能探针分诊服务

    使用中心扩散探针策略（Center-Out Exponential/Linear Probe）
    快速判断视频是否需要全量分离
    """

    def __init__(
        self,
        brouhaha_service,
        snr_threshold: float = 15.0,
        max_step_chunks: int = 30
    ):
        """
        初始化智能探针服务

        Args:
            brouhaha_service: Brouhaha SNR 检测服务实例
            snr_threshold: SNR 阈值，低于此值触发分离（默认 15.0 dB）
            max_step_chunks: 最大步长（chunk数），达到此步长后转为线性扫描
                           假设1个chunk=0.5s，30个chunk=15s，保证不漏掉长于15s的BGM
        """
        self.brouhaha = brouhaha_service
        self.threshold = snr_threshold
        self.max_step = max_step_chunks

    def _fibonacci_generator(self):
        """生成斐波那契数列: 1, 1, 2, 3, 5, 8..."""
        a, b = 0, 1
        while True:
            yield b
            a, b = b, a + b

    def run_probe(
        self,
        chunks: List,
        progress_callback: Optional[callable] = None
    ) -> Tuple[str, Dict[int, Dict]]:
        """
        执行中心扩散探针 (Center-Out Exponential/Linear Probe)

        Args:
            chunks: AudioChunk 列表
            progress_callback: 进度回调函数，接收 (current, total) 参数

        Returns:
            decision: 'SEPARATE_ALL' | 'PASS_ALL'
            cache: {chunk_index: {'snr': float, 'c50': float, 'decision': str}}
        """
        n = len(chunks)
        if n == 0:
            return "PASS_ALL", {}

        center = n // 2
        cache = {}  # 缓存结果

        # 1. 优先探测中心
        is_dirty, result = self._check_chunk(chunks[center])
        cache[center] = result

        if progress_callback:
            progress_callback(1, n)

        if is_dirty:
            logger.info(
                f"探针: 中心点 [{center}] 发现干扰 (SNR={result['snr']:.1f}dB)，"
                f"触发全量分离"
            )
            return "SEPARATE_ALL", cache

        # 2. 斐波那契扩散 + 线性扫描
        fib_gen = self._fibonacci_generator()
        visited_indices = {center}
        cumulative_radius = 0  # 累加半径

        while True:
            try:
                # 获取下一个斐波那契数
                fib_step = next(fib_gen)
            except StopIteration:
                break  # 理论上斐波那契无限，这里防御性编程

            # 关键逻辑修复：
            # 增量 = min(斐波那契增长, 最大步长限制)
            # 效果：前期按 1,1,2,3,5... 加速扩散
            #      一旦超过 max_step，后期按 max_step, max_step... 匀速线性扩散
            current_increment = min(fib_step, self.max_step)

            # 累加总半径
            cumulative_radius += current_increment

            left = center - cumulative_radius
            right = center + cumulative_radius

            # 检查是否全部越界（探测结束）
            if left < 0 and right >= n:
                break

            # 探测左翼
            if left >= 0 and left not in visited_indices:
                is_dirty, result = self._check_chunk(chunks[left])
                cache[left] = result
                visited_indices.add(left)

                if progress_callback:
                    progress_callback(len(visited_indices), n)

                if is_dirty:
                    logger.info(
                        f"探针: 左翼 [{left}] 发现干扰 (SNR={result['snr']:.1f}dB)，"
                        f"触发全量分离"
                    )
                    return "SEPARATE_ALL", cache

            # 探测右翼
            if right < n and right not in visited_indices:
                is_dirty, result = self._check_chunk(chunks[right])
                cache[right] = result
                visited_indices.add(right)

                if progress_callback:
                    progress_callback(len(visited_indices), n)

                if is_dirty:
                    logger.info(
                        f"探针: 右翼 [{right}] 发现干扰 (SNR={result['snr']:.1f}dB)，"
                        f"触发全量分离"
                    )
                    return "SEPARATE_ALL", cache

        coverage = len(cache) / n
        logger.info(
            f"探针: 通过智能快筛 (覆盖率 {coverage:.1%})，判定为纯净视频"
        )
        return "PASS_ALL", cache

    def _check_chunk(self, chunk) -> Tuple[bool, Dict]:
        """
        实际推理逻辑

        Args:
            chunk: AudioChunk 对象

        Returns:
            is_dirty: 是否需要分离
            result: 检测结果字典
        """
        # 调用 Brouhaha 检测 SNR 和 C50
        brouhaha_result = self.brouhaha.detect(
            chunk.audio,
            chunk.sample_rate,
            chunk_id=chunk.index
        )

        snr = brouhaha_result.snr
        c50 = brouhaha_result.c50

        is_dirty = snr < self.threshold

        return is_dirty, {
            "snr": snr,
            "c50": c50,
            "calculated": True,
            "probe_decision": "separate" if is_dirty else "pass"
        }


# 单例访问
_smart_probe_instance: Optional[SmartProbeService] = None


def get_smart_probe_service(
    snr_threshold: Optional[float] = None,
    max_step_chunks: Optional[int] = None
) -> SmartProbeService:
    """
    获取智能探针服务单例

    Args:
        snr_threshold: SNR 阈值（默认 15.0 dB）
        max_step_chunks: 最大步长（默认 30 chunks）

    Returns:
        SmartProbeService: 智能探针服务实例
    """
    global _smart_probe_instance
    runtime = get_smart_probe_runtime_params()
    effective_snr = snr_threshold if snr_threshold is not None else runtime.get("snr_threshold", 15.0)
    effective_max_step = max_step_chunks if max_step_chunks is not None else 30

    if _smart_probe_instance is None:
        from app.services.brouhaha_service import get_brouhaha_service
        brouhaha = get_brouhaha_service()
        _smart_probe_instance = SmartProbeService(
            brouhaha_service=brouhaha,
            snr_threshold=effective_snr,
            max_step_chunks=effective_max_step
        )
    else:
        _smart_probe_instance.threshold = effective_snr
        _smart_probe_instance.max_step = effective_max_step
    return _smart_probe_instance


def reset_smart_probe_service():
    """重置智能探针服务单例（用于测试）"""
    global _smart_probe_instance
    _smart_probe_instance = None
