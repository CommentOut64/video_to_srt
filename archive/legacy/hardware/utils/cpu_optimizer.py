"""
CPU 架构检测与 ONNX 线程优化模块

根据 Intel/AMD 架构差异，智能配置 ONNX Runtime 线程数
集成现有的 hardware_service.py 获取真实硬件数据
"""
import platform
import multiprocessing
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class CPUArchitectureDetector:
    """CPU 架构检测器"""

    @staticmethod
    def detect_cpu_vendor(cpu_name: Optional[str] = None) -> str:
        """
        检测 CPU 厂商

        Args:
            cpu_name: CPU 名称（可选），优先从此识别

        Returns:
            str: 'intel', 'amd', 或 'unknown'
        """
        try:
            # 优先使用提供的 CPU 名称
            if cpu_name:
                cpu_name_lower = cpu_name.lower()
                if 'intel' in cpu_name_lower:
                    return 'intel'
                elif 'amd' in cpu_name_lower:
                    return 'amd'

            # 回退：使用 platform.processor()
            cpu_info = platform.processor().lower()
            if 'intel' in cpu_info:
                return 'intel'
            elif 'amd' in cpu_info:
                return 'amd'

            # Windows 注册表
            if platform.system() == "Windows":
                try:
                    import winreg
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                      r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                        cpu_name_reg = winreg.QueryValueEx(key, "ProcessorNameString")[0].lower()
                        if 'intel' in cpu_name_reg:
                            return 'intel'
                        elif 'amd' in cpu_name_reg:
                            return 'amd'
                except:
                    pass

            # Linux /proc/cpuinfo
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()
                    if 'genuineintel' in cpuinfo:
                        return 'intel'
                    elif 'authenticamd' in cpuinfo:
                        return 'amd'
            except:
                pass

            return 'unknown'

        except Exception as e:
            logger.warning(f"检测 CPU 厂商失败: {e}")
            return 'unknown'

    @staticmethod
    def is_intel_hybrid_architecture(cpu_name: Optional[str] = None) -> bool:
        """
        检测是否为 Intel 混合架构（12代及以后）

        Args:
            cpu_name: CPU 名称

        Returns:
            bool: 是否为混合架构
        """
        if not cpu_name:
            return False

        cpu_name_lower = cpu_name.lower()

        # Intel 12代及以后的型号标识
        hybrid_indicators = [
            '12th gen', '13th gen', '14th gen', '15th gen',  # 代数
            'core ultra',  # Ultra 系列
            '-12', '-13', '-14', '-15',  # 型号后缀
            '12100', '12400', '12600', '12700', '12900',  # 12代
            '13100', '13400', '13600', '13700', '13900',  # 13代
            '14100', '14400', '14600', '14700', '14900',  # 14代
        ]

        return any(indicator in cpu_name_lower for indicator in hybrid_indicators)

    @staticmethod
    def get_physical_cores(hardware_info=None) -> int:
        """
        获取物理核心数（不含超线程）

        Args:
            hardware_info: HardwareInfo 对象（可选）

        Returns:
            int: 物理核心数
        """
        try:
            # 优先使用 hardware_info
            if hardware_info and hasattr(hardware_info, 'cpu_cores'):
                return hardware_info.cpu_cores

            # 回退：使用 psutil
            import psutil
            return psutil.cpu_count(logical=False) or 1
        except ImportError:
            # psutil 不可用，假设超线程为2倍
            logical_cores = multiprocessing.cpu_count()
            return max(1, logical_cores // 2)

    @staticmethod
    def detect_intel_p_cores(cpu_name: Optional[str] = None, physical_cores: int = None) -> int:
        """
        检测 Intel P-Core 数量（性能核）

        对于混合架构，尝试从 CPU 名称推断 P-Core 数量

        Args:
            cpu_name: CPU 名称
            physical_cores: 物理核心总数

        Returns:
            int: P-Core 数量
        """
        if not cpu_name or not physical_cores:
            return physical_cores or 1

        # 如果不是混合架构，返回全部物理核心
        if not CPUArchitectureDetector.is_intel_hybrid_architecture(cpu_name):
            return physical_cores

        # 混合架构：根据型号推断 P-Core 数量
        # 简化规则（基于常见型号）
        cpu_name_lower = cpu_name.lower()

        # i9 系列：通常 8 P-Core
        if 'i9' in cpu_name_lower:
            return min(8, physical_cores)
        # i7 系列：通常 6-8 P-Core
        elif 'i7' in cpu_name_lower:
            return min(6, physical_cores)
        # i5 系列：通常 4-6 P-Core
        elif 'i5' in cpu_name_lower:
            return min(4, physical_cores)
        # i3 系列：通常 4 P-Core
        elif 'i3' in cpu_name_lower:
            return min(4, physical_cores)

        # 保守估计：物理核心数的 50%
        return max(1, physical_cores // 2)


class ONNXThreadOptimizer:
    """ONNX Runtime 线程优化器"""

    @staticmethod
    def calculate_optimal_threads(
        vendor: Optional[str] = None,
        physical_cores: Optional[int] = None,
        cpu_name: Optional[str] = None,
        usage_ratio: float = 0.6,
        hardware_info=None
    ) -> Tuple[int, Dict[str, any]]:
        """
        计算最优线程数

        Args:
            vendor: CPU 厂商，None 则自动检测
            physical_cores: 物理核心数，None 则自动检测
            cpu_name: CPU 名称，用于精确识别
            usage_ratio: 核心使用比例（0.5-0.6 推荐）
            hardware_info: HardwareInfo 对象（优先使用）

        Returns:
            Tuple[int, Dict]: (最优线程数, 配置信息)
        """
        detector = CPUArchitectureDetector()

        # 从 hardware_info 获取数据（优先）
        if hardware_info:
            if not cpu_name and hasattr(hardware_info, 'cpu_name'):
                cpu_name = hardware_info.cpu_name
            if not physical_cores and hasattr(hardware_info, 'cpu_cores'):
                physical_cores = hardware_info.cpu_cores

        # 自动检测
        if vendor is None:
            vendor = detector.detect_cpu_vendor(cpu_name)

        if physical_cores is None:
            physical_cores = detector.get_physical_cores(hardware_info)

        logical_cores = multiprocessing.cpu_count()

        # 配置信息
        info = {
            'vendor': vendor,
            'physical_cores': physical_cores,
            'logical_cores': logical_cores,
            'cpu_name': cpu_name or 'Unknown',
            'usage_ratio': usage_ratio,
            'strategy': ''
        }

        # Intel 策略
        if vendor == 'intel':
            is_hybrid = detector.is_intel_hybrid_architecture(cpu_name)

            if is_hybrid:
                # 混合架构：仅使用 P-Core
                p_cores = detector.detect_intel_p_cores(cpu_name, physical_cores)
                optimal_threads = max(1, int(p_cores * usage_ratio))
                info['strategy'] = f'Intel 混合架构：仅使用 {p_cores} 个 P-Core 的 {usage_ratio*100:.0f}%'
                info['p_cores'] = p_cores
            else:
                # 传统架构：使用物理核心数的 60%
                optimal_threads = max(1, int(physical_cores * usage_ratio))
                info['strategy'] = f'Intel 传统架构：使用物理核心数的 {usage_ratio*100:.0f}%'

        # AMD 策略
        elif vendor == 'amd':
            # AMD 全大核：使用物理核心数的 50-60%
            optimal_threads = max(1, int(physical_cores * usage_ratio))
            info['strategy'] = f'AMD 全大核：使用物理核心数的 {usage_ratio*100:.0f}%，避免跨 CCX 开销'

        # 未知架构
        else:
            # 保守策略：使用物理核心数的 50%
            optimal_threads = max(1, int(physical_cores * 0.5))
            info['strategy'] = f'未知架构：保守策略，使用物理核心数的 50%'

        # 安全限制
        optimal_threads = max(1, min(optimal_threads, physical_cores))
        info['optimal_threads'] = optimal_threads

        return optimal_threads, info

    @staticmethod
    def get_onnx_session_options(
        vendor: Optional[str] = None,
        optimal_threads: Optional[int] = None,
        hardware_info=None
    ):
        """
        获取配置好的 ONNX Runtime SessionOptions

        Args:
            vendor: CPU 厂商
            optimal_threads: 最优线程数
            hardware_info: HardwareInfo 对象

        Returns:
            ort.SessionOptions: 配置好的 SessionOptions
        """
        import onnxruntime as ort

        # 自动计算
        if optimal_threads is None:
            optimal_threads, _ = ONNXThreadOptimizer.calculate_optimal_threads(
                vendor=vendor,
                hardware_info=hardware_info
            )

        if vendor is None:
            cpu_name = hardware_info.cpu_name if hardware_info else None
            vendor = CPUArchitectureDetector.detect_cpu_vendor(cpu_name)

        # 创建 SessionOptions
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 线程配置
        sess_options.intra_op_num_threads = optimal_threads
        sess_options.inter_op_num_threads = 1

        # 通用 CPU 优化（适用于 Intel/AMD）
        try:
            # 1. 关闭线程自旋等待，降低空转功耗
            sess_options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            sess_options.add_session_config_entry('session.inter_op.allow_spinning', '0')

            # 2. 将极小浮点数（denormal）视为 0，提升性能
            #    音频模型中常出现极小浮点数，Intel/AMD CPU 处理 denormal 都会有性能损失
            sess_options.add_session_config_entry('session.set_denormal_as_zero', '1')

            logger.debug(f"CPU 优化配置已启用: allow_spinning=0, denormal_as_zero=1")
        except Exception as e:
            logger.debug(f"CPU 优化配置失败（可能不支持）: {e}")

        return sess_options


def get_hardware_aware_thread_config(hardware_info=None) -> Tuple[int, Dict]:
    """
    获取硬件感知的线程配置（便捷函数）

    Args:
        hardware_info: HardwareInfo 对象

    Returns:
        Tuple[int, Dict]: (最优线程数, 配置信息)
    """
    return ONNXThreadOptimizer.calculate_optimal_threads(hardware_info=hardware_info)


def print_cpu_optimization_info(hardware_info=None):
    """打印 CPU 优化信息（用于调试）"""
    detector = CPUArchitectureDetector()
    optimizer = ONNXThreadOptimizer()

    # 获取 CPU 信息
    cpu_name = hardware_info.cpu_name if hardware_info else None
    physical_cores = hardware_info.cpu_cores if hardware_info else detector.get_physical_cores()
    logical_cores = multiprocessing.cpu_count()
    vendor = detector.detect_cpu_vendor(cpu_name)

    print("=" * 60)
    print("CPU 架构检测与优化配置")
    print("=" * 60)
    print(f"CPU 型号: {cpu_name or platform.processor()}")
    print(f"CPU 厂商: {vendor.upper()}")
    print(f"物理核心: {physical_cores}")
    print(f"逻辑核心: {logical_cores}")

    if vendor == 'intel' and cpu_name:
        is_hybrid = detector.is_intel_hybrid_architecture(cpu_name)
        print(f"混合架构: {'是' if is_hybrid else '否'}")
        if is_hybrid:
            p_cores = detector.detect_intel_p_cores(cpu_name, physical_cores)
            print(f"P-Core 数量: {p_cores}")
    print()

    # 计算不同使用比例的线程数
    for ratio in [0.5, 0.6, 0.75]:
        threads, info = optimizer.calculate_optimal_threads(
            vendor=vendor,
            physical_cores=physical_cores,
            cpu_name=cpu_name,
            usage_ratio=ratio,
            hardware_info=hardware_info
        )
        print(f"使用比例 {ratio*100:.0f}%: {threads} 线程")
        print(f"  策略: {info['strategy']}")
        print()

    # 推荐配置
    optimal_threads, info = optimizer.calculate_optimal_threads(
        hardware_info=hardware_info
    )
    print("=" * 60)
    print("推荐配置")
    print("=" * 60)
    print(f"最优线程数: {optimal_threads}")
    print(f"配置策略: {info['strategy']}")
    print("=" * 60)


if __name__ == "__main__":
    # 测试：尝试使用 hardware_service
    import sys
    from pathlib import Path

    # 添加 backend 目录到 sys.path
    backend_dir = Path(__file__).resolve().parent.parent.parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    try:
        from app.services.hardware_service import CoreHardwareDetector
        detector = CoreHardwareDetector()
        hardware_info = detector.detect()
        print("使用真实硬件检测数据：")
        print_cpu_optimization_info(hardware_info)
    except Exception as e:
        print(f"无法加载 hardware_service: {e}")
        import traceback
        traceback.print_exc()
        print("\n使用回退检测：")
        print_cpu_optimization_info()
