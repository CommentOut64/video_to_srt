"""
资源管理器

管理模型生命周期和显存分配，防止 OOM
"""
import logging
import asyncio
import gc
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class ModelType(Enum):
    """模型类型"""
    SENSEVOICE = "sensevoice"
    WHISPER = "whisper"
    DEMUCS = "demucs"
    VAD = "vad"


@dataclass
class ModelInfo:
    """模型信息"""
    model_type: ModelType
    model_id: str                    # 模型标识（如 "medium", "large-v3"）
    instance: Any = None             # 模型实例
    vram_usage_mb: int = 0           # 显存占用（MB）
    loaded_at: float = 0.0           # 加载时间戳
    last_used_at: float = 0.0        # 最后使用时间戳
    use_count: int = 0               # 使用次数

    @property
    def idle_time(self) -> float:
        """空闲时间（秒）"""
        return time.time() - self.last_used_at


@dataclass
class ResourceBudget:
    """资源预算"""
    max_vram_mb: int = 8000          # 最大显存预算
    reserved_vram_mb: int = 500      # 保留显存（系统用）
    max_models: int = 3              # 最大同时加载模型数

    @property
    def available_vram_mb(self) -> int:
        return self.max_vram_mb - self.reserved_vram_mb


# 模型显存估算（MB）
MODEL_VRAM_ESTIMATES: Dict[str, Dict[str, int]] = {
    "sensevoice": {
        "small": 800,
        "small_int8": 500,
    },
    "whisper": {
        "tiny": 400,
        "base": 500,
        "small": 1000,
        "medium": 2000,
        "large": 3000,
        "large-v2": 3000,
        "large-v3": 3500,
    },
    "demucs": {
        "htdemucs": 1500,
        "htdemucs_ft": 1500,
    },
    "vad": {
        "silero": 100,
    }
}


class ResourceManager:
    """
    资源管理器

    功能:
    - 模型生命周期管理（加载/卸载）
    - 显存预算控制
    - LRU 淘汰策略
    - 自动清理
    """

    def __init__(
        self,
        budget: Optional[ResourceBudget] = None,
        auto_cleanup: bool = True,
        idle_timeout: float = 300.0  # 5分钟空闲自动卸载
    ):
        """
        初始化资源管理器

        Args:
            budget: 资源预算配置
            auto_cleanup: 是否启用自动清理
            idle_timeout: 空闲超时时间（秒）
        """
        self.logger = logging.getLogger(__name__)
        self.budget = budget or ResourceBudget()
        self.auto_cleanup = auto_cleanup
        self.idle_timeout = idle_timeout

        self._models: Dict[str, ModelInfo] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

        # 初始化显存预算
        self._init_vram_budget()

    def _init_vram_budget(self):
        """初始化显存预算"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                total_vram = props.total_memory // (1024 * 1024)
                # 使用 80% 的显存作为预算
                self.budget.max_vram_mb = int(total_vram * 0.8)
                self.logger.info(f"显存预算: {self.budget.max_vram_mb}MB (总计: {total_vram}MB)")
            except Exception as e:
                self.logger.warning(f"获取显存信息失败: {e}")

    def _get_model_key(self, model_type: ModelType, model_id: str) -> str:
        """生成模型唯一键"""
        return f"{model_type.value}:{model_id}"

    def _estimate_vram(self, model_type: ModelType, model_id: str) -> int:
        """估算模型显存需求"""
        type_estimates = MODEL_VRAM_ESTIMATES.get(model_type.value, {})
        return type_estimates.get(model_id, 1000)  # 默认 1GB

    def get_current_vram_usage(self) -> int:
        """获取当前显存使用量（MB）"""
        return sum(m.vram_usage_mb for m in self._models.values())

    def get_available_vram(self) -> int:
        """获取可用显存（MB）"""
        return self.budget.available_vram_mb - self.get_current_vram_usage()

    def can_load_model(self, model_type: ModelType, model_id: str) -> bool:
        """检查是否可以加载模型"""
        # 检查是否已加载
        key = self._get_model_key(model_type, model_id)
        if key in self._models:
            return True

        # 检查模型数量限制
        if len(self._models) >= self.budget.max_models:
            return False

        # 检查显存预算
        required_vram = self._estimate_vram(model_type, model_id)
        return self.get_available_vram() >= required_vram

    async def register_model(
        self,
        model_type: ModelType,
        model_id: str,
        instance: Any,
        vram_usage_mb: Optional[int] = None
    ) -> bool:
        """
        注册已加载的模型

        Args:
            model_type: 模型类型
            model_id: 模型标识
            instance: 模型实例
            vram_usage_mb: 实际显存占用（可选）

        Returns:
            是否注册成功
        """
        async with self._lock:
            key = self._get_model_key(model_type, model_id)

            # 如果已存在，更新实例
            if key in self._models:
                self._models[key].instance = instance
                self._models[key].last_used_at = time.time()
                return True

            # 估算显存
            if vram_usage_mb is None:
                vram_usage_mb = self._estimate_vram(model_type, model_id)

            # 检查是否需要腾出空间
            if self.get_available_vram() < vram_usage_mb:
                await self._evict_models(vram_usage_mb)

            # 再次检查
            if self.get_available_vram() < vram_usage_mb:
                self.logger.warning(f"显存不足，无法加载 {key}")
                return False

            # 注册模型
            self._models[key] = ModelInfo(
                model_type=model_type,
                model_id=model_id,
                instance=instance,
                vram_usage_mb=vram_usage_mb,
                loaded_at=time.time(),
                last_used_at=time.time(),
                use_count=0
            )

            self.logger.info(f"模型已注册: {key}, 显存: {vram_usage_mb}MB")
            return True

    async def acquire(self, model_type: ModelType, model_id: str) -> Optional[Any]:
        """
        获取模型实例（更新使用时间）

        Args:
            model_type: 模型类型
            model_id: 模型标识

        Returns:
            模型实例，未加载返回 None
        """
        async with self._lock:
            key = self._get_model_key(model_type, model_id)
            if key in self._models:
                model_info = self._models[key]
                model_info.last_used_at = time.time()
                model_info.use_count += 1
                return model_info.instance
            return None

    async def unload(self, model_type: ModelType, model_id: str) -> bool:
        """
        卸载模型

        Args:
            model_type: 模型类型
            model_id: 模型标识

        Returns:
            是否卸载成功
        """
        async with self._lock:
            key = self._get_model_key(model_type, model_id)
            if key not in self._models:
                return False

            model_info = self._models.pop(key)
            await self._cleanup_model(model_info)
            self.logger.info(f"模型已卸载: {key}")
            return True

    async def unload_all(self):
        """卸载所有模型"""
        async with self._lock:
            for key in list(self._models.keys()):
                model_info = self._models.pop(key)
                await self._cleanup_model(model_info)
            self.logger.info("所有模型已卸载")

    async def _cleanup_model(self, model_info: ModelInfo):
        """清理模型资源"""
        try:
            # 删除模型实例
            del model_info.instance
            model_info.instance = None

            # 清理 GPU 缓存
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 强制垃圾回收
            gc.collect()

        except Exception as e:
            self.logger.error(f"清理模型失败: {e}")

    async def _evict_models(self, required_vram: int):
        """
        淘汰模型以腾出空间（LRU 策略）

        Args:
            required_vram: 需要的显存（MB）
        """
        # 按最后使用时间排序
        sorted_models = sorted(
            self._models.items(),
            key=lambda x: x[1].last_used_at
        )

        freed_vram = 0
        for key, model_info in sorted_models:
            if self.get_available_vram() + freed_vram >= required_vram:
                break

            # 卸载模型
            self._models.pop(key)
            await self._cleanup_model(model_info)
            freed_vram += model_info.vram_usage_mb
            self.logger.info(f"LRU 淘汰模型: {key}, 释放: {model_info.vram_usage_mb}MB")

    async def start_auto_cleanup(self):
        """启动自动清理任务"""
        if not self.auto_cleanup:
            return

        self._cleanup_task = asyncio.create_task(self._auto_cleanup_loop())
        self.logger.info(f"自动清理已启动，空闲超时: {self.idle_timeout}s")

    async def stop_auto_cleanup(self):
        """停止自动清理任务"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _auto_cleanup_loop(self):
        """自动清理循环"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次

                async with self._lock:
                    now = time.time()
                    to_unload = []

                    for key, model_info in self._models.items():
                        if model_info.idle_time > self.idle_timeout:
                            to_unload.append(key)

                    for key in to_unload:
                        model_info = self._models.pop(key)
                        await self._cleanup_model(model_info)
                        self.logger.info(f"自动卸载空闲模型: {key}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"自动清理异常: {e}")

    def get_status(self) -> Dict:
        """获取资源管理器状态"""
        return {
            "budget": {
                "max_vram_mb": self.budget.max_vram_mb,
                "reserved_vram_mb": self.budget.reserved_vram_mb,
                "available_vram_mb": self.budget.available_vram_mb
            },
            "usage": {
                "current_vram_mb": self.get_current_vram_usage(),
                "free_vram_mb": self.get_available_vram(),
                "loaded_models": len(self._models)
            },
            "models": {
                key: {
                    "type": info.model_type.value,
                    "id": info.model_id,
                    "vram_mb": info.vram_usage_mb,
                    "idle_time": round(info.idle_time, 1),
                    "use_count": info.use_count
                }
                for key, info in self._models.items()
            }
        }

    def is_model_loaded(self, model_type: ModelType, model_id: str) -> bool:
        """检查模型是否已加载"""
        key = self._get_model_key(model_type, model_id)
        return key in self._models


# 单例实例
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """获取资源管理器单例"""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager
