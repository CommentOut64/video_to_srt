"""
模型预加载和缓存管理器
实现模型预加载、LRU缓存、内存监控等功能
"""

import os
import gc
import logging
import threading
import time
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import psutil
import torch
# 延迟导入 faster_whisper，避免启动时加载 ctranslate2 导致首次启动卡死
# from faster_whisper import WhisperModel  # 已移至 get_model() 内部延迟导入

# 使用完整路径导入
from app.models.job_models import JobSettings


@dataclass
class ModelCacheInfo:
    """模型缓存信息"""
    model: Any
    key: Tuple[str, str, str]
    load_time: float
    last_used: float
    memory_size: int  # 估算的内存占用(MB)


@dataclass
class PreloadConfig:
    """预加载配置"""
    enabled: bool = True
    default_models: List[str] = None
    max_cache_size: int = 3  # 最大缓存模型数量
    memory_threshold: float = 0.8  # 内存使用阈值(80%)
    preload_timeout: int = 300  # 预加载超时时间(秒)
    warmup_enabled: bool = True  # 是否启用预热

    def __post_init__(self):
        if self.default_models is None:
            self.default_models = ["medium"]


class ModelPreloadManager:
    """简化版模型预加载和缓存管理器 - 方案二实现
    
    核心改进:
    1. 统一锁机制避免死锁
    2. 幂等性预加载避免重复执行
    3. 缓存版本号确保状态同步
    4. 标准化日志便于调试
    """
    
    def __init__(self, config: PreloadConfig = None):
        self.config = config or PreloadConfig()
        self.logger = self._setup_logger()

        # 模型缓存 (LRU)
        self._whisper_cache: OrderedDict[Tuple[str, str, str], ModelCacheInfo] = OrderedDict()

        # SenseVoice 模型缓存（单例模式）
        self._sensevoice_service = None

        # 统一锁 - 简化并发控制，避免多锁死锁
        self._global_lock = threading.RLock()

        # 简化的预加载状态 - 单一数据源
        self._preload_status = {
            "is_preloading": False,
            "progress": 0.0,
            "current_model": "",
            "total_models": 0,
            "loaded_models": 0,
            "errors": [],
            "failed_attempts": 0,
            "last_attempt_time": 0,
            "max_retry_attempts": 3,
            "retry_cooldown": 30,
            "cache_version": int(time.time())  # 缓存版本号，用于状态同步
        }

        # 预加载任务管理 - 实现幂等性
        self._preload_promise: Optional[asyncio.Task] = None

        # 内存监控
        self._memory_monitor = MemoryMonitor()

        self.logger.info("ModelPreloadManager初始化完成 - 简化架构")
    
    def _setup_logger(self) -> logging.Logger:
        """设置标准化的日志记录器"""
        logger = logging.getLogger(f"{__name__}.ModelPreloadManager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - [模型管理] - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def get_preload_status(self) -> Dict[str, Any]:
        """获取预加载状态 - 线程安全版本"""
        with self._global_lock:
            status = self._preload_status.copy()
            self.logger.debug(f"状态查询: 预加载={status['is_preloading']}, 进度={status['progress']:.1f}%, 已加载={status['loaded_models']}")
            return status
    
    def get_cache_status(self) -> Dict[str, Any]:
        """获取缓存状态 - 线程安全版本"""
        with self._global_lock:
            whisper_models = [
                {
                    "key": info.key,
                    "memory_mb": info.memory_size,
                    "last_used": info.last_used,
                    "load_time": info.load_time
                }
                for info in self._whisper_cache.values()
            ]

            total_memory = sum(info.memory_size for info in self._whisper_cache.values())

            cache_status = {
                "whisper_models": whisper_models,
                "total_memory_mb": total_memory,
                "max_cache_size": self.config.max_cache_size,
                "memory_info": self._memory_monitor.get_memory_info(),
                "cache_version": self._preload_status["cache_version"]
            }

            self.logger.debug(f"缓存查询: Whisper模型={len(whisper_models)}个, 内存={total_memory}MB")
            return cache_status
    
    async def preload_models(self, progress_callback=None) -> Dict[str, Any]:
        """预加载默认模型 - 重构版：Demucs + VAD，无 Whisper

        预加载逻辑：
        1. 预加载 Silero VAD 模型（内置，快速）
        2. 预加载默认 Demucs 模型（htdemucs）
        3. 【移除】不再预加载 Whisper 模型（按需加载）
        """
        with self._global_lock:
            if self._preload_status["is_preloading"]:
                self.logger.info("预加载已在进行中，返回已有任务")
                return {"success": True, "message": "预加载已在进行中"}

            if not self.config.enabled:
                self.logger.warning("模型预加载功能已禁用")
                return {"success": False, "message": "预加载功能已禁用"}

            self._preload_status.update({
                "is_preloading": True,
                "progress": 0.0,
                "current_model": "",
                "total_models": 2,  # VAD + Demucs
                "loaded_models": 0,
                "errors": [],
                "last_attempt_time": time.time()
            })

        try:
            success_count = 0

            # ===== 1. 预加载 Silero VAD 模型（保留原逻辑）=====
            try:
                self.logger.info("[预加载] 加载 Silero VAD 模型...")
                with self._global_lock:
                    self._preload_status.update({
                        "current_model": "Silero VAD",
                        "progress": 0.0
                    })

                from pathlib import Path as PathlibPath
                from silero_vad.utils_vad import OnnxWrapper

                builtin_model_path = PathlibPath(__file__).parent.parent / "assets" / "silero" / "silero_vad.onnx"

                if builtin_model_path.exists():
                    _ = OnnxWrapper(str(builtin_model_path), force_onnx_cpu=False)
                    self.logger.info("Silero VAD 模型预加载成功")
                    success_count += 1
                    with self._global_lock:
                        self._preload_status["loaded_models"] += 1
                else:
                    self.logger.warning(f"Silero VAD 模型缺失: {builtin_model_path}")

            except Exception as e:
                self.logger.warning(f"Silero VAD 预加载失败（非致命）: {e}")

            # ===== 2. 预加载 Demucs 模型（新增）=====
            try:
                self.logger.info("[预加载] 加载 Demucs 模型...")
                with self._global_lock:
                    self._preload_status.update({
                        "current_model": "Demucs (htdemucs)",
                        "progress": 50.0
                    })

                from app.services.demucs_service import get_demucs_service
                demucs_service = get_demucs_service()

                # 在后台线程中执行同步的模型加载，避免阻塞主线程
                loop = asyncio.get_event_loop()
                preload_success = await loop.run_in_executor(
                    None,
                    demucs_service.preload_model,
                    "htdemucs"
                )

                if preload_success:
                    self.logger.info("Demucs 模型预加载成功")
                    success_count += 1
                    with self._global_lock:
                        self._preload_status["loaded_models"] += 1
                else:
                    self.logger.warning("Demucs 模型预加载返回失败")
                    with self._global_lock:
                        self._preload_status["errors"].append("Demucs 预加载失败")

            except Exception as e:
                self.logger.warning(f"Demucs 预加载失败（非致命）: {e}")
                with self._global_lock:
                    self._preload_status["errors"].append(f"Demucs 预加载失败: {e}")

            # ===== 【移除】不再预加载 Whisper 模型 =====
            # 原来的 Whisper 预加载代码已删除
            # Whisper 仅在后处理复核阶段按需加载

            # 完成预加载
            success = success_count > 0

            with self._global_lock:
                self._preload_status.update({
                    "is_preloading": False,
                    "progress": 100.0,
                    "current_model": "",
                    "cache_version": int(time.time())
                })

            result = {
                "success": success,
                "loaded_models": success_count,
                "total_models": 2,
                "errors": self._preload_status["errors"].copy()
            }

            if success:
                self.logger.info(f"预加载任务成功完成: {success_count}/2 个模型")
            else:
                self.logger.warning("预加载任务完成但无成功加载的模型")

            if progress_callback:
                progress_callback(self._preload_status.copy())

            return result

        except Exception as e:
            self.logger.error(f"预加载异常: {e}", exc_info=True)
            with self._global_lock:
                self._preload_status.update({
                    "is_preloading": False,
                    "progress": 0.0,
                    "errors": [str(e)]
                })
            return {"success": False, "message": str(e)}

    def reset_preload_attempts(self):
        """重置预加载失败计数 - 线程安全版本"""
        with self._global_lock:
            old_attempts = self._preload_status["failed_attempts"]
            self._preload_status["failed_attempts"] = 0
            self._preload_status["last_attempt_time"] = 0
            self._preload_status["cache_version"] = int(time.time())
            
        self.logger.info(f"预加载失败计数已重置: {old_attempts} -> 0")
    
    def get_model(self, settings: JobSettings):
        """获取Whisper模型 (带LRU缓存) - 简化版本"""
        key = (settings.model, settings.compute_type, settings.device)
        
        with self._global_lock:
            # 命中缓存
            if key in self._whisper_cache:
                info = self._whisper_cache[key]
                info.last_used = time.time()
                # 移到最后 (最近使用)
                self._whisper_cache.move_to_end(key)
                self.logger.debug(f"命中模型缓存: {key}")
                return info.model
            
            # 缓存未命中，加载新模型
            self.logger.info(f"需要加载新模型: {key}")
            return self._load_whisper_model(settings)
    
    def _load_whisper_model(self, settings: JobSettings):
        """加载Whisper模型 - 简化版本带并发保护"""
        key = (settings.model, settings.compute_type, settings.device)

        # 再次检查缓存（避免并发加载同一模型）
        with self._global_lock:
            if key in self._whisper_cache:
                info = self._whisper_cache[key]
                info.last_used = time.time()
                self._whisper_cache.move_to_end(key)
                self.logger.debug(f"并发检查命中缓存，避免重复加载: {key}")
                return info.model

        self.logger.info(f"开始加载新Whisper模型: {key}")

        # 检查内存
        if not self._memory_monitor.check_memory_available():
            self.logger.warning("内存不足，尝试清理缓存")
            self._cleanup_old_models()

        # 检查缓存大小
        with self._global_lock:
            if len(self._whisper_cache) >= self.config.max_cache_size:
                self._evict_lru_model()

        try:
            start_time = time.time()

            # 处理 auto 模式：解析为具体的计算类型
            compute_type_resolved = settings.compute_type
            if compute_type_resolved == "auto":
                from app.services.whisper_service import get_auto_compute_type
                compute_type_resolved = get_auto_compute_type(settings.device)
                self.logger.info(f"auto模式已解析为: {compute_type_resolved}")

            self.logger.info(f"正在从磁盘加载模型 {settings.model} (device={settings.device}, compute_type={compute_type_resolved})")

            # 导入配置以获取缓存路径
            from app.core.config import config

            # 延迟导入 faster_whisper，避免启动时加载 ctranslate2 导致首次启动卡死
            from faster_whisper import WhisperModel

            # 使用 faster_whisper 的 WhisperModel 加载模型
            model = WhisperModel(
                settings.model,
                device=settings.device,
                compute_type=compute_type_resolved,  # 使用解析后的计算类型
                download_root=str(config.HF_CACHE_DIR),
                local_files_only=True
            )
            load_time = time.time() - start_time

            # 估算内存使用
            memory_size = self._estimate_model_memory(model)

            # 添加到缓存
            info = ModelCacheInfo(
                model=model,
                key=key,
                load_time=load_time,
                last_used=time.time(),
                memory_size=memory_size
            )

            with self._global_lock:
                self._whisper_cache[key] = info
                # 更新缓存版本号
                self._preload_status["cache_version"] = int(time.time())

            self.logger.info(f"成功加载并缓存Whisper模型 {key} (内存: {memory_size}MB, 耗时: {load_time:.2f}s)")
            return model

        except Exception as e:
            self.logger.error(f"加载Whisper模型失败 {key}: {str(e)}", exc_info=True)
            raise

    def _warmup_model(self, model):
        """预热模型 - 空跑一次确保完全加载"""
        try:
            self.logger.debug("开始模型预热")

            # 创建虚拟音频数据 (1秒静音)
            import numpy as np
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 16kHz 1秒

            # 使用 transcribe 方法预热模型
            segments, _ = model.transcribe(dummy_audio)
            _ = list(segments)  # 触发生成器

            self.logger.debug("模型预热完成")

        except Exception as e:
            self.logger.warning(f"模型预热失败: {str(e)}")
    
    def _evict_lru_model(self):
        """驱逐最久未使用的模型 - 需要在锁内调用"""
        if not self._whisper_cache:
            return
        
        # 最久未使用的在开头
        oldest_key = next(iter(self._whisper_cache))
        info = self._whisper_cache.pop(oldest_key)
        
        self.logger.info(f"驱逐LRU模型: {oldest_key}, 释放内存: {info.memory_size}MB")
        
        # 释放内存
        del info.model
        del info
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _cleanup_old_models(self):
        """清理旧模型释放内存 - 需要在锁外调用"""
        current_time = time.time()
        to_remove = []
        
        with self._global_lock:
            for key, info in self._whisper_cache.items():
                # 超过10分钟未使用的模型
                if current_time - info.last_used > 600:
                    to_remove.append(key)
            
            for key in to_remove:
                info = self._whisper_cache.pop(key)
                self.logger.info(f"清理旧模型: {key}")
                del info.model
                del info
        
        if to_remove:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info(f"💫 清理了 {len(to_remove)} 个旧模型")
    
    def _estimate_model_memory(self, model) -> int:
        """估算模型内存使用 (MB)"""
        try:
            # 简单估算，基于模型参数
            if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
                total_params = sum(p.numel() for p in model.model.parameters())
                # 假设每个参数4字节 (float32) 或 2字节 (float16)
                bytes_per_param = 2  # float16
                total_bytes = total_params * bytes_per_param
                return int(total_bytes / (1024 * 1024))  # 转换为MB
        except:
            pass
        
        # 默认估算值
        return 500  # 默认500MB
    
    def clear_cache(self):
        """清空所有缓存 - 简化版本，立即同步状态"""
        with self._global_lock:
            # 记录清理前的缓存状态
            whisper_count = len(self._whisper_cache)
            total_memory = sum(info.memory_size for info in self._whisper_cache.values())

            # 清理Whisper模型缓存
            for info in self._whisper_cache.values():
                del info.model
            self._whisper_cache.clear()

            # 清理 SenseVoice 模型缓存
            self.unload_sensevoice()

            # 立即更新预加载状态 - 解决状态同步问题
            self._preload_status.update({
                "loaded_models": 0,
                "is_preloading": False,
                "progress": 0.0,
                "current_model": "",
                "errors": [],
                "cache_version": int(time.time())  # 更新缓存版本号
            })

        # 垃圾回收和GPU内存清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info(f"已清空所有模型缓存: Whisper={whisper_count}个, 释放内存={total_memory}MB")

    def evict_model(self, model_id: str, device: str = "cuda", compute_type: str = "float16"):
        """
        清理指定Whisper模型的缓存

        Args:
            model_id: 模型ID
            device: 设备类型
            compute_type: 计算类型
        """
        key = (model_id, compute_type, device)

        with self._global_lock:
            if key in self._whisper_cache:
                info = self._whisper_cache.pop(key)
                self.logger.info(f"清理模型缓存: {key}, 释放内存: {info.memory_size}MB")

                # 释放内存
                del info.model
                del info

                # 更新预加载状态中的loaded_models计数
                self._preload_status["loaded_models"] = len(self._whisper_cache)
                self._preload_status["cache_version"] = int(time.time())

        # 垃圾回收和GPU内存清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========== SenseVoice 模型管理 ==========

    def get_sensevoice_model(self):
        """获取 SenseVoice 模型（单例）"""
        with self._global_lock:
            if self._sensevoice_service is None:
                try:
                    from app.services.sensevoice_onnx_service import get_sensevoice_service
                    self._sensevoice_service = get_sensevoice_service()

                    if not self._sensevoice_service.is_loaded:
                        self._sensevoice_service.load_model()

                    # 更新缓存版本
                    self._preload_status["cache_version"] = int(time.time())
                    self.logger.info("SenseVoice 模型已加载到缓存")

                except Exception as e:
                    self.logger.error(f"加载 SenseVoice 模型失败: {e}")
                    self._sensevoice_service = None
                    raise

            return self._sensevoice_service

    def unload_sensevoice(self):
        """卸载 SenseVoice 模型"""
        with self._global_lock:
            if self._sensevoice_service is not None:
                try:
                    self._sensevoice_service.unload_model()
                    self._sensevoice_service = None

                    # 更新缓存版本
                    self._preload_status["cache_version"] = int(time.time())

                    self.logger.info("SenseVoice 模型已卸载")
                except Exception as e:
                    self.logger.error(f"卸载 SenseVoice 模型失败: {e}")

        # 垃圾回收和GPU内存清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_demucs(self):
        """
        卸载 Demucs 模型（显式释放显存）

        注意：Demucs 模型通常在 transcription_service 中管理
        此方法提供统一的显存释放接口
        """
        with self._global_lock:
            self.logger.info("触发 Demucs 显存释放")

        # 垃圾回收和GPU内存清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("PyTorch 显存已释放")

    # ========== 单模型管理接口 - 委托给模型管理服务 ==========

    def download_whisper_model(self, model_id: str) -> bool:
        """
        下载单个Whisper模型（委托给模型管理服务）

        Args:
            model_id: 模型ID (tiny, base, small, medium, large-v2, large-v3)

        Returns:
            bool: 是否成功启动下载
        """
        try:
            from app.services.model_manager_service import get_model_manager
            model_mgr = get_model_manager()
            success = model_mgr.download_whisper_model(model_id)

            if success:
                self.logger.info(f"已委托模型管理服务下载Whisper模型: {model_id}")
            return success

        except Exception as e:
            self.logger.error(f"下载Whisper模型失败: {model_id} - {e}")
            return False

    def delete_whisper_model(self, model_id: str) -> bool:
        """
        删除Whisper模型（委托给模型管理服务，并清理缓存）

        Args:
            model_id: 模型ID

        Returns:
            bool: 是否删除成功
        """
        try:
            from app.services.model_manager_service import get_model_manager
            model_mgr = get_model_manager()

            # 先从缓存中移除
            with self._global_lock:
                keys_to_remove = [k for k in self._whisper_cache.keys() if k[0] == model_id]
                for key in keys_to_remove:
                    info = self._whisper_cache.pop(key)
                    del info.model
                    self.logger.debug(f"从缓存中移除模型: {key}")

                # 更新缓存版本号
                self._preload_status["cache_version"] = int(time.time())

            # 清理GPU内存
            if keys_to_remove:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 委托给模型管理服务删除磁盘文件
            success = model_mgr.delete_whisper_model(model_id)

            if success:
                self.logger.info(f"已删除Whisper模型: {model_id}")
            return success

        except Exception as e:
            self.logger.error(f"删除Whisper模型失败: {model_id} - {e}")
            return False

    def list_all_models(self) -> Dict[str, Any]:
        """
        列出所有模型的状态（整合磁盘状态和缓存状态）

        Returns:
            Dict: 包含whisper模型的状态信息
        """
        try:
            from app.services.model_manager_service import get_model_manager
            model_mgr = get_model_manager()

            # 获取磁盘上的模型状态
            whisper_models = [
                {
                    "model_id": m.model_id,
                    "size_mb": m.size_mb,
                    "status": m.status,
                    "download_progress": m.download_progress,
                    "local_path": m.local_path,
                    "description": m.description,
                    "cached": any(k[0] == m.model_id for k in self._whisper_cache.keys())
                }
                for m in model_mgr.list_whisper_models()
            ]

            return {
                "whisper_models": whisper_models,
                "cache_info": {
                    "whisper_cached": len(self._whisper_cache),
                    "total_memory_mb": sum(info.memory_size for info in self._whisper_cache.values())
                }
            }

        except Exception as e:
            self.logger.error(f"列出模型失败: {e}")
            return {"error": str(e)}


class MemoryMonitor:
    """内存监控器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        try:
            # 系统内存
            memory = psutil.virtual_memory()

            # GPU内存 (如果可用)
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3),  # GB
                    "gpu_memory_allocated": torch.cuda.memory_allocated() / (1024**3),  # GB
                    "gpu_memory_cached": torch.cuda.memory_reserved() / (1024**3),  # GB
                }

            return {
                "system_memory_total": memory.total / (1024**3),  # GB
                "system_memory_used": memory.used / (1024**3),  # GB
                "system_memory_percent": memory.percent,
                **gpu_info
            }
        except Exception as e:
            self.logger.error(f"获取内存信息失败: {str(e)}")
            return {}

    def check_memory_available(self, threshold: float = 0.85) -> bool:
        """检查内存是否充足"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent < (threshold * 100)
        except:
            return True  # 默认认为内存充足


# ========== 全局单例模式 - 提供统一的模型管理器接口 ==========

_model_manager: Optional[ModelPreloadManager] = None


def initialize_model_manager(config: PreloadConfig = None) -> ModelPreloadManager:
    """
    初始化全局模型管理器

    Args:
        config: 预加载配置

    Returns:
        ModelPreloadManager: 模型管理器实例
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelPreloadManager(config)
        logging.getLogger(__name__).info(" 全局模型预加载管理器已初始化")
    return _model_manager


def get_model_manager() -> Optional[ModelPreloadManager]:
    """
    获取全局模型管理器

    Returns:
        Optional[ModelPreloadManager]: 模型管理器实例，未初始化则返回None
    """
    return _model_manager


async def preload_default_models(progress_callback=None) -> Dict[str, Any]:
    """
    预加载默认模型

    Args:
        progress_callback: 进度回调函数

    Returns:
        Dict: 预加载结果
    """
    if _model_manager is None:
        return {"success": False, "message": "模型管理器未初始化"}

    return await _model_manager.preload_models(progress_callback)


def get_preload_status() -> Dict[str, Any]:
    """
    获取预加载状态

    Returns:
        Dict: 预加载状态信息
    """
    if _model_manager is None:
        return {"is_preloading": False, "message": "模型管理器未初始化"}

    return _model_manager.get_preload_status()


def get_cache_status() -> Dict[str, Any]:
    """
    获取缓存状态

    Returns:
        Dict: 缓存状态信息
    """
    if _model_manager is None:
        return {"message": "模型管理器未初始化"}

    return _model_manager.get_cache_status()
