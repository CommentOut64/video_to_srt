"""
统一SSE (Server-Sent Events) 管理服务
支持多频道、多客户端、线程安全的实时事件推送

使用场景：
1. 模型下载进度推送（频道：models）
2. 转录任务进度推送（频道：project:{project_id}）
3. 转录文字流式输出（频道：project:{project_id}）

核心原则：
- 单通道原则：每个资源只建立一个SSE连接
- 轻量推送原则：只推送小数据和信号，大数据用HTTP GET
- 重连即全量：断线重连后，推送initial_state，客户端全量拉取
- 线程安全：支持从后台线程广播消息到异步事件循环
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Callable
from collections import defaultdict
from fastapi import Request

logger = logging.getLogger(__name__)


class SSEManager:
    """统一SSE连接管理器"""

    def __init__(self, heartbeat_interval: int = 10, max_queue_size: int = 1000):
        """
        初始化SSE管理器

        Args:
            heartbeat_interval: 心跳间隔（秒）
            max_queue_size: 每个连接的消息队列最大容量
        """
        # 连接池：{channel_id: [queue1, queue2, ...]}
        self.connections: Dict[str, List[asyncio.Queue]] = defaultdict(list)

        # 配置
        self.heartbeat_interval = heartbeat_interval
        self.max_queue_size = max_queue_size

        # 统计信息
        self.total_connections = 0
        self.total_messages_sent = 0

        # 主事件循环引用（在应用启动时设置）
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info(f"SSE管理器已初始化 (心跳: {heartbeat_interval}s, 队列: {max_queue_size})")

    async def subscribe(
        self,
        channel_id: str,
        request: Request,
        initial_state_callback: Optional[Callable] = None
    ):
        """
        订阅指定频道的SSE事件流

        Args:
            channel_id: 频道ID（如 "models", "project:abc123"）
            request: FastAPI请求对象
            initial_state_callback: 初始状态回调函数（可选，用于推送initial_state）

        Yields:
            SSE格式的消息字符串
        """
        # 如果已存在同频道连接，主动踢掉旧连接，确保单活跃连接（避免多编辑器状态混乱）
        if channel_id in self.connections and self.connections[channel_id]:
            logger.info(f"SSE频道已有连接，踢掉旧连接后重建: {channel_id}")
            old_connections = list(self.connections[channel_id])
            self.connections[channel_id].clear()
            for q in old_connections:
                try:
                    q.put_nowait({"event": "force_disconnect", "data": {"reason": "new_connection"}})
                except Exception as e:
                    logger.debug(f"推送强制断开事件失败: {e}")

        # 创建此连接的专用队列
        event_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.connections[channel_id].append(event_queue)
        self.total_connections += 1

        connection_id = f"{channel_id}#{len(self.connections[channel_id])}"

        try:
            logger.debug(f"SSE连接已建立: {connection_id} (频道: {channel_id}, 总连接: {self._get_total_active_connections()})")

            # 1. 发送连接成功消息
            yield self._format_sse("connected", {
                "channel_id": channel_id,
                "message": "SSE连接已建立",
                "timestamp": time.time()
            })

            # 2. 发送初始状态（如果提供了回调）
            if initial_state_callback:
                try:
                    initial_state = initial_state_callback()
                    if initial_state:
                        yield self._format_sse("initial_state", initial_state)
                        logger.debug(f"发送初始状态: {connection_id}")
                except Exception as e:
                    logger.error(f"获取初始状态失败: {e}")

            # 3. 持续推送消息
            heartbeat_counter = 0
            while True:
                # 检查客户端是否断开
                if await request.is_disconnected():
                    logger.debug(f"客户端已断开: {connection_id}")
                    break

                try:
                    # 等待新消息（超时后发送心跳）
                    message = await asyncio.wait_for(
                        event_queue.get(),
                        timeout=self.heartbeat_interval
                    )

                    # 特殊事件：强制断开，用于踢掉旧连接
                    if message.get("event") == "force_disconnect":
                        yield self._format_sse("force_disconnect", message.get("data", {}))
                        logger.debug(f"收到强制断开指令，关闭连接: {connection_id}")
                        break

                    # 发送事件
                    formatted = self._format_sse(message["event"], message["data"])
                    yield formatted
                    self.total_messages_sent += 1

                except asyncio.TimeoutError:
                    # 超时，发送心跳
                    heartbeat_counter += 1
                    yield self._format_sse("ping", {
                        "timestamp": time.time(),
                        "count": heartbeat_counter
                    })

        except asyncio.CancelledError:
            logger.debug(f"SSE连接被取消: {connection_id}")
        except Exception as e:
            logger.error(f"SSE错误: {connection_id} - {e}")
        finally:
            # 清理连接
            try:
                self.connections[channel_id].remove(event_queue)
                if not self.connections[channel_id]:
                    del self.connections[channel_id]
                logger.debug(f"SSE连接已断开: {connection_id} (剩余连接: {self._get_total_active_connections()})")
            except (ValueError, KeyError):
                pass

    async def broadcast(self, channel_id: str, event: str, data: dict):
        """
        向指定频道的所有订阅者广播消息（异步安全）

        Args:
            channel_id: 频道ID
            event: 事件类型（如 "progress", "fragment", "signal"）
            data: 事件数据（字典）
        """
        if channel_id not in self.connections:
            logger.debug(f"频道无连接，跳过广播: {channel_id}")
            return

        message = {
            "event": event,
            "data": data
        }

        success_count = 0
        failed_count = 0

        for queue in self.connections[channel_id][:]:  # 使用切片避免遍历时修改
            try:
                # 检查队列容量，避免阻塞
                if queue.qsize() >= self.max_queue_size * 0.95:
                    # 队列接近满，跳过此次更新
                    failed_count += 1
                    logger.debug(f"队列已满，跳过更新: {channel_id}")
                    continue

                # 非阻塞放入队列
                queue.put_nowait(message)
                success_count += 1

            except asyncio.QueueFull:
                failed_count += 1
                logger.debug(f"队列满，放入失败: {channel_id}")
            except Exception as e:
                failed_count += 1
                logger.error(f"广播失败: {channel_id} - {e}")

        if success_count > 0:
            logger.debug(f"广播完成: {channel_id} - {event} (成功: {success_count}, 失败: {failed_count})")

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """
        设置主事件循环引用（在应用启动时调用）

        Args:
            loop: uvicorn/FastAPI 的主事件循环
        """
        self.loop = loop
        logger.info("SSE管理器已绑定主事件循环")

    def broadcast_sync(self, channel_id: str, event: str, data: dict):
        """
        从同步上下文（后台线程）广播消息（线程安全）

        用于从后台线程（如转录任务、模型下载任务）向SSE推送消息

        Args:
            channel_id: 频道ID
            event: 事件类型
            data: 事件数据

        注意：此方法依赖于在应用启动时设置的主事件循环
        """
        # 优先使用预先保存的主事件循环
        loop = self.loop

        if loop is None:
            logger.warning(f"SSE主事件循环未设置，跳过推送: {channel_id}/{event}")
            # 回退：尝试获取事件循环（可能不可靠）
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        logger.warning("事件循环已关闭，无法推送SSE消息")
                        return
                except Exception:
                    logger.warning("无法获取事件循环，跳过SSE推送")
                    return

        if loop is None or loop.is_closed():
            logger.warning("事件循环不可用，跳过SSE推送")
            return

        # 检查频道是否有连接
        if channel_id not in self.connections or not self.connections[channel_id]:
            logger.debug(f"频道无活跃连接，跳过推送: {channel_id}")
            return

        # 使用 run_coroutine_threadsafe 从后台线程安全地调度协程
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.broadcast(channel_id, event, data),
                loop
            )
            # 不等待结果，避免阻塞后台线程
        except RuntimeError as e:
            # 忽略事件循环关闭时的错误（_call_connection_lost异常）
            if "Event loop is closed" in str(e) or "_ProactorBasePipeTransport" in str(e):
                logger.debug(f"事件循环已关闭，跳过SSE推送: {channel_id}")
            else:
                logger.warning(f"SSE推送调度失败: {e}")
        except Exception as e:
            logger.warning(f"SSE推送调度失败: {e}")

    def _format_sse(self, event: str, data: dict) -> str:
        """
        格式化为标准SSE消息格式

        Args:
            event: 事件名称
            data: 数据字典

        Returns:
            SSE格式字符串: "event: xxx\ndata: {...}\n\n"
        """
        json_data = json.dumps(data, ensure_ascii=False)
        return f"event: {event}\ndata: {json_data}\n\n"

    def _get_total_active_connections(self) -> int:
        """获取当前活跃连接总数"""
        return sum(len(queues) for queues in self.connections.values())

    def get_channel_stats(self, channel_id: str) -> Dict:
        """
        获取指定频道的统计信息

        Args:
            channel_id: 频道ID

        Returns:
            统计信息字典
        """
        if channel_id not in self.connections:
            return {
                "channel_id": channel_id,
                "active_connections": 0,
                "exists": False
            }

        queues = self.connections[channel_id]
        return {
            "channel_id": channel_id,
            "active_connections": len(queues),
            "queue_sizes": [q.qsize() for q in queues],
            "exists": True
        }

    def get_global_stats(self) -> Dict:
        """
        获取全局统计信息

        Returns:
            全局统计字典
        """
        return {
            "total_channels": len(self.connections),
            "total_connections": self._get_total_active_connections(),
            "total_messages_sent": self.total_messages_sent,
            "channels": list(self.connections.keys())
        }


# ========== SSE 事件 Tag 定义（SenseVoice 扩展）==========

# 进度事件 Tag
SSE_PROGRESS_TAGS = {
    "progress.overall": "总体进度更新",
    "progress.extract": "音频提取进度",
    "progress.bgm_detect": "BGM 检测进度",
    "progress.spectrum_analysis": "频谱分诊进度",
    "progress.demucs": "人声分离进度",
    "progress.vad": "VAD 分段进度",
    "progress.sensevoice": "SenseVoice 转录进度（流式）",
    "progress.whisper": "Whisper 复核进度",
    "progress.llm_proof": "LLM 校对进度",
    "progress.llm_trans": "LLM 翻译进度",
    "progress.srt": "SRT 生成进度",
}

# 字幕流式事件 Tag
SSE_SUBTITLE_TAGS = {
    "subtitle.sv_segment": "SenseVoice 完成一个 VAD 段",
    "subtitle.sv_sentence": "SenseVoice 完成一个句子",
    "subtitle.draft": "草稿字幕（快流）",
    "subtitle.replace_chunk": "Chunk 替换（慢流定稿）",
    "subtitle.restored": "断点恢复字幕",
    "subtitle.finalized": "极速模式定稿字幕",
    "subtitle.revised": "句级修订（含说话人改绑）",
    "subtitle.speaker_profiles": "说话人资料变更",
    "subtitle.whisper_patch": "Whisper 复核覆盖一个句子",
    "subtitle.llm_proof": "LLM 校对覆盖一个句子",
    "subtitle.llm_trans": "LLM 翻译完成一个句子",
    "subtitle.batch_update": "批量更新多个句子",
}

# 信号事件 Tag
SSE_SIGNAL_TAGS = {
    "signal.job_start": "任务开始",
    "signal.job_complete": "任务完成",
    "signal.job_failed": "任务失败",
    "signal.phase_start": "阶段开始",
    "signal.phase_complete": "阶段完成",
    "signal.pause_pending": "暂停请求已接收，等待流水线响应",
    "signal.pause_ack": "暂停已确认（检查点落盘完成）",
    "signal.circuit_breaker": "熔断触发",
    "signal.model_upgrade": "模型升级",
    "signal.cache_gc_recommend": "缓存清理提示",
}

# 模型下载事件 Tag
SSE_MODEL_TAGS = {
    "model.download.start": "模型下载开始",
    "model.download.progress": "模型下载进度",
    "model.download.complete": "模型下载完成",
    "model.download.error": "模型下载失败",
    "model.download.cache_hit": "模型已在本地",
}

# 预处理事件 Tag
SSE_PREPROCESS_TAGS = {
    "preprocessing.langid.started": "LangID 语言检测开始",
    "preprocessing.langid.progress": "LangID 语言检测进度",
    "preprocessing.langid.completed": "LangID 语言检测完成",
    "preprocessing.langid.error": "LangID 语言检测失败",
    "preprocessing.speaker.started": "Speaker 声纹提取开始",
    "preprocessing.speaker.progress": "Speaker 声纹提取进度",
    "preprocessing.speaker.completed": "Speaker 声纹提取完成",
    "preprocessing.speaker.error": "Speaker 声纹提取失败",
}

# 调试事件 Tag
SSE_DEBUG_TAGS = {
    "debug.punctuation": "标点调试输出",
    "debug.punctuation_scheduler": "标点调度决策调试",
}


# ========== SSE 推送辅助方法 ==========

def push_progress_event(sse_manager: SSEManager, job_id: str, phase: str, data: dict):
    """推送进度事件"""
    tag = f"progress.{phase}"
    channel_id = f"project:{job_id}"
    sse_manager.broadcast_sync(channel_id, tag, data)


def push_subtitle_event(sse_manager: SSEManager, job_id: str, event_type: str, sentence_data: dict):
    """推送字幕流式事件"""
    tag = f"subtitle.{event_type}"
    channel_id = f"project:{job_id}"
    sse_manager.broadcast_sync(channel_id, tag, sentence_data)


def push_signal_event(sse_manager: SSEManager, job_id: str, signal_type: str, message: str = ""):
    """推送信号事件"""
    tag = f"signal.{signal_type}"
    channel_id = f"project:{job_id}"
    sse_manager.broadcast_sync(channel_id, tag, {"job_id": job_id, "signal": signal_type, "message": message})


# ========== 单例模式 ==========

_sse_manager_instance: Optional[SSEManager] = None


def get_sse_manager() -> SSEManager:
    """
    获取SSE管理器单例

    Returns:
        SSEManager: SSE管理器实例
    """
    global _sse_manager_instance
    if _sse_manager_instance is None:
        # 从配置中读取参数（如果需要）
        from app.core.config import config
        heartbeat_interval = getattr(config, 'SSE_HEARTBEAT_INTERVAL', 10)
        max_queue_size = getattr(config, 'SSE_MAX_QUEUE_SIZE', 1000)

        _sse_manager_instance = SSEManager(
            heartbeat_interval=heartbeat_interval,
            max_queue_size=max_queue_size
        )
    return _sse_manager_instance
