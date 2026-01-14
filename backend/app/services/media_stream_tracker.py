"""
媒体流占用追踪器

用于标记正在被前端流式读取的任务文件，删除任务前可检查是否仍有占用。
"""

from collections import defaultdict
from threading import Lock
from typing import Dict

_active_streams: Dict[str, int] = defaultdict(int)
_lock = Lock()


def register_stream(job_id: str) -> int:
    """注册一次媒体流访问，返回当前活跃数量。"""
    with _lock:
        _active_streams[job_id] += 1
        return _active_streams[job_id]


def unregister_stream(job_id: str) -> int:
    """取消注册媒体流访问，返回剩余活跃数量。"""
    with _lock:
        if _active_streams.get(job_id, 0) > 1:
            _active_streams[job_id] -= 1
        else:
            _active_streams.pop(job_id, None)
        return _active_streams.get(job_id, 0)


def get_active_streams(job_id: str) -> int:
    """获取指定任务当前活跃媒体流数量。"""
    with _lock:
        return _active_streams.get(job_id, 0)
