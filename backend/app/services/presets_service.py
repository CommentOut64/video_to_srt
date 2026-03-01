"""
自定义预设持久化服务

职责：
- 管理 data/user_presets.json 的读写
- 提供 GET / POST / DELETE 操作
- 单用户场景，无需鉴权
"""

import json
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from app.core.config import config

logger = logging.getLogger(__name__)

# 预设存储路径
_PRESETS_FILE: Path = config.BASE_DIR / "data" / "user_presets.json"

# 预设数量上限
_MAX_PRESETS = 20


def _ensure_data_dir() -> None:
    """确保 data 目录存在"""
    _PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)


def _read_presets_file() -> List[dict]:
    """从磁盘读取预设列表"""
    if not _PRESETS_FILE.exists():
        return []
    try:
        text = _PRESETS_FILE.read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, list):
            return data
        logger.warning("[Presets] 预设文件格式异常，返回空列表")
        return []
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"[Presets] 读取预设文件失败: {e}")
        return []


def _write_presets_file(presets: List[dict]) -> None:
    """写入预设列表到磁盘"""
    _ensure_data_dir()
    try:
        _PRESETS_FILE.write_text(
            json.dumps(presets, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        logger.error(f"[Presets] 写入预设文件失败: {e}")
        raise


def get_all_presets() -> List[dict]:
    """获取所有自定义预设"""
    return _read_presets_file()


def add_preset(name: str, preset_config: dict) -> dict:
    """
    新增自定义预设

    Args:
        name: 预设名称
        preset_config: 预设配置（包含 preprocessing / transcription / refinement 等）

    Returns:
        新创建的预设对象

    Raises:
        ValueError: 名称为空或超出数量上限
    """
    name = name.strip()
    if not name:
        raise ValueError("预设名称不能为空")

    presets = _read_presets_file()

    if len(presets) >= _MAX_PRESETS:
        raise ValueError(f"自定义预设数量已达上限 ({_MAX_PRESETS})")

    # 生成时间戳 ID
    preset_id = f"custom_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    new_preset = {
        "id": preset_id,
        "name": name,
        "config": preset_config,
        "created_at": datetime.now().isoformat(),
    }

    presets.append(new_preset)
    _write_presets_file(presets)

    logger.info(f"[Presets] 新增预设: {name} (id={preset_id})")
    return new_preset


def delete_preset(preset_id: str) -> bool:
    """
    删除自定义预设

    Args:
        preset_id: 预设 ID

    Returns:
        是否成功删除
    """
    presets = _read_presets_file()
    original_len = len(presets)
    presets = [p for p in presets if p.get("id") != preset_id]

    if len(presets) == original_len:
        return False

    _write_presets_file(presets)
    logger.info(f"[Presets] 已删除预设: {preset_id}")
    return True
