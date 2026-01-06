"""
模型完整性验证工具
负责验证模型文件的完整性
"""

from pathlib import Path
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ModelValidator:
    """模型完整性验证器"""

    # Whisper 模型必需的文件
    # 注意：vocabulary 文件可能是 .txt 或 .json 格式（如 large-v3 使用 .json）
    WHISPER_REQUIRED_FILES = [
        "model.bin",
        "config.json",
        "tokenizer.json"
    ]
    
    # vocabulary 文件的可选格式（只需要其中一个）
    WHISPER_VOCABULARY_FILES = [
        "vocabulary.txt",
        "vocabulary.json"
    ]

    # 对齐模型相关常量已移除

    @staticmethod
    def validate_whisper_model(model_path: Path) -> Tuple[bool, List[str], str]:
        """
        验证 Whisper 模型完整性

        Args:
            model_path: 模型目录路径

        Returns:
            Tuple[bool, List[str], str]: (是否完整, 缺失的文件列表, 详细信息)
        """
        if not model_path.exists():
            return False, [], f"模型目录不存在: {model_path}"

        missing_files = []
        file_info = []

        # 检查必需文件
        for file_name in ModelValidator.WHISPER_REQUIRED_FILES:
            file_path = model_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
                file_info.append(f"  x {file_name}: 缺失")
            else:
                size = file_path.stat().st_size
                if size == 0:
                    missing_files.append(file_name)
                    file_info.append(f"  x {file_name}: 0 字节（损坏）")
                else:
                    file_info.append(f"  v {file_name}: {size:,} 字节")

        # 检查 vocabulary 文件（支持 .txt 或 .json 格式）
        vocab_found = False
        vocab_info = None
        for vocab_file in ModelValidator.WHISPER_VOCABULARY_FILES:
            vocab_path = model_path / vocab_file
            if vocab_path.exists():
                size = vocab_path.stat().st_size
                if size > 0:
                    vocab_found = True
                    vocab_info = f"  v {vocab_file}: {size:,} 字节"
                    break
                else:
                    vocab_info = f"  x {vocab_file}: 0 字节（损坏）"
        
        if not vocab_found:
            missing_files.append("vocabulary.txt/json")
            if vocab_info:
                file_info.append(vocab_info)
            else:
                file_info.append(f"  x vocabulary.txt/json: 缺失")
        else:
            file_info.append(vocab_info)

        is_complete = len(missing_files) == 0
        detail = "\n".join(file_info)

        return is_complete, missing_files, detail

    # validate_align_model 方法已移除

    @staticmethod
    def find_model_snapshots(hub_dir: Path, model_name: str) -> List[Path]:
        """
        查找模型的所有快照目录

        Args:
            hub_dir: HuggingFace hub 缓存目录
            model_name: 模型名称（如 "models--Systran--faster-whisper-medium"）

        Returns:
            List[Path]: 快照目录列表
        """
        model_dir = hub_dir / model_name
        if not model_dir.exists():
            return []

        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            return []

        # 返回所有快照目录
        return [d for d in snapshots_dir.iterdir() if d.is_dir()]

