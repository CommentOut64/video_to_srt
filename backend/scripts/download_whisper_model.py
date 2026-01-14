"""
Faster-Whisper 模型下载脚本

功能：
- 从 HuggingFace 镜像源下载 CTranslate2 格式的 Whisper 模型
- 下载到项目 models 目录
- 支持断点续传
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from huggingface_hub import snapshot_download
import logging

from backend.app.services.model_validator import ModelValidator  # 直接使用服务端同款校验器

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def download_faster_whisper_model(
    model_size: str = "medium",
    cache_dir: str = None
):
    """
    下载 Faster-Whisper CTranslate2 格式模型

    Args:
        model_size: 模型大小 (tiny, base, small, medium, large-v2, large-v3)
        cache_dir: 缓存目录，默认使用项目 models/huggingface
    """
    if cache_dir is None:
        cache_dir = BASE_DIR / "models" / "huggingface"

    # 确保目录存在
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 配置 HuggingFace 镜像源
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    # Faster-Whisper 的 CTranslate2 模型仓库
    repo_id = f"Systran/faster-whisper-{model_size}"

    logger.info(f"开始下载 Faster-Whisper 模型: {repo_id}")
    logger.info(f"下载目录: {cache_dir}")
    logger.info(f"使用镜像源: {os.environ.get('HF_ENDPOINT')}")

    try:
        # 下载模型
        model_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            resume_download=True,  # 支持断点续传
            local_files_only=False
        )

        logger.info(f"模型下载完成！")
        logger.info(f"模型路径: {model_path}")

        # 使用统一的模型校验器，确保 vocabulary.json 这类新格式不会误判
        model_path = Path(model_path)
        is_complete, missing_files, detail = ModelValidator.validate_whisper_model(model_path)

        if not is_complete:
            logger.warning(f"警告：缺少文件 {missing_files}")
            logger.info(f"模型文件详情:\n{detail}")
            return False
        else:
            logger.info("模型文件完整性验证通过")
            logger.info(f"模型文件详情:\n{detail}")
            return True

    except Exception as e:
        logger.error(f"模型下载失败: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="下载 Faster-Whisper 模型")
    parser.add_argument(
        "--size",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="模型大小"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="缓存目录（默认: models/huggingface）"
    )

    args = parser.parse_args()

    success = download_faster_whisper_model(
        model_size=args.size,
        cache_dir=args.cache_dir
    )

    sys.exit(0 if success else 1)
