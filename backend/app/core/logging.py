"""
统一的日志系统配置（v3.0 增强版）

新增功能:
- 日志轮转（按大小和时间）
- 结构化日志输出（JSON 格式）
- 日志上下文管理器
- 性能日志记录
"""

import logging
import sys
import json
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from app.core.config import config
try:
    from loguru import logger as loguru_logger
except ImportError:  # pragma: no cover - 允许无 Loguru 环境运行
    loguru_logger = None


class _FallbackLoggerAdapter:
    """Loguru 缺失时的最小兼容层，提供 bind 接口。"""

    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None) -> None:
        self._logger = logger
        self._extra = dict(extra or {})

    def bind(self, **fields: Any) -> "_FallbackLoggerAdapter":
        merged = dict(self._extra)
        merged.update(fields)
        return _FallbackLoggerAdapter(self._logger, merged)

    def _log(self, level: int, message: str, *args: Any, **kwargs: Any) -> None:
        if self._extra:
            extra = dict(kwargs.get("extra") or {})
            extra.update(self._extra)
            kwargs["extra"] = extra
        if (args or kwargs) and "{" in message:
            try:
                message = message.format(*args, **kwargs)
                args = ()
            except (IndexError, KeyError, ValueError):
                pass
        self._logger.log(level, message, *args, **kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.DEBUG, message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.INFO, message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.ERROR, message, *args, **kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("exc_info", True)
        self._log(logging.ERROR, message, *args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, message, *args, **kwargs)


class MillisecondFormatter(logging.Formatter):
    """包含毫秒精度的日志格式化器"""

    def format(self, record):
        # 时间戳：精确到毫秒
        import datetime
        ct = datetime.datetime.fromtimestamp(record.created)
        timestamp = ct.strftime('%H:%M:%S') + '.%03d' % (record.msecs)

        # 提取日志来源（模块名）
        logger_name = record.name.split('.')[-1]

        # 统一格式：时间戳 [级别] [来源] 信息
        return f"{timestamp} [{record.levelname}] [{logger_name}] {record.getMessage()}"


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器（JSON 格式）"""

    def format(self, record):
        import datetime

        log_data = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage()
        }

        # 添加额外字段
        if hasattr(record, 'job_id'):
            log_data['job_id'] = record.job_id
        if hasattr(record, 'duration'):
            log_data['duration'] = record.duration
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data

        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class ThirdPartyFilter(logging.Filter):
    """过滤第三方库的多余日志"""

    # 需要完全禁止的日志
    BLOCKED_MESSAGES = [
        "No language specified, language will be first be detected",
        "Performing voice activity detection using Pyannote",
        "Lightning automatically upgraded your loaded checkpoint",
        "Model was trained with pyannote.audio",
        "Model was trained with torch",
        "TensorFloat-32 (TF32) has been disabled",
        "No active speech found in audio",
        "Warning: audio is shorter than 30s, language detection may be inaccurate",
        "Using `TRANSFORMERS_CACHE` is deprecated",
        "ReproducibilityWarning",
        "FutureWarning",
        # Windows asyncio ProactorEventLoop 已知问题，无害
        "_ProactorBasePipeTransport._call_connection_lost",
        "Exception in callback _ProactorBasePipeTransport",
        # V3.2.0+dev.20260218.01: SpeechBrain 模型加载时的冗余日志
        "Using file found at",       # speechbrain.utils.fetching
        "Using symlink found at",    # speechbrain.utils.fetching (Windows)
        "Loading pretrained files for:",  # speechbrain.utils.parameter_transfer
        "CategoricalEncoder.expect_len was never called",  # speechbrain.dataio.encoder
    ]

    def filter(self, record):
        msg = record.getMessage()
        # 检查是否是被阻止的消息
        for blocked in self.BLOCKED_MESSAGES:
            if blocked in msg:
                return False
        return True


def setup_logging(
    enable_rotation: bool = True,
    enable_structured: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """
    配置日志系统（v3.0 增强版）

    Args:
        enable_rotation: 是否启用日志轮转
        enable_structured: 是否启用结构化日志（JSON 格式）
        max_bytes: 单个日志文件最大大小（字节）
        backup_count: 保留的备份文件数量
    """

    # 创建根日志处理器
    root_logger = logging.getLogger()
    log_level = getattr(logging, config.LOG_LEVEL)
    root_logger.setLevel(logging.DEBUG)  # 设置根logger为DEBUG，让处理器来控制级别

    # 清除已有的处理器
    root_logger.handlers.clear()

    # 创建格式化器
    console_formatter = MillisecondFormatter()

    # 控制台输出：固定 INFO 级别（给用户看）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(ThirdPartyFilter())
    root_logger.addHandler(console_handler)

    # 文件输出：固定 DEBUG 级别（给开发者看）
    if enable_rotation:
        # 使用轮转文件处理器
        file_handler = RotatingFileHandler(
            config.LOG_FILE,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
    else:
        # 使用普通文件处理器
        file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(console_formatter)
    file_handler.addFilter(ThirdPartyFilter())
    root_logger.addHandler(file_handler)

    # 结构化日志输出（可选）
    if enable_structured:
        structured_log_file = config.LOG_DIR / "app_structured.log"
        structured_handler = RotatingFileHandler(
            structured_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        structured_handler.setLevel(log_level)
        structured_handler.setFormatter(StructuredFormatter())
        structured_handler.addFilter(ThirdPartyFilter())
        root_logger.addHandler(structured_handler)

    # 设置第三方库日志级别为WARNING
    third_party_loggers = [
        'urllib3', 'multipart', 'transformers',
        'faster_whisper', 'ctranslate2',  # Faster-Whisper 及其底层库
        'silero', 'torch', 'pytorch_lightning', 'pyannote',
        'speechbrain', 'whisper', 'onnxruntime',
        # V3.2.0+dev.20260218.01: SpeechBrain 子模块（from_hparams 可能重置父级别）
        'speechbrain.utils.fetching',
        'speechbrain.utils.parameter_transfer',
        'speechbrain.dataio.encoder',
        # pyannote 子模块
        'pyannote.audio.core.io',
        'pyannote.audio.utils.reproducibility',
        # lightning 子模块
        'lightning', 'lightning.fabric',
    ]

    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
        logging.getLogger(logger_name).addFilter(ThirdPartyFilter())

    # 禁用 uvicorn 的访问日志（INFO级别的请求日志）
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"日志系统已初始化 - 控制台: INFO, 文件: DEBUG")
    logger.info(f"日志文件: {config.LOG_FILE}")
    if enable_rotation:
        logger.info(f"日志轮转已启用 - 最大: {max_bytes // (1024*1024)}MB, 备份: {backup_count}个")
    if enable_structured:
        logger.info(f"结构化日志已启用")

    # V3.2.0+dev.20260203.10: Loguru 迁移准备（分层推进）
    _configure_loguru_if_available(
        level=config.LOG_LEVEL,
        max_bytes=max_bytes,
        backup_count=backup_count,
    )

    return get_loguru_logger(__name__)


def _configure_loguru_if_available(
    level: str,
    max_bytes: int,
    backup_count: int,
) -> None:
    """初始化 Loguru（不影响现有 logging 体系）。"""
    if loguru_logger is None:
        return
    loguru_logger.remove()
    loguru_logger.configure(extra={"logger": "app"})
    log_text_dir = config.LOG_DIR / "log"
    log_json_dir = config.LOG_DIR / "jsonl"
    log_text_dir.mkdir(parents=True, exist_ok=True)
    log_json_dir.mkdir(parents=True, exist_ok=True)
    loguru_logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:HH:mm:ss.SSS}</green> "
               "[<level>{level}</level>] "
               "[<cyan>{extra[logger]}</cyan>] "
               "<level>{message}</level>",
        colorize=True,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    loguru_text_file = log_text_dir / f"loguru_{config.LOG_FILE.stem}.log"
    loguru_logger.add(
        loguru_text_file,
        level="DEBUG",
        rotation=max_bytes,
        retention=backup_count,
        enqueue=True,
        serialize=False,
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} [{level}] [{extra[logger]}] {message}",
    )
    loguru_json_file = log_json_dir / f"loguru_{config.LOG_FILE.stem}.jsonl"
    loguru_logger.add(
        loguru_json_file,
        level="DEBUG",
        rotation=max_bytes,
        retention=backup_count,
        enqueue=True,
        serialize=True,
        backtrace=False,
        diagnose=False,
    )

def _shorten_logger_name(name: str) -> str:
    """将 logger 名称精简为可识别的最短形式。"""
    if not name:
        return "app"
    normalized = str(name).strip()
    if not normalized:
        return "app"
    if "." in normalized:
        return normalized.split(".")[-1]
    if "\\" in normalized or "/" in normalized:
        normalized = normalized.replace("\\", "/")
        return normalized.split("/")[-1]
    return normalized

def get_loguru_logger(name: str, **bind_fields: Any):
    """获取 Loguru 记录器并绑定上下文字段。"""
    if loguru_logger is None:
        base_logger = logging.getLogger(name)
        return _FallbackLoggerAdapter(base_logger, bind_fields)
    base_logger = loguru_logger.bind(logger=_shorten_logger_name(name))
    if bind_fields:
        base_logger = base_logger.bind(**bind_fields)
    return base_logger


def resolve_loguru_logger(
    logger: Optional[Any],
    name: str,
    **bind_fields: Any,
):
    """强制使用 Loguru 记录器，忽略非 Loguru 的入参。"""
    if loguru_logger is None:
        if logger is not None and hasattr(logger, "bind"):
            return logger.bind(**bind_fields) if bind_fields else logger
        base_logger = logging.getLogger(name)
        return _FallbackLoggerAdapter(base_logger, bind_fields)
    if logger is not None and hasattr(logger, "bind"):
        base_logger = logger.bind(logger=_shorten_logger_name(name))
    else:
        base_logger = loguru_logger.bind(logger=_shorten_logger_name(name))
    if bind_fields:
        base_logger = base_logger.bind(**bind_fields)
    return base_logger


@contextmanager
def log_context(
    logger: logging.Logger,
    operation: str,
    level: int = logging.INFO,
    **extra_fields
):
    """
    日志上下文管理器，自动记录操作开始/结束和耗时

    用法:
        with log_context(logger, "模型加载", job_id="job_001"):
            model = load_model()

    Args:
        logger: 日志记录器
        operation: 操作名称
        level: 日志级别
        **extra_fields: 额外的上下文字段
    """
    start_time = time.time()

    # 记录开始
    extra_data = {"operation": operation, **extra_fields}
    logger.log(level, f"{operation} 开始", extra={"extra_data": extra_data})

    try:
        yield
        # 记录成功
        duration = time.time() - start_time
        extra_data["duration"] = round(duration, 3)
        extra_data["status"] = "success"
        logger.log(level, f"{operation} 完成 (耗时: {duration:.3f}s)",
                  extra={"extra_data": extra_data, "duration": duration})
    except Exception as e:
        # 记录失败
        duration = time.time() - start_time
        extra_data["duration"] = round(duration, 3)
        extra_data["status"] = "failed"
        extra_data["error"] = str(e)
        logger.error(f"{operation} 失败 (耗时: {duration:.3f}s): {e}",
                    extra={"extra_data": extra_data, "duration": duration})
        raise


class PerformanceLogger:
    """
    性能日志记录器

    用于记录关键操作的性能指标
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._metrics: Dict[str, list] = {}

    def record(self, metric_name: str, value: float, unit: str = "ms"):
        """记录性能指标"""
        if metric_name not in self._metrics:
            self._metrics[metric_name] = []
        self._metrics[metric_name].append(value)

    def log_summary(self):
        """输出性能摘要"""
        if not self._metrics:
            return

        self.logger.info("=== 性能摘要 ===")
        for metric_name, values in self._metrics.items():
            if values:
                avg = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                self.logger.info(
                    f"{metric_name}: 平均={avg:.2f}, 最小={min_val:.2f}, "
                    f"最大={max_val:.2f}, 次数={len(values)}"
                )

    def clear(self):
        """清除所有指标"""
        self._metrics.clear()


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器实例
    """
    return logging.getLogger(name)
