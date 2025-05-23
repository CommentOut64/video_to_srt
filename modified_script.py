import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 允许重复的 OpenMP 库, 避免某些环境下的错误

import subprocess
import platform # 未直接使用, 但可用于更细致的平台判断
import json
import threading
import shutil
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc # 垃圾回收
from contextlib import redirect_stdout, redirect_stderr # 用于重定向标准输出和错误
import warnings
import sys
import io
import re # 正则表达式, 未见明显使用, 但可能是未来或间接使用
import logging # <--- 新增: 日志模块

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, SpinnerColumn
from rich.table import Table
from rich.live import Live
from rich.box import ROUNDED

# --- Rich Console ---
console = Console() # Rich 控制台对象

# --- Global Logger Instance ---
# 在 setup_logging() 中配置
logger = logging.getLogger(__name__)
LOG_FILE_NAME = "app_runtime.log"
# log_file_path 将在 APP_BASE_PATH 定义后设置

# 忽略特定警告
warnings.filterwarnings("ignore", message="TensorFloat-32 \\(TF32\\) has been disabled")
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

class StderrFilter:
    """
    一个自定义的 stderr 包装器, 用于过滤不希望看到的警告信息 (例如 'libpng warning').
    它还尝试模拟标准错误流的文件描述符和 TTY 属性, 以兼容某些库.
    """
    def __init__(self):
        self.old_stderr = sys.stderr # 保存原始的 stderr
        self._fileno = None # 初始化文件描述符编号
        try:
            self._fileno = self.old_stderr.fileno()
        except (AttributeError, io.UnsupportedOperation):
            self._fileno = None

    def __enter__(self):
        sys.stderr = self # 替换全局 stderr 为当前实例
        return self

    def __exit__(self, *args):
        sys.stderr = self.old_stderr # 恢复原始的 stderr

    def write(self, text):
        if 'libpng warning' not in text:
            self.old_stderr.write(text)

    def flush(self):
        self.old_stderr.flush() # 调用原始 stderr 的 flush

    def fileno(self):
        if self._fileno is not None:
            return self._fileno
        if hasattr(self.old_stderr, 'fileno'):
             current_fileno = self.old_stderr.fileno()
             if isinstance(current_fileno, int):
                 return current_fileno
        raise io.UnsupportedOperation("underlying stream does not support fileno or returned non-integer")

    def isatty(self):
        if hasattr(self.old_stderr, 'isatty'):
            return self.old_stderr.isatty()
        return False

if not isinstance(sys.stderr, StderrFilter):
    sys.stderr = StderrFilter()

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PILLOW_SILENCE_LIBPNG'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    import torch
except ImportError:
    pass

TORCH_VERSION = "2.5.1"
TORCHAUDIO_VERSION = "2.0.2"
TORCHVISION_VERSION = "0.15.2"
WHISPERX_REPO = "git+https://github.com/m-bain/whisperx.git"

def get_app_base_path():
    if hasattr(sys, 'frozen'):
        return os.path.dirname(sys.executable)
    return os.getcwd()

APP_BASE_PATH = get_app_base_path()
log_file_path = os.path.join(APP_BASE_PATH, LOG_FILE_NAME) # <--- 设置日志文件完整路径

def setup_logging():
    """配置全局日志记录器."""
    global logger, log_file_path
    logger.setLevel(logging.DEBUG) # 记录 DEBUG 及以上级别的信息

    # 文件处理器 (FileHandler)
    fh = logging.FileHandler(log_file_path, encoding='utf-8', mode='a') # 'a' 为追加模式
    fh.setLevel(logging.DEBUG)

    # 日志格式器 (Formatter)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
    fh.setFormatter(formatter)

    # 将处理器添加到记录器 (避免重复添加)
    if not logger.handlers:
        logger.addHandler(fh)
        # 可以选择性地添加一个 StreamHandler 以同时在控制台输出日志 (主要用于调试脚本本身)
        # sh = logging.StreamHandler(sys.stdout)
        # sh.setLevel(logging.INFO) # 例如, 控制台只显示 INFO 及以上
        # sh.setFormatter(formatter)
        # logger.addHandler(sh)

MODEL_CACHE_DIR = os.path.join(APP_BASE_PATH, "model_cache")
TEMP_DIR = os.path.join(APP_BASE_PATH, "temp_files")
STATUS_FILE = os.path.join(TEMP_DIR, "status.json")

SEGMENT_LENGTH_MS = 60 * 1000
SILENCE_SEARCH_DURATION_MS = 2000
MIN_SILENCE_LEN = 300
SILENCE_THRESH = -40

DEVICE = "cpu"
BATCH_SIZE = 16
COMPUTE_TYPE = "int8"
WHISPER_MODEL = "medium"
USE_WORD_TIMESTAMPS_CONFIG = False
CPU_THREADS_PER_MODEL_INSTANCE = max(1, (os.cpu_count() or 4) // 2)

whisper_model_global = None
model_lock = threading.Lock()

def check_dependencies(verbose=True):
    if verbose:
        console.print("[bold blue][INFO][/bold blue] 正在检查必要的依赖项...")
    logger.info("开始检查依赖项...")
    
    dependencies = ["pydub", "tqdm", "transformers", "ffmpeg-python", "faster-whisper", "pytorch-lightning"]
    missing_deps = []
    installed_deps_messages = []

    try:
        import torch
        import torchaudio
        msg = f"PyTorch (版本: {torch.__version__}) 已安装"
        installed_deps_messages.append(f"[bold green]PyTorch[/bold green] (版本: {torch.__version__}) 已安装")
        logger.info(msg)
        msg = f"torchaudio (版本: {torchaudio.__version__}) 已安装"
        installed_deps_messages.append(f"[bold green]torchaudio[/bold green] (版本: {torchaudio.__version__}) 已安装")
        logger.info(msg)
        if torch.cuda.is_available():
            installed_deps_messages.append("CUDA 可用, 但此程序配置为仅使用 CPU.")
            logger.info("CUDA 可用, 但此程序配置为仅使用 CPU.")
        else:
            installed_deps_messages.append("CUDA 不可用. 程序将按预期使用 CPU.")
            logger.info("CUDA 不可用. 程序将按预期使用 CPU.")
        installed_deps_messages.append(f"[bold blue]程序将仅使用 CPU ({DEVICE}) 进行处理.[/bold blue]")
        logger.info(f"程序将仅使用 CPU ({DEVICE}) 进行处理.")
    except ImportError:
        missing_deps.extend([f"torch=={TORCH_VERSION}", f"torchaudio=={TORCHAUDIO_VERSION}", f"torchvision=={TORCHVISION_VERSION}"])
        installed_deps_messages.append("[bold yellow]PyTorch 或 torchaudio 未安装. 程序将仅使用 CPU.[/bold yellow]")
        logger.warning("PyTorch 或 torchaudio 未安装.")

    for dep in dependencies:
        try:
            __import__(dep.replace("-", "_").split("==")[0])
            msg = f"{dep} 已安装"
            installed_deps_messages.append(f"[bold green]{dep}[/bold green] 已安装")
            logger.info(msg)
        except ImportError:
            missing_deps.append(dep)
            logger.warning(f"依赖项缺失: {dep}")

    try:
        import whisperx
        installed_deps_messages.append("[bold green]whisperx[/bold green] 已安装")
        logger.info("whisperx 已安装")
    except ImportError:
        missing_deps.append(WHISPERX_REPO)
        logger.warning(f"依赖项缺失: whisperx (仓库: {WHISPERX_REPO})")

    try:
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode == 0:
            installed_deps_messages.append("[bold green]ffmpeg[/bold green] 已安装")
            logger.info("ffmpeg 已安装")
        else:
            msg = "ffmpeg 检测失败 (命令执行但返回非零). 可能需要手动安装或检查配置."
            if verbose: console.print(f"[bold yellow][WARNING][/bold yellow] {msg}")
            logger.warning(msg)
            missing_deps.append("ffmpeg (手动安装)")
    except FileNotFoundError:
        msg = "未找到 ffmpeg 命令. 请安装 ffmpeg 并确保其在系统路径中."
        if verbose: console.print(f"[bold yellow][WARNING][/bold yellow] {msg}")
        logger.warning(msg)
        missing_deps.append("ffmpeg (手动安装)")

    if verbose:
        for msg_console in installed_deps_messages: # Renamed to avoid conflict
            console.print(f"[INFO] {msg_console}")

    if missing_deps:
        if verbose:
            console.print("\n[bold yellow][WARNING][/bold yellow] 以下依赖项缺失或存在问题:")
            for dep in missing_deps:
                console.print(f"  - [yellow]{dep}[/yellow]")
            console.print("\n请尝试使用菜单中的 “检查与管理依赖项” 功能, 或手动安装以上依赖项.")
        logger.warning(f"依赖检查完成. 缺失的依赖项: {', '.join(missing_deps)}")
        return False
    
    if verbose:
        console.print("[bold green][INFO][/bold green] 依赖检查完成. 所有必要的依赖项均已安装.")
    logger.info("依赖检查完成. 所有必要的依赖项均已安装.")
    return True

def ensure_app_dirs():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        logger.info(f"临时目录已创建: {TEMP_DIR}")
        console.print(f"[bold blue][INFO][/bold blue] 临时目录已创建: {TEMP_DIR}")
    if not os.path.exists(MODEL_CACHE_DIR):
        os.makedirs(MODEL_CACHE_DIR)
        logger.info(f"模型缓存目录已创建: {MODEL_CACHE_DIR}")
        console.print(f"[bold blue][INFO][/bold blue] 模型缓存目录已创建: {MODEL_CACHE_DIR}")

def load_status():
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
                logger.info(f"状态已从 {STATUS_FILE} 加载.")
                return status_data
        except json.JSONDecodeError as e:
            logger.warning(f"状态文件 {STATUS_FILE} 已损坏: {e}. 将重新开始.")
            console.print("[bold yellow][WARNING][/bold yellow] 状态文件已损坏. 将重新开始.")
            return {}
    logger.info(f"状态文件 {STATUS_FILE} 未找到. 返回空状态.")
    return {}

def save_status(status):
    ensure_app_dirs()
    try:
        with open(STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(status, f, ensure_ascii=False, indent=4)
        logger.info(f"状态已保存到 {STATUS_FILE}.")
    except IOError as e:
        logger.error(f"保存状态到 {STATUS_FILE} 失败: {e}")
        console.print(f"[bold red][ERROR][/bold red] 保存状态文件失败: {e}")


def cleanup_temp():
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
            logger.info(f"已清理临时目录: {TEMP_DIR}")
            console.print(f"[bold blue][INFO][/bold blue] 已清理临时目录: {TEMP_DIR}")
        except OSError as e:
            logger.error(f"清理临时文件失败: {e}")
            console.print(f"[bold red][ERROR][/bold red] 清理临时文件失败: {e}")
    else:
        logger.info(f"临时目录 {TEMP_DIR} 不存在, 无需清理.")

def extract_audio(input_file_path, audio_output_path, force_extract=False):
    logger.info(f"请求提取音频: input='{input_file_path}', output='{audio_output_path}', force={force_extract}")
    if not force_extract and os.path.exists(audio_output_path):
        logger.info(f"标准化音频文件已存在, 跳过提取: {audio_output_path}")
        console.print(f"[bold blue][INFO][/bold blue] 标准化音频文件已存在. 跳过提取: {audio_output_path}")
        return True

    command = [
        "ffmpeg", "-y", "-i", input_file_path,
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        audio_output_path
    ]
    logger.info(f"执行 ffmpeg 命令: {' '.join(command)}")
    console.print(f"[bold blue][INFO][/bold blue] 正在处理输入文件 '{os.path.basename(input_file_path)}' 以生成标准化音频...")
    
    try:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
            progress.add_task("FFmpeg 处理中...", total=None)
            process_result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        logger.info(f"标准化音频已成功生成: {audio_output_path}. FFmpeg stdout: {process_result.stdout[:200]}...") # Log snippet of stdout
        console.print(f"[bold green][INFO][/bold green] 标准化音频已成功生成: {audio_output_path}")
        return True
    except FileNotFoundError:
        logger.error("未找到 ffmpeg. 请确保已安装 ffmpeg 并将其添加到系统 PATH.")
        console.print("[bold red][ERROR][/bold red] 未找到 ffmpeg. 请确保已安装 ffmpeg 并将其添加到系统 PATH.")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"使用 ffmpeg 处理文件 '{os.path.basename(input_file_path)}' 时出错. "
                     f"命令: {' '.join(e.cmd)}, 返回码: {e.returncode}\n"
                     f"Stderr: {e.stderr}\nStdout: {e.stdout}", exc_info=False) # exc_info=False as we are logging details manually
        console.print(f"[bold red][ERROR][/bold red] 使用 ffmpeg 处理文件 '{os.path.basename(input_file_path)}' 时出错.")
        console.print(f"命令: {' '.join(e.cmd)}\n返回码: {e.returncode}")
        if e.stderr:
            console.print(f"FFmpeg 错误输出:\n[dim]{e.stderr.strip()}[/dim]")
        else:
            console.print("FFmpeg 未产生 stderr 输出 (或输出为空).")
        if e.stdout:
             console.print(f"FFmpeg 标准输出:\n[dim]{e.stdout.strip()}[/dim]")
        return False

def get_existing_segments(temp_dir_path):
    segments = sorted(glob.glob(os.path.join(temp_dir_path, "segment_*.wav")))
    logger.debug(f"在 {temp_dir_path} 中找到 {len(segments)} 个现有分段.")
    return segments

def split_audio(audio_path, force_split=False):
    logger.info(f"请求分割音频: path='{audio_path}', force={force_split}")
    try:
        from pydub import AudioSegment, silence
        logger.debug("pydub 导入成功.")
    except ImportError as e:
        logger.critical("pydub 未安装, 无法进行音频分割.", exc_info=True)
        console.print("[bold red][CRITICAL][/bold red] pydub 库未安装, 无法分割音频. 请安装 pydub.")
        return []


    existing_segment_files = get_existing_segments(TEMP_DIR)
    status = load_status()

    if not force_split and status.get("segments_info") and existing_segment_files:
        if len(existing_segment_files) == len(status["segments_info"]):
            logger.info("使用先前保存的分段信息.")
            console.print("[bold blue][INFO][/bold blue] 使用先前保存的分段信息.")
            return status["segments_info"]
        else:
            logger.info("现有的分段文件与状态记录不匹配. 将重新分段.")
            console.print("[bold blue][INFO][/bold blue] 现有的分段文件与状态记录不匹配. 将重新分段.")
    elif not force_split and existing_segment_files:
        logger.info("检测到现有的分段文件但无状态信息. 将重新分段以确保时间戳正确.")
        console.print("[bold blue][INFO][/bold blue] 检测到现有的分段文件但无状态信息. 将重新分段以确保时间戳正确.")

    console.print("[bold blue][INFO][/bold blue] 开始音频分段...")
    logger.info("开始音频分段...")
    try:
        audio = AudioSegment.from_wav(audio_path)
        logger.info(f"音频文件 {audio_path} 加载成功, 长度: {len(audio)} ms.")
    except FileNotFoundError:
        logger.error(f"音频文件未找到: {audio_path}")
        console.print(f"[bold red][ERROR][/bold red] 音频文件未找到: {audio_path}")
        return []
    except Exception as e:
        logger.error(f"加载音频文件失败 ({audio_path}): {e}", exc_info=True)
        console.print(f"[bold red][ERROR][/bold red] 加载音频文件失败 ({audio_path}): {e}")
        return []

    audio_length_ms = len(audio)
    segments_info = []
    current_pos_ms = 0
    segment_idx = 0

    if existing_segment_files:
        logger.info("正在清理旧的分段文件...")
        console.print("[bold blue][INFO][/bold blue] 正在清理旧的分段文件...")
        for f_path in existing_segment_files:
            try:
                os.remove(f_path)
                logger.debug(f"已删除旧分段文件: {f_path}")
            except OSError as e:
                logger.warning(f"清理旧分段文件 {f_path} 失败: {e}")
                console.print(f"[bold yellow][WARNING][/bold yellow] 清理旧分段文件 {f_path} 失败: {e}")

    with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TextColumn("({task.completed}/{task.total} ms)"),
                  TimeRemainingColumn(), TimeElapsedColumn(),
                  console=console, transient=False) as progress_bar:
        
        segment_task = progress_bar.add_task("音频分段进度", total=audio_length_ms)
        logger.info(f"开始循环分段, 总长度: {audio_length_ms} ms.")
        while current_pos_ms < audio_length_ms:
            end_pos_ms = current_pos_ms + SEGMENT_LENGTH_MS
            actual_end_pos_ms = min(end_pos_ms, audio_length_ms)

            if actual_end_pos_ms <= current_pos_ms:
                logger.debug(f"分段结束: actual_end_pos_ms ({actual_end_pos_ms}) <= current_pos_ms ({current_pos_ms}).")
                break

            if actual_end_pos_ms < audio_length_ms and \
               (actual_end_pos_ms - current_pos_ms) > SILENCE_SEARCH_DURATION_MS:
                
                search_start_ms = max(current_pos_ms, actual_end_pos_ms - SILENCE_SEARCH_DURATION_MS)
                search_chunk = audio[search_start_ms:actual_end_pos_ms]
                logger.debug(f"分段 {segment_idx}: 在 {search_start_ms}-{actual_end_pos_ms} ms 范围内搜索静音.")
                try:
                    silence_ranges = silence.detect_silence(
                        search_chunk, 
                        min_silence_len=MIN_SILENCE_LEN, 
                        silence_thresh=SILENCE_THRESH
                    )
                    if silence_ranges:
                        first_silence_start_in_search_chunk = silence_ranges[0][0]
                        potential_new_end_ms = search_start_ms + first_silence_start_in_search_chunk
                        if potential_new_end_ms > (current_pos_ms + MIN_SILENCE_LEN):
                            logger.debug(f"分段 {segment_idx}: 在 {potential_new_end_ms} ms 处找到静音点, 更新结束时间.")
                            actual_end_pos_ms = potential_new_end_ms
                        else:
                            logger.debug(f"分段 {segment_idx}: 静音点太靠前, 忽略.")
                    else:
                        logger.debug(f"分段 {segment_idx}: 未找到静音点.")
                except Exception as e:
                    logger.warning(f"在分段 {segment_idx} 静音检测时出错: {e}", exc_info=True)
                    progress_bar.console.print(f"\n[bold yellow][WARNING][/bold yellow] 在分段 {segment_idx} 静音检测时出错: {e}")
            
            segment_audio_chunk = audio[current_pos_ms:actual_end_pos_ms]
            segment_filename = os.path.join(TEMP_DIR, f"segment_{segment_idx}.wav")
            
            try:
                segment_audio_chunk.export(segment_filename, format="wav")
                segments_info.append({"file": segment_filename, "start_ms": current_pos_ms})
                logger.info(f"已导出分段 {segment_idx}: {segment_filename}, start_ms: {current_pos_ms}, end_ms: {actual_end_pos_ms}")
            except Exception as e:
                logger.error(f"导出分段 {segment_filename} 失败: {e}", exc_info=True)
                progress_bar.console.print(f"\n[bold red][ERROR][/bold red] 导出分段 {segment_filename} 失败: {e}")
            
            progress_bar.update(segment_task, advance=(actual_end_pos_ms - current_pos_ms))
            current_pos_ms = actual_end_pos_ms
            segment_idx += 1
    
    logger.info(f"音频分段完成. 共创建 {len(segments_info)} 个分段.")
    console.print(f"[bold green][INFO][/bold green] 音频分段完成. 共创建 {len(segments_info)} 个分段.")
    
    status = load_status()
    status["segments_info"] = segments_info
    save_status(status)
    
    return segments_info

def load_whisper_model_rich():
    logger.info("请求加载 WhisperX 模型.")
    try:
        import whisperx
        logger.debug("whisperx 导入成功.")
    except ImportError as e:
        logger.critical("whisperx 未安装, 无法加载模型.", exc_info=True)
        console.print("[bold red][CRITICAL][/bold red] whisperx 库未安装. 请安装 whisperx.")
        return None

    global whisper_model_global, WHISPER_MODEL, COMPUTE_TYPE, DEVICE, CPU_THREADS_PER_MODEL_INSTANCE, MODEL_CACHE_DIR
    
    if whisper_model_global is None:
        logger.info(f"全局 Whisper 模型为 None, 开始加载. 模型: {WHISPER_MODEL}, 计算类型: {COMPUTE_TYPE}, 设备: {DEVICE}")
        effective_compute_type = COMPUTE_TYPE
        if DEVICE == "cpu" and COMPUTE_TYPE not in ["int8", "float32", "int16"]:
            msg = f"当前计算类型 '{COMPUTE_TYPE}' 可能不是 CPU 上的最优选择. CPU 通常推荐 'int8', 'float32', 或 'int16'. WhisperX 可能会调整或报错."
            logger.warning(msg)
            console.print(f"[bold yellow][WARNING][/bold yellow] {msg}")

        console.print(f"[bold blue][INFO][/bold blue] 正在加载 WhisperX 模型 ([cyan]{WHISPER_MODEL}[/cyan], compute: [cyan]{effective_compute_type}[/cyan], device: [cyan]{DEVICE}[/cyan])...")
        console.print(f"[bold blue][INFO][/bold blue] (首次加载新模型需要从镜像源 '{os.environ.get('HF_ENDPOINT', '默认 Hugging Face 源')}' 下载到 '{MODEL_CACHE_DIR}', 请耐心等待...)")
        logger.info(f"模型下载源: {os.environ.get('HF_ENDPOINT', '默认 Hugging Face 源')}, 缓存目录: {MODEL_CACHE_DIR}")

        model_load_kwargs = {
            "device": DEVICE,
            "compute_type": effective_compute_type,
            "download_root": MODEL_CACHE_DIR,
        }
        if DEVICE == "cpu":
            model_load_kwargs["threads"] = CPU_THREADS_PER_MODEL_INSTANCE
            logger.info(f"WhisperX 模型将使用 {model_load_kwargs['threads']} 个 CPU 线程.")
            console.print(f"[INFO] WhisperX 模型将使用 [cyan]{model_load_kwargs['threads']}[/cyan] 个 CPU 线程 (通过 'threads' 参数传递给 faster-whisper).")
        
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
                progress.add_task("加载 Whisper 模型中...", total=None)
                with model_lock:
                    if whisper_model_global is None:
                         logger.info(f"实际执行 whisperx.load_model with kwargs: {model_load_kwargs}")
                         whisper_model_global = whisperx.load_model(WHISPER_MODEL, **model_load_kwargs)
            logger.info("WhisperX 模型加载成功.")
            console.print("[bold green][INFO][/bold green] WhisperX 模型加载成功.")
        except Exception as e:
            logger.error(f"加载 WhisperX 模型失败: {e}", exc_info=True)
            console.print(f"[bold red][ERROR][/bold red] 加载 WhisperX 模型失败: {e}")
            console.print(f"请检查模型名称 ('{WHISPER_MODEL}'), 计算类型 ('{effective_compute_type}'), 设备 ('{DEVICE}') 设置, 以及模型缓存路径 ('{MODEL_CACHE_DIR}').")
            console.print("确保 WhisperX 和相关依赖 (如 PyTorch, faster-whisper) 已正确安装且与 CPU 兼容.")
            return None
    else:
        logger.info("WhisperX 模型已加载, 使用缓存实例.")
    return whisper_model_global

def transcribe_and_align_segment(segment_info, model_instance, align_model_cache):
    segment_file = segment_info["file"]
    segment_start_ms = segment_info["start_ms"]
    segment_basename = os.path.basename(segment_file)
    logger.info(f"开始处理分段: {segment_basename}, start_ms: {segment_start_ms}")
    
    try:
        import whisperx
    except ImportError as e:
        logger.critical("whisperx 未安装, 无法转录和对齐.", exc_info=True)
        return {"error": "whisperx not installed", "segment_basename": segment_basename}


    detected_language_for_segment = segment_info.get("detected_language")
    logger.debug(f"分段 {segment_basename}: 预检测语言: {detected_language_for_segment}")

    audio_data = None
    try:
        audio_data = whisperx.load_audio(segment_file)
        logger.debug(f"分段 {segment_basename}: 音频加载成功.")
        
        transcription_result = model_instance.transcribe(
            audio_data, 
            batch_size=BATCH_SIZE, 
            verbose=False,
            language=detected_language_for_segment
        )
        logger.debug(f"分段 {segment_basename}: 转录完成. 结果片段数: {len(transcription_result.get('segments', [])) if transcription_result else 0}, 检测语言: {transcription_result.get('language') if transcription_result else 'N/A'}")


        if not transcription_result or not transcription_result.get("segments"):
            logger.warning(f"分段 {segment_basename}: 转录结果为空或无片段.")
            return None

        lang_code = detected_language_for_segment if detected_language_for_segment else transcription_result["language"]
        logger.debug(f"分段 {segment_basename}: 用于对齐的语言代码: {lang_code}")
        
        align_model, align_metadata = align_model_cache.get(lang_code, (None, None))
        
        if align_model is None:
            logger.info(f"分段 {segment_basename}: 语言 '{lang_code}' 的对齐模型不在缓存中, 开始加载.")
            try:
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=lang_code,
                    device=DEVICE
                )
                align_model_cache[lang_code] = (align_model, align_metadata)
                logger.info(f"分段 {segment_basename}: 语言 '{lang_code}' 的对齐模型加载并缓存成功.")
            except Exception as e:
                logger.error(f"加载语言 '{lang_code}' 的对齐模型失败: {e}", exc_info=True)
                raise Exception(f"加载语言 '{lang_code}' 的对齐模型失败 (模型将尝试从默认 HuggingFace 缓存路径下载): {e}")
        else:
            logger.debug(f"分段 {segment_basename}: 使用缓存的语言 '{lang_code}' 对齐模型.")
                
        aligned_result = whisperx.align(
            transcription_result["segments"], 
            align_model, 
            align_metadata, 
            audio_data, 
            device=DEVICE,
        )
        logger.debug(f"分段 {segment_basename}: 对齐完成.")

        segment_start_sec = segment_start_ms / 1000.0
        final_adjusted_alignment = {"segments": []}

        if "word_segments" in aligned_result:
            final_adjusted_alignment["word_segments"] = []
            for word_info in aligned_result["word_segments"]:
                if "start" in word_info and word_info["start"] is not None:
                    word_info["start"] += segment_start_sec
                if "end" in word_info and word_info["end"] is not None:
                    word_info["end"] += segment_start_sec
                final_adjusted_alignment["word_segments"].append(word_info)
        
        for seg in aligned_result["segments"]:
            if "start" in seg and seg["start"] is not None:
                seg["start"] += segment_start_sec
            if "end" in seg and seg["end"] is not None:
                seg["end"] += segment_start_sec
            final_adjusted_alignment["segments"].append(seg)
        
        logger.info(f"分段 {segment_basename}: 成功处理并调整时间戳.")
        return final_adjusted_alignment
    except Exception as e:
        logger.error(f"处理分段 {segment_basename} 时发生错误: {e}", exc_info=True)
        return {"error": str(e), "segment_basename": segment_basename}
    finally:
        if audio_data is not None:
            del audio_data
            gc.collect()
            logger.debug(f"分段 {segment_basename}: 音频数据已清理.")

def process_all_segments(segments_info_list, current_status):
    logger.info(f"开始处理所有 {len(segments_info_list)} 个分段.")
    try:
        import whisperx 
    except ImportError as e:
        logger.critical("whisperx 未安装, 无法处理分段.", exc_info=True)
        console.print("[bold red][CRITICAL][/bold red] whisperx 库未安装.")
        return None

    model_instance = load_whisper_model_rich()
    if model_instance is None:
        logger.error("Whisper 模型未能加载. 中止处理所有分段.")
        console.print("[bold red][ERROR][/bold red] Whisper 模型未能加载. 中止处理.")
        return None

    processed_results_map = current_status.get("processed_results", {})
    all_segment_results = [None] * len(segments_info_list)
    tasks_to_submit_for_processing = []
    already_processed_count = 0

    overall_detected_language = current_status.get("detected_language")
    if overall_detected_language is None and segments_info_list and os.path.exists(segments_info_list[0]["file"]):
        logger.info("正在使用第一个分段检测音频语言...")
        console.print(f"[bold blue][INFO][/bold blue] 正在使用第一个分段检测音频语言...")
        first_segment_audio = None
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress_lang:
                progress_lang.add_task("语言检测中...", total=None)
                first_segment_audio = whisperx.load_audio(segments_info_list[0]["file"])
                buffer_stdout = io.StringIO(); buffer_stderr = io.StringIO()
                with redirect_stdout(buffer_stdout), redirect_stderr(buffer_stderr):
                     initial_transcription_result = model_instance.transcribe(first_segment_audio, batch_size=BATCH_SIZE, verbose=False)
            
            if initial_transcription_result and "language" in initial_transcription_result:
                overall_detected_language = initial_transcription_result["language"]
                logger.info(f"音频语言已检测: {overall_detected_language}")
                console.print(f"[bold green][INFO][/bold green] 音频语言已检测: [cyan]{overall_detected_language}[/cyan]")
                current_status["detected_language"] = overall_detected_language
                save_status(current_status)
            else:
                logger.warning("无法从第一个分段检测语言.")
                console.print(f"[bold yellow][WARNING][/bold yellow] 无法从第一个分段检测语言. 转录时将由模型自动判断各分段语言.")
        except Exception as e:
            logger.warning(f"初始语言检测失败: {e}. 转录时将由模型自动判断各分段语言.", exc_info=True)
            console.print(f"[bold yellow][WARNING][/bold yellow] 初始语言检测失败: {e}. 转录时将由模型自动判断各分段语言.")
        finally:
            if first_segment_audio is not None: 
                del first_segment_audio; gc.collect()
                logger.debug("第一个分段的音频数据已清理 (用于语言检测).")

    if overall_detected_language:
        logger.info(f"将整体检测到的语言 '{overall_detected_language}' 应用于未指定语言的分段.")
        for seg_info in segments_info_list:
            if "detected_language" not in seg_info:
                seg_info["detected_language"] = overall_detected_language
    
    alignment_model_cache = {} 
    if overall_detected_language:
        logger.info(f"正在为语言 '{overall_detected_language}' 预加载对齐模型 (设备: {DEVICE})...")
        console.print(f"[bold blue][INFO][/bold blue] 正在为语言 [cyan]{overall_detected_language}[/cyan] 预加载对齐模型 (设备: {DEVICE})...")
        console.print(f"[INFO] 对齐模型将通过镜像 ('{os.environ.get('HF_ENDPOINT', '默认Hugging Face源')}') 下载, 并缓存到默认 Hugging Face 路径 (通常在用户主目录).")
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress_align:
                progress_align.add_task(f"加载 {overall_detected_language} 对齐模型中...", total=None)
                with model_lock:
                    align_model, align_metadata = whisperx.load_align_model(
                        language_code=overall_detected_language, 
                        device=DEVICE
                    )
                alignment_model_cache[overall_detected_language] = (align_model, align_metadata)
            logger.info(f"语言 '{overall_detected_language}' 的对齐模型已预加载.")
            console.print(f"[bold green][INFO][/bold green] 语言 [cyan]{overall_detected_language}[/cyan] 的对齐模型已预加载.")
        except Exception as e:
            logger.warning(f"为语言 '{overall_detected_language}' 预加载对齐模型失败: {e}. 后续处理中将按需加载.", exc_info=True)
            console.print(f"[bold yellow][WARNING][/bold yellow] 为语言 '{overall_detected_language}' 预加载对齐模型失败: {e}. 后续处理中将按需加载.")

    for idx, seg_info in enumerate(segments_info_list):
        idx_str = str(idx)
        if idx_str in processed_results_map and \
           processed_results_map[idx_str] is not None and \
           "error" not in processed_results_map[idx_str]:
            all_segment_results[idx] = processed_results_map[idx_str]
            already_processed_count += 1
            logger.debug(f"分段 {idx}: 使用缓存的结果.")
        else:
            if os.path.exists(seg_info["file"]):
                tasks_to_submit_for_processing.append((idx, seg_info))
                logger.debug(f"分段 {idx}: 添加到待处理列表.")
            else:
                logger.warning(f"分段文件 {seg_info['file']} 未找到. 跳过索引 {idx}.")
                console.print(f"[bold yellow][WARNING][/bold yellow] 分段文件 {seg_info['file']} 未找到. 跳过索引 {idx}.")
                error_info = {"error": "File not found", "segment_basename": os.path.basename(seg_info.get("file", f"segment_{idx}_NA"))}
                all_segment_results[idx] = error_info
                processed_results_map[idx_str] = error_info

    if not tasks_to_submit_for_processing and already_processed_count == len(segments_info_list):
        logger.info("所有分段先前均已成功处理.")
        console.print("[bold blue][INFO][/bold blue] 所有分段先前均已成功处理.")
        save_status(current_status)
        return all_segment_results
    
    cpu_core_count = os.cpu_count() or 1
    num_workers = max(1, cpu_core_count // CPU_THREADS_PER_MODEL_INSTANCE) 
    num_workers = min(num_workers, 4)
    
    logger.info(f"正在转录 {len(tasks_to_submit_for_processing)} 个新的/失败的分段 (共 {len(segments_info_list)} 个). "
                f"使用 {num_workers} 个 Python 工作线程. 每个任务内部使用 {CPU_THREADS_PER_MODEL_INSTANCE} 个 CPU 线程.")
    console.print(f"[bold blue][INFO][/bold blue] 正在转录 [cyan]{len(tasks_to_submit_for_processing)}[/cyan] 个新的/失败的分段 (共 {len(segments_info_list)} 个).")
    console.print(f"[INFO] 使用 [cyan]{num_workers}[/cyan] 个 Python 工作线程进行调度.")
    console.print(f"[INFO] 每个转录任务内部 (faster-whisper) 将尝试使用 [cyan]{CPU_THREADS_PER_MODEL_INSTANCE}[/cyan] 个 CPU 线程.")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
                  TextColumn("[cyan]{task.completed}/{task.total}[/cyan] 段"), TimeElapsedColumn(), TimeRemainingColumn(),
                  console=console, transient=False, refresh_per_second=1) as progress_bar:
        
        transcribe_task = progress_bar.add_task("转录进度", total=len(segments_info_list), completed=already_processed_count)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx_map = {
                executor.submit(transcribe_and_align_segment, seg_info_item, model_instance, alignment_model_cache): original_idx 
                for original_idx, seg_info_item in tasks_to_submit_for_processing
            }
            
            for future in as_completed(future_to_idx_map):
                idx = future_to_idx_map[future]
                idx_str = str(idx)
                segment_basename_for_error = os.path.basename(segments_info_list[idx].get("file", f"segment_{idx}_NA"))
                try:
                    segment_result = future.result()
                    if segment_result is not None:
                        if "error" in segment_result:
                            logger.warning(f"分段 {idx} ({segment_result.get('segment_basename', segment_basename_for_error)}) 处理错误: {segment_result['error']}")
                            progress_bar.console.print(f"\n[bold yellow][WARNING][/bold yellow] 分段 {idx} ({segment_result.get('segment_basename', segment_basename_for_error)}) 处理错误: {segment_result['error']}")
                            all_segment_results[idx] = segment_result
                            processed_results_map[idx_str] = segment_result
                        else:
                            logger.info(f"分段 {idx} ({segment_basename_for_error}) 处理成功.")
                            all_segment_results[idx] = segment_result
                            processed_results_map[idx_str] = segment_result
                    else:
                        logger.warning(f"分段 {idx} ({segment_basename_for_error}) 未能成功产生结果 (返回 None).")
                        progress_bar.console.print(f"\n[bold yellow][WARNING][/bold yellow] 分段 {idx} ({segment_basename_for_error}) 未能成功产生结果 (返回 None).")
                        error_info = {"error": "No result returned", "segment_basename": segment_basename_for_error}
                        all_segment_results[idx] = error_info
                        processed_results_map[idx_str] = error_info
                    
                    current_status["processed_results"] = processed_results_map
                    save_status(current_status)
                except Exception as exc:
                    logger.error(f'分段 {idx} ({segment_basename_for_error}) 产生意外异常: {exc}', exc_info=True)
                    progress_bar.console.print(f'\n[bold red][ERROR][/bold red] 分段 {idx} ({segment_basename_for_error}) 产生意外异常: {exc}')
                    error_info = {"error": str(exc), "segment_basename": segment_basename_for_error, "unexpected": True}
                    all_segment_results[idx] = error_info
                    processed_results_map[idx_str] = error_info
                    current_status["processed_results"] = processed_results_map
                    save_status(current_status)
                finally:
                    progress_bar.update(transcribe_task, advance=1)
    
    successful_count = sum(1 for r in all_segment_results if r is not None and "error" not in r)
    failed_count = len(segments_info_list) - successful_count
    if failed_count > 0:
        logger.warning(f"转录完成. 但有 {failed_count} 个分段未能成功处理.")
        console.print(f"[bold yellow][WARNING][/bold yellow] 转录完成. 但有 [cyan]{failed_count}[/cyan] 个分段未能成功处理.")
    else:
        logger.info("所有分段均已成功转录和对齐.")
        console.print("[bold green][INFO][/bold green] 所有分段均已成功转录和对齐.")
    
    del alignment_model_cache; gc.collect()
    logger.debug("对齐模型缓存已清理.")
    return all_segment_results

def format_timestamp(seconds_float):
    if seconds_float is None or seconds_float < 0:
        logger.debug(f"无效的输入时间戳: {seconds_float}, 将其视为 0.0")
        seconds_float = 0.0
    
    total_seconds_int = int(seconds_float)
    milliseconds = int(round((seconds_float - total_seconds_int) * 1000))
    
    hours, remainder = divmod(total_seconds_int, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if milliseconds >= 1000: seconds += 1; milliseconds -= 1000
    if seconds >= 60: minutes += 1; seconds -= 60
    if minutes >= 60: hours += 1; minutes -= 60
        
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def generate_srt(all_transcription_results, srt_output_path, use_word_timestamps=False):
    logger.info(f"开始生成 SRT 文件: {srt_output_path}, 使用词级别时间戳: {use_word_timestamps}")
    console.print("[bold blue][INFO][/bold blue] 正在生成 SRT 字幕文件...")
    srt_content_lines = []
    subtitle_entry_counter = 1

    if all_transcription_results is None:
        logger.error("转录结果为空, 无法生成 SRT.")
        console.print("[bold red][ERROR][/bold red] 转录结果为空, 无法生成 SRT.")
        return False

    for i, single_segment_result in enumerate(all_transcription_results):
        if single_segment_result is None or ("error" in single_segment_result and single_segment_result["error"]):
            logger.warning(f"跳过生成 SRT 的分段 {i} (处理失败或为空): {single_segment_result.get('error', '空结果') if single_segment_result else '空结果'}")
            continue

        segments_to_process_for_srt = []
        if use_word_timestamps and single_segment_result.get("word_segments"):
            logger.debug(f"分段 {i}: 使用词级别时间戳.")
            for word_info in single_segment_result["word_segments"]:
                start_time = word_info.get("start")
                end_time = word_info.get("end")
                text = word_info.get("word", "").strip()
                if start_time is not None and end_time is not None and text:
                    segments_to_process_for_srt.append({"start": start_time, "end": end_time, "text": text})
        elif single_segment_result.get("segments"):
            logger.debug(f"分段 {i}: 使用句子/短语级别时间戳.")
            for seg_data in single_segment_result["segments"]:
                start_time = seg_data.get("start")
                end_time = seg_data.get("end")
                text = seg_data.get("text", "").strip()
                if start_time is not None and end_time is not None and text:
                    segments_to_process_for_srt.append({"start": start_time, "end": end_time, "text": text})
        
        for srt_entry_data in segments_to_process_for_srt:
            start_time_sec, end_time_sec, text_content = srt_entry_data["start"], srt_entry_data["end"], srt_entry_data["text"]
            
            if end_time_sec <= start_time_sec or not text_content:
                logger.debug(f"跳过无效的 SRT 条目: start={start_time_sec}, end={end_time_sec}, text='{text_content}'")
                continue
            
            srt_content_lines.append(str(subtitle_entry_counter))
            srt_content_lines.append(f"{format_timestamp(start_time_sec)} --> {format_timestamp(end_time_sec)}")
            srt_content_lines.append(text_content)
            srt_content_lines.append("")
            subtitle_entry_counter += 1

    if not srt_content_lines:
        logger.warning("未生成任何有效的字幕行. SRT 文件可能为空.")
        console.print("[bold yellow][WARNING][/bold yellow] 未生成任何有效的字幕行. SRT 文件可能为空或不符合标准.")
        return False

    try:
        with open(srt_output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content_lines))
        
        abs_srt_path = os.path.abspath(srt_output_path)
        logger.info(f"SRT 字幕文件已生成: {abs_srt_path}")
        console.print(f"[bold green][SUCCESS][/bold green] SRT 字幕文件已生成: [link=file://{abs_srt_path}]{srt_output_path}[/link]")
        return True
    except IOError as e:
        logger.error(f"写入 SRT 文件 '{srt_output_path}' 失败: {e}", exc_info=True)
        console.print(f"[bold red][ERROR][/bold red] 写入 SRT 文件 '{srt_output_path}' 失败: {e}")
        return False

def get_file_path_with_dialog():
    logger.info("尝试打开文件选择对话框.")
    selected_file_path = ""
    original_stderr_fileno = -1 
    saved_stderr_fd_copy = -1
    dev_null_fd = -1

    try:
        try:
            current_stderr_fd = sys.stderr.fileno()
            if isinstance(current_stderr_fd, int): original_stderr_fileno = current_stderr_fd
            elif hasattr(sys.__stderr__, 'fileno') and isinstance(sys.__stderr__.fileno(), int): original_stderr_fileno = sys.__stderr__.fileno()
            else: original_stderr_fileno = 2
        except (AttributeError, io.UnsupportedOperation): original_stderr_fileno = 2
        logger.debug(f"原始 stderr 文件描述符: {original_stderr_fileno}")

        saved_stderr_fd_copy = os.dup(original_stderr_fileno)
        dev_null_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(dev_null_fd, original_stderr_fileno)
        logger.debug("stderr 已重定向到 null 以进行 Tkinter 操作.")

        import tkinter as tk
        from tkinter import filedialog
        os.environ['TK_SILENCE_DEPRECATION'] = '1'
        root = tk.Tk()
        root.withdraw()
        console.print("[bold blue][INFO][/bold blue] 请选择要处理的视频或音频文件...")
        
        dialog_selected_path = filedialog.askopenfilename(
            title="选择视频或音频文件",
            filetypes=[
                ("媒体文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.mp3 *.wav *.flac *.m4a *.aac *.ogg"),
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("音频文件", "*.mp3 *.wav *.flac *.m4a *.aac *.ogg"),
                ("所有文件", "*.*")
            ]
        )
        root.destroy()
        selected_file_path = dialog_selected_path if dialog_selected_path else ""
        if selected_file_path:
            logger.info(f"用户选择了文件: {selected_file_path}")
        else:
            logger.info("用户取消了文件选择.")

    except Exception as e:
        logger.error(f"打开文件选择对话框时出错: {e}", exc_info=True)
        if saved_stderr_fd_copy != -1 and original_stderr_fileno != -1:
            try: os.dup2(saved_stderr_fd_copy, original_stderr_fileno)
            except OSError as oe_restore:
                sys.__stderr__.write(f"CRITICAL: 恢复 stderr 失败 (在 Tkinter 异常处理中): {oe_restore}\n")
                logger.critical(f"恢复 stderr 失败 (在 Tkinter 异常处理中): {oe_restore}")
        
        console.print(f"[bold red][ERROR][/bold red] 打开文件选择对话框时出错: {e}")
        console.print("[INFO] 如果无法使用图形化文件选择器, 请考虑修改脚本以接受命令行参数或手动输入文件路径.")
        selected_file_path = ""
    finally:
        if dev_null_fd != -1:
            try: os.close(dev_null_fd)
            except OSError: pass
        
        if saved_stderr_fd_copy != -1 and original_stderr_fileno != -1:
            try:
                os.dup2(saved_stderr_fd_copy, original_stderr_fileno)
                logger.debug("stderr 已恢复.")
            except OSError as oe_final_restore:
                 sys.__stderr__.write(f"CRITICAL: 恢复 stderr 失败 (在 finally 块中): {oe_final_restore}\n")
                 logger.critical(f"恢复 stderr 失败 (在 finally 块中): {oe_final_restore}")
        
        if saved_stderr_fd_copy != -1:
            try: os.close(saved_stderr_fd_copy)
            except OSError: pass
            
    return selected_file_path

def handle_import_and_process():
    logger.info("开始导入和处理流程.")
    global WHISPER_MODEL, COMPUTE_TYPE
    
    selected_input_file = ""
    while True:
        selected_input_file = get_file_path_with_dialog()
        if not selected_input_file:
            logger.info("未选择文件.")
            if not Confirm.ask("未选择文件. 是否重试选择文件?", default=True, console=console):
                logger.info("用户选择不重试, 返回主菜单.")
                console.print("[bold blue][INFO][/bold blue] 返回主菜单.")
                return
            else:
                logger.info("用户选择重试文件选择.")
                continue
        
        if not os.path.isfile(selected_input_file):
            logger.warning(f"选择的路径不是有效文件: '{selected_input_file}'.")
            if not Confirm.ask(f"选择的路径不是有效文件: '{selected_input_file}'.\n是否重试选择文件?", default=True, console=console):
                logger.info("用户选择不重试 (无效文件路径), 返回主菜单.")
                console.print("[bold blue][INFO][/bold blue] 返回主菜单.")
                return
            else:
                logger.info("用户选择重试文件选择 (无效文件路径).")
                continue
        else:
            logger.info(f"已选择有效文件: {selected_input_file}")
            console.print(f"[bold blue][INFO][/bold blue] 已选择文件: [cyan]{selected_input_file}[/cyan]")
            break

    input_filename_base = os.path.basename(selected_input_file)
    srt_output_filepath = os.path.splitext(selected_input_file)[0] + ".srt"
    temp_audio_filepath = os.path.join(TEMP_DIR, "audio.wav")
    logger.info(f"输入文件名: {input_filename_base}, SRT 输出: {srt_output_filepath}, 临时音频: {temp_audio_filepath}")

    status_data = load_status()
    force_audio_extraction, force_audio_split = False, False

    if status_data.get("input_file") == input_filename_base and \
       status_data.get("model_used") == WHISPER_MODEL and \
       status_data.get("compute_type_used") == COMPUTE_TYPE:
        logger.info(f"找到与 '{input_filename_base}' (模型: {WHISPER_MODEL}, 计算类型: {COMPUTE_TYPE}) 相关的先前状态.")
        console.print(f"[bold blue][INFO][/bold blue] 找到与 '{input_filename_base}' (模型: {WHISPER_MODEL}, 计算类型: {COMPUTE_TYPE}) 相关的先前状态.")
        if not Confirm.ask("是否尝试从上次中断处继续处理?", default=True, console=console):
            logger.info("用户选择不继续处理, 将重新开始.")
            console.print("[bold blue][INFO][/bold blue] 将重新开始处理 (清理先前状态).")
            status_data = {}
            force_audio_extraction = True
            force_audio_split = True
            cleanup_temp()
            ensure_app_dirs()
        else:
            logger.info("用户选择从上次中断处继续处理.")
    else:
        if status_data:
            logger.info("输入文件、模型或计算类型已更改, 或无先前状态. 开始新的处理.")
            console.print("[bold blue][INFO][/bold blue] 输入文件、模型或计算类型已更改, 或无先前状态. 开始新的处理.")
        else:
            logger.info("无先前状态. 开始新的处理.")
        status_data = {}
        force_audio_extraction = True
        force_audio_split = True
        cleanup_temp()
        ensure_app_dirs()

    status_data["input_file"] = input_filename_base
    status_data["model_used"] = WHISPER_MODEL 
    status_data["compute_type_used"] = COMPUTE_TYPE
    save_status(status_data)
    logger.info(f"状态已更新: input_file='{input_filename_base}', model_used='{WHISPER_MODEL}', compute_type_used='{COMPUTE_TYPE}'")

    logger.info("步骤 1: 提取音频")
    if not extract_audio(selected_input_file, temp_audio_filepath, force_audio_extraction):
        logger.error("音频提取失败. 中止处理.")
        console.print("[bold red][ERROR][/bold red] 音频提取失败. 中止处理.")
        return

    logger.info("步骤 2: 音频分段")
    audio_segments = split_audio(temp_audio_filepath, force_audio_split)
    if not audio_segments:
        logger.error("音频分段失败或未产生任何分段. 中止处理.")
        console.print("[bold red][ERROR][/bold red] 音频分段失败或未产生任何分段. 中止处理.")
        return

    logger.info("步骤 3: 转录和对齐所有分段")
    transcription_results = process_all_segments(audio_segments, status_data)
    if not transcription_results:
        logger.error("转录和对齐过程发生严重错误. 中止处理.")
        console.print("[bold red][ERROR][/bold red] 转录和对齐过程发生严重错误 (例如模型加载失败). 中止处理.")
        return
    
    successful_segment_count = sum(1 for r in transcription_results if r is not None and "error" not in r)
    if successful_segment_count == 0:
        logger.error("没有任何分段被成功转录. 无法生成 SRT 文件.")
        console.print("[bold red][ERROR][/bold red] 没有任何分段被成功转录. 无法生成 SRT 文件.")
        return
    elif successful_segment_count < len(audio_segments):
        logger.warning(f"只有 {successful_segment_count}/{len(audio_segments)} 个分段成功处理.")
        console.print(f"[bold yellow][WARNING][/bold yellow] 只有 {successful_segment_count}/{len(audio_segments)} 个分段成功处理.")
        if not Confirm.ask("是否仍要根据已成功处理的部分生成 SRT 文件?", default=False, console=console):
            logger.info("用户取消生成部分 SRT 文件.")
            console.print("[bold blue][INFO][/bold blue] 已取消生成 SRT 文件.")
            return
        logger.info("用户同意生成部分 SRT 文件.")
            
    logger.info("步骤 4: 生成 SRT 文件")
    if not generate_srt(transcription_results, srt_output_filepath, USE_WORD_TIMESTAMPS_CONFIG):
        logger.warning("SRT 文件生成失败或未生成任何内容.")
        console.print("[bold yellow][WARNING][/bold yellow] SRT 文件生成失败或未生成任何内容.")
    
    logger.info("步骤 5: 清理临时文件 (可选)")
    if Confirm.ask("是否清理本次处理产生的临时文件?", default=True, console=console):
        logger.info("用户选择清理临时文件.")
        cleanup_temp()
    else:
        logger.info("用户选择保留临时文件.")
        console.print("[bold blue][INFO][/bold blue] 保留临时文件. 您可以稍后手动删除或在下次处理不同文件时自动清理.")
    
    logger.info("处理流程结束. 返回主菜单.")
    console.print("\n[bold blue][INFO][/bold blue] 处理流程结束. 返回主菜单.")

def handle_dependencies_check_ui():
    logger.info("进入依赖检查与管理 UI.")
    console.print(Panel(Text("依赖检查与管理", justify="center", style="bold cyan"), box=ROUNDED, expand=False))
    if Confirm.ask("是否现在执行依赖项检查?", default=True, console=console):
        logger.info("用户选择执行依赖项检查.")
        check_dependencies(verbose=True)
    else:
        logger.info("用户跳过依赖检查.")
        console.print("[bold blue][INFO][/bold blue] 跳过依赖检查.")
        return

    if Confirm.ask("是否尝试自动下载并安装缺失的依赖项 (实验性功能, 当前为占位符)?", default=False, console=console):
        logger.info("用户选择尝试自动安装依赖 (占位符).")
        console.print("[bold yellow][INFO][/bold yellow] 自动依赖安装功能目前是一个占位符. 请根据上面的检查结果, 手动使用 pip 或 conda 安装缺失的依赖项.")
    else:
        logger.info("用户选择不自动安装依赖.")
        console.print("[bold blue][INFO][/bold blue] 请根据检查结果手动安装任何缺失的依赖项.")
    
    logger.info("依赖检查与管理 UI 结束.")
    console.print("\n[bold blue][INFO][/bold blue] 返回主菜单.")

def handle_model_selection_ui():
    logger.info("进入模型选择与参数配置 UI.")
    global WHISPER_MODEL, COMPUTE_TYPE, BATCH_SIZE, whisper_model_global, CPU_THREADS_PER_MODEL_INSTANCE
    
    console.print(Panel(Text("Whisper 模型与参数配置 (CPU)", justify="center", style="bold cyan"), box=ROUNDED, expand=False))
    
    settings_table = Table(title="当前模型设置", box=ROUNDED, show_lines=True)
    settings_table.add_column("参数", style="magenta", no_wrap=True)
    settings_table.add_column("当前值", style="green")
    settings_table.add_row("Whisper 模型 (WHISPER_MODEL)", WHISPER_MODEL)
    settings_table.add_row("计算类型 (COMPUTE_TYPE)", COMPUTE_TYPE)
    settings_table.add_row("运行设备 (DEVICE)", DEVICE)
    settings_table.add_row("批处理大小 (BATCH_SIZE)", str(BATCH_SIZE))
    settings_table.add_row("模型内部 CPU 线程数 (CPU_THREADS_PER_MODEL_INSTANCE)", str(CPU_THREADS_PER_MODEL_INSTANCE))
    console.print(settings_table)
    logger.info(f"当前模型设置: Model={WHISPER_MODEL}, Compute={COMPUTE_TYPE}, Batch={BATCH_SIZE}, CPUThreads={CPU_THREADS_PER_MODEL_INSTANCE}")


    console.print("\n[bold]可配置选项:[/bold]")
    console.print("1. 选择 Whisper 模型大小 (例如: tiny, base, small, medium, large-v1/v2/v3)")
    console.print("2. 设置计算类型 (CPU 推荐: int8, float32, int16)")
    console.print("3. 设置批处理大小 (整数, 例如: 8, 16, 32)")
    console.print(f"4. 设置模型内部 CPU 线程数 (整数, 1 到 {os.cpu_count() or 'N/A'})")
    console.print("5. 返回主菜单")
    
    choice = Prompt.ask("请选择要修改的配置项", choices=["1", "2", "3", "4", "5"], default="5", console=console)
    logger.info(f"用户选择配置项: {choice}")
    
    temp_model = WHISPER_MODEL
    temp_compute = COMPUTE_TYPE
    temp_batch = BATCH_SIZE
    temp_cpu_threads = CPU_THREADS_PER_MODEL_INSTANCE

    if choice == "1":
        new_model = Prompt.ask(f"输入新的 Whisper 模型名称 (当前: {temp_model})", default=temp_model, console=console).strip()
        if new_model: temp_model = new_model
        logger.info(f"临时模型更改为: {temp_model}")
    elif choice == "2":
        allowed_cpu_compute_types = ["int8", "float32", "int16"]
        new_compute = Prompt.ask(f"输入新的计算类型 (当前: {temp_compute}, CPU推荐: {', '.join(allowed_cpu_compute_types)})", default=temp_compute, console=console).strip()
        if new_compute in allowed_cpu_compute_types or new_compute == "auto":
            temp_compute = new_compute
            logger.info(f"临时计算类型更改为: {temp_compute}")
        else:
            logger.warning(f"无效的计算类型输入: '{new_compute}'. 保持当前设置.")
            console.print(f"[yellow]无效的计算类型: '{new_compute}'. 推荐使用 {', '.join(allowed_cpu_compute_types)} 或 'auto'. 保持当前设置.[/yellow]")
    elif choice == "3":
        new_batch_size = IntPrompt.ask(f"输入新的批处理大小 (当前: {temp_batch})", default=temp_batch, console=console)
        if new_batch_size > 0:
            temp_batch = new_batch_size
            logger.info(f"临时批处理大小更改为: {temp_batch}")
        else:
            logger.warning(f"无效的批处理大小输入: {new_batch_size}. 保持当前设置.")
            console.print(f"[yellow]无效的批处理大小: {new_batch_size}. 必须大于 0. 保持当前设置.[/yellow]")
    elif choice == "4":
        max_threads = os.cpu_count() or 8
        new_cpu_threads = IntPrompt.ask(f"输入新的模型内部 CPU 线程数 (当前: {temp_cpu_threads}, 建议 1-{max_threads})", default=temp_cpu_threads, console=console)
        if 0 < new_cpu_threads <= max_threads:
            temp_cpu_threads = new_cpu_threads
            logger.info(f"临时内部 CPU 线程数更改为: {temp_cpu_threads}")
        elif new_cpu_threads > max_threads:
             logger.warning(f"设置的线程数 ({new_cpu_threads}) 超过 CPU 核心数 ({max_threads}). 保持当前设置.")
             console.print(f"[yellow]设置的线程数 ({new_cpu_threads}) 超过 CPU 核心数 ({max_threads}). 可能导致性能下降. 请谨慎设置. 保持当前设置.[/yellow]")
        else:
            logger.warning(f"无效的内部 CPU 线程数输入: {new_cpu_threads}. 保持当前设置.")
            console.print(f"[yellow]无效的内部 CPU 线程数: {new_cpu_threads}. 必须大于 0. 保持当前设置.[/yellow]")
    elif choice == "5":
        logger.info("用户选择返回主菜单, 未作更改.")
        console.print("[bold blue][INFO][/bold blue] 未作更改. 返回主菜单.")
        return

    if choice != "5":
        console.print(f"\n[bold]建议的更改:[/bold]\n"
                      f"  模型: {temp_model}\n"
                      f"  计算类型: {temp_compute}\n"
                      f"  批处理大小: {temp_batch}\n"
                      f"  内部CPU线程: {temp_cpu_threads}")
        if Confirm.ask("\n是否应用以上更改?", default=True, console=console):
            logger.info("用户确认应用更改.")
            model_config_changed = (WHISPER_MODEL != temp_model or 
                                    COMPUTE_TYPE != temp_compute or 
                                    CPU_THREADS_PER_MODEL_INSTANCE != temp_cpu_threads)
            
            WHISPER_MODEL = temp_model
            COMPUTE_TYPE = temp_compute
            BATCH_SIZE = temp_batch
            CPU_THREADS_PER_MODEL_INSTANCE = temp_cpu_threads
            logger.info(f"模型配置已更新: Model={WHISPER_MODEL}, Compute={COMPUTE_TYPE}, Batch={BATCH_SIZE}, CPUThreads={CPU_THREADS_PER_MODEL_INSTANCE}")
            
            if model_config_changed and whisper_model_global is not None:
                logger.info("模型关键配置已更改. 清除已缓存的全局模型实例.")
                console.print("[INFO] 模型关键配置已更改. 清除已缓存的全局模型实例. 下次使用时将重新加载.")
                del whisper_model_global
                whisper_model_global = None
                gc.collect()
            console.print("[bold green][SUCCESS][/bold green] 模型配置已更新!")
        else:
            logger.info("用户取消应用更改.")
            console.print("[bold blue][INFO][/bold blue] 更改已取消.")
    
    logger.info("模型选择与参数配置 UI 结束.")
    console.print("\n[bold blue][INFO][/bold blue] 返回主菜单.")

def handle_advanced_settings_ui():
    logger.info("进入高级设置 UI.")
    global USE_WORD_TIMESTAMPS_CONFIG
    
    console.print(Panel(Text("高级设置", justify="center", style="bold cyan"), box=ROUNDED, expand=False))
    
    current_status_text = "[bold green]启用 (词级别)[/bold green]" if USE_WORD_TIMESTAMPS_CONFIG else "[bold red]禁用 (句子/短语级别)[/bold red]"
    console.print(f"当前 SRT 时间戳模式: {current_status_text}")
    logger.info(f"当前 SRT 时间戳模式: {'词级别' if USE_WORD_TIMESTAMPS_CONFIG else '句子/短语级别'}")
    
    prompt_text = f"是否 {'禁用词级别时间戳 (切换到句子/短语级别)' if USE_WORD_TIMESTAMPS_CONFIG else '启用词级别时间戳'}?"
    
    if Confirm.ask(prompt_text, default=True, console=console):
        USE_WORD_TIMESTAMPS_CONFIG = not USE_WORD_TIMESTAMPS_CONFIG
        new_status_text = "[bold green]启用 (词级别)[/bold green]" if USE_WORD_TIMESTAMPS_CONFIG else "[bold red]禁用 (句子/短语级别)[/bold red]"
        logger.info(f"SRT 时间戳模式已更改为: {'词级别' if USE_WORD_TIMESTAMPS_CONFIG else '句子/短语级别'}")
        console.print(f"[bold green][SUCCESS][/bold green] SRT 时间戳模式已设置为: {new_status_text}")
    else:
        logger.info("SRT 时间戳模式设置未更改.")
        console.print("[bold blue][INFO][/bold blue] SRT 时间戳模式设置未更改.")
        
    logger.info("高级设置 UI 结束.")
    console.print("\n[bold blue][INFO][/bold blue] 返回主菜单.")

def display_main_menu_ui():
    console.rule("[bold cyan]Video-to-SRT 字幕生成工具 (CPU 版)[/bold cyan]", style="cyan")
    
    menu_text_obj = Text("\n请选择操作:\n\n", justify="left")
    menu_text_obj.append("[1] 导入媒体文件并生成字幕 (默认)\n", style="yellow")
    menu_text_obj.append("[2] 检查与管理依赖项\n", style="yellow")
    menu_text_obj.append("[3] 配置 Whisper 模型与参数\n", style="yellow")
    menu_text_obj.append("[4] 高级设置 (例如 SRT 时间戳级别)\n", style="yellow")
    menu_text_obj.append("[5] 退出程序\n\n", style="yellow")
    
    console.print(Panel(menu_text_obj, title="主菜单", border_style="magenta", padding=(1, 2), expand=False, box=ROUNDED))
    choice = Prompt.ask("输入选项 [1-5]", choices=["1", "2", "3", "4", "5"], default="1", console=console)
    logger.debug(f"主菜单用户选择: {choice}")
    return choice

def main_cli_loop():
    ensure_app_dirs()
    logger.info("进入主命令行界面循环.")
    
    while True:
        user_action = display_main_menu_ui()
        logger.info(f"用户操作: {user_action}")
        if user_action == "1":
            handle_import_and_process()
        elif user_action == "2":
            handle_dependencies_check_ui()
        elif user_action == "3":
            handle_model_selection_ui()
        elif user_action == "4":
            handle_advanced_settings_ui()
        elif user_action == "5":
            if Confirm.ask("您确定要退出程序吗?", default=True, console=console): 
                logger.info("用户确认退出程序.")
                console.print("[bold blue]感谢使用, 程序已退出.[/bold blue]")
                break
            else:
                logger.info("用户取消退出程序.")
        
        console.line(2)

if __name__ == "__main__":
    setup_logging() # <--- 在程序开始时配置日志
    logger.info("======================================================================")
    logger.info("应用程序启动")
    logger.info(f"应用程序基础路径 (APP_BASE_PATH): {APP_BASE_PATH}")
    logger.info(f"日志文件路径: {log_file_path}")
    logger.info(f"模型缓存目录 (MODEL_CACHE_DIR): {MODEL_CACHE_DIR}")
    logger.info(f"临时文件目录 (TEMP_DIR): {TEMP_DIR}")
    logger.info(f"状态文件路径 (STATUS_FILE): {STATUS_FILE}")
    logger.info(f"Python 版本: {sys.version}")
    logger.info(f"操作系统: {platform.system()} {platform.release()} ({platform.version()})")
    logger.info(f"CPU 核心数: {os.cpu_count()}")
    logger.info(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")
    logger.info("======================================================================")


    try:
        main_cli_loop()
    except KeyboardInterrupt:
        logger.warning("用户通过 KeyboardInterrupt (Ctrl+C) 中断了程序.", exc_info=False) # exc_info=False as it's expected
        console.print("\n[bold yellow]用户中断了进程 (Ctrl+C). 正在退出.[/bold yellow]")
    except Exception as e:
        logger.critical("在 main_cli_loop 中发生未处理的严重错误:", exc_info=True) # exc_info=True 会自动记录堆栈跟踪
        console.print("\n[bold red][CRITICAL ERROR][/bold red] 程序遇到未处理的严重错误, 即将终止.")
        console.print_exception(show_locals=True, width=None) # Rich 的异常打印到控制台
        # console.log(f"[bold red]错误详情: {str(e)}[/bold red]") # logger.critical 已经记录了错误
        console.print(f"[bold red]错误详情 (也已记录到日志文件 '{LOG_FILE_NAME}'): {str(e)}[/bold red]")
        console.print(f"建议将上述错误信息截图或复制, 或查看日志文件 '{LOG_FILE_NAME}' 以便排查问题.")
    finally:
        logger.info("应用程序终止.")
        logger.info("======================================================================\n")
        console.print("程序已终止.", style="dim")