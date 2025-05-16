import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 允许重复的 OpenMP 库, 这是某些环境下的常见修复方法

import subprocess
import platform
import json
import threading
import shutil
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import warnings
import sys
import io
import re

# Rich 库导入, 用于命令行界面 (CLI UI)
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, SpinnerColumn
from rich.table import Table
from rich.live import Live
from rich.box import ROUNDED

# 初始化 Rich Console
console = Console()

# 过滤特定的警告信息
warnings.filterwarnings("ignore", message="TensorFloat-32 \\(TF32\\) has been disabled") # 常见的 TF32 警告
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.") # 来自 WhisperX 或其依赖项的警告
# warnings.filterwarnings("ignore", category=UserWarning, module='torchvision') # 如果 torchvision 警告过于频繁, 可以取消注释此行

# StderrFilter 类, 用于选择性地过滤 stderr 消息 (例如, libpng 警告)
# Rich 可能会通过这个过滤器将信息写入 stderr.
class StderrFilter:
    def __init__(self):
        self.old_stderr = sys.stderr
        try:
            # 尝试获取文件描述符, 以兼容需要它的 C 库或工具
            self._fileno = self.old_stderr.fileno()
        except (AttributeError, io.UnsupportedOperation):
            self._fileno = None 
        
    def __enter__(self):
        sys.stderr = self
        return self
        
    def __exit__(self, *args):
        sys.stderr = self.old_stderr
        
    def write(self, text):
        # 过滤掉所有 libpng 警告, 可以调整以适应其他特定警告
        if 'libpng warning' not in text: 
            self.old_stderr.write(text)
    
    def flush(self):
        self.old_stderr.flush()
        
    def fileno(self):
        # 如果文件描述符可用, 则提供它, 这对于某些 C 库的交互至关重要
        if self._fileno is not None:
            return self._fileno
        if hasattr(self.old_stderr, 'fileno'):
             current_fileno = self.old_stderr.fileno()
             if isinstance(current_fileno, int):
                 return current_fileno
        raise io.UnsupportedOperation("underlying stream does not support fileno or returned non-integer")

    def isatty(self):
        # 将 isatty 检查委托给原始的 stderr
        if hasattr(self.old_stderr, 'isatty'):
            return self.old_stderr.isatty()
        return False

# 如果 StderrFilter 尚未应用, 则应用它
if not isinstance(sys.stderr, StderrFilter):
    sys.stderr = StderrFilter()

# 设置环境变量以确保一致的行为
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PILLOW_SILENCE_LIBPNG'] = '1' # 尝试静默 Pillow 的 libpng 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # TensorFlow 日志级别 (1=INFO, 2=WARNING, 3=ERROR)

# PyTorch 后端设置以提高性能 (TF32 允许在 Ampere 及更新的 GPU 上使用 TensorFloat32)
try:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except ImportError:
    pass # PyTorch 尚未安装

# ========================
# 配置区域 (Configuration Area)
# ========================
TORCH_VERSION = "2.5.1"         # 目标 PyTorch 版本, 用于依赖检查
TORCHAUDIO_VERSION = "2.0.2"    # 目标 torchaudio 版本
TORCHVISION_VERSION = "0.15.2"  # 目标 torchvision 版本
WHISPERX_REPO = "git+https://github.com/m-bain/whisperx.git" # WhisperX 代码库, 用于安装
TEMP_DIR = os.path.join(os.getcwd(), "temp")  # 临时目录, 用于存放中间文件
STATUS_FILE = os.path.join(TEMP_DIR, "status.json") # 文件, 用于存储处理状态以便恢复

# 音频处理参数 (Audio processing parameters)
SEGMENT_LENGTH_MS = 60 * 1000       # 音频分段的目标长度 (毫秒)
SILENCE_SEARCH_DURATION_MS = 2000 # 在分段末尾搜索静音的时长
MIN_SILENCE_LEN = 300             # 用于分割的最小静音长度 (毫秒)
SILENCE_THRESH = -40                # 静音阈值 (dBFS)

# WhisperX 模型和处理参数 (可通过用户界面更改)
DEVICE = "cuda" if ('torch' in sys.modules and torch.cuda.is_available()) else "cpu" # 默认设备
BATCH_SIZE = 16                     # 转录时的批处理大小
COMPUTE_TYPE = "float16"            # Whisper 模型的计算类型 (例如, float16, int8, float32)
WHISPER_MODEL = "medium"            # 默认 Whisper 模型大小

# 全局变量, 用于存储已加载的 WhisperX 模型实例
whisper_model_global = None
# GPU 操作的线程锁, 防止竞争条件
gpu_lock = threading.Lock()

# ========================
# 依赖检查 (Dependency Check)
# ========================
def check_dependencies(verbose=True):
    """检查必要的依赖项并打印其状态."""
    if verbose:
        console.print("[bold blue][INFO][/bold blue] 正在检查必要的依赖项...")
    
    dependencies = [
        "pydub", "tqdm", "transformers", "ffmpeg-python",
        "faster-whisper", "pytorch-lightning"
    ]
    missing_deps = []
    installed_deps_messages = []

    try:
        import torch
        import torchaudio
        # 如果在此处找到 PyTorch, 确保启用 TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        installed_deps_messages.append(f"[bold green]PyTorch[/bold green] (版本: {torch.__version__}) 已安装")
        installed_deps_messages.append(f"[bold green]torchaudio[/bold green] (版本: {torchaudio.__version__}) 已安装")
        installed_deps_messages.append(f"CUDA 是否可用: {torch.cuda.is_available()}")
    except ImportError:
        missing_deps.extend([f"torch=={TORCH_VERSION}", f"torchaudio=={TORCHAUDIO_VERSION}", f"torchvision=={TORCHVISION_VERSION}"])

    for dep in dependencies:
        try:
            __import__(dep.replace("-", "_").split("==")[0]) # 规范化名称以便导入
            installed_deps_messages.append(f"[bold green]{dep}[/bold green] 已安装")
        except ImportError:
            missing_deps.append(dep)
    
    try:
        import whisperx
        installed_deps_messages.append("[bold green]whisperx[/bold green] 已安装")
    except ImportError:
        missing_deps.append(WHISPERX_REPO) # 添加 whisperx 的 git 仓库地址
    
    try:
        # 检查 ffmpeg 命令行工具
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode == 0:
            installed_deps_messages.append("[bold green]ffmpeg[/bold green] 已安装")
        else:
            if verbose: console.print("[bold yellow][WARNING][/bold yellow] ffmpeg 检测失败. 可能需要手动安装.")
            missing_deps.append("ffmpeg (手动安装)")
    except FileNotFoundError:
        if verbose: console.print("[bold yellow][WARNING][/bold yellow] 未找到 ffmpeg. 请安装 ffmpeg 并确保其在系统路径中.")
        missing_deps.append("ffmpeg (手动安装)")

    if verbose:
        for msg in installed_deps_messages:
            console.print(f"[INFO] {msg}")

    if missing_deps:
        if verbose:
            console.print("\n[bold yellow][WARNING][/bold yellow] 以下依赖项缺失或存在问题:")
            for dep in missing_deps:
                console.print(f"  - [yellow]{dep}[/yellow]")
            console.print("\n请尝试使用菜单中的 “检查与管理依赖项” 功能, 或手动安装以上依赖项.")
        return False
    
    if verbose:
        console.print("[bold green][INFO][/bold green] 依赖检查完成. 所有必要的依赖项均已安装.")
    return True

# ========================
# 工具函数 (Utility Functions)
# ========================
def ensure_temp_dir():
    """确保临时目录存在."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        console.print(f"[bold blue][INFO][/bold blue] 临时目录已创建: {TEMP_DIR}")

def load_status():
    """从状态文件中加载处理状态."""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            console.print("[bold yellow][WARNING][/bold yellow] 状态文件已损坏. 将重新开始.")
            return {}
    return {}

def save_status(status):
    """将处理状态保存到状态文件."""
    ensure_temp_dir()
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(status, f, ensure_ascii=False, indent=4)

def cleanup_temp():
    """移除临时目录及其内容."""
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
            console.print(f"[bold blue][INFO][/bold blue] 已清理临时目录: {TEMP_DIR}")
        except OSError as e:
            console.print(f"[bold red][ERROR][/bold red] 清理临时文件失败: {e}")

# ========================
# 步骤 1: 音频提取 (Step 1: Audio Extraction)
# ========================
def extract_audio(input_file_path, audio_output_path, force_extract=False):
    """使用 ffmpeg 从输入文件中提取音频到标准化的 WAV 格式."""
    if not force_extract and os.path.exists(audio_output_path):
        console.print(f"[bold blue][INFO][/bold blue] 标准化音频文件已存在. 跳过提取: {audio_output_path}")
        return True

    # ffmpeg 命令: 覆盖输出文件, 指定输入文件, 无视频, 单声道, 16kHz 采样率, pcm_s16le 编码
    command = [
        "ffmpeg", "-y", "-i", input_file_path, "-vn", "-ac", "1", 
        "-ar", "16000", "-acodec", "pcm_s16le", audio_output_path
    ]
    console.print(f"[bold blue][INFO][/bold blue] 正在处理输入文件 '{os.path.basename(input_file_path)}' 以生成标准化音频...")
    try:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
            progress.add_task("FFmpeg 处理中...", total=None)
            # 对 ffmpeg 输出使用 utf-8 编码并替换错误字符, 以增强鲁棒性
            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        console.print(f"[bold green][INFO][/bold green] 标准化音频已成功生成: {audio_output_path}")
        return True
    except FileNotFoundError:
        console.print("[bold red][ERROR][/bold red] 未找到 ffmpeg. 请确保已安装 ffmpeg 并将其添加到系统 PATH.")
        return False
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red][ERROR][/bold red] 使用 ffmpeg 处理文件 '{os.path.basename(input_file_path)}' 时出错.")
        console.print(f"命令: {' '.join(e.cmd)}")
        console.print(f"返回码: {e.returncode}")
        if e.stderr:
            console.print(f"FFmpeg 错误输出:\n[dim]{e.stderr}[/dim]")
        else:
            console.print("FFmpeg 未产生 stderr 输出.")
        return False

# ========================
# 步骤 2: 音频分段 (Step 2: Audio Segmentation)
# ========================
def get_existing_segments(temp_dir_path):
    """从临时目录中检索已存在的音频分段文件列表 (已排序)."""
    segment_pattern = os.path.join(temp_dir_path, "segment_*.wav")
    return sorted(glob.glob(segment_pattern))

def split_audio(audio_path, force_split=False):
    """根据时长和静音检测将音频文件分割成段."""
    from pydub import AudioSegment, silence # pydub 在此处局部导入
    
    existing_segment_files = get_existing_segments(TEMP_DIR)
    status = load_status()

    # 根据状态文件检查是否可以重用分段
    if not force_split and status.get("segments_info") and existing_segment_files:
        if len(existing_segment_files) == len(status["segments_info"]):
            console.print("[bold blue][INFO][/bold blue] 使用先前保存的分段信息.")
            return status["segments_info"]
        else:
            console.print("[bold blue][INFO][/bold blue] 现有的分段文件与状态记录不匹配. 将重新分段.")
    elif not force_split and existing_segment_files:
        console.print("[bold blue][INFO][/bold blue] 检测到现有的分段文件但无状态信息. 将重新分段以确保时间戳正确.")

    console.print("[bold blue][INFO][/bold blue] 开始音频分段...")
    try:
        audio = AudioSegment.from_wav(audio_path)
    except FileNotFoundError:
        console.print(f"[bold red][ERROR][/bold red] 音频文件未找到: {audio_path}")
        return []
    except Exception as e: # 捕获其他 pydub 加载错误
        console.print(f"[bold red][ERROR][/bold red] 加载音频文件失败: {e}")
        return []

    audio_length_ms = len(audio)
    segments_info = []
    current_pos_ms = 0
    segment_idx = 0

    # 如果重新分段, 清理旧的分段文件
    if existing_segment_files:
        console.print("[bold blue][INFO][/bold blue] 正在清理旧的分段文件...")
        for f_path in existing_segment_files:
            try:
                os.remove(f_path)
            except OSError:
                pass # 如果文件被锁定或已删除, 忽略错误

    with Progress(
        TextColumn("[progress.description]{task.description}"), BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total} ms)"),
        TimeRemainingColumn(), TimeElapsedColumn(),
        console=console, transient=False # 保持进度条可见直到下一个主要步骤
    ) as progress_bar:
        segment_task = progress_bar.add_task("音频分段进度", total=audio_length_ms)
        while current_pos_ms < audio_length_ms:
            end_pos_ms = current_pos_ms + SEGMENT_LENGTH_MS
            actual_end_pos_ms = min(end_pos_ms, audio_length_ms)

            if actual_end_pos_ms > current_pos_ms:
                # 尝试在分段末尾的静音处分割, 以获得更自然的断点
                if actual_end_pos_ms < audio_length_ms and (actual_end_pos_ms - current_pos_ms) > SILENCE_SEARCH_DURATION_MS:
                    search_start_ms = max(current_pos_ms, actual_end_pos_ms - SILENCE_SEARCH_DURATION_MS)
                    search_chunk = audio[search_start_ms:actual_end_pos_ms]
                    try:
                        # detect_silence 返回静音的 [开始, 结束] 列表 (毫秒), 相对于 search_chunk
                        silence_ranges = silence.detect_silence(
                            search_chunk, min_silence_len=MIN_SILENCE_LEN, silence_thresh=SILENCE_THRESH
                        )
                        if silence_ranges:
                            # 使用第一个检测到的静音的开始点作为分割点
                            first_silence_start_in_search_chunk = silence_ranges[0][0]
                            potential_new_end_ms = search_start_ms + first_silence_start_in_search_chunk
                            # 确保调整静音后的分段不会太短
                            if potential_new_end_ms > current_pos_ms + MIN_SILENCE_LEN: 
                                actual_end_pos_ms = potential_new_end_ms
                    except Exception as e: # 捕获静音检测期间的错误
                        progress_bar.console.print(f"\n[bold yellow][WARNING][/bold yellow] 静音检测时出错: {e}")

                segment_audio_chunk = audio[current_pos_ms:actual_end_pos_ms]
                segment_filename = os.path.join(TEMP_DIR, f"segment_{segment_idx}.wav")
                try:
                    segment_audio_chunk.export(segment_filename, format="wav")
                    segments_info.append({"file": segment_filename, "start_ms": current_pos_ms})
                except Exception as e:
                    progress_bar.console.print(f"\n[bold red][ERROR][/bold red] 导出分段 {segment_filename} 失败: {e}")

                progress_bar.update(segment_task, advance=(actual_end_pos_ms - current_pos_ms))
                current_pos_ms = actual_end_pos_ms
                segment_idx += 1
            else: # 如果 audio_length_ms > 0, 则不应发生此情况
                break 
    
    console.print(f"[bold green][INFO][/bold green] 音频分段完成. 共创建 {len(segments_info)} 个分段.")
    status = load_status() # 重新加载状态, 以防在其他地方被修改 (此处不太可能)
    status["segments_info"] = segments_info
    save_status(status)
    return segments_info

# ========================
# 步骤 3: 转录与对齐 (Step 3: Transcription & Alignment)
# ========================
def load_whisper_model_rich():
    """使用 Rich 进度显示加载 WhisperX 模型, 并使用全局设置."""
    import whisperx # 此特定功能的局部导入
    global whisper_model_global, WHISPER_MODEL, COMPUTE_TYPE, DEVICE
    
    if whisper_model_global is None:
        console.print(f"[bold blue][INFO][/bold blue] 正在加载 WhisperX 模型 ([cyan]{WHISPER_MODEL}[/cyan], compute: [cyan]{COMPUTE_TYPE}[/cyan], device: [cyan]{DEVICE}[/cyan])...")
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
                progress.add_task("加载 Whisper 模型中...", total=None)
                with gpu_lock: # 确保线程安全的模型加载
                    whisper_model_global = whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE)
            console.print("[bold green][INFO][/bold green] WhisperX 模型加载成功.")
        except Exception as e:
            console.print(f"[bold red][ERROR][/bold red] 加载 WhisperX 模型失败: {e}")
            console.print("请检查模型名称, 计算类型, 设备设置以及 WhisperX 和相关依赖 (如 PyTorch) 是否正确安装.")
            if DEVICE == "cuda":
                console.print("如果使用 CUDA, 请确保 CUDA 环境已正确配置.")
            return None # 加载失败时返回 None
    return whisper_model_global

def transcribe_and_align_segment(segment_info, model_instance, align_model_cache, device_to_use=DEVICE):
    """转录并对齐单个音频分段."""
    import whisperx # 局部导入
    from contextlib import redirect_stdout # <--- 确保 redirect_stdout 在此导入
    import io # <--- 确保 io 在此导入, 因为 io.StringIO 被使用

    segment_file = segment_info["file"]
    segment_start_ms = segment_info["start_ms"]
    segment_basename = os.path.basename(segment_file)
    # 此分段的语言, 可能是预先检测的或每个分段检测一次
    detected_language_for_segment = segment_info.get("detected_language")

    try:
        audio_data = whisperx.load_audio(segment_file)
        
        # 抑制 whisperx 内部 print 输出的详细信息
        temp_stdout_buffer = io.StringIO()
        with redirect_stdout(temp_stdout_buffer): # 现在 redirect_stdout 应该已定义
            with gpu_lock: # 线程安全的转录
                transcription_result = model_instance.transcribe(
                    audio_data, 
                    batch_size=BATCH_SIZE, 
                    verbose=False, # 这是 WhisperX 的 verbose, 不是我们的
                    language=detected_language_for_segment # 如果已知语言, 则传递给模型
                )

        if not transcription_result or not transcription_result.get("segments"):
            return None # 没有有效的转录结果

        # 确定对齐模型的语言代码
        # 如果语言是预先检测的, 则使用它. 否则, 使用此分段转录结果中的语言.
        lang_code = detected_language_for_segment if detected_language_for_segment else transcription_result["language"]
        
        align_model, align_metadata = align_model_cache.get(lang_code, (None, None))

        if align_model is None: # 此语言的对齐模型尚未缓存
            try:
                # console.print(f"正在加载语言 {lang_code} 的对齐模型") # 调试信息
                with gpu_lock: # 线程安全的模型加载
                    align_model, align_metadata = whisperx.load_align_model(language_code=lang_code, device=device_to_use)
                align_model_cache[lang_code] = (align_model, align_metadata) # 缓存模型
            except Exception as e:
                # 此错误将由调用者捕获并打印上下文信息
                raise Exception(f"加载语言 '{lang_code}' 的对齐模型失败: {e}")
        
        with gpu_lock: # 线程安全的对齐
            aligned_result = whisperx.align(
                transcription_result["segments"], 
                align_model, 
                align_metadata, 
                audio_data, 
                device_to_use
            )

        # 将时间戳调整为绝对时间 (相对于原始完整音频)
        segment_start_sec = segment_start_ms / 1000.0
        final_adjusted_alignment = {"segments": []}

        if "word_segments" in aligned_result:
            final_adjusted_alignment["word_segments"] = []
            for word_info in aligned_result["word_segments"]:
                if "start" in word_info: word_info["start"] += segment_start_sec
                if "end" in word_info: word_info["end"] += segment_start_sec
                final_adjusted_alignment["word_segments"].append(word_info)

        for seg in aligned_result["segments"]:
            if "start" in seg: seg["start"] += segment_start_sec
            if "end" in seg: seg["end"] += segment_start_sec
            final_adjusted_alignment["segments"].append(seg)
        
        del audio_data # 释放内存
        gc.collect()
        return final_adjusted_alignment

    except Exception as e:
        # 返回错误信息, 由主处理循环处理
        return {"error": str(e), "segment_basename": segment_basename}

def process_all_segments(segments_info_list, current_status):
    """使用线程池处理所有音频分段的转录和对齐."""
    import whisperx # 局部导入
    from contextlib import redirect_stdout, redirect_stderr # 用于捕获输出

    model_instance = load_whisper_model_rich()
    if model_instance is None:
        console.print("[bold red][ERROR][/bold red] Whisper 模型未能加载. 中止处理.")
        return None

    # 从先前运行中获取的结果, 存储在状态文件中
    processed_results_map = current_status.get("processed_results", {})
    # 初始化列表以存储所有分段的结果
    all_segment_results = [None] * len(segments_info_list)
    
    tasks_to_submit_for_processing = []
    already_processed_count = 0
    
    # 如果尚未知晓, 尝试从第一个分段检测整体音频语言
    overall_detected_language = current_status.get("detected_language")
    if overall_detected_language is None and segments_info_list and os.path.exists(segments_info_list[0]["file"]):
        console.print(f"[bold blue][INFO][/bold blue] 正在使用第一个分段检测音频语言...")
        first_segment_audio = None
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress_lang:
                progress_lang.add_task("语言检测中...", total=None)
                first_segment_audio = whisperx.load_audio(segments_info_list[0]["file"])
                
                # 在此敏感的检测阶段捕获 whisperx 的 stdout/stderr
                buffer_stdout = io.StringIO()
                buffer_stderr = io.StringIO()
                with redirect_stdout(buffer_stdout), redirect_stderr(buffer_stderr), gpu_lock:
                     initial_transcription_result = model_instance.transcribe(first_segment_audio, batch_size=BATCH_SIZE, verbose=False)

            if initial_transcription_result and "language" in initial_transcription_result:
                overall_detected_language = initial_transcription_result["language"]
                console.print(f"[bold green][INFO][/bold green] 音频语言已检测: [cyan]{overall_detected_language}[/cyan]")
                current_status["detected_language"] = overall_detected_language
                save_status(current_status) # 保存检测到的语言
            else:
                console.print(f"[bold yellow][WARNING][/bold yellow] 无法从第一个分段检测语言. 将尝试逐段检测或使用模型默认设置.")
        except Exception as e:
            console.print(f"[bold yellow][WARNING][/bold yellow] 初始语言检测失败: {e}")
        finally:
            del first_segment_audio # 释放内存
            gc.collect()

    # 将整体检测到的语言传播到所有尚未设置语言的 segment_info 对象
    if overall_detected_language:
        for seg_info in segments_info_list:
            if "detected_language" not in seg_info:
                seg_info["detected_language"] = overall_detected_language
    
    # 对齐模型的缓存 (每种语言一个)
    alignment_model_cache = {} 
    if overall_detected_language: # 如果已知语言, 则预加载对齐模型
        console.print(f"[bold blue][INFO][/bold blue] 正在为语言 [cyan]{overall_detected_language}[/cyan] 预加载对齐模型...")
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress_align:
                progress_align.add_task(f"加载 {overall_detected_language} 对齐模型中...", total=None)
                with gpu_lock:
                    align_model, align_metadata = whisperx.load_align_model(language_code=overall_detected_language, device=DEVICE)
                alignment_model_cache[overall_detected_language] = (align_model, align_metadata)
            console.print(f"[bold green][INFO][/bold green] 语言 [cyan]{overall_detected_language}[/cyan] 的对齐模型已预加载.")
        except Exception as e:
            console.print(f"[bold yellow][WARNING][/bold yellow] 为 {overall_detected_language} 预加载对齐模型失败: {e}. 如有需要, 将尝试逐段加载.")

    # 识别需要处理的分段
    for idx, seg_info in enumerate(segments_info_list):
        idx_str = str(idx) # JSON 键是字符串
        if idx_str in processed_results_map and processed_results_map[idx_str] is not None and "error" not in processed_results_map[idx_str]:
            all_segment_results[idx] = processed_results_map[idx_str]
            already_processed_count += 1
        else:
            if os.path.exists(seg_info["file"]):
                tasks_to_submit_for_processing.append((idx, seg_info))
            else:
                console.print(f"[bold yellow][WARNING][/bold yellow] 分段文件 {seg_info['file']} 未找到. 跳过索引 {idx}.")
                all_segment_results[idx] = {"error": "File not found", "segment_basename": os.path.basename(seg_info.get("file","segment_NA"))}
                processed_results_map[idx_str] = all_segment_results[idx]


    if not tasks_to_submit_for_processing and already_processed_count == len(segments_info_list):
        console.print("[bold blue][INFO][/bold blue] 所有分段先前均已成功处理 (根据状态文件).")
        save_status(current_status) # 确保状态是最新的
        return all_segment_results
    
    num_workers = min(4, os.cpu_count() or 1) # 工作线程的合理默认值
    console.print(f"[bold blue][INFO][/bold blue] 正在转录 [cyan]{len(tasks_to_submit_for_processing)}[/cyan] 个新的/失败的分段 (共 {len(segments_info_list)} 个), 使用 [cyan]{num_workers}[/cyan] 个工作线程.")

    with Progress(
        TextColumn("[progress.description]{task.description}"), BarColumn(),
        TextColumn("[cyan]{task.completed}/{task.total}[/cyan] 段"),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeRemainingColumn(), console=console, transient=False
    ) as progress_bar:
        transcribe_task = progress_bar.add_task("转录进度", total=len(segments_info_list), completed=already_processed_count)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 将 future 映射到其原始索引, 以便正确放置结果
            future_to_idx_map = {
                executor.submit(transcribe_and_align_segment, seg_info_item, model_instance, alignment_model_cache, DEVICE): original_idx 
                for original_idx, seg_info_item in tasks_to_submit_for_processing
            }

            for future in as_completed(future_to_idx_map):
                idx = future_to_idx_map[future]
                idx_str = str(idx)
                try:
                    segment_result = future.result()
                    if segment_result is not None:
                        if "error" in segment_result: # transcribe_and_align_segment 返回的错误字典
                            progress_bar.console.print(f"\n[bold yellow][WARNING][/bold yellow] 分段 {idx} ({segment_result.get('segment_basename', 'N/A')}) 处理错误: {segment_result['error']}")
                            all_segment_results[idx] = None # 标记为失败
                            processed_results_map[idx_str] = {"error": str(segment_result['error'])}
                        else:
                            all_segment_results[idx] = segment_result
                            processed_results_map[idx_str] = segment_result
                    else: # Transcribe_and_align_segment 返回 None (例如, 没有转录结果)
                        progress_bar.console.print(f"\n[bold yellow][WARNING][/bold yellow] 分段 {idx} ({segments_info_list[idx]['file']}) 未能成功产生结果.")
                        all_segment_results[idx] = None
                        processed_results_map[idx_str] = None # 在状态中标记为 None
                    
                    current_status["processed_results"] = processed_results_map # 更新状态
                    save_status(current_status) # 增量保存进度

                except Exception as exc: # future 本身产生的意外异常
                    segment_basename_for_error = os.path.basename(segments_info_list[idx].get("file", f"segment_{idx}"))
                    progress_bar.console.print(f'\n[bold red][ERROR][/bold red] 分段 {idx} ({segment_basename_for_error}) 产生意外异常: {exc}')
                    all_segment_results[idx] = None
                    processed_results_map[idx_str] = {"error": str(exc), "unexpected": True}
                    current_status["processed_results"] = processed_results_map
                    save_status(current_status)
                finally:
                    progress_bar.update(transcribe_task, advance=1)
    
    successful_count = sum(1 for r in all_segment_results if r is not None and "error" not in r)
    failed_count = len(segments_info_list) - successful_count
    
    if failed_count > 0:
        console.print(f"[bold yellow][WARNING][/bold yellow] 转录完成. [cyan]{failed_count}[/cyan] 个分段未能成功处理.")
    else:
        console.print("[bold green][INFO][/bold green] 所有分段均已成功转录和对齐.")

    del alignment_model_cache # 清理模型缓存
    gc.collect()
    return all_segment_results

# ========================
# 步骤 4: 字幕文件生成 (Step 4: Subtitle File Generation)
# ========================
def format_timestamp(seconds_float):
    """将秒数 (浮点型) 格式化为 SRT 时间戳字符串 (HH:MM:SS,mmm)."""
    if seconds_float is None or seconds_float < 0: seconds_float = 0.0 # 处理 null 或负值
    
    # 计算小时, 分钟, 秒和毫秒
    total_seconds_int = int(seconds_float)
    milliseconds = int(round((seconds_float - total_seconds_int) * 1000))
    
    hours = total_seconds_int // 3600
    minutes = (total_seconds_int % 3600) // 60
    secs = total_seconds_int % 60

    # 处理可能级联到秒的毫秒舍入
    if milliseconds >= 1000:
        secs += 1
        milliseconds -= 1000 # 或者 milliseconds = 0
        if secs >= 60:
            minutes += 1
            secs -= 60 # 或者 secs = 0
            if minutes >= 60:
                hours += 1
                minutes -= 60 # 或者 minutes = 0
                
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def generate_srt(all_transcription_results, srt_output_path, use_word_timestamps=False):
    """从转录结果生成 SRT 字幕文件."""
    console.print("[bold blue][INFO][/bold blue] 正在生成 SRT 字幕文件...")
    srt_content_lines = []
    subtitle_entry_counter = 1

    if all_transcription_results is None:
        console.print("[bold red][ERROR][/bold red] 转录结果为空. 无法生成 SRT 文件.")
        return False

    for single_segment_result in all_transcription_results:
        if single_segment_result is None or ("error" in single_segment_result and single_segment_result["error"]):
            continue # 跳过失败或空的分段

        segments_to_process_for_srt = []
        # 选择使用词级时间戳还是分段/句子级时间戳
        if use_word_timestamps and single_segment_result.get("word_segments"):
            for word_info in single_segment_result["word_segments"]:
                start_time = word_info.get("start")
                end_time = word_info.get("end")
                text_content = word_info.get("word", "").strip() # .word 通常包含前导/尾随空格
                if start_time is not None and end_time is not None and text_content:
                    segments_to_process_for_srt.append({"start": start_time, "end": end_time, "text": text_content})
        elif single_segment_result.get("segments"): # 默认使用短语/句子分段
            for seg_data in single_segment_result["segments"]:
                start_time = seg_data.get("start")
                end_time = seg_data.get("end")
                text_content = seg_data.get("text", "").strip()
                if start_time is not None and end_time is not None and text_content:
                     segments_to_process_for_srt.append({"start": start_time, "end": end_time, "text": text_content})
        
        for srt_entry_data in segments_to_process_for_srt:
            start_time_sec = srt_entry_data["start"]
            end_time_sec = srt_entry_data["end"]
            text = srt_entry_data["text"]

            # 确保字幕条目具有有效的持续时间
            if end_time_sec <= start_time_sec:
                # console.print(f"[bold yellow][DEBUG][/bold yellow] 文本 '{text[:30]}...' 的时间戳无效或持续时间为零. 跳过.") # 可选的调试信息
                continue

            start_timestamp_str = format_timestamp(start_time_sec)
            end_timestamp_str = format_timestamp(end_time_sec)

            srt_content_lines.append(str(subtitle_entry_counter))
            srt_content_lines.append(f"{start_timestamp_str} --> {end_timestamp_str}")
            srt_content_lines.append(text)
            srt_content_lines.append("") # 空行分隔符
            subtitle_entry_counter += 1
    
    if not srt_content_lines:
        console.print("[bold yellow][WARNING][/bold yellow] 未生成任何有效的字幕行. 所有分段可能都处理失败或内容为空.")
        return False

    try:
        with open(srt_output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content_lines))
        # 在支持的终端中提供可点击的链接
        abs_srt_path = os.path.abspath(srt_output_path)
        console.print(f"[bold green][SUCCESS][/bold green] SRT 字幕文件已生成: [link=file://{abs_srt_path}]{srt_output_path}[/link]")
        return True
    except IOError as e:
        console.print(f"[bold red][ERROR][/bold red] 写入 SRT 文件失败: {e}")
        return False

# ========================
# Tkinter 文件对话框 (包含用于抑制 C 库警告的 stderr 重定向)
# ========================
def get_file_path_with_dialog():
    """使用 Tkinter 打开文件选择对话框.
    包含鲁棒的 stderr 重定向, 以抑制来自 Tk/Tcl 的潜在 C 库警告.
    """
    selected_file_path = ""
    original_stderr_fileno = -1 # 存储原始 stderr 文件描述符
    saved_stderr_fd_copy = -1   # 原始 stderr fd 的副本
    dev_null_fd = -1            # /dev/null 的文件描述符

    try:
        # 步骤 1: 获取当前 sys.stderr 的文件描述符
        # sys.stderr 可能是我们的 StderrFilter 实例或原始的 sys.__stderr__
        try:
            # 如果我们的过滤器处于活动状态并提供文件描述符, 则优先使用它
            current_stderr_fd = sys.stderr.fileno() 
            if isinstance(current_stderr_fd, int):
                original_stderr_fileno = current_stderr_fd
            else: # 如果 fileno() 返回非整数 (对于 StderrFilter 不应发生), 则回退
                original_stderr_fileno = sys.__stderr__.fileno() if hasattr(sys.__stderr__, 'fileno') else 2
        except (AttributeError, io.UnsupportedOperation):
            # 如果 sys.stderr 或 sys.__stderr__ 没有 fileno (例如, 在某些测试环境中), 则回退
            original_stderr_fileno = 2 # 默认为标准错误 fd

        # 步骤 2: 复制原始 stderr fd 以便稍后恢复
        saved_stderr_fd_copy = os.dup(original_stderr_fileno)
        
        # 步骤 3: 打开 /dev/null 并将 stderr 重定向到它
        dev_null_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(dev_null_fd, original_stderr_fileno)

        # 步骤 4: 现在, 导入并使用 Tkinter (希望 C 警告已被抑制)
        import tkinter as tk
        from tkinter import filedialog
        os.environ['TK_SILENCE_DEPRECATION'] = '1' # 针对 macOS Tkinter 弃用警告
        
        root = tk.Tk()
        root.withdraw() # 隐藏主 Tk 窗口
        console.print("[bold blue][INFO][/bold blue] 请选择要处理的视频或音频文件...")
        
        # 此调用将阻塞, 直到选择文件或关闭对话框
        dialog_selected_path = filedialog.askopenfilename(
            title="选择视频或音频文件",
            filetypes=[
                ("媒体文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.mp3 *.wav *.flac *.m4a *.aac *.ogg"),
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("音频文件", "*.mp3 *.wav *.flac *.m4a *.aac *.ogg"),
                ("所有文件", "*.*")
            ]
        )
        root.destroy() # 清理并关闭 Tkinter 根窗口
        selected_file_path = dialog_selected_path if dialog_selected_path else ""

    except Exception as e:
        # 如果发生错误, 确保在打印错误之前恢复 stderr
        if saved_stderr_fd_copy != -1 and original_stderr_fileno != -1:
            try: os.dup2(saved_stderr_fd_copy, original_stderr_fileno)
            except OSError as oe_restore: # 这是恢复 stderr 的严重失败
                  sys.__stderr__.write(f"CRITICAL: 在 except 块中恢复 stderr 失败: {oe_restore}\n") # 使用原始 stderr
        
        console.print(f"[bold red][ERROR][/bold red] 打开文件选择对话框时出错: {e}")
        selected_file_path = "" # 确保在出错时返回空路径
    finally:
        # 步骤 5: 始终恢复原始 stderr 并关闭复制的/打开的 fd
        if dev_null_fd != -1: # 如果 dev_null_fd 被 dup2 到 original_stderr_fileno 上, 必须先关闭它
            try: os.close(dev_null_fd)
            except OSError: pass
        
        if saved_stderr_fd_copy != -1 and original_stderr_fileno != -1:
            try: os.dup2(saved_stderr_fd_copy, original_stderr_fileno)
            except OSError as oe_final_restore: # 恢复 stderr 的严重失败
                sys.__stderr__.write(f"CRITICAL: 在 finally 块中恢复 stderr 失败: {oe_final_restore}\n")

        if saved_stderr_fd_copy != -1:
            try: os.close(saved_stderr_fd_copy) # 关闭复制的原始 fd
            except OSError: pass
            
    return selected_file_path

# ========================
# Rich UI - 主流程处理程序 (Main Process Handler)
# ========================
def handle_import_and_process():
    """处理主要工作流程: 文件选择, 处理和 SRT 生成."""
    global WHISPER_MODEL # 访问全局模型设置, 因为它可能影响恢复逻辑

    ensure_temp_dir()
    selected_input_file = ""
    
    # 文件选择循环, 直到选择有效文件或用户取消
    while True:
        selected_input_file = get_file_path_with_dialog()

        if not selected_input_file: # 用户取消了对话框
            console.print("[bold yellow][INFO][/bold yellow] 未选择文件.")
            if Confirm.ask("是否重试选择文件?", default=True):
                continue
            else:
                console.print("[bold blue][INFO][/bold blue] 返回主菜单.")
                return 
        
        if not os.path.isfile(selected_input_file):
            console.print(f"[bold red][ERROR][/bold red] 选择的路径不是有效文件: {selected_input_file}")
            if Confirm.ask("是否重试选择文件?", default=True):
                continue
            else:
                console.print("[bold blue][INFO][/bold blue] 返回主菜单.")
                return 
        else: # 文件有效
            console.print(f"[bold blue][INFO][/bold blue] 已选择文件: [cyan]{selected_input_file}[/cyan]")
            break 

    input_filename_base = os.path.basename(selected_input_file)
    # 输出的 SRT 文件将与输入文件同名, 扩展名为 .srt, 位于同一目录
    srt_output_filepath = os.path.splitext(selected_input_file)[0] + ".srt"
    # 临时目录中的标准化音频文件路径
    temp_audio_filepath = os.path.join(TEMP_DIR, "audio.wav")

    status_data = load_status()
    # 强制重新处理步骤的标志
    force_audio_extraction, force_audio_split, force_transcription = False, False, False

    # 根据输入文件名和使用的模型检查是否可以恢复
    if status_data.get("input_file") == input_filename_base and status_data.get("model_used") == WHISPER_MODEL:
        console.print(f"[bold blue][INFO][/bold blue] 找到与 '{input_filename_base}' (模型: {WHISPER_MODEL}) 相关的先前状态.")
        if Confirm.ask("是否尝试从上次中断处继续?", default=True):
            # 如果恢复, force 标志保持 False, 依赖状态文件获取进度
            pass
        else: # 用户选择不恢复
            console.print("[bold blue][INFO][/bold blue] 将为此文件重新开始处理. 所有步骤都将重做.")
            status_data = {} # 清空状态
            force_audio_extraction, force_audio_split, force_transcription = True, True, True
            cleanup_temp() # 清理临时文件以重新开始
            ensure_temp_dir() # 重新创建临时目录
    else: # 新文件或不同模型, 因此强制重新处理
        if status_data.get("input_file") != input_filename_base and status_data.get("input_file") is not None:
            console.print("[bold blue][INFO][/bold blue] 输入文件已更改. 开始新的处理.")
        if status_data.get("model_used") != WHISPER_MODEL and status_data.get("model_used") is not None:
             console.print(f"[bold blue][INFO][/bold blue] Whisper 模型已从 '{status_data.get('model_used', 'N/A')}' 更改为 '{WHISPER_MODEL}'. 需要重新处理.")
        status_data = {}
        force_audio_extraction, force_audio_split, force_transcription = True, True, True
        cleanup_temp()
        ensure_temp_dir()

    # 使用当前文件和模型更新状态, 然后保存
    status_data["input_file"] = input_filename_base
    status_data["model_used"] = WHISPER_MODEL 
    save_status(status_data)

    # --- 核心处理流程 (Core Processing Pipeline) ---
    if not extract_audio(selected_input_file, temp_audio_filepath, force_audio_extraction):
        console.print("[bold red][ERROR][/bold red] 音频提取失败. 处理中止.")
        return

    audio_segments = split_audio(temp_audio_filepath, force_audio_split)
    if not audio_segments:
        console.print("[bold red][ERROR][/bold red] 音频分段失败. 处理中止.")
        return
    
    # 注意: force_transcription 不直接传递给 process_all_segments.
    # process_all_segments 会智能地使用 status_data (如果 force_transcription 为 true, 则 status_data 已被清除)
    # 来确定哪些分段需要处理.
    transcription_results = process_all_segments(audio_segments, status_data) # status_data 在内部重新加载/管理
    
    if not transcription_results: 
        console.print("[bold red][ERROR][/bold red] 转录和对齐过程发生严重错误. 处理中止.")
        return # 如果结果为 None 或因严重故障导致空列表, 则无法继续

    # 检查部分成功的情况
    successful_segment_count = sum(1 for r in transcription_results if r is not None and "error" not in r)
    if successful_segment_count == 0:
        console.print("[bold red][ERROR][/bold red] 没有任何分段被成功转录. 无法生成 SRT.")
        return
    elif successful_segment_count < len(audio_segments):
        console.print(f"[bold yellow][WARNING][/bold yellow] 只有 {successful_segment_count}/{len(audio_segments)} 个分段成功处理. SRT 文件可能不完整.")
        if not Confirm.ask("是否仍要生成可能不完整的 SRT 文件?", default=False):
            return

    # 询问用户 SRT 的首选时间戳粒度
    use_word_level_timestamps = Confirm.ask(
        "SRT 是否使用 [词级别] 时间戳 (更精确但可能更零碎)?\n"
        "(选择 '否' 将使用 [句子/短语级别] 时间戳)", 
        default=False
    )
    
    if not generate_srt(transcription_results, srt_output_filepath, use_word_level_timestamps):
        console.print("[bold yellow][WARNING][/bold yellow] SRT 文件生成失败或未生成任何内容.")
    # 成功消息在 generate_srt 内部打印

    if Confirm.ask("是否清理临时文件?", default=True):
        cleanup_temp()
    else:
        console.print("[bold blue][INFO][/bold blue] 保留临时文件.")
    
    console.print("\n[bold blue][INFO][/bold blue] 处理流程结束. 返回主菜单.")

# ========================
# Rich UI - 其他处理程序 (Other Handlers)
# ========================
def handle_dependencies_check_ui():
    """用于检查和管理依赖项的 UI 处理程序."""
    console.print(Panel(Text("依赖检查与管理", justify="center", style="bold cyan"), box=ROUNDED, expand=False))
    
    if Confirm.ask("是否现在执行依赖项检查?", default=True):
        check_dependencies(verbose=True) # verbose=True 会打印详细信息
        # 消息由 check_dependencies 本身打印
    else:
        console.print("[bold blue][INFO][/bold blue] 跳过依赖检查. 返回主菜单.")
        return

    # 未来自动依赖安装的占位符
    if Confirm.ask("是否尝试自动下载并安装缺失的依赖项 (实验性功能)?", default=False):
        console.print("[bold yellow][INFO][/bold yellow] 自动依赖安装功能是占位符. 目前请手动安装.")
        # 示例代码:
        # missing = get_missing_dependencies_list() # 一个返回缺失依赖列表的函数
        # for dep in missing:
        #     console.print(f"尝试安装 {dep}...")
        #     subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
    else:
        console.print("[bold blue][INFO][/bold blue] 请根据检查结果手动安装任何缺失的依赖项.")
            
    console.print("\n[bold blue][INFO][/bold blue] 返回主菜单.")

def handle_model_selection_ui():
    """用于配置 Whisper 模型和处理参数的 UI 处理程序."""
    global WHISPER_MODEL, COMPUTE_TYPE, DEVICE, BATCH_SIZE, whisper_model_global # 允许修改全局变量
    
    console.print(Panel(Text("Whisper 模型与参数配置", justify="center", style="bold cyan"), box=ROUNDED, expand=False))
    
    # 显示当前设置
    settings_table = Table(title="当前模型设置", box=ROUNDED, show_lines=True)
    settings_table.add_column("参数", style="magenta")
    settings_table.add_column("当前值", style="green")
    settings_table.add_row("Whisper 模型 (WHISPER_MODEL)", WHISPER_MODEL)
    settings_table.add_row("计算类型 (COMPUTE_TYPE)", COMPUTE_TYPE)
    settings_table.add_row("运行设备 (DEVICE)", DEVICE)
    settings_table.add_row("批处理大小 (BATCH_SIZE)", str(BATCH_SIZE))
    console.print(settings_table)

    console.print("\n[bold]可配置选项:[/bold]")
    console.print("1. 选择 Whisper 模型大小 (例如: tiny, base, small, medium, large, large-v2, large-v3)")
    console.print("2. 设置计算类型 (例如: float16, float32, int8)")
    console.print("3. 设置运行设备 (例如: cuda, cpu)")
    console.print("4. 设置批处理大小 (整数)")
    console.print("5. 返回主菜单")

    choice = Prompt.ask("请选择要修改的配置项", choices=["1", "2", "3", "4", "5"], default="5")

    # 存储临时值以便在应用前确认
    temp_model, temp_compute, temp_device, temp_batch = WHISPER_MODEL, COMPUTE_TYPE, DEVICE, BATCH_SIZE

    if choice == "1":
        new_model = Prompt.ask(f"输入新的 Whisper 模型名称 (当前: {temp_model})", default=temp_model).strip()
        if new_model: temp_model = new_model
    elif choice == "2":
        new_compute = Prompt.ask(f"输入新的计算类型 (当前: {temp_compute}, 例如: float16, int8, float32)", default=temp_compute).strip()
        if new_compute in ["float16", "float32", "int8", "int8_float16"]: # 添加常用类型
            temp_compute = new_compute
        else:
            console.print(f"[yellow]无效的计算类型: {new_compute}. 保持当前设置.[/yellow]")
    elif choice == "3":
        new_device = Prompt.ask(f"输入新的运行设备 (当前: {temp_device}, 例如: cuda, cpu)", default=temp_device).strip()
        if new_device in ["cuda", "cpu"]:
            if new_device == "cuda":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        console.print("[bold yellow][WARNING][/bold yellow] CUDA 不可用! 选择 'cuda' 设备可能导致错误.")
                        if not Confirm.ask("仍要将设备设置为 'cuda' 吗?", default=False):
                            new_device = temp_device # 恢复原值
                except ImportError:
                     console.print("[bold yellow][WARNING][/bold yellow] 未找到 PyTorch, 无法验证 CUDA 可用性.")
            temp_device = new_device
        else:
            console.print(f"[yellow]无效的设备: {new_device}. 保持当前设置.[/yellow]")
    elif choice == "4":
        new_batch_size = IntPrompt.ask(f"输入新的批处理大小 (当前: {temp_batch})", default=temp_batch)
        if new_batch_size > 0:
            temp_batch = new_batch_size
        else:
            console.print(f"[yellow]无效的批处理大小: {new_batch_size}. 保持当前设置.[/yellow]")
    elif choice == "5":
        console.print("[bold blue][INFO][/bold blue] 未作更改. 返回主菜单.")
        return

    # 如果不是直接返回, 则确认并应用更改
    if choice != "5":
        console.print(f"\n[bold]建议的更改:[/bold]\n"
                      f"  模型: {temp_model}\n"
                      f"  计算类型: {temp_compute}\n"
                      f"  设备: {temp_device}\n"
                      f"  批处理大小: {temp_batch}")
        if Confirm.ask("\n是否应用以上更改?", default=True):
            # 检查是否有任何需要重新加载模型的关键设置已更改
            model_config_changed = (WHISPER_MODEL != temp_model or 
                                    COMPUTE_TYPE != temp_compute or 
                                    DEVICE != temp_device)

            WHISPER_MODEL, COMPUTE_TYPE, DEVICE, BATCH_SIZE = temp_model, temp_compute, temp_device, temp_batch
            
            if model_config_changed and whisper_model_global is not None:
                console.print("[INFO] 模型配置已更改. 清除已缓存的模型. 下次使用时将重新加载.")
                del whisper_model_global
                whisper_model_global = None 
                gc.collect()
                if DEVICE == "cuda" and 'torch' in sys.modules: # 检查是否已导入 torch
                    try:
                        torch.cuda.empty_cache() # 如果使用 CUDA, 清空 GPU 缓存
                    except Exception: pass # 如果 torch 或 cuda 未完全可用, 忽略错误

            console.print("[bold green][SUCCESS][/bold green] 模型配置已更新!")
        else:
            console.print("[bold blue][INFO][/bold blue] 更改已取消.")

    console.print("\n[bold blue][INFO][/bold blue] 返回主菜单.")

# ========================
# 主界面循环 (Main UI Loop)
# ========================
def display_main_menu_ui():
    """显示主菜单并获取用户选择."""
    console.rule("[bold cyan]Video-to-SRT 字幕生成工具[/bold cyan]", style="cyan")
    menu_text_obj = Text("\n请选择操作:\n\n", justify="left")
    menu_text_obj.append("[1] 导入并生成字幕 (默认)\n", style="yellow")
    menu_text_obj.append("[2] 检查与管理依赖项\n", style="yellow")
    menu_text_obj.append("[3] 配置 Whisper 模型与参数\n", style="yellow")
    menu_text_obj.append("[4] 退出程序\n\n", style="yellow")
    
    console.print(Panel(
        menu_text_obj, 
        title="主菜单", 
        border_style="magenta", 
        padding=(1,2),
        expand=False, # 不将面板扩展到全宽
        box=ROUNDED
    ))
    user_choice = Prompt.ask("输入选项 [1-4]", choices=["1", "2", "3", "4"], default="1")
    return user_choice

def main_cli_loop():
    """主命令行界面循环."""
    # 初始静默依赖检查 (可选, 可能有些打扰)
    # if not check_dependencies(verbose=False):
    #     console.print("[bold yellow][WARNING][/bold yellow] 检测到潜在的依赖项问题. "
    #                   "请使用选项 [2] 检查详细信息.")

    while True:
        user_action = display_main_menu_ui()
        if user_action == "1":
            handle_import_and_process()
        elif user_action == "2":
            handle_dependencies_check_ui()
        elif user_action == "3":
            handle_model_selection_ui()
        elif user_action == "4":
            if Confirm.ask("您确定要退出程序吗?", default=True):
                console.print("[bold blue]感谢使用, 程序已退出.[/bold blue]")
                break
        console.line(2) # 在再次显示菜单前添加一些间隔

if __name__ == "__main__":
    try:
        main_cli_loop()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]用户中断了进程. 正在退出.[/bold yellow]")
    except Exception as e:
        # 为未处理的异常打印 rich 格式的回溯信息
        console.print_exception(show_locals=True, width=None) # 自动宽度
        console.log(f"[bold red]发生未捕获的严重错误: {e}[/bold red]")
        console.log("请查看上面的错误详细信息. 您可能需要重新启动程序.")
    finally:
        # 任何最终的清理工作都可以放在这里
        # 使用 'with' 语句的 ThreadPoolExecutor 会自行处理关闭.
        console.print("程序已终止.", style="dim")
