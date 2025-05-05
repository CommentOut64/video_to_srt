import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

import subprocess
import sys
import platform
import json
import threading
import shutil
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# 添加尝试导入必要库，如果导入失败，通过install_dependencies()安装
try:
    from pydub import AudioSegment, silence
    from tqdm import tqdm
    import whisperx
except ImportError:
    # 导入失败会在check_environment和install_dependencies中处理
    pass

# ========================
# 配置区域
# ========================

# 临时文件夹路径
TEMP_DIR = os.path.join(os.getcwd(), "temp")
# 用于记录上一次处理状态的文件
STATUS_FILE = os.path.join(TEMP_DIR, "status.json")
# 每个音频分段时长（毫秒）
SEGMENT_LENGTH_MS = 60 * 1000
# 分段时检测静音的区间参数
SILENCE_SEARCH_DURATION_MS = 2000
# 最低静音时长（毫秒）
MIN_SILENCE_LEN = 300
# 静音阈值（单位：dBFS）
SILENCE_THRESH = -40

# Whisperx 全局参数
DEVICE = "cuda"  # 或 "cpu"
BATCH_SIZE = 16
COMPUTE_TYPE = "float16"  # 或 "int8", "float32"
WHISPER_MODEL = "medium"  # 指定 Whisper 模型大小



# ========================
# 环境检测
# ========================

def install_dependencies():
    """检查并安装必要的依赖库"""
    print("[INFO] 检查并安装必要的依赖...")
    
    # 定义依赖项及其版本要求
    required_packages = {
        "pydub": "pydub",
        "tqdm": "tqdm",
        # 指定与CUDA 11.8兼容的PyTorch 1.13.1版本
        "torch": None,  # 特殊处理，见下方
        "transformers": "transformers",
        "ffmpeg-python": "ffmpeg-python"
    }
    
    # 检查并安装pip
    try:
        import pip
    except ImportError:
        print("[ERROR] pip未安装，请先安装pip")
        return False
    
    # 使用清华源
    pip_command = [sys.executable, "-m", "pip", "install", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]
    
    # 检查tkinter (系统级依赖)
    try:
        import tkinter
        print("[INFO] tkinter已安装")
    except ImportError:
        print("[WARNING] tkinter未安装，这是系统级依赖，需要单独安装")
        if platform.system() == "Windows":
            print("  在Windows上，tkinter通常随Python一起安装")
        elif platform.system() == "Linux":
            print("  在Linux上，请使用系统包管理器安装python-tk包")
            print("  例如: sudo apt-get install python3-tk")
        elif platform.system() == "Darwin":
            print("  在macOS上，请使用系统包管理器安装python-tk包")
            print("  例如: brew install python-tk")
    
    # 特别处理PyTorch - 安装CUDA 11.7版本(与CUDA 11.8兼容)
    try:
        import torch
        print(f"[INFO] PyTorch已安装，版本:{torch.__version__}")
        # 检查是否有完整兼容性问题
        try:
            import torchaudio
            print(f"[INFO] torchaudio已安装，版本:{torchaudio.__version__}")
            torch_installed = True
        except (ImportError, OSError) as e:
            print(f"[WARNING] torchaudio导入失败: {e}")
            torch_installed = False
    except ImportError:
        torch_installed = False
    
    if not torch_installed:
        print("[INFO] 卸载现有PyTorch相关包...")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
        
        print("[INFO] 安装与CUDA 11.8兼容的PyTorch 1.13.1...")
        try:
            # 确保安装完全匹配的版本集
            torch_cmd = pip_command + [
                "torch==1.13.1+cu117", 
                "torchvision==0.14.1+cu117", 
                "torchaudio==0.13.1+cu117", 
                "--extra-index-url", 
                "https://download.pytorch.org/whl/cu117"
            ]
            subprocess.check_call(torch_cmd)
            
            # 验证安装
            import torch
            import torchaudio
            print(f"[INFO] PyTorch安装成功，版本: {torch.__version__}")
            print(f"[INFO] torchaudio安装成功，版本: {torchaudio.__version__}")
            print(f"[INFO] CUDA是否可用: {torch.cuda.is_available()}")
        except subprocess.CalledProcessError:
            print("[ERROR] PyTorch安装失败")
            return False
    
    # 安装一般依赖
    for package, install_name in required_packages.items():
        if package == "torch":  # 已单独处理
            continue
        try:
            __import__(package)
            print(f"[INFO] {package}已安装")
        except ImportError:
            print(f"[INFO] 安装{package}...")
            if install_name:
                try:
                    subprocess.check_call(pip_command + [install_name])
                    print(f"[INFO] {package}安装成功")
                except subprocess.CalledProcessError:
                    print(f"[ERROR] {package}安装失败")
                    return False
    
    # 特殊处理pyannote.audio，安装0.x版本
    try:
        import pyannote.audio
        version = getattr(pyannote.audio, "__version__", "unknown")
        print(f"[INFO] pyannote.audio已安装，版本:{version}")
        if not version.startswith("0."):
            print("[WARNING] pyannote.audio版本不兼容，需要0.x版本")
            pyannote_installed = False
        else:
            pyannote_installed = True
    except ImportError:
        pyannote_installed = False
    
    if not pyannote_installed:
        print("[INFO] 安装pyannote.audio 0.x版本...")
        try:
            # 尝试安装特定版本，如果失败则尝试0.x系列
            try:
                subprocess.check_call(pip_command + ["pyannote.audio==0.0.1"])
            except subprocess.CalledProcessError:
                print("[INFO] 尝试安装pyannote.audio 0.x系列版本...")
                subprocess.check_call(pip_command + ["'pyannote.audio>=0.0.1,<1.0.0'"])
            print("[INFO] pyannote.audio安装成功")
        except subprocess.CalledProcessError:
            print("[ERROR] pyannote.audio安装失败")
            return False
    
# 安装whisperx
    try:
        import whisperx
        print("[INFO] whisperx已安装")
        # 额外验证whisperx是否可用
        try:
            # 尝试访问一个关键函数或属性
            getattr(whisperx, "load_model")
            print("[INFO] whisperx功能测试通过")
            whisperx_installed = True
        except (AttributeError, ImportError) as e:
            print(f"[WARNING] whisperx虽然已导入但功能测试失败: {e}")
            whisperx_installed = False
    except ImportError:
        print("[INFO] 无法导入whisperx，将尝试安装")
        whisperx_installed = False
    
    if not whisperx_installed:
        print("[INFO] 安装whisperx...")
        try:
            # 确保相关依赖包已经安装且可用
            subprocess.check_call(pip_command + ["faster-whisper"])
            
            # 卸载可能存在的有问题的whisperx版本
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "whisperx"])
            
            # 重新安装whisperx
            subprocess.check_call(pip_command + ["git+https://github.com/m-bain/whisperx.git"])
            
            # 清除sys.modules缓存中的whisperx，强制下次导入时重新加载
            if "whisperx" in sys.modules:
                del sys.modules["whisperx"]
                
            # 验证安装
            import whisperx
            print("[INFO] whisperx安装成功并可导入")
        except subprocess.CalledProcessError:
            print("[ERROR] whisperx安装失败")
            return False
        except ImportError as e:
            print(f"[ERROR] whisperx安装后仍然无法导入: {e}")
            return False
    
    print("[INFO] 所有依赖检查和安装完成")
    return True

def check_environment():
    """检查运行环境和必要组件"""
    print("[INFO] 检查运行环境...")

    # 检查 Python 版本
    python_version = sys.version.split()[0]
    print(f"[INFO] Python 版本: {python_version}")
    if int(python_version.split('.')[0]) < 3 or (
            int(python_version.split('.')[0]) == 3 and int(python_version.split('.')[1]) < 8):
        print("[ERROR] 需要 Python 3.8 或更高版本")
        return False

    # 检查必要的库
    required_packages = {
        "tkinter": "用于文件选择对话框",
        "pydub": "用于音频处理",
        "tqdm": "用于显示进度条",
        "whisperx": "用于语音识别和强制对齐"
    }

    missing_packages = []
    for package, desc in required_packages.items():
        try:
            if package == "tkinter":
                import tkinter
            elif package == "pydub":
                import pydub
            elif package == "tqdm":
                import tqdm
            elif package == "whisperx":
                # 清理缓存以确保重新导入
                if "whisperx" in sys.modules:
                    del sys.modules["whisperx"]
                import whisperx
                # 尝试访问一个关键功能以验证完整性
                getattr(whisperx, "load_model")
            print(f"[INFO] 已安装 {package}")
        except (ImportError, AttributeError) as e:
            missing_packages.append(f"{package} ({desc}) - 错误: {e}")

    if missing_packages:
        print("[ERROR] 缺少以下必要库:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\n请使用以下命令安装缺失的库:")
        print("pip install " + " ".join([pkg.split()[0] for pkg in missing_packages]))
        return False

    # 检查 ffmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=False)
        if result.returncode == 0:
            print("[INFO] ffmpeg 已安装")
        else:
            print("[ERROR] ffmpeg 测试失败")
            return False
    except FileNotFoundError:
        print("[ERROR] ffmpeg 未找到，请安装 ffmpeg 并确保其在系统路径中")
        print("下载地址: https://ffmpeg.org/download.html")
        return False

    # 如果设置为使用 CUDA，检查 CUDA 环境
    if "DEVICE" in globals() and DEVICE == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "未知"
                print(f"[INFO] CUDA 可用: 版本 {cuda_version}, 设备 {device_count}, 名称: {device_name}")
            else:
                print("[WARNING] CUDA 不可用")
                
        except ImportError:
            print("[WARNING] 无法导入 torch 以检查 CUDA 状态")
            

    print("[INFO] 环境检查完成")
    return True




# ========================
# 工具函数
# ========================
def ensure_temp_dir():
    """检查并创建临时文件夹"""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)


def load_status():
    """加载状态信息用于断点续传"""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("[WARNING] 状态文件格式错误，将重新开始。")
            return {}
    return {}


def save_status(status):
    """保存状态信息"""
    ensure_temp_dir()  # 确保目录存在
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(status, f, ensure_ascii=False, indent=4)


def cleanup_temp():
    """删除临时文件夹"""
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
            print(f"[INFO] 已清理临时文件夹：{TEMP_DIR}")
        except OSError as e:
            print(f"[ERROR] 清理临时文件失败: {e}")


# ========================
# 第一步：视频导入与音频提取
# ========================

def extract_audio(video_path, audio_output_path, force_extract=False):
    """
    调用 ffmpeg 提取音频为 WAV 格式 (单声道, 16kHz)。
    如果文件已存在且 force_extract 为 False，则跳过。
    """
    if not force_extract and os.path.exists(audio_output_path):
        print(f"[INFO] 音频文件已存在，跳过提取：{audio_output_path}")
        return True

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",  # 不输出视频
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",  # 明确指定WAV编码
        audio_output_path
    ]
    print("[INFO] 正在提取音频...")
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True)
        print(f"[INFO] 音频已成功提取到：{audio_output_path}")
        return True
    except FileNotFoundError:
        print("[ERROR] ffmpeg 未找到。请确保 ffmpeg 已安装并添加到系统 PATH。")
        return False
    except subprocess.CalledProcessError as e:
        print("[ERROR] ffmpeg 提取音频时出错。")
        print(f"命令: {' '.join(e.cmd)}")
        print(f"返回码: {e.returncode}")
        print(f"错误输出:\n{e.stderr}")
        return False


# ========================
# 第二步：音频分段
# ========================

def get_existing_segments(temp_dir):
    """查找临时目录中已存在的分段文件并排序"""
    segment_pattern = os.path.join(temp_dir, "segment_*.wav")
    return sorted(glob.glob(segment_pattern))


def split_audio(audio_path, force_split=False):
    """
    将长音频切分为较短的片段。
    如果分段文件已存在且 force_split 为 False，则尝试加载信息。
    返回：
        segments_info: 列表，每个元素为字典 {"file": path, "start_ms": start_time}
    """
    existing_segment_files = get_existing_segments(TEMP_DIR)

    # 检查：如果段文件存在且不强制分割，尝试从状态文件中获取分段信息
    status = load_status()
    if not force_split and status.get("segments_info") and existing_segment_files:
        # 检查现有文件是否与状态中的数量匹配
        if len(existing_segment_files) == len(status["segments_info"]):
            print("[INFO] 使用之前保存的分段信息。")
            return status["segments_info"]
        else:
            print("[INFO] 现有分段文件与状态记录不匹配，将重新分段。")
    elif not force_split and existing_segment_files:
        print("[INFO] 检测到现有分段文件，但无状态信息，将重新分段以确保时间戳正确。")

    print("[INFO] 开始音频分段...")
    try:
        audio = AudioSegment.from_wav(audio_path)
    except FileNotFoundError:
        print(f"[ERROR] 音频文件未找到：{audio_path}")
        return []
    except Exception as e:
        print(f"[ERROR] 加载音频文件失败: {e}")
        return []

    audio_length = len(audio)
    segments_info = []
    start = 0
    index = 0

    # 如果重新分割，先清理旧的分段文件
    if existing_segment_files:
        print("[INFO] 清理旧的分段文件...")
        for f in existing_segment_files:
            try:
                os.remove(f)
            except OSError:
                pass  # 忽略清理过程中的错误

    pbar = tqdm(total=audio_length, unit='ms', desc="音频分段进度")
    while start < audio_length:
        end = start + SEGMENT_LENGTH_MS
        actual_end = min(end, audio_length)  # 临时结束点

        if actual_end > start:  # 确保还有音频
            # 尝试在结束点附近寻找静音，但只有当不是最后一段时
            if actual_end < audio_length and (actual_end - start) > SILENCE_SEARCH_DURATION_MS:
                search_start = max(start, actual_end - SILENCE_SEARCH_DURATION_MS)
                search_segment = audio[search_start:actual_end]
                try:
                    # 找到相对于search_segment开始的静音
                    silence_ranges = silence.detect_silence(
                        search_segment,
                        min_silence_len=MIN_SILENCE_LEN,
                        silence_thresh=SILENCE_THRESH
                    )
                    if silence_ranges:
                        # 找到搜索窗口内第一个静音的开始
                        first_silence_start_in_search = silence_ranges[0][0]
                        # 计算原始音频时间线中的结束点
                        potential_end = search_start + first_silence_start_in_search
                        # 只有在合理的情况下使用这个静音点（不会太短）
                        if potential_end > start + MIN_SILENCE_LEN:
                            actual_end = potential_end
                except Exception as e:
                    print(f"\n[WARNING] 检测静音时出错: {e}")

            segment_audio = audio[start:actual_end]
            segment_filename = os.path.join(TEMP_DIR, f"segment_{index}.wav")
            try:
                segment_audio.export(segment_filename, format="wav")
                segments_info.append({"file": segment_filename, "start_ms": start})
            except Exception as e:
                print(f"\n[ERROR] 导出分段 {segment_filename} 失败: {e}")

            pbar.update(actual_end - start)
            start = actual_end
            index += 1
        else:
            break  # 没有更多音频或零长度段

    pbar.close()
    print(f"[INFO] 音频分段完成，共 {len(segments_info)} 个分段。")

    # 保存分段信息到状态用于潜在的恢复
    status = load_status()
    status["segments_info"] = segments_info
    save_status(status)

    return segments_info


# ========================
# 第三步：转录与强制对齐
# ========================

# GPU锁，在模型加载/卸载时或VRAM紧张时很有用
gpu_lock = threading.Lock()
# 全局模型变量，只加载一次
whisper_model = None


def load_whisper_model():
    """全局加载WhisperX模型"""
    global whisper_model
    if whisper_model is None:
        print(f"[INFO] 加载 WhisperX 模型 ({WHISPER_MODEL}, {COMPUTE_TYPE})...")
        try:
            whisper_model = whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE)
            print("[INFO] WhisperX 模型加载成功。")
        except Exception as e:
            print(f"[ERROR] 加载 WhisperX 模型失败: {e}")
            print("请检查模型名称、计算类型、设备设置以及 WhisperX 和相关依赖（如 PyTorch）是否正确安装。")
            if DEVICE == "cuda":
                print("确保 CUDA 环境已正确配置。")
            sys.exit(1)  # 如果模型加载失败，退出
    return whisper_model


def transcribe_and_align(segment_info, model, align_model_cache, device=DEVICE):
    """
    对单个音频分段进行转录和强制对齐。
    使用缓存的对齐模型（如果可用）。
    """
    segment_file = segment_info["file"]
    segment_start_ms = segment_info["start_ms"]
    segment_basename = os.path.basename(segment_file)

    try:
        # 使用whisperx函数加载音频
        audio = whisperx.load_audio(segment_file)

        # 1. 使用全局加载的模型进行转录
        with gpu_lock:  # 锁可能仍然需要，取决于GPU使用模式
            # 添加verbose=False以禁止语言检测消息
            result = model.transcribe(audio, batch_size=BATCH_SIZE, verbose=False)

        # 检查转录是否产生了段落
        if not result or not result.get("segments"):
            print(f"[WARNING] 分段 {segment_basename} 未生成有效转录结果。")
            return None  # 表示此段落失败

        # 2. 对齐
        lang_code = result["language"]
        align_model, align_metadata = align_model_cache.get(lang_code, (None, None))

        if align_model is None:
            try:
                with gpu_lock:  # 加载对齐模型时可能也需要锁
                    align_model, align_metadata = whisperx.load_align_model(language_code=lang_code, device=device)
                align_model_cache[lang_code] = (align_model, align_metadata)  # 缓存它
            except Exception as e:
                print(f"\n[ERROR] 加载语言 '{lang_code}' 的对齐模型失败: {e}")
                return None  # 表示失败

        with gpu_lock:  # 对齐过程的锁
            # 执行对齐
            alignment = whisperx.align(result["segments"], align_model, align_metadata, audio, device)

        # 3. 调整时间戳（转换毫秒为秒进行调整）
        segment_start_sec = segment_start_ms / 1000.0
        adjusted_alignment = {"segments": []}  # 确保结构存在

        # 如果存在，调整单词时间戳 (whisperx >= 3.0)
        if "word_segments" in alignment:
            adjusted_alignment["word_segments"] = []
            for word_info in alignment["word_segments"]:
                # 确保在调整前存在start/end
                if "start" in word_info: word_info["start"] += segment_start_sec
                if "end" in word_info: word_info["end"] += segment_start_sec
                adjusted_alignment["word_segments"].append(word_info)

        # 调整段落时间戳（对齐后始终存在）
        for seg in alignment["segments"]:
            # 确保在调整前存在start/end
            if "start" in seg: seg["start"] += segment_start_sec
            if "end" in seg: seg["end"] += segment_start_sec
            adjusted_alignment["segments"].append(seg)

        # 清理音频对象
        del audio
        gc.collect()  # 强制进行更积极的垃圾回收

        return adjusted_alignment  # 返回带有调整时间的结构

    except Exception as e:
        print(f"\n[ERROR] 处理分段 {segment_basename} 时出错: {e}")
        return None  # 表示失败


def process_all_segments(segments_info, status):
    """
    使用多线程处理所有音频分段的转录和对齐。
    利用 status 文件进行断点续传。
    """
    model = load_whisper_model()  # 确保在开始线程前加载模型
    if model is None:
        print("[ERROR] Whisper 模型未能加载，无法继续处理。")
        return None  # 返回None表示失败

    processed_results_map = status.get("processed_results", {})  # 加载映射 {idx_str: result}
    all_results = [None] * len(segments_info)  # 初始化结果列表
    tasks_to_submit = []
    processed_count = 0

    # 从状态填充结果并为未处理的段准备任务
    for idx, seg_info in enumerate(segments_info):
        idx_str = str(idx)  # 使用字符串键以兼容JSON
        if idx_str in processed_results_map:
            all_results[idx] = processed_results_map[idx_str]
            processed_count += 1
        else:
            # 在添加任务前检查段文件是否存在
            if os.path.exists(seg_info["file"]):
                tasks_to_submit.append((idx, seg_info))
            else:
                print(f"[WARNING] 分段文件 {seg_info['file']} 未找到，跳过处理索引 {idx}。")

    if not tasks_to_submit:
        print("[INFO] 所有分段均已处理完成（根据状态文件）。")
        # 验证all_results是否有每个段的条目
        if processed_count == len(segments_info):
            return all_results
        else:
            print("[WARNING] 状态文件显示完成，但结果数量与分段数不匹配。建议检查临时文件或重新运行。")
            return all_results  # 返回我们拥有的内容

    num_workers = min(4, os.cpu_count() or 1)  # 限制工作线程数，尤其是使用GPU锁时
    print(
        f"[INFO] 开始转录 {len(tasks_to_submit)} 个未处理的分段（共 {len(segments_info)} 个），使用 {num_workers} 个工作线程。")

    # 对齐模型缓存（每种语言一个）
    align_model_cache = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 将future映射到索引
        future_to_index = {executor.submit(transcribe_and_align, seg_info, model, align_model_cache, DEVICE): idx for
                           idx, seg_info in tasks_to_submit}

        # 进度条设置
        pbar = tqdm(total=len(segments_info), initial=processed_count, desc="转录进度", unit="段")

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            idx_str = str(idx)
            try:
                result = future.result()  # 从transcribe_and_align获取结果
                if result is not None:
                    all_results[idx] = result
                    # 处理成功后立即更新状态
                    processed_results_map[idx_str] = result
                    status["processed_results"] = processed_results_map
                    save_status(status)
                else:
                    print(f"\n[WARNING] 分段 {idx} 未能成功处理。")

            except Exception as exc:
                # 处理future本身引发的异常（如果在任务内处理，则很少发生）
                print(f'\n[ERROR] 分段 {idx} 执行时产生意外异常: {exc}')
            finally:
                # 无论段落成功/失败，都更新进度条
                pbar.update(1)

        pbar.close()

    # 处理后的最终检查
    completed_count = sum(1 for r in all_results if r is not None)
    if completed_count != len(segments_info):
        print(f"[WARNING] 转录完成，但有 {len(segments_info) - completed_count} 个分段未能成功处理。")
    else:
        print("[INFO] 所有分段转录处理完成。")

    # 清理对齐模型缓存
    del align_model_cache
    gc.collect()

    return all_results


# ========================
# 第四步：字幕文件生成
# ========================

def format_timestamp(seconds):
    """格式化时间戳为 SRT 格式 hh:mm:ss,ms"""
    if seconds is None or seconds < 0: seconds = 0  # 处理潜在的None或负值
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    # 确保毫秒不超过999
    if milliseconds >= 1000:
        secs += 1
        milliseconds = 0
        if secs >= 60:
            minutes += 1
            secs = 0
            if minutes >= 60:
                hours += 1
                minutes = 0
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def generate_srt(all_results, srt_output_path, use_word_timestamps=False):
    """
    将所有转录结果组合成 SRT 文件。
    优先使用 'segments'，如果 use_word_timestamps=True 且 'word_segments' 存在，则使用词级别时间戳。
    """
    print("[INFO] 正在生成 SRT 字幕文件...")
    srt_lines = []
    counter = 1

    if all_results is None:
        print("[ERROR] 转录结果为空，无法生成 SRT 文件。")
        return False

    for result in all_results:
        if result is None:
            continue

        # 选项 1：如果请求并可用，使用词级别时间戳
        if use_word_timestamps and result.get("word_segments"):
            for word_info in result["word_segments"]:
                start_time = word_info.get("start")
                end_time = word_info.get("end")
                text = word_info.get("word", "").strip()
                # 如果基本信息缺失或文本为空，则跳过
                if start_time is None or end_time is None or not text:
                    continue

                start_ts = format_timestamp(start_time)
                end_ts = format_timestamp(end_time)

                srt_lines.append(str(counter))
                srt_lines.append(f"{start_ts} --> {end_ts}")
                srt_lines.append(text)
                srt_lines.append("")
                counter += 1
        # 选项 2：使用段落级别时间戳（默认）
        elif result.get("segments"):
            for seg in result["segments"]:
                start_time = seg.get("start")
                end_time = seg.get("end")
                text = seg.get("text", "").strip()
                # 如果基本信息缺失或文本为空，则跳过
                if start_time is None or end_time is None or not text:
                    continue

                # 检查无效时间戳（结束时间早于开始时间）
                if end_time < start_time:
                    print(f"[WARNING] 检测到无效时间戳 (end < start) 在段落: {text[:30]}...，跳过此行。")
                    continue

                start_ts = format_timestamp(start_time)
                end_ts = format_timestamp(end_time)

                srt_lines.append(str(counter))
                srt_lines.append(f"{start_ts} --> {end_ts}")
                srt_lines.append(text)
                srt_lines.append("")
                counter += 1
        else:
            pass  # 此结果段中没有可用数据

    if not srt_lines:
        print("[WARNING] 未生成任何有效的字幕行。")
        return False

    try:
        with open(srt_output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))
        print(f"[INFO] SRT 字幕文件已成功生成：{srt_output_path}")
        return True
    except IOError as e:
        print(f"[ERROR] 写入 SRT 文件失败: {e}")
        return False


# ========================
# 主程序
# ========================
def main():
    """主函数：处理流程"""

    # 安装依赖
    if not install_dependencies():
        print("[ERROR] 依赖安装失败，程序无法继续。")
        sys.exit(1)
        
    # 检查环境
    if not check_environment():
        print("[ERROR] 环境检查失败，程序无法继续。")
        sys.exit(1)
    ensure_temp_dir()

    # 确保导入必要的库（在环境检查后）
    try:
        import tkinter as tk
        from tkinter import filedialog
        from pydub import AudioSegment, silence
        from tqdm import tqdm
        import whisperx
        import gc
    except ImportError as e:
        print(f"[ERROR] 导入必要库失败: {e}")
        print("请尝试重新运行程序或手动安装缺失的依赖。")
        sys.exit(1)

    # 添加文件选择对话框
    try:
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        print("[INFO] 请选择要处理的视频文件...")
        video_path = filedialog.askopenfilename(
            title="选择要处理的视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("所有文件", "*.*")]
        )
        root.destroy()
    except Exception as e:
        print(f"[ERROR] 打开文件选择对话框时出错: {e}")
        print("[INFO] 请手动输入视频文件路径:")
        video_path = input("请输入视频文件的完整路径: ").strip('"')
        if not os.path.isfile(video_path):
            print(f"[ERROR] 无法找到文件: {video_path}")
            sys.exit(1)

    if not video_path:  # 如果用户取消选择
        print("[INFO] 未选择文件，程序退出。")
        sys.exit(0)

    print(f"[INFO] 已选择文件: {video_path}")
    # --- 文件选择结束 ---

    # 检查选择的路径是否为有效文件
    if not os.path.isfile(video_path):
        print(f"[ERROR] 选择的路径不是有效文件：{video_path}")
        sys.exit(1)

    video_filename = os.path.basename(video_path)
    srt_output_path = os.path.splitext(video_path)[0] + ".srt"
    audio_path = os.path.join(TEMP_DIR, "audio.wav")

    # --- 断点续传逻辑 ---
    status = load_status()
    force_extract = False
    force_split = False
    force_process = False  # 强制重新转录的标志

    # 检查状态文件是否用于相同的视频
    if status.get("video_file") == video_filename:
        print(f"[INFO] 检测到与 '{video_filename}' 相关的先前状态。")
        # 询问用户是否要恢复或重新开始
        resume_choice = input("是否尝试从上次中断处继续？(y/n，默认为 y): ").strip().lower()
        if resume_choice == 'n':
            print("[INFO] 用户选择重新开始处理。将清理旧状态和临时文件。")
            status = {}  # 清除状态
            cleanup_temp()  # 清理所有临时文件
            ensure_temp_dir()  # 重新创建临时目录
            force_extract = True
            force_split = True
            force_process = True

    # 1. 更新状态文件
    status["video_file"] = video_filename
    save_status(status)

    # 2. 提取音频
    if not extract_audio(video_path, audio_path, force_extract):
        print("[ERROR] 音频提取失败，程序终止。")
        sys.exit(1)

    # 3. 分割音频
    segments_info = split_audio(audio_path, force_split)
    if not segments_info:
        print("[ERROR] 音频分段失败，程序终止。")
        sys.exit(1)

    # 4. 转录和强制对齐
    all_results = process_all_segments(segments_info, status)
    if not all_results:
        print("[ERROR] 转录过程失败，程序终止。")
        sys.exit(1)

    # 5. 生成字幕文件
    if generate_srt(all_results, srt_output_path):
        print(f"[SUCCESS] 字幕生成成功：{srt_output_path}")

    # 6. 询问是否清理临时文件
    cleanup_choice = input("是否清理临时文件？(y/n，默认为 n): ").strip().lower()
    if cleanup_choice == 'y':
        cleanup_temp()
        print("[INFO] 临时文件已清理。")
    else:
        print("[INFO] 保留临时文件以供后续使用。")


if __name__ == "__main__":
    main()
