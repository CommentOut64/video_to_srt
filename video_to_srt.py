import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import subprocess
import sys
import platform
import json
import threading
import shutil
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import warnings
# 已有的警告过滤
warnings.filterwarnings("ignore", message="TensorFloat-32 \\(TF32\\) has been disabled")
# 添加对 libpng 警告的过滤
warnings.filterwarnings("ignore", message=".*libpng warning.*")
# 忽略 PyAnnote Audio 的 TF32 相关警告
warnings.filterwarnings("ignore", message="TensorFloat-32 \\(TF32\\) has been disabled")
# 提前导入和设置 PyTorch
try:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except ImportError:
    pass

# ========================
# 配置区域
# ========================

# 依赖版本配置
TORCH_VERSION = "2.5.1"
TORCHAUDIO_VERSION = "2.0.2"
TORCHVISION_VERSION = "0.15.2"
WHISPERX_REPO = "git+https://github.com/m-bain/whisperx.git"

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
# 依赖检查
# ========================

def check_dependencies():
    """检查是否安装了必要的依赖库"""
    print("[INFO] 正在检查必要的依赖...")
    
    # 定义关键依赖项
    dependencies = [
        "pydub",
        "tqdm",
        "transformers",
        "ffmpeg-python",
        "faster-whisper",
        "pytorch-lightning"
    ]
    
    missing_deps = []
    
    # 检查PyTorch安装
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        import torchaudio
        print(f"[INFO] PyTorch已安装,版本: {torch.__version__}")
        print(f"[INFO] torchaudio已安装,版本: {torchaudio.__version__}")
        print(f"[INFO] CUDA是否可用: {torch.cuda.is_available()}")
    except ImportError:
        missing_deps.append(f"torch=={TORCH_VERSION}")
        missing_deps.append(f"torchaudio=={TORCHAUDIO_VERSION}")
        missing_deps.append(f"torchvision=={TORCHVISION_VERSION}")
    
    # 检查其他依赖
    for dep in dependencies:
        try:
            __import__(dep.replace("-", "_").split("==")[0])
            print(f"[INFO] {dep}已安装")
        except ImportError:
            missing_deps.append(dep)
    
    # 检查whisperx
    try:
        import whisperx
        print("[INFO] whisperx已安装")
    except ImportError:
        missing_deps.append(WHISPERX_REPO)
    
    # 检查ffmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode == 0:
            print("[INFO] ffmpeg已安装")
        else:
            print("[WARNING] ffmpeg检测失败,需要手动安装")
            missing_deps.append("ffmpeg")
    except FileNotFoundError:
        print("[WARNING] ffmpeg未找到,请安装ffmpeg并确保其在系统路径中")
        missing_deps.append("ffmpeg")
    
    if missing_deps:
        print("\n[WARNING] 以下依赖未安装:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\n请手动安装以上依赖后再运行程序.")
        return False
    
    print("[INFO] 依赖检查完成,所有必要依赖已安装")
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
            print("[WARNING] 状态文件格式错误,将重新开始.")
            return {}
    return {}


def save_status(status):
    """保存状态信息"""
    ensure_temp_dir()
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
    """调用 ffmpeg 提取音频为 WAV 格式 (单声道, 16kHz)"""
    if not force_extract and os.path.exists(audio_output_path):
        print(f"[INFO] 音频文件已存在,跳过提取：{audio_output_path}")
        return True

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        audio_output_path
    ]
    print("[INFO] 正在提取音频...")
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[INFO] 音频已成功提取到：{audio_output_path}")
        return True
    except FileNotFoundError:
        print("[ERROR] ffmpeg 未找到.请确保 ffmpeg 已安装并添加到系统 PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print("[ERROR] ffmpeg 提取音频时出错.")
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
    """将长音频切分为较短的片段"""
    from pydub import AudioSegment, silence
    from tqdm import tqdm
    
    existing_segment_files = get_existing_segments(TEMP_DIR)

    status = load_status()
    if not force_split and status.get("segments_info") and existing_segment_files:
        if len(existing_segment_files) == len(status["segments_info"]):
            print("[INFO] 使用之前保存的分段信息.")
            return status["segments_info"]
        else:
            print("[INFO] 现有分段文件与状态记录不匹配,将重新分段.")
    elif not force_split and existing_segment_files:
        print("[INFO] 检测到现有分段文件,但无状态信息,将重新分段以确保时间戳正确.")

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

    if existing_segment_files:
        print("[INFO] 清理旧的分段文件...")
        for f in existing_segment_files:
            try:
                os.remove(f)
            except OSError:
                pass

    pbar = tqdm(total=audio_length, unit='ms', desc="音频分段进度")
    while start < audio_length:
        end = start + SEGMENT_LENGTH_MS
        actual_end = min(end, audio_length)

        if actual_end > start:
            if actual_end < audio_length and (actual_end - start) > SILENCE_SEARCH_DURATION_MS:
                search_start = max(start, actual_end - SILENCE_SEARCH_DURATION_MS)
                search_segment = audio[search_start:actual_end]
                try:
                    silence_ranges = silence.detect_silence(
                        search_segment,
                        min_silence_len=MIN_SILENCE_LEN,
                        silence_thresh=SILENCE_THRESH
                    )
                    if silence_ranges:
                        first_silence_start_in_search = silence_ranges[0][0]
                        potential_end = search_start + first_silence_start_in_search
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
            break

    pbar.close()
    print(f"[INFO] 音频分段完成,共 {len(segments_info)} 个分段.")

    status = load_status()
    status["segments_info"] = segments_info
    save_status(status)

    return segments_info


# ========================
# 第三步：转录与强制对齐
# ========================

gpu_lock = threading.Lock()
whisper_model = None


def load_whisper_model():
    """全局加载WhisperX模型"""
    import whisperx
    
    global whisper_model
    if whisper_model is None:
        print(f"[INFO] 加载 WhisperX 模型 ({WHISPER_MODEL}, {COMPUTE_TYPE})...")
        try:
            whisper_model = whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE)
            print("[INFO] WhisperX 模型加载成功.")
        except Exception as e:
            print(f"[ERROR] 加载 WhisperX 模型失败: {e}")
            print("请检查模型名称、计算类型、设备设置以及 WhisperX 和相关依赖（如 PyTorch）是否正确安装.")
            if DEVICE == "cuda":
                print("确保 CUDA 环境已正确配置.")
            sys.exit(1)
    return whisper_model


def transcribe_and_align(segment_info, model, align_model_cache, device=DEVICE):
    """对单个音频分段进行转录和强制对齐"""
    import whisperx
    import io
    from contextlib import redirect_stdout
    
    segment_file = segment_info["file"]
    segment_start_ms = segment_info["start_ms"]
    segment_basename = os.path.basename(segment_file)
    detected_language = segment_info.get("detected_language")

    try:
        audio = whisperx.load_audio(segment_file)

        # 使用重定向stdout来隐藏语言检测输出
        temp_stdout = io.StringIO()
        with redirect_stdout(temp_stdout):
            with gpu_lock:
                # 如果已知语言，则指定语言进行转录
                if detected_language:
                    result = model.transcribe(audio, batch_size=BATCH_SIZE, verbose=False, language=detected_language)
                else:
                    result = model.transcribe(audio, batch_size=BATCH_SIZE, verbose=False)

        if not result or not result.get("segments"):
            print(f"[WARNING] 分段 {segment_basename} 未生成有效转录结果.")
            return None

        lang_code = detected_language if detected_language else result["language"]
        align_model, align_metadata = align_model_cache.get(lang_code, (None, None))

        if align_model is None:
            try:
                with gpu_lock:
                    align_model, align_metadata = whisperx.load_align_model(language_code=lang_code, device=device)
                align_model_cache[lang_code] = (align_model, align_metadata)
            except Exception as e:
                print(f"\n[ERROR] 加载语言 '{lang_code}' 的对齐模型失败: {e}")
                return None 
                
        with gpu_lock:
            result = model.transcribe(audio, batch_size=BATCH_SIZE, verbose=False)


        lang_code = detected_language if detected_language else result["language"]
        align_model, align_metadata = align_model_cache.get(lang_code, (None, None))

        if align_model is None:
            try:
                with gpu_lock:
                    align_model, align_metadata = whisperx.load_align_model(language_code=lang_code, device=device)
                align_model_cache[lang_code] = (align_model, align_metadata)
            except Exception as e:
                print(f"\n[ERROR] 加载语言 '{lang_code}' 的对齐模型失败: {e}")
                return None

        with gpu_lock:
            alignment = whisperx.align(result["segments"], align_model, align_metadata, audio, device)

        segment_start_sec = segment_start_ms / 1000.0
        adjusted_alignment = {"segments": []}

        if "word_segments" in alignment:
            adjusted_alignment["word_segments"] = []
            for word_info in alignment["word_segments"]:
                if "start" in word_info: word_info["start"] += segment_start_sec
                if "end" in word_info: word_info["end"] += segment_start_sec
                adjusted_alignment["word_segments"].append(word_info)

        for seg in alignment["segments"]:
            if "start" in seg: seg["start"] += segment_start_sec
            if "end" in seg: seg["end"] += segment_start_sec
            adjusted_alignment["segments"].append(seg)

        del audio
        gc.collect()

        return adjusted_alignment

    except Exception as e:
        print(f"\n[ERROR] 处理分段 {segment_basename} 时出错: {e}")
        return None


def process_all_segments(segments_info, status):
    """使用多线程处理所有音频分段的转录和对齐"""
    from tqdm import tqdm
    import sys
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    model = load_whisper_model()
    if model is None:
        print("[ERROR] Whisper 模型未能加载,无法继续处理.")
        return None

    processed_results_map = status.get("processed_results", {})
    all_results = [None] * len(segments_info)
    tasks_to_submit = []
    processed_count = 0

    # 从状态文件中获取已识别的语言信息
    detected_language = status.get("detected_language", None)

    # 如果还没有识别语言，且有音频切片需要处理，先识别一个切片的语言
    if detected_language is None and segments_info and os.path.exists(segments_info[0]["file"]):
        import whisperx
        
        # 同时重定向 stdout 和 stderr 以完全隐藏 WhisperX 输出
        temp_stdout = io.StringIO()
        temp_stderr = io.StringIO()
        
        try:
            # 加载音频但不显示任何输出
            audio = whisperx.load_audio(segments_info[0]["file"])
            
            # 完全捕获所有输出
            with redirect_stdout(temp_stdout), redirect_stderr(temp_stderr):
                initial_result = model.transcribe(audio, batch_size=BATCH_SIZE, verbose=False)
            
            # 从结果中提取语言信息并以自定义格式显示
            if initial_result and "language" in initial_result:
                detected_language = initial_result["language"]
                # 使用我们自己的格式输出语言检测结果
                print(f"[INFO] 音频语言识别完成: {detected_language}")
                # 保存检测到的语言供后续使用
                status["detected_language"] = detected_language
                save_status(status)
                
                # 让每个分段信息都包含检测到的语言
                for seg_info in segments_info:
                    seg_info["detected_language"] = detected_language

        except Exception as e:
            print(f"[WARNING] 初始语言检测失败: {e}")
            # 继续使用普通模式
            detected_language = None
    
    # 加载对齐模型（如果已知语言）
    align_model_cache = {}
    if detected_language:
        import whisperx
        print(f"[INFO] 使用预先检测的语言: {detected_language}")
        try:
            with gpu_lock:
                align_model, align_metadata = whisperx.load_align_model(language_code=detected_language, device=DEVICE)
            align_model_cache[detected_language] = (align_model, align_metadata)
            print(f"[INFO] {detected_language}语言的对齐模型预加载完成")
        except Exception as e:
            print(f"[WARNING] 预加载对齐模型失败: {e}")

    for idx, seg_info in enumerate(segments_info):
        idx_str = str(idx)
        if idx_str in processed_results_map:
            all_results[idx] = processed_results_map[idx_str]
            processed_count += 1
        else:
            if os.path.exists(seg_info["file"]):
                tasks_to_submit.append((idx, seg_info))
            else:
                if os.path.exists(seg_info["file"]):
                    # 确保每个段都带有语言信息
                    if detected_language and "detected_language" not in seg_info:
                        seg_info["detected_language"] = detected_language
                else:
                    print(f"[WARNING] 分段文件 {seg_info['file']} 未找到,跳过处理索引 {idx}.")

    if not tasks_to_submit:
        print("[INFO] 所有分段均已处理完成（根据状态文件）.")
        if processed_count == len(segments_info):
            return all_results
        else:
            print("[WARNING] 状态文件显示完成,但结果数量与分段数不匹配.建议检查临时文件或重新运行.")
            return all_results

    num_workers = min(4, os.cpu_count() or 1)
    print(
        f"[INFO] 开始转录 {len(tasks_to_submit)} 个未处理的分段（共 {len(segments_info)} 个）,使用 {num_workers} 个工作线程.")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {executor.submit(transcribe_and_align, seg_info, model, align_model_cache, DEVICE): idx for
                           idx, seg_info in tasks_to_submit}

        pbar = tqdm(total=len(segments_info), initial=processed_count, desc="转录进度", unit="段")

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            idx_str = str(idx)
            try:
                result = future.result()
                if result is not None:
                    all_results[idx] = result
                    processed_results_map[idx_str] = result
                    status["processed_results"] = processed_results_map
                    save_status(status)
                else:
                    print(f"\n[WARNING] 分段 {idx} 未能成功处理.")

            except Exception as exc:
                print(f'\n[ERROR] 分段 {idx} 执行时产生意外异常: {exc}')
            finally:
                pbar.update(1)

        pbar.close()

    completed_count = sum(1 for r in all_results if r is not None)
    if completed_count != len(segments_info):
        print(f"[WARNING] 转录完成,但有 {len(segments_info) - completed_count} 个分段未能成功处理.")
    else:
        print("[INFO] 所有分段转录处理完成.")

    del align_model_cache
    gc.collect()

    return all_results


# ========================
# 第四步：字幕文件生成
# ========================

def format_timestamp(seconds):
    """格式化时间戳为 SRT 格式 hh:mm:ss,ms"""
    if seconds is None or seconds < 0: seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
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
    """将所有转录结果组合成 SRT 文件"""
    print("[INFO] 正在生成 SRT 字幕文件...")
    srt_lines = []
    counter = 1

    if all_results is None:
        print("[ERROR] 转录结果为空,无法生成 SRT 文件.")
        return False

    for result in all_results:
        if result is None:
            continue

        if use_word_timestamps and result.get("word_segments"):
            for word_info in result["word_segments"]:
                start_time = word_info.get("start")
                end_time = word_info.get("end")
                text = word_info.get("word", "").strip()
                if start_time is None or end_time is None or not text:
                    continue

                start_ts = format_timestamp(start_time)
                end_ts = format_timestamp(end_time)

                srt_lines.append(str(counter))
                srt_lines.append(f"{start_ts} --> {end_ts}")
                srt_lines.append(text)
                srt_lines.append("")
                counter += 1
        elif result.get("segments"):
            for seg in result["segments"]:
                start_time = seg.get("start")
                end_time = seg.get("end")
                text = seg.get("text", "").strip()
                if start_time is None or end_time is None or not text:
                    continue

                if end_time < start_time:
                    print(f"[WARNING] 检测到无效时间戳 (end < start) 在段落: {text[:30]}...,跳过此行.")
                    continue

                start_ts = format_timestamp(start_time)
                end_ts = format_timestamp(end_time)

                srt_lines.append(str(counter))
                srt_lines.append(f"{start_ts} --> {end_ts}")
                srt_lines.append(text)
                srt_lines.append("")
                counter += 1
        else:
            pass

    if not srt_lines:
        print("[WARNING] 未生成任何有效的字幕行.")
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
    
    # 询问是否检查依赖
    check_deps_choice = input("是否检查依赖项？(y/n, 默认为n): ").strip().lower()
    if check_deps_choice != 'y':
        print("[INFO] 跳过依赖检查.")
    else:
        if not check_dependencies():
            print("[WARNING] 依赖检查发现问题,请安装所需依赖后重试.")
            return
        
    ensure_temp_dir()

    # 确保导入必要的库
    # 更完善的 stderr 重定向
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        # 使用环境变量来抑制 libpng 警告（适用于某些系统）
        os.environ['TK_SILENCE_DEPRECATION'] = '1'
        
        # 重定向标准错误输出
        with open(os.devnull, 'w') as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            
            # 创建并隐藏 root 窗口
            root = tk.Tk()
            root.withdraw()
            
            print("[INFO] 请选择要处理的视频文件...")
            video_path = filedialog.askopenfilename(
                title="选择要处理的视频文件",
                filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("所有文件", "*.*")]
            )
            root.destroy()
            
            # 恢复标准错误输出
            sys.stderr = old_stderr

    except Exception as e:
        # 确保恢复标准错误输出
        if 'old_stderr' in locals():
            sys.stderr = old_stderr
            
        print(f"[ERROR] 打开文件选择对话框时出错: {e}")
        print("[INFO] 请手动输入视频文件路径:")
        video_path_input = input("请输入视频文件的完整路径: ").strip('"')
        if not os.path.isfile(video_path_input):
            print(f"[ERROR] 无法找到文件: {video_path_input}")
            sys.exit(1)
        video_path = video_path_input # 赋值给 video_path
    finally:
        # 确保恢复 stderr
        sys.stderr = old_stderr
        # captured_stderr.close()

    if not video_path:
        print("[INFO] 未选择文件,程序退出.")
        sys.exit(0)

    print(f"[INFO] 已选择文件: {video_path}")

    # 检查选择的路径是否为有效文件
    if not os.path.isfile(video_path):
        print(f"[ERROR] 选择的路径不是有效文件：{video_path}")
        sys.exit(1)

    video_filename = os.path.basename(video_path)
    srt_output_path = os.path.splitext(video_path)[0] + ".srt"
    audio_path = os.path.join(TEMP_DIR, "audio.wav")

    # 断点续传逻辑
    status = load_status()
    force_extract = False
    force_split = False
    force_process = False

    if status.get("video_file") == video_filename:
        print(f"[INFO] 检测到与 '{video_filename}' 相关的先前状态.")
        resume_choice = input("是否尝试从上次中断处继续？(y/n,默认为 y): ").strip().lower()
        if resume_choice == 'n':
            print("[INFO] 用户选择重新开始处理.将清理旧状态和临时文件.")
            status = {}
            cleanup_temp()
            ensure_temp_dir()
            force_extract = True
            force_split = True
            force_process = True

    # 1. 更新状态文件
    status["video_file"] = video_filename
    save_status(status)

    # 2. 提取音频
    if not extract_audio(video_path, audio_path, force_extract):
        print("[ERROR] 音频提取失败,程序终止.")
        sys.exit(1)

    # 3. 分割音频
    segments_info = split_audio(audio_path, force_split)
    if not segments_info:
        print("[ERROR] 音频分段失败,程序终止.")
        sys.exit(1)

    # 4. 转录和强制对齐
    all_results = process_all_segments(segments_info, status)
    if not all_results:
        print("[ERROR] 转录过程失败,程序终止.")
        sys.exit(1)

    # 5. 生成字幕文件
    if generate_srt(all_results, srt_output_path):
        print(f"[SUCCESS] 字幕生成成功：{srt_output_path}")

    # 6. 询问是否清理临时文件
    cleanup_choice = input("是否清理临时文件？(y/n,默认为 n): ").strip().lower()
    if cleanup_choice == 'y':
        cleanup_temp()
        print("[INFO] 临时文件已清理.")
    else:
        print("[INFO] 保留临时文件以供后续使用.")


if __name__ == "__main__":
    main()