# VideoToSRT_CPU.spec
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# 获取当前激活的 Conda 环境的 site-packages 路径
# sys.prefix 在激活的 Conda 环境中会指向该环境的根目录
conda_env_site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')

# --- 收集二进制文件 (DLLs, pyd) ---
binaries_to_collect = []

# 明确收集 PyTorch 的核心 DLLs
# PyInstaller 的 PyTorch hook 应该会处理大部分，但有时需要手动补充
# 目标路径 '.' 表示将它们放在打包后输出目录的根级别
torch_lib_path = os.path.join(conda_env_site_packages, 'torch', 'lib')
if os.path.isdir(torch_lib_path):
    critical_torch_dlls = [
        'c10.dll', 'torch.dll', 'torch_cpu.dll',
        'torch_global_deps.dll', 'fbgemm.dll', 'asmjit.dll',
        # OpenMP runtime, PyInstaller 的 hook 通常会处理，但可以明确指定
        'libiomp5md.dll', # Intel OpenMP
        # 'vcomp140.dll', # MSVC OpenMP, 如果你的 PyTorch 是用 MSVC 编译的
    ]
    for dll_name in critical_torch_dlls:
        dll_src_path = os.path.join(torch_lib_path, dll_name)
        if os.path.exists(dll_src_path):
            binaries_to_collect.append((dll_src_path, '.'))
        else:
            print(f"WARNING: Critical PyTorch DLL not found: {dll_src_path}", file=sys.stderr)

# 明确收集 torchvision._C.pyd
torchvision_c_pyd_path = os.path.join(conda_env_site_packages, 'torchvision', '_C.pyd')
if os.path.exists(torchvision_c_pyd_path):
    # 将 _C.pyd 放入名为 'torchvision' 的子目录中，以匹配其原始的相对导入路径
    binaries_to_collect.append((torchvision_c_pyd_path, 'torchvision'))
else:
    print(f"WARNING: torchvision._C.pyd not found: {torchvision_c_pyd_path}", file=sys.stderr)

# 明确收集 torchaudio 的 pyd (如果存在且需要)
# torchaudio_pyd_path = os.path.join(conda_env_site_packages, 'torchaudio', '_torchaudio.pyd') # 确认实际文件名
# if os.path.exists(torchaudio_pyd_path):
#     binaries_to_collect.append((torchaudio_pyd_path, 'torchaudio'))

# --- 收集数据文件 (例如 onnxruntime 可能需要的 .onnx 模型，transformers 的 tokenizer 文件等) ---
# PyInstaller hooks 通常会处理 transformers 和 onnxruntime 的数据文件
# datas_to_collect = []
# datas_to_collect += collect_data_files('onnxruntime')
# datas_to_collect += collect_data_files('transformers')
# datas_to_collect += collect_data_files('huggingface_hub')

a = Analysis(
    ['modified_script.py'],
    pathex=[conda_env_site_packages, '.'], # 确保首先搜索激活环境的 site-packages
    binaries=binaries_to_collect,
    datas=[], # 如果上面 datas_to_collect 有内容，则替换为 datas_to_collect
    hiddenimports=[
        'tkinter', 'tkinter.filedialog',
        'pydub', 'pydub.utils', 'pydub.scipy_effects', # pydub 及其子模块
        'soundfile', 'audioread', # pydub 可能的后端
        'torch', 'torchvision', 'torchvision.ops', 'torchaudio', # 核心 Torch 库
        'transformers', # 主 transformers 包
        # 显式包含脚本中可能间接使用的 transformers 模型组件
        'transformers.models.auto.modeling_auto',
        'transformers.models.auto.configuration_auto',
        'transformers.models.wav2vec2.modeling_wav2vec2',
        'transformers.models.wav2vec2.configuration_wav2vec2',
        'transformers.models.whisper.modeling_whisper',
        'transformers.models.whisper.processing_whisper',
        'transformers.models.whisper.feature_extraction_whisper',
        'ctranslate2', # faster-whisper 依赖
        'onnxruntime', # whisperx/faster-whisper 可能使用
        'onnxruntime.capi._pybind_state', # onnxruntime 的一部分
        'safetensors',
        'huggingface_hub',
        'huggingface_hub.inference_api',
        'pkg_resources', 'pkg_resources.py2_warn', # Setuptools 的一部分，常被需要
        'rich',
        # Python 标准库，有时 PyInstaller 会遗漏
        'shutil', 'json', 'threading', 'glob', 'gc', 'warnings', 'io', 'subprocess',
        'concurrent.futures',
        'platform',
        'importlib_metadata', # pkg_resources 的现代替代品，某些库可能需要
    ],
    hookspath=[], # 可以指定自定义 hook 目录
    hooksconfig={},
    runtime_hooks=[],
    excludes=[], # 可以排除不需要的模块以减小体积
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries, # 确保这里的 a.binaries 包含了我们手动添加的
    a.zipfiles,
    a.datas,    # 确保这里的 a.datas 包含了我们手动添加的
    [],
    name='VideoToSRT_CPU',
    debug=False, # 打包用于分发时设为 False，调试时可设为 True
    bootloader_ignore_signals=False,
    strip=False, # 尝试不剥离符号，有时有助于解决 DLL 问题，但会增加体积
    upx=False,   # 初始阶段禁用 UPX，以排除其干扰并加快启动（体积会变大）
    console=True, # 这是一个控制台应用
    icon=None,    # 可选：指定图标路径
)

# 如果是 --onedir 模式，使用 COLLECT
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False, # 对收集的文件也禁用 UPX
    name='VideoToSRT_CPU' # 输出的文件夹名称
)