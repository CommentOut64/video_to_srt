# VideoToSRT_CPU.spec
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None
conda_env_site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')

binaries_to_collect = []

# PyTorch 核心 DLLs
torch_lib_path = os.path.join(conda_env_site_packages, 'torch', 'lib')
if os.path.isdir(torch_lib_path):
    critical_torch_dlls = [
        'c10.dll', 'torch.dll', 'torch_cpu.dll',
        'torch_global_deps.dll', 'fbgemm.dll', 'asmjit.dll',
        'libiomp5md.dll',
    ]
    for dll_name in critical_torch_dlls:
        dll_src_path = os.path.join(torch_lib_path, dll_name)
        if os.path.exists(dll_src_path):
            binaries_to_collect.append((dll_src_path, '.'))

# torchvision._C.pyd
torchvision_c_pyd_path = os.path.join(conda_env_site_packages, 'torchvision', '_C.pyd')
if os.path.exists(torchvision_c_pyd_path):
    binaries_to_collect.append((torchvision_c_pyd_path, 'torchvision'))
else:
    # 保留一个信息性的打印，说明文件未找到，但这可能不是关键问题
    print(f"INFO: torchvision._C.pyd not found (this might be okay if torchvision's C extensions are not strictly needed): {torchvision_c_pyd_path}", file=sys.stderr)


# --- 收集数据文件 ---
datas_to_collect = []

def add_data_files_to_subdir(package_name, subdir_name, current_datas_list, include_py_files=False):
    try:
        collected_files = collect_data_files(package_name, include_py_files=include_py_files)
        for src_path, relative_dest_path in collected_files:
            current_datas_list.append((src_path, os.path.join(subdir_name, relative_dest_path)))
    except Exception as e:
        print(f"Error collecting data files for {package_name} into {subdir_name}: {e}", file=sys.stderr)

add_data_files_to_subdir('transformers', 'transformers', datas_to_collect, include_py_files=False)
add_data_files_to_subdir('huggingface_hub', 'huggingface_hub', datas_to_collect, include_py_files=False)
add_data_files_to_subdir('whisperx', 'whisperx', datas_to_collect, include_py_files=False)
add_data_files_to_subdir('faster_whisper', 'faster_whisper', datas_to_collect, include_py_files=False)
add_data_files_to_subdir('soundfile', 'soundfile', datas_to_collect, include_py_files=False)


a = Analysis(
    ['modified_script.py'],
    pathex=[conda_env_site_packages, '.'],
    binaries=binaries_to_collect,
    datas=datas_to_collect,
    hiddenimports=[
        'tkinter', 'tkinter.filedialog',
        'pydub', 'pydub.utils', 'pydub.scipy_effects',
        'soundfile', 'audioread', '_soundfile_data',
        'torch', 'torchvision', 'torchvision.ops', 'torchaudio',
        'transformers',
        'transformers.models.auto.modeling_auto',
        'transformers.models.auto.configuration_auto',
        'transformers.models.wav2vec2.modeling_wav2vec2',
        'transformers.models.wav2vec2.configuration_wav2vec2',
        'transformers.models.whisper.modeling_whisper',
        'transformers.models.whisper.processing_whisper',
        'transformers.models.whisper.feature_extraction_whisper',
        'ctranslate2',
        'faster_whisper',
        'whisperx', 'whisperx.asr', 'whisperx.alignment', 'whisperx.utils',
        'onnxruntime', 'onnxruntime.capi._pybind_state',
        'safetensors',
        'pytorch_lightning',
        'huggingface_hub', 'huggingface_hub.inference_api', 'huggingface_hub.utils',
        'filelock',
        'requests',
        'tqdm',
        'regex',
        'packaging',
        'tiktoken',
        'pkg_resources', 'pkg_resources.py2_warn',
        'rich', 'rich.themes',
        'shutil', 'json', 'threading', 'glob', 'gc', 'warnings', 'io', 'subprocess',
        'concurrent.futures',
        'platform',
        'importlib_metadata',
        'charset_normalizer',
        'idna',
        'pdb',
        'unittest',         # <--- 添加 unittest
        'unittest.mock',    # <--- 添加 unittest.mock
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'doctest', 'test', 'tests', # 'unittest' 已从此列表中移除, 'pdb' 也已移除
        'tkinter.test', 'tkinter.tix', 'FixTk',
        'PIL.ImageTk', 'PIL._tkinter_finder',
        'cv2',
        'matplotlib', 'pandas', 'scipy.spatial', 'scipy.linalg',
        'IPython', 'jupyter_client', 'jupyter_core',
        'PyQt5', 'PySide2', 'wx',
        'torch.utils.tensorboard',
        'onnxruntime.training',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='VideoToSRT_CPU',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='VideoToSRT_CPU'
)