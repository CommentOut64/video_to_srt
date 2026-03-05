# -*- coding: utf-8 -*-
"""
uv 依赖管理模块
V3.2.4+dev.20260306.02

使用 uv 替代 pip 进行依赖管理（仅开发模式），特性：
- uv sync --check 智能检测依赖状态
- 开发模式使用 --all-extras
- 生产模式使用基础依赖
- 无需标志文件（.env_installed, .req_hash）
"""

import os
import json
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("launcher.uv")


@dataclass
class UvResult:
    """uv 命令执行结果"""
    success: bool
    message: str
    needs_sync: bool = False


@dataclass
class OnnxRuntimeProviderStatus:
    """ONNX Runtime 提供者状态快照。"""

    is_ort_available: bool
    is_cuda_provider_available: bool
    ort_version: str
    providers: Tuple[str, ...]
    error_message: str = ""


class UvManager:
    """uv 依赖管理器"""

    def __init__(self, project_root: Path, dev_mode: bool = False):
        self.project_root = project_root
        self.dev_mode = dev_mode
        self.uv_exec = self._find_uv()

    def _find_uv(self) -> Optional[Path]:
        """查找 uv 可执行文件"""
        # 1. 项目内置的 uv
        tools_uv = self.project_root / "tools" / "uv.exe"
        if tools_uv.exists():
            return tools_uv

        # 2. 系统 PATH 中的 uv
        uv_path = shutil.which("uv")
        if uv_path:
            return Path(uv_path)

        return None

    def _find_embedded_python(self) -> Optional[Path]:
        """
        查找项目嵌入式 Python。

        V3.2.4+dev.20260305.01: 生产模式优先使用嵌入式 Python
        """
        embedded_python = self.project_root / "tools" / "python" / "python.exe"
        if embedded_python.exists():
            return embedded_python
        return None

    def is_available(self) -> bool:
        """检查 uv 是否可用"""
        return self.uv_exec is not None

    def _windows_no_window_kwargs(self) -> Dict[str, Any]:
        """
        Windows 下返回“无控制台窗口”子进程参数。

        设计取舍：
        - 仅在生产模式启用，开发模式保留默认行为便于调试。
        - 统一用于 run/Popen，避免 uv/python 探针偶发闪窗。
        """
        if os.name != "nt" or self.dev_mode:
            return {}

        kwargs: Dict[str, Any] = {}
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if creationflags:
            kwargs["creationflags"] = creationflags

        startupinfo_cls = getattr(subprocess, "STARTUPINFO", None)
        startf_use_showwindow = getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
        sw_hide = getattr(subprocess, "SW_HIDE", 0)
        if startupinfo_cls and startf_use_showwindow:
            startupinfo = startupinfo_cls()
            startupinfo.dwFlags |= startf_use_showwindow
            startupinfo.wShowWindow = sw_hide
            kwargs["startupinfo"] = startupinfo

        return kwargs

    def _build_uv_env(self) -> Dict[str, str]:
        """
        构建 uv 命令所需环境变量。

        设计取舍：
        - 强制锁定项目内 `.venv`，避免使用用户目录中的 uv 环境。
        - 生产模式默认使用 copy 链接，避免 .venv 依赖用户缓存路径。
        """
        env = os.environ.copy()
        venv_dir = self.project_root / ".venv"
        env.setdefault("UV_PROJECT_ENVIRONMENT", str(venv_dir))
        if not self.dev_mode:
            env.setdefault("UV_LINK_MODE", "copy")
        return env

    def _build_uv_python_args(self) -> Tuple[str, ...]:
        """
        为 uv 命令构建 --python 参数。

        V3.2.4+dev.20260305.01: 生产模式优先使用嵌入式 Python
        """
        if not self.dev_mode:
            embedded_python = self._find_embedded_python()
            if embedded_python:
                return ("--python", str(embedded_python))

        python_exec = self.get_python_path()
        if python_exec:
            return ("--python", str(python_exec))
        return ()

    def _ensure_local_python_home(self) -> None:
        """
        修正 pyvenv.cfg 中的 Python 基路径，避免指向用户缓存目录。

        适配场景：打包产物携带 tools/python 时，优先使用本地 Python。
        """
        venv_dir = self.project_root / ".venv"
        cfg_path = venv_dir / "pyvenv.cfg"
        if not cfg_path.exists():
            return

        local_home = self.project_root / "tools" / "python"
        if not local_home.exists():
            return

        try:
            text = cfg_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = cfg_path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            logger.warning("读取 pyvenv.cfg 失败: %s", exc)
            return

        lines = text.splitlines()
        if not lines:
            return

        updated_lines = []
        updated = False
        local_home_str = str(local_home)
        local_python_exe = local_home / "python.exe"

        def _should_rewrite(value: str) -> bool:
            if not value:
                return False
            if value == local_home_str:
                return False
            if "uv\\python" in value.lower():
                return True
            return not Path(value).exists()

        for line in lines:
            if "=" not in line:
                updated_lines.append(line)
                continue
            key, raw_value = line.split("=", 1)
            key_name = key.strip().lower()
            value = raw_value.strip()
            if key_name in {"home", "base-prefix", "base-exec-prefix"}:
                if _should_rewrite(value):
                    updated_lines.append(f"{key.strip()} = {local_home_str}")
                    updated = True
                else:
                    updated_lines.append(line)
                continue
            if key_name == "base-executable":
                if _should_rewrite(value) and local_python_exe.exists():
                    updated_lines.append(f"{key.strip()} = {local_python_exe}")
                    updated = True
                else:
                    updated_lines.append(line)
                continue
            updated_lines.append(line)

        if not updated:
            return

        try:
            cfg_path.write_text("\n".join(updated_lines), encoding="utf-8")
            logger.info("已修复 pyvenv.cfg Python 基路径: %s", local_home_str)
        except OSError as exc:
            logger.warning("写入 pyvenv.cfg 失败: %s", exc)

    def check_sync_status(self) -> UvResult:
        """
        检查依赖同步状态

        使用 uv sync --check 快速检测：
        - 返回码 0：环境已同步，无需操作
        - 返回码 1：环境过期，需要运行 uv sync
        """
        if not self.dev_mode:
            return UvResult(
                success=True,
                message="生产模式跳过 uv 依赖检查",
                needs_sync=False,
            )
        if not self.is_available():
            return UvResult(
                success=False,
                message="uv 未安装或不可用",
                needs_sync=True
            )

        try:
            self._ensure_local_python_home()
            python_args = self._build_uv_python_args()
            cmd = [
                str(self.uv_exec),
                "sync",
                "--check",
                "--quiet",
                "--no-install-project",
                *python_args,
            ]

            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                timeout=60,
                env=self._build_uv_env(),
                **self._windows_no_window_kwargs(),
            )

            if result.returncode == 0:
                return UvResult(
                    success=True,
                    message="依赖已同步，无需更新",
                    needs_sync=False
                )
            else:
                return UvResult(
                    success=True,
                    message="依赖需要同步",
                    needs_sync=True
                )

        except subprocess.TimeoutExpired:
            return UvResult(
                success=False,
                message="检查依赖状态超时",
                needs_sync=True
            )
        except Exception as e:
            return UvResult(
                success=False,
                message=f"检查依赖状态失败: {e}",
                needs_sync=True
            )

    def sync_dependencies(self, progress_callback=None) -> UvResult:
        """
        同步依赖

        Args:
            progress_callback: 进度回调函数 (message: str, progress: float)

        Returns:
            UvResult: 同步结果
        """
        if not self.dev_mode:
            return UvResult(success=True, message="生产模式跳过 uv 依赖同步")
        if not self.is_available():
            return UvResult(
                success=False,
                message="uv 未安装或不可用"
            )

        self._ensure_local_python_home()

        # 构建命令
        python_args = self._build_uv_python_args()
        cmd = [str(self.uv_exec), "sync", "--no-install-project", *python_args]

        if self.dev_mode:
            # 开发模式：安装所有可选依赖
            cmd.append("--all-extras")
            logger.info("开发模式：同步所有依赖（包括开发工具）")
        else:
            # 生产模式：仅安装核心依赖 + launcher 依赖
            cmd.extend(["--extra", "launcher"])
            logger.info("生产模式：同步核心依赖")

        if progress_callback:
            progress_callback("正在同步依赖...", 0.1)

        try:
            # 实时输出进度
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=self._build_uv_env(),
                **self._windows_no_window_kwargs(),
            )

            output_lines = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    line = line.strip()
                    output_lines.append(line)
                    logger.debug(line)

                    # 解析进度信息
                    if progress_callback:
                        if "Resolved" in line:
                            progress_callback("正在解析依赖...", 0.3)
                        elif "Prepared" in line or "Downloaded" in line:
                            progress_callback("正在下载包...", 0.5)
                        elif "Installed" in line:
                            progress_callback("正在安装包...", 0.7)
                        elif "Uninstalled" in line:
                            progress_callback("正在清理旧包...", 0.6)

            if process.returncode == 0:
                if progress_callback:
                    progress_callback("依赖同步完成", 1.0)
                return UvResult(
                    success=True,
                    message="依赖同步完成"
                )
            else:
                error_msg = "\n".join(output_lines[-10:])  # 最后10行
                return UvResult(
                    success=False,
                    message=f"依赖同步失败:\n{error_msg}"
                )

        except Exception as e:
            return UvResult(
                success=False,
                message=f"依赖同步异常: {e}"
            )

    def get_python_path(self) -> Optional[Path]:
        """
        获取 Python 路径。

        V3.2.4+dev.20260306.02: 生产模式仅使用嵌入式 Python
        """
        # 生产模式：仅使用嵌入式 Python
        if not self.dev_mode:
            embedded_python = self._find_embedded_python()
            return embedded_python

        # 开发模式：使用 venv Python
        venv_python = self.project_root / ".venv" / "Scripts" / "python.exe"
        if venv_python.exists():
            return venv_python

        # Linux/macOS
        venv_python_unix = self.project_root / ".venv" / "bin" / "python"
        if venv_python_unix.exists():
            return venv_python_unix

        return None

    def get_site_packages(self) -> Optional[Path]:
        """获取 site-packages 路径"""
        # 生产模式：嵌入式 Python
        if not self.dev_mode:
            embedded_site = self.project_root / "tools" / "python" / "Lib" / "site-packages"
            if embedded_site.exists():
                return embedded_site
            return None

        # 开发模式：venv
        site_packages = self.project_root / ".venv" / "Lib" / "site-packages"
        if site_packages.exists():
            return site_packages

        # Linux/macOS (需要检测 Python 版本)
        venv_lib = self.project_root / ".venv" / "lib"
        if venv_lib.exists():
            for item in venv_lib.iterdir():
                if item.name.startswith("python"):
                    sp = item / "site-packages"
                    if sp.exists():
                        return sp

        return None

    def ensure_venv(self) -> UvResult:
        """
        确保虚拟环境存在。

        V3.2.4+dev.20260306.02: 仅开发模式允许创建 venv
        """
        venv_dir = self.project_root / ".venv"

        if not self.dev_mode:
            return UvResult(success=False, message="生产模式禁止创建虚拟环境")

        if venv_dir.exists():
            return UvResult(success=True, message="虚拟环境已存在")

        if not self.is_available():
            return UvResult(
                success=False,
                message="uv 未安装，无法创建虚拟环境"
            )

        try:
            cmd = [str(self.uv_exec), "venv", str(venv_dir)]
            if not self.dev_mode:
                embedded_python = self._find_embedded_python()
                if embedded_python:
                    cmd.extend(["--python", str(embedded_python)])

            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                timeout=120,
                env=self._build_uv_env(),
                **self._windows_no_window_kwargs(),
            )

            if result.returncode == 0:
                return UvResult(success=True, message="虚拟环境创建成功")
            else:
                stderr = result.stderr.decode('utf-8', errors='replace')
                return UvResult(success=False, message=f"创建虚拟环境失败: {stderr}")

        except Exception as e:
            return UvResult(success=False, message=f"创建虚拟环境异常: {e}")

    def _build_python_runtime_env(self, site_packages: Optional[Path]) -> Dict[str, str]:
        """
        构建运行 Python 探针脚本时的环境变量。

        设计考虑：
        - 优先将 site-packages 下的 torch/lib 注入 PATH，减少 Windows 下 DLL 搜索歧义。
        - 保持与启动器主流程一致，仅做最小必要补充，不覆盖用户已有 PATH。
        """
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        if site_packages:
            torch_lib = site_packages / "torch" / "lib"
            if torch_lib.exists():
                env["PATH"] = f"{torch_lib};{env.get('PATH', '')}"
        return env

    @staticmethod
    def _extract_json_from_stdout(stdout_text: str) -> Dict[str, Any]:
        """从 stdout 中提取最后一行 JSON。"""
        for line in reversed(stdout_text.splitlines()):
            text = line.strip()
            if not text:
                continue
            if text.startswith("{") and text.endswith("}"):
                return json.loads(text)
        return {}

    def _inspect_onnxruntime_provider_status(
        self,
        python_exec: Path,
        site_packages: Optional[Path],
    ) -> OnnxRuntimeProviderStatus:
        """
        检查当前环境 onnxruntime 的 Provider 状态。

        返回值不抛异常，统一收敛为结构化状态，便于上层稳定决策。
        """
        probe_script = (
            "import json\n"
            "result = {\n"
            "  'is_ort_available': False,\n"
            "  'is_cuda_provider_available': False,\n"
            "  'ort_version': '',\n"
            "  'providers': [],\n"
            "  'error_message': ''\n"
            "}\n"
            "try:\n"
            "  import onnxruntime as ort\n"
            "  providers = list(ort.get_available_providers())\n"
            "  result['is_ort_available'] = True\n"
            "  result['ort_version'] = getattr(ort, '__version__', '')\n"
            "  result['providers'] = providers\n"
            "  result['is_cuda_provider_available'] = 'CUDAExecutionProvider' in providers\n"
            "except Exception as exc:\n"
            "  result['error_message'] = str(exc)\n"
            "print(json.dumps(result, ensure_ascii=False))\n"
        )

        try:
            result = subprocess.run(
                [str(python_exec), "-c", probe_script],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                env=self._build_python_runtime_env(site_packages),
                **self._windows_no_window_kwargs(),
            )
        except Exception as exc:
            return OnnxRuntimeProviderStatus(
                is_ort_available=False,
                is_cuda_provider_available=False,
                ort_version="",
                providers=(),
                error_message=f"探针执行失败: {exc}",
            )

        payload = self._extract_json_from_stdout(result.stdout)
        if not payload:
            stderr_tail = "\n".join(result.stderr.splitlines()[-3:]) if result.stderr else ""
            return OnnxRuntimeProviderStatus(
                is_ort_available=False,
                is_cuda_provider_available=False,
                ort_version="",
                providers=(),
                error_message=f"探针输出不可解析(returncode={result.returncode}) {stderr_tail}".strip(),
            )

        providers_raw = payload.get("providers") or []
        providers = tuple(str(item) for item in providers_raw)
        return OnnxRuntimeProviderStatus(
            is_ort_available=bool(payload.get("is_ort_available")),
            is_cuda_provider_available=bool(payload.get("is_cuda_provider_available")),
            ort_version=str(payload.get("ort_version") or ""),
            providers=providers,
            error_message=str(payload.get("error_message") or ""),
        )

    def _get_installed_distribution_version(
        self,
        python_exec: Path,
        package_name: str,
    ) -> Optional[str]:
        """读取指定 distribution 的版本；未安装时返回 None。"""
        script = (
            "import importlib.metadata as md\n"
            f"name = {package_name!r}\n"
            "try:\n"
            "  print(md.version(name))\n"
            "except Exception:\n"
            "  print('')\n"
        )
        try:
            result = subprocess.run(
                [str(python_exec), "-c", script],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                **self._windows_no_window_kwargs(),
            )
        except Exception as exc:
            logger.warning("读取包版本失败: package=%s error=%s", package_name, exc)
            return None

        version_text = result.stdout.strip()
        return version_text or None

    def _reinstall_onnxruntime_gpu(
        self,
        python_exec: Path,
        package_version: str,
    ) -> UvResult:
        """使用 uv pip 对 onnxruntime-gpu 做无依赖强制重装。"""
        if not self.is_available():
            return UvResult(success=False, message="uv 不可用，无法执行 ORT-GPU 自动修正")

        cmd = [
            str(self.uv_exec),
            "pip",
            "install",
            "--python",
            str(python_exec),
            "--force-reinstall",
            "--no-deps",
            f"onnxruntime-gpu=={package_version}",
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                env=self._build_uv_env(),
                **self._windows_no_window_kwargs(),
            )
        except Exception as exc:
            return UvResult(success=False, message=f"执行 ORT-GPU 自动修正异常: {exc}")

        if result.returncode == 0:
            return UvResult(success=True, message=f"onnxruntime-gpu=={package_version} 重装完成")

        err_tail = "\n".join(result.stdout.splitlines()[-8:])
        if result.stderr:
            err_tail = f"{err_tail}\n{result.stderr}".strip()
        return UvResult(
            success=False,
            message=f"onnxruntime-gpu 重装失败(returncode={result.returncode}):\n{err_tail}",
        )

    def ensure_onnxruntime_gpu_runtime(
        self,
        python_exec: Path,
        site_packages: Optional[Path],
    ) -> UvResult:
        """
        确保运行时 `onnxruntime` 指向可用的 GPU Provider 版本。

        设计取舍：
        - 不尝试从依赖图移除 `onnxruntime`（上游包强依赖，风险高）。
        - 采用“uv sync 后重装 onnxruntime-gpu（无依赖）+ provider 探针”稳定结果。
        """
        if not self.dev_mode:
            return UvResult(success=True, message="生产模式跳过 ONNX Runtime GPU 自动修正")

        gpu_version = self._get_installed_distribution_version(python_exec, "onnxruntime-gpu")
        if not gpu_version:
            return UvResult(success=True, message="未检测到 onnxruntime-gpu，跳过自动修正")

        status_before = self._inspect_onnxruntime_provider_status(python_exec, site_packages)
        if status_before.is_cuda_provider_available:
            providers = ",".join(status_before.providers)
            return UvResult(
                success=True,
                message=f"ONNX Runtime CUDA Provider 已可用: version={status_before.ort_version} providers=[{providers}]",
            )

        logger.warning(
            "检测到 ONNX Runtime CUDA Provider 不可用，开始自动修正: version=%s providers=%s error=%s",
            status_before.ort_version or "unknown",
            list(status_before.providers),
            status_before.error_message,
        )
        reinstall_result = self._reinstall_onnxruntime_gpu(python_exec, gpu_version)
        if not reinstall_result.success:
            return reinstall_result

        status_after = self._inspect_onnxruntime_provider_status(python_exec, site_packages)
        providers_after = ",".join(status_after.providers)
        if status_after.is_cuda_provider_available:
            return UvResult(
                success=True,
                message=f"ONNX Runtime GPU 自动修正成功: version={status_after.ort_version} providers=[{providers_after}]",
            )

        return UvResult(
            success=False,
            message=(
                "ONNX Runtime GPU 自动修正后仍无 CUDA Provider: "
                f"version={status_after.ort_version or 'unknown'} "
                f"providers=[{providers_after}] "
                f"error={status_after.error_message}"
            ),
        )


def fix_pytorch_dll(site_packages: Path):
    """
    修复 PyTorch 在 Windows 上的 DLL 依赖问题

    PyTorch 2.x+cu121 的 fbgemm.dll 依赖 libomp140.x86_64.dll (LLVM OpenMP)，
    但 Windows 系统默认不包含此 DLL。
    """
    import shutil as shutil_mod

    torch_lib = site_packages / "torch" / "lib"
    source_dll = torch_lib / "libiomp5md.dll"
    target_dll = torch_lib / "libomp140.x86_64.dll"

    if not torch_lib.exists():
        return  # PyTorch 未安装

    if target_dll.exists():
        return  # 已修复

    if source_dll.exists():
        logger.info("修复 PyTorch DLL 依赖...")
        shutil_mod.copy(source_dll, target_dll)
        logger.info("DLL 修复完成")
