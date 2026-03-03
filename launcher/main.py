# -*- coding: utf-8 -*-
"""
AnchorFlux Launcher - 主入口
V3.2.4+dev.20260303.04

新架构启动器，特性：
- uv 依赖管理（替代 pip，智能检测依赖状态）
- Go Stub + onedir 打包方案（快速启动）
- 自更新系统（支持启动器自身更新）
- CustomTkinter 更新界面（轻量美观）
- 优化的进程清理机制（快速退出）
"""

import os
import sys
import time
import signal
import logging
import subprocess
from pathlib import Path
from typing import Optional

from .config import (
    VERSION, APP_NAME, LauncherConfig,
    get_project_root, load_env_config, detect_dev_mode,
    DEFAULT_UI_MODE, DEFAULT_RUNTIME_POLICY, DEFAULT_FLAVOR,
    DEFAULT_BACKEND_PORT, DEFAULT_FRONTEND_PORT,
    normalize_ui_mode, normalize_runtime_policy, normalize_flavor
)
from .uv_manager import UvManager, fix_pytorch_dll
from .updater import SelfUpdater
from .shell_manager import (
    wait_backend_ready,
    launch_electron,
    open_browser_fallback,
    resolve_shell_path,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("launcher")


class SingleInstanceGuard:
    """
    启动器单实例锁。

    设计取舍：
    - 使用文件锁保证同一目录仅允许一个 launcher 主循环运行；
    - 第二次启动不抢占，不杀旧进程，仅走“激活已有实例”路径。
    """

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._handle: Optional[object] = None

    def acquire(self) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.lock_path, "a+b")
        try:
            self._handle.seek(0)
            self._handle.write(b"\0")
            self._handle.flush()
            self._handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._close_handle()
            return False

        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(str(os.getpid()).encode("utf-8"))
        self._handle.flush()
        return True

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self._close_handle()

    def _close_handle(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None


class ProcessManager:
    """
    进程管理器（优化版）
    V3.2.0+dev.20260209.04: 优化清理逻辑，减少退出时间
    """

    def __init__(self):
        self.backend_process: Optional[subprocess.Popen] = None
        self.frontend_process: Optional[subprocess.Popen] = None
        self._shutdown_requested = False
        self._cleanup_done = False  # 标记是否已清理过

    def _find_pid_on_port(self, port: int) -> Optional[int]:
        """
        快速查找占用端口的 PID
        使用 netstat 命令，比 psutil.net_connections() 快很多
        """
        try:
            result = subprocess.run(
                f'netstat -ano | findstr ":{port}" | findstr "LISTENING"',
                shell=True, capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split('\n'):
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            return int(parts[-1])
                        except ValueError:
                            continue
        except Exception:
            pass
        return None

    def cleanup_old_processes(self, backend_port: int, frontend_port: int):
        """
        清理残留进程（优化版）
        - 只在首次启动时执行
        - 使用快速端口查找
        - 跳过 FFmpeg 全局扫描（短期进程，通常已退出）
        """
        # 只在首次启动时清理
        if self._cleanup_done:
            return

        if os.name != 'nt':
            self._cleanup_done = True
            return

        logger.info("检查残留进程...")
        cleaned_any = False

        try:
            import psutil
            _psutil_available = True
        except ImportError:
            _psutil_available = False
            logger.warning("psutil 不可用，使用 taskkill 清理")

        for port in [backend_port, frontend_port]:
            pid = self._find_pid_on_port(port)
            if pid:
                logger.info(f"终止占用端口 {port} 的进程: PID={pid}")
                try:
                    if _psutil_available:
                        import psutil
                        proc = psutil.Process(pid)
                        proc.terminate()
                        proc.wait(timeout=2)
                    else:
                        subprocess.run(
                            ['taskkill', '/F', '/PID', str(pid)],
                            capture_output=True, timeout=2
                        )
                    cleaned_any = True
                except Exception as e:
                    logger.debug(f"清理进程 {pid} 失败: {e}")

        self._cleanup_done = True

        # 只有真正清理了才等待
        if cleaned_any:
            time.sleep(0.5)

    def start_backend(self, config: LauncherConfig) -> bool:
        """启动后端服务"""
        logger.info(f"启动后端服务 (端口 {config.backend_port})...")

        backend_dir = config.project_root / "backend"

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        env['DEV_MODE'] = 'true' if config.dev_mode else 'false'

        if config.hf_mirror:
            env['HF_ENDPOINT'] = 'https://hf-mirror.com'

        env['ANCHORFLUX_UI_MODE'] = config.ui_mode
        env['ANCHORFLUX_RUNTIME_POLICY'] = config.runtime_policy
        env['ANCHORFLUX_FLAVOR'] = config.flavor
        env['ANCHORFLUX_LITE'] = 'true' if config.flavor == 'lite' else 'false'

        # 设置 PATH（PyTorch DLL 等）
        if config.site_packages:
            torch_lib = config.site_packages / "torch" / "lib"
            path_additions = []
            if torch_lib.exists():
                path_additions.append(str(torch_lib))
            if config.tools_dir and config.tools_dir.exists():
                path_additions.append(str(config.tools_dir))
            if path_additions:
                env['PATH'] = ';'.join(path_additions) + ';' + env.get('PATH', '')

        cmd = [
            str(config.python_exec),
            '-m', 'uvicorn',
            'app.main:app',
            '--host', '0.0.0.0',
            '--port', str(config.backend_port)
        ]

        try:
            if config.dev_mode and os.name == 'nt':
                creationflags = subprocess.CREATE_NEW_CONSOLE
                self.backend_process = subprocess.Popen(
                    cmd, cwd=str(backend_dir), env=env,
                    creationflags=creationflags
                )
            else:
                self.backend_process = subprocess.Popen(
                    cmd, cwd=str(backend_dir), env=env
                )

            logger.info(f"后端已启动: PID={self.backend_process.pid}")
            logger.info(
                "后端运行模式: flavor=%s ui_mode=%s runtime_policy=%s",
                config.flavor,
                config.ui_mode,
                config.runtime_policy,
            )

            time.sleep(3)
            if self.backend_process.poll() is not None:
                logger.error(f"后端立即崩溃，退出码: {self.backend_process.returncode}")
                return False

            return True

        except Exception as e:
            logger.error(f"启动后端失败: {e}")
            return False

    def start_frontend(self, config: LauncherConfig) -> bool:
        """启动前端服务（仅开发模式）"""
        if not config.dev_mode:
            return True

        frontend_dir = config.project_root / "frontend"

        if not (frontend_dir / "node_modules").exists():
            logger.warning("node_modules 不存在，运行 npm install...")
            try:
                subprocess.run('npm install', cwd=str(frontend_dir),
                             check=True, timeout=300, shell=True)
            except Exception as e:
                logger.error(f"npm install 失败: {e}")
                return False

        logger.info(f"启动前端服务 (端口 {config.frontend_port})...")

        try:
            creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            self.frontend_process = subprocess.Popen(
                'npm run dev',
                cwd=str(frontend_dir),
                creationflags=creationflags,
                shell=True
            )

            logger.info(f"前端已启动: PID={self.frontend_process.pid}")

            time.sleep(3)
            if self.frontend_process.poll() is not None:
                logger.error(f"前端立即崩溃，退出码: {self.frontend_process.returncode}")
                return False

            return True

        except Exception as e:
            logger.error(f"启动前端失败: {e}")
            return False

    def wait_for_backend_exit(self) -> int:
        """等待后端退出"""
        if self.backend_process is None:
            return -1
        return self.backend_process.wait()

    def is_backend_running(self) -> bool:
        """检查后端是否运行中"""
        if self.backend_process is None:
            return False
        return self.backend_process.poll() is None

    def terminate_all(self):
        """
        终止所有进程（优化版）
        V3.2.0+dev.20260209.04: 并行终止，减少退出时间
        """
        self._shutdown_requested = True

        # 收集需要终止的进程
        processes_to_terminate = []

        if self.frontend_process and self.frontend_process.poll() is None:
            processes_to_terminate.append(("前端", self.frontend_process))

        if self.backend_process and self.backend_process.poll() is None:
            processes_to_terminate.append(("后端", self.backend_process))

        if not processes_to_terminate:
            return  # 没有需要终止的进程

        # 并行发送 terminate 信号
        for name, proc in processes_to_terminate:
            logger.info(f"终止{name}进程...")
            proc.terminate()

        # 等待所有进程退出（最多 3 秒）
        start = time.time()
        while time.time() - start < 3:
            all_exited = all(
                proc.poll() is not None
                for _, proc in processes_to_terminate
            )
            if all_exited:
                break
            time.sleep(0.1)

        # 强制终止仍在运行的进程
        for name, proc in processes_to_terminate:
            if proc.poll() is None:
                logger.warning(f"{name}进程未响应，强制终止")
                proc.kill()


class Launcher:
    """主启动器"""

    def __init__(self):
        self.project_root = get_project_root()
        self.config: Optional[LauncherConfig] = None
        self.uv_manager: Optional[UvManager] = None
        self.updater: Optional[SelfUpdater] = None
        self.process_manager: Optional[ProcessManager] = None
        self.instance_guard: Optional[SingleInstanceGuard] = None
        self._exit_code_override: Optional[int] = None
        self._running = True

    def _signal_handler(self, signum, frame):
        """信号处理"""
        logger.info(f"收到信号 {signum}，正在关闭...")
        self._running = False
        if self.process_manager:
            self.process_manager.terminate_all()

    def _activate_running_instance(
        self,
        *,
        ui_mode: str,
        dev_mode: bool,
        backend_port: int,
        frontend_port: int,
        shell_path: Optional[Path],
    ) -> None:
        """激活已运行实例的界面，不重启后端。"""
        backend_url = f"http://127.0.0.1:{backend_port}"
        browser_url = (
            f"http://127.0.0.1:{frontend_port}"
            if dev_mode
            else backend_url
        )
        normalized_ui_mode = normalize_ui_mode(ui_mode)

        logger.info("检测到已运行实例，尝试激活现有窗口...")
        if normalized_ui_mode == "none":
            logger.info("当前模式为 none，不拉起 UI。")
            return

        if normalized_ui_mode == "electron":
            wait_backend_ready(backend_url, timeout_sec=8)
            target_shell_path = resolve_shell_path(self.project_root, shell_path)
            if launch_electron(target_shell_path):
                return
            logger.warning("激活 Electron 失败，回退浏览器")
            open_browser_fallback(browser_url)
            return

        open_browser_fallback(browser_url)

    def initialize(self) -> bool:
        """初始化启动器"""
        # 检测开发模式
        dev_mode = detect_dev_mode(self.project_root)

        # 配置日志级别
        if dev_mode:
            logging.getLogger().setLevel(logging.DEBUG)

        logger.info("=" * 50)
        logger.info(f"{APP_NAME} Launcher v{VERSION}")
        logger.info(f"模式: {'开发' if dev_mode else '生产'}")
        logger.info("=" * 50)

        # 清理旧版本启动器（如果从自更新重启）
        if "--cleanup-old" in sys.argv:
            self.updater = SelfUpdater(self.project_root)
            self.updater.cleanup_old_launcher()

        # 加载环境配置
        env_config = load_env_config(self.project_root)
        raw_ui_mode = os.environ.get(
            "ANCHORFLUX_UI_MODE",
            env_config.get("ANCHORFLUX_UI_MODE", DEFAULT_UI_MODE),
        )
        raw_runtime_policy = os.environ.get(
            "ANCHORFLUX_RUNTIME_POLICY",
            env_config.get("ANCHORFLUX_RUNTIME_POLICY", DEFAULT_RUNTIME_POLICY),
        )
        raw_flavor = os.environ.get(
            "ANCHORFLUX_FLAVOR",
            env_config.get("ANCHORFLUX_FLAVOR", ""),
        )
        if not str(raw_flavor).strip():
            raw_lite = os.environ.get(
                "ANCHORFLUX_LITE",
                env_config.get("ANCHORFLUX_LITE", ""),
            )
            if str(raw_lite).strip().lower() in ("true", "1", "yes"):
                raw_flavor = "lite"
            else:
                raw_flavor = DEFAULT_FLAVOR
        raw_shell_path = os.environ.get(
            "ANCHORFLUX_SHELL_PATH",
            env_config.get("ANCHORFLUX_SHELL_PATH", ""),
        ).strip()
        shell_path: Optional[Path] = None
        if raw_shell_path:
            shell_path = Path(raw_shell_path)
            if not shell_path.is_absolute():
                shell_path = (self.project_root / shell_path).resolve()

        # 单实例守卫：第二次启动仅激活已有实例，不抢占也不重启。
        self.instance_guard = SingleInstanceGuard(self.project_root / "data" / "launcher.lock")
        if not self.instance_guard.acquire():
            self._activate_running_instance(
                ui_mode=raw_ui_mode,
                dev_mode=dev_mode,
                backend_port=DEFAULT_BACKEND_PORT,
                frontend_port=DEFAULT_FRONTEND_PORT,
                shell_path=shell_path,
            )
            self._exit_code_override = 0
            return False

        # 初始化 uv 管理器
        self.uv_manager = UvManager(self.project_root, dev_mode)

        if not self.uv_manager.is_available():
            logger.error("uv 未安装或不可用！请安装 uv: https://docs.astral.sh/uv/")
            return False

        logger.info(f"uv 路径: {self.uv_manager.uv_exec}")

        # 检查依赖状态
        sync_result = self.uv_manager.check_sync_status()

        if sync_result.needs_sync:
            logger.info("依赖需要同步...")

            # 尝试使用 GUI
            from .ui import is_gui_available, run_dependency_sync_gui

            if is_gui_available():
                # 使用 GUI 显示进度
                def sync_func(progress_callback):
                    return self.uv_manager.sync_dependencies(progress_callback)

                run_dependency_sync_gui(sync_func)
            else:
                # 命令行模式
                result = self.uv_manager.sync_dependencies(
                    lambda msg, prog: logger.info(f"[{int(prog*100):3d}%] {msg}")
                )
                if not result.success:
                    logger.error(f"依赖同步失败: {result.message}")
                    return False
        else:
            logger.info("依赖已同步 (快速启动模式)")

        # 获取 Python 路径
        python_exec = self.uv_manager.get_python_path()
        site_packages = self.uv_manager.get_site_packages()

        if not python_exec:
            logger.error("未找到 Python 环境！")
            return False

        logger.info(f"Python: {python_exec}")

        # 修复 PyTorch DLL（如果需要）
        if site_packages:
            fix_pytorch_dll(site_packages)

        # ONNX Runtime GPU 自动修正：
        # 在 uv sync 后（或快速启动）做 provider 探针，必要时重装 onnxruntime-gpu，
        # 规避 Windows 下 onnxruntime/onnxruntime-gpu 覆盖顺序导致的 CUDA Provider 丢失。
        ort_fix_result = self.uv_manager.ensure_onnxruntime_gpu_runtime(
            python_exec=python_exec,
            site_packages=site_packages,
        )
        if ort_fix_result.success:
            logger.info(ort_fix_result.message)
        else:
            # 不阻断启动：允许 CPU 回退，同时给出明确告警供排查。
            logger.warning("ONNX Runtime GPU 自动修正失败，将继续启动（可回退 CPU）: %s", ort_fix_result.message)

        # 创建配置
        self.config = LauncherConfig(
            project_root=self.project_root,
            dev_mode=dev_mode,
            python_exec=python_exec,
            python_mode="venv",
            site_packages=site_packages,
            tools_dir=self.project_root / "tools",
            hf_mirror=env_config.get('USE_HF_MIRROR', 'true').lower() == 'true',
            log_level='DEBUG' if dev_mode else 'INFO',
            ui_mode=normalize_ui_mode(raw_ui_mode),
            runtime_policy=normalize_runtime_policy(raw_runtime_policy),
            flavor=normalize_flavor(raw_flavor),
            shell_path=shell_path,
        )
        logger.info(
            "启动器配置: flavor=%s ui_mode=%s runtime_policy=%s shell_path=%s",
            self.config.flavor,
            self.config.ui_mode,
            self.config.runtime_policy,
            self.config.shell_path or "<default>",
        )

        # 初始化管理器
        self.process_manager = ProcessManager()
        self.updater = SelfUpdater(self.project_root)

        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        return True

    def _launch_ui(self) -> None:
        """
        根据 ui_mode 触发 UI 启动。

        设计取舍：
        - `browser` 模式沿用后端自带的自动打开逻辑，Launcher 不重复打开。
        - `electron` 模式由 Launcher 负责等待后端就绪并拉起 Shell。
        """
        if self.config is None:
            return

        backend_url = f"http://127.0.0.1:{self.config.backend_port}"
        browser_url = (
            f"http://127.0.0.1:{self.config.frontend_port}"
            if self.config.dev_mode
            else backend_url
        )

        if self.config.ui_mode == "none":
            logger.info("UI 模式为 none，跳过 UI 启动。")
            return

        if self.config.ui_mode == "browser":
            logger.info("UI 模式为 browser，由后端负责自动打开浏览器。")
            return

        if self.config.ui_mode != "electron":
            logger.warning("未知 UI 模式，回退为 browser: %s", self.config.ui_mode)
            return

        logger.info("UI 模式为 electron，等待后端就绪后启动 Shell...")
        if not wait_backend_ready(backend_url, timeout_sec=60):
            logger.warning("后端就绪探针超时，回退浏览器。")
            open_browser_fallback(browser_url)
            return

        shell_path = resolve_shell_path(self.project_root, self.config.shell_path)
        if not launch_electron(shell_path):
            logger.warning("Electron Shell 启动失败，回退浏览器。")
            open_browser_fallback(browser_url)

    def run(self) -> int:
        """运行主循环"""
        try:
            if not self.initialize():
                if self._exit_code_override is not None:
                    return self._exit_code_override
                input("按 Enter 键退出...")
                return 1

            while self._running:
                # 清理旧进程
                self.process_manager.cleanup_old_processes(
                    self.config.backend_port,
                    self.config.frontend_port
                )

                # 启动后端
                if not self.process_manager.start_backend(self.config):
                    logger.error("后端启动失败")
                    input("按 Enter 键退出...")
                    return 1

                # 启动前端（开发模式）
                if self.config.dev_mode:
                    if not self.process_manager.start_frontend(self.config):
                        logger.warning("前端启动失败，继续运行...")

                self._launch_ui()

                # 显示启动信息
                logger.info("")
                logger.info("=" * 50)
                logger.info("服务启动成功!")
                logger.info("=" * 50)

                if self.config.dev_mode:
                    logger.info(f"前端: http://localhost:{self.config.frontend_port}")
                logger.info(f"应用: http://localhost:{self.config.backend_port}")
                logger.info(f"API 文档: http://localhost:{self.config.backend_port}/docs")
                logger.info("")
                logger.info("按 Ctrl+C 停止，或使用应用内的「退出系统」按钮")
                logger.info("=" * 50)

                # 等待后端退出
                exit_code = self.process_manager.wait_for_backend_exit()
                logger.info(f"后端退出，代码: {exit_code}")

                # 检查更新信号
                update_info = self.updater.check_signal()

                if update_info:
                    logger.info(f"检测到更新: {update_info.version}")
                    self.process_manager.terminate_all()

                    # 使用 GUI 执行更新
                    from .ui import is_gui_available, run_update_gui

                    if is_gui_available():
                        def update_func(progress_callback):
                            return self.updater.execute_update(update_info, progress_callback)

                        run_update_gui(update_func)
                    else:
                        result = self.updater.execute_update(
                            update_info,
                            lambda msg, prog: logger.info(f"[{int(prog*100):3d}%] {msg}")
                        )
                        if not result.success:
                            logger.error(f"更新失败: {result.message}")

                    time.sleep(2)
                    continue
                else:
                    logger.info("正常关闭")
                    self._running = False

            # 清理
            if self.process_manager:
                self.process_manager.terminate_all()

            logger.info("启动器关闭完成")
            return 0
        finally:
            if self.instance_guard is not None:
                self.instance_guard.release()


def main():
    """主入口"""
    if os.name == 'nt':
        os.system('chcp 65001 >nul 2>&1')

    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'

    # 抑制 libpng 的 iCCP 警告
    os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'
    os.environ['OPENCV_IO_ENABLE_JASPER'] = '0'

    launcher = Launcher()
    sys.exit(launcher.run())


if __name__ == "__main__":
    main()
