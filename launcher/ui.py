# -*- coding: utf-8 -*-
"""
CustomTkinter 更新界面
V3.2.0+dev.20260209.04

使用 CustomTkinter 实现现代化更新进度界面
优化视觉效果，更接近 Fluent Design 风格
"""

import sys
import os
import logging
import threading
import warnings

# V3.2.0+dev.20260209.04: 抑制 libpng iCCP 警告
# 该警告由 PNG 图片中不正确的 sRGB ICC 配置文件引起，不影响功能
# libpng 警告是 C 层面输出，需要重定向 stderr 来抑制
os.environ['TK_SILENCE_DEPRECATION'] = '1'
warnings.filterwarnings("ignore", message=".*iCCP.*")

def _suppress_libpng_warnings():
    """抑制 libpng 的 C 层面警告输出"""
    if sys.platform == 'win32':
        import ctypes
        # 临时重定向 stderr 来抑制 libpng 警告
        # 仅在 Tkinter 初始化时生效
        try:
            ctypes.windll.kernel32.SetStdHandle(-12, 0)  # STD_ERROR_HANDLE
        except Exception:
            pass

_suppress_libpng_warnings()
from typing import Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger("launcher.ui")

# 尝试导入 CustomTkinter
_CTK_AVAILABLE = False
try:
    import customtkinter as ctk
    _CTK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CustomTkinter 不可用: {e}")
    _CTK_AVAILABLE = False


def is_gui_available() -> bool:
    """检查 GUI 是否可用"""
    return _CTK_AVAILABLE


@dataclass
class UpdateResult:
    """更新结果（用于类型提示）"""
    success: bool
    message: str


# 颜色主题
class Colors:
    """颜色定义"""
    # 主色调
    PRIMARY = "#0078D4"  # Windows 蓝
    PRIMARY_HOVER = "#106EBE"
    PRIMARY_DARK = "#005A9E"

    # 状态色
    SUCCESS = "#107C10"  # 绿色
    SUCCESS_LIGHT = "#13A10E"
    ERROR = "#D13438"  # 红色
    WARNING = "#CA5010"  # 橙色

    # 中性色
    TEXT_PRIMARY = "#1A1A1A"
    TEXT_SECONDARY = "#666666"
    TEXT_DISABLED = "#A0A0A0"

    # 背景色（浅色模式）
    BG_PRIMARY = "#FFFFFF"
    BG_SECONDARY = "#F5F5F5"
    BG_TERTIARY = "#E8E8E8"

    # 背景色（深色模式）
    BG_DARK_PRIMARY = "#1F1F1F"
    BG_DARK_SECONDARY = "#2D2D2D"
    BG_DARK_TERTIARY = "#3D3D3D"


class UpdateWindow(ctk.CTk if _CTK_AVAILABLE else object):
    """
    更新进度窗口

    特性：
    - 现代化 Fluent Design 风格
    - 自适应深色/浅色模式
    - 动画效果
    - 精细的视觉层次
    """

    def __init__(self, title: str = "AnchorFlux 更新"):
        if not _CTK_AVAILABLE:
            raise RuntimeError("CustomTkinter 不可用")

        super().__init__()

        # 窗口配置
        self.title(title)
        self.geometry("520x340")
        self.minsize(480, 320)
        self.resizable(False, False)

        # 设置外观模式
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        # 设置窗口背景色
        self.configure(fg_color=(Colors.BG_PRIMARY, Colors.BG_DARK_PRIMARY))

        # 居中显示
        self.update_idletasks()
        self._center_window()

        # 状态
        self._worker_thread: Optional[threading.Thread] = None
        self._update_func: Optional[Callable] = None

        self._setup_ui()

    def _center_window(self):
        """窗口居中"""
        width = 520
        height = 340
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _setup_ui(self):
        """设置界面"""
        # 主容器
        self.main_frame = ctk.CTkFrame(
            self,
            corner_radius=0,
            fg_color="transparent"
        )
        self.main_frame.pack(fill="both", expand=True, padx=0, pady=0)

        # 内容区域
        self.content_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="transparent"
        )
        self.content_frame.pack(fill="both", expand=True, padx=50, pady=50)

        # 标题
        self.title_label = ctk.CTkLabel(
            self.content_frame,
            text="正在检查更新...",
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=22, weight="bold"),
            text_color=(Colors.TEXT_PRIMARY, "#FFFFFF")
        )
        self.title_label.pack(pady=(0, 8))

        # 状态信息
        self.status_label = ctk.CTkLabel(
            self.content_frame,
            text="请稍候...",
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=13),
            text_color=(Colors.TEXT_SECONDARY, "#AAAAAA")
        )
        self.status_label.pack(pady=(0, 25))

        # 进度条容器
        self.progress_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color="transparent",
            height=6
        )
        self.progress_frame.pack(fill="x", pady=(0, 8))

        # 进度条
        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame,
            height=6,
            corner_radius=3,
            progress_color=Colors.PRIMARY,
            fg_color=(Colors.BG_TERTIARY, Colors.BG_DARK_TERTIARY)
        )
        self.progress_bar.pack(fill="x")
        self.progress_bar.set(0)

        # 进度信息行
        self.progress_info_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color="transparent"
        )
        self.progress_info_frame.pack(fill="x", pady=(0, 15))

        # 进度百分比（左侧）
        self.percent_label = ctk.CTkLabel(
            self.progress_info_frame,
            text="0%",
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=12, weight="bold"),
            text_color=Colors.PRIMARY
        )
        self.percent_label.pack(side="left")

        # 详细信息（右侧）
        self.detail_label = ctk.CTkLabel(
            self.progress_info_frame,
            text="",
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=11),
            text_color=(Colors.TEXT_DISABLED, "#888888")
        )
        self.detail_label.pack(side="right")

        # 弹性空间
        spacer = ctk.CTkFrame(self.content_frame, fg_color="transparent", height=10)
        spacer.pack(fill="x", expand=True)

        # 按钮区域
        self.button_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color="transparent"
        )
        self.button_frame.pack(fill="x", pady=(15, 0))

        # 取消按钮（次要按钮样式）
        self.cancel_button = ctk.CTkButton(
            self.button_frame,
            text="取消",
            width=80,
            height=40,
            corner_radius=4,
            fg_color="transparent",
            border_width=1,
            border_color=(Colors.BG_TERTIARY, Colors.BG_DARK_TERTIARY),
            text_color=(Colors.TEXT_PRIMARY, "#FFFFFF"),
            hover_color=(Colors.BG_SECONDARY, Colors.BG_DARK_SECONDARY),
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=13),
            command=self._on_cancel
        )
        self.cancel_button.pack(side="right", padx=(10, 0))

        # 确定按钮（主要按钮样式）
        self.confirm_button = ctk.CTkButton(
            self.button_frame,
            text="确定",
            width=80,
            height=40,
            corner_radius=4,
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=13),
            command=self._on_confirm
        )
        # 初始隐藏

    def set_title(self, title: str):
        """设置标题"""
        self.title_label.configure(text=title)

    def set_status(self, status: str):
        """设置状态"""
        self.status_label.configure(text=status)

    def set_progress(self, value: float):
        """设置进度 (0.0-1.0)"""
        self.progress_bar.set(value)
        self.percent_label.configure(text=f"{int(value * 100)}%")

    def set_detail(self, detail: str):
        """设置详细信息"""
        self.detail_label.configure(text=detail)

    def start_update(self, update_func: Callable):
        """开始更新"""
        self.set_title("正在更新...")
        self.icon_label.configure(text="⟳")
        self.cancel_button.configure(state="disabled")
        self._update_func = update_func

        self._worker_thread = threading.Thread(target=self._run_update, daemon=True)
        self._worker_thread.start()

    def _run_update(self):
        """在后台线程运行更新"""
        try:
            def progress_callback(msg: str, prog: float):
                self.after(0, lambda: self._on_progress(msg, prog))

            result = self._update_func(progress_callback)
            self.after(0, lambda: self._on_finished(result.success, result.message))
        except Exception as e:
            self.after(0, lambda: self._on_finished(False, str(e)))

    def _on_progress(self, message: str, progress: float):
        """进度更新"""
        self.set_status(message)
        self.set_progress(progress)

    def _on_finished(self, success: bool, message: str):
        """更新完成"""
        self.cancel_button.pack_forget()
        self.confirm_button.pack(side="right")

        if success:
            self.set_title("更新完成")
            self.set_status("更新已成功安装，请重启应用")
            self.set_progress(1.0)
            self.set_detail("")
            self.progress_bar.configure(progress_color=Colors.SUCCESS)
            self.percent_label.configure(text_color=Colors.SUCCESS)
        else:
            self.set_title("更新失败")
            self.set_status("更新过程中出现错误")
            self.set_detail(message[:50] + "..." if len(message) > 50 else message)
            self.progress_bar.configure(progress_color=Colors.ERROR)
            self.percent_label.configure(text_color=Colors.ERROR)

    def _on_cancel(self):
        """取消"""
        self.destroy()

    def _on_confirm(self):
        """确定"""
        self.destroy()


class DependencyWindow(ctk.CTk if _CTK_AVAILABLE else object):
    """依赖安装进度窗口"""

    def __init__(self, title: str = "AnchorFlux 初始化"):
        if not _CTK_AVAILABLE:
            raise RuntimeError("CustomTkinter 不可用")

        super().__init__()

        self.title(title)
        self.geometry("580x380")
        self.minsize(520, 350)
        self.resizable(False, False)

        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self.configure(fg_color=(Colors.BG_PRIMARY, Colors.BG_DARK_PRIMARY))

        self.update_idletasks()
        self._center_window()

        self._worker_thread: Optional[threading.Thread] = None
        self._on_complete_callback: Optional[Callable] = None

        self._setup_ui()

    def _center_window(self):
        """窗口居中"""
        width = 580
        height = 380
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _setup_ui(self):
        """设置界面"""
        # 主容器
        self.main_frame = ctk.CTkFrame(
            self,
            corner_radius=0,
            fg_color="transparent"
        )
        self.main_frame.pack(fill="both", expand=True)

        # 顶部装饰条
        self.accent_bar = ctk.CTkFrame(
            self.main_frame,
            height=4,
            corner_radius=0,
            fg_color=Colors.PRIMARY
        )
        self.accent_bar.pack(fill="x", side="top")

        # 内容区域
        self.content_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="transparent"
        )
        self.content_frame.pack(fill="both", expand=True, padx=50, pady=40)

        # 图标
        self.icon_label = ctk.CTkLabel(
            self.content_frame,
            text="⚙",
            font=ctk.CTkFont(size=52),
            text_color=Colors.PRIMARY
        )
        self.icon_label.pack(pady=(0, 20))

        # 标题
        self.title_label = ctk.CTkLabel(
            self.content_frame,
            text="正在初始化环境...",
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=22, weight="bold"),
            text_color=(Colors.TEXT_PRIMARY, "#FFFFFF")
        )
        self.title_label.pack(pady=(0, 8))

        # 状态信息
        self.status_label = ctk.CTkLabel(
            self.content_frame,
            text="首次运行需要下载依赖，请耐心等待...",
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=13),
            text_color=(Colors.TEXT_SECONDARY, "#AAAAAA")
        )
        self.status_label.pack(pady=(0, 30))

        # 进度条容器
        self.progress_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color="transparent",
            height=8
        )
        self.progress_frame.pack(fill="x", pady=(0, 10))

        # 进度条
        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame,
            height=8,
            corner_radius=4,
            progress_color=Colors.PRIMARY,
            fg_color=(Colors.BG_TERTIARY, Colors.BG_DARK_TERTIARY)
        )
        self.progress_bar.pack(fill="x")
        self.progress_bar.set(0)

        # 进度信息行
        self.progress_info_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color="transparent"
        )
        self.progress_info_frame.pack(fill="x", pady=(0, 20))

        # 进度百分比
        self.percent_label = ctk.CTkLabel(
            self.progress_info_frame,
            text="0%",
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=13, weight="bold"),
            text_color=Colors.PRIMARY
        )
        self.percent_label.pack(side="left")

        # 详细信息
        self.detail_label = ctk.CTkLabel(
            self.content_frame,
            text="这可能需要 5-30 分钟，取决于网络速度...",
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=11),
            text_color=(Colors.TEXT_DISABLED, "#888888"),
            wraplength=480
        )
        self.detail_label.pack(pady=(0, 10))

    def set_title(self, title: str):
        self.title_label.configure(text=title)

    def set_status(self, status: str):
        self.status_label.configure(text=status)

    def set_progress(self, value: float):
        self.progress_bar.set(value)
        self.percent_label.configure(text=f"{int(value * 100)}%")

    def set_detail(self, detail: str):
        self.detail_label.configure(text=detail)

    def start_sync(self, sync_func: Callable, on_complete: Callable = None):
        """开始同步依赖"""
        self._on_complete_callback = on_complete

        self._worker_thread = threading.Thread(target=self._run_sync, args=(sync_func,), daemon=True)
        self._worker_thread.start()

    def _run_sync(self, sync_func: Callable):
        """在后台线程运行同步"""
        try:
            def progress_callback(msg: str, prog: float):
                self.after(0, lambda: self._on_progress(msg, prog))

            result = sync_func(progress_callback)
            self.after(0, lambda: self._on_finished(result.success, result.message))
        except Exception as e:
            self.after(0, lambda: self._on_finished(False, str(e)))

    def _on_progress(self, message: str, progress: float):
        self.set_status(message)
        self.set_progress(progress)

    def _on_finished(self, success: bool, message: str):
        if success:
            self.set_title("初始化完成")
            self.set_status("依赖安装成功")
            self.set_progress(1.0)
            self.icon_label.configure(text="✓", text_color=Colors.SUCCESS)
            self.progress_bar.configure(progress_color=Colors.SUCCESS)
            self.accent_bar.configure(fg_color=Colors.SUCCESS)
            self.percent_label.configure(text_color=Colors.SUCCESS)

            self.after(1500, self._complete)
        else:
            self.set_title("初始化失败")
            self.set_status("依赖安装失败")
            self.set_detail(message)
            self.icon_label.configure(text="✕", text_color=Colors.ERROR)
            self.progress_bar.configure(progress_color=Colors.ERROR)
            self.accent_bar.configure(fg_color=Colors.ERROR)
            self.percent_label.configure(text_color=Colors.ERROR)

    def _complete(self):
        self.destroy()
        if self._on_complete_callback:
            self._on_complete_callback()


def run_dependency_sync_gui(sync_func: Callable, on_complete: Callable = None):
    """运行依赖同步 GUI"""
    if not _CTK_AVAILABLE:
        logger.warning("GUI 不可用，使用命令行模式")

        def cli_callback(msg, prog):
            print(f"[{int(prog*100):3d}%] {msg}")

        result = sync_func(cli_callback)
        if on_complete and result.success:
            on_complete()
        return result

    window = DependencyWindow()
    window.after(100, lambda: window.start_sync(sync_func, on_complete))
    window.mainloop()


def run_update_gui(update_func: Callable) -> bool:
    """运行更新 GUI"""
    if not _CTK_AVAILABLE:
        logger.warning("GUI 不可用，使用命令行模式")

        def cli_callback(msg, prog):
            print(f"[{int(prog*100):3d}%] {msg}")

        result = update_func(cli_callback)
        return result.success

    window = UpdateWindow()
    window.after(100, lambda: window.start_update(update_func))
    window.mainloop()
    return True
