"""
文件管理服务
"""
import os
import sys
import subprocess
from typing import List, Dict
from datetime import datetime


class FileManagementService:
    """文件管理服务"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        # 确保目录存在
        for dir_path in [self.input_dir, self.output_dir]:
            os.makedirs(dir_path, exist_ok=True)

    # 支持的文件扩展名集合
    SUBTITLE_EXTENSIONS = {'.srt', '.ass', '.vtt'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}

    def is_supported_file(self, filename: str) -> bool:
        """检查是否为支持的视频或音频文件"""
        ext = os.path.splitext(filename.lower())[1]
        return ext in self.VIDEO_EXTENSIONS or ext in self.AUDIO_EXTENSIONS

    def is_supported_subtitle(self, filename: str) -> bool:
        """检查是否为支持的字幕文件"""
        ext = os.path.splitext(filename.lower())[1]
        return ext in self.SUBTITLE_EXTENSIONS

    def list_input_files(self) -> List[Dict]:
        """获取输入目录中的所有媒体文件"""
        files = []
        if os.path.exists(self.input_dir):
            for filename in os.listdir(self.input_dir):
                file_path = os.path.join(self.input_dir, filename)
                if os.path.isfile(file_path) and self.is_supported_file(filename):
                    stat = os.stat(file_path)
                    files.append({
                        'name': filename,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'modified_timestamp': int(stat.st_mtime * 1000),  # 添加时间戳（毫秒）
                        'created_timestamp': int(stat.st_ctime * 1000),  # 添加创建时间戳（毫秒）
                        'path': file_path
                    })

        # 按修改时间倒序排列
        files.sort(key=lambda x: x['modified_timestamp'], reverse=True)
        return files

    def list_input_subtitle_files(self) -> List[Dict]:
        """获取输入目录中的所有字幕文件"""
        files = []
        if os.path.exists(self.input_dir):
            for filename in os.listdir(self.input_dir):
                file_path = os.path.join(self.input_dir, filename)
                if os.path.isfile(file_path) and self.is_supported_subtitle(filename):
                    stat = os.stat(file_path)
                    files.append({
                        'name': filename,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'modified_timestamp': int(stat.st_mtime * 1000),
                        'path': file_path,
                    })

        # 按修改时间倒序排列
        files.sort(key=lambda x: x['modified_timestamp'], reverse=True)
        return files

    def delete_input_file(self, filename: str) -> bool:
        """删除输入目录中的文件"""
        file_path = os.path.join(self.input_dir, filename)
        if not os.path.exists(file_path):
            return False
        try:
            os.remove(file_path)
            return True
        except Exception:
            return False

    def get_input_file_path(self, filename: str) -> str:
        """获取输入文件的完整路径"""
        return os.path.join(self.input_dir, filename)

    def get_output_file_path(self, filename: str) -> str:
        """获取输出文件的完整路径"""
        return os.path.join(self.output_dir, filename)

    def open_input_folder(self) -> bool:
        """使用系统文件管理器打开input目录"""
        try:
            # 确保目录存在
            if not os.path.exists(self.input_dir):
                os.makedirs(self.input_dir, exist_ok=True)

            # 根据操作系统选择打开方式
            if sys.platform == 'win32':
                # Windows: 使用 explorer
                os.startfile(self.input_dir)
            elif sys.platform == 'darwin':
                # macOS: 使用 open
                subprocess.run(['open', self.input_dir], check=True)
            else:
                # Linux: 使用 xdg-open
                subprocess.run(['xdg-open', self.input_dir], check=True)

            return True
        except Exception as e:
            print(f"打开文件夹失败: {str(e)}")
            return False