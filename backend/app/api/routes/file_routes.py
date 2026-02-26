"""
文件管理相关API路由
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List

from app.services.file_service import FileManagementService

router = APIRouter(prefix="/api", tags=["files"])


def create_file_router(file_service: FileManagementService):
    """创建文件管理路由"""
    
    @router.get("/files")
    async def list_files():
        """获取输入目录中的所有媒体文件"""
        try:
            files = file_service.list_input_files()
            if len(files) == 0:
                return {"files": [], "input_dir": file_service.input_dir, "message": "input 目录中没有找到支持的媒体文件"}
            return {"files": files, "input_dir": file_service.input_dir}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")

    @router.get("/files/subtitles")
    async def list_subtitle_files():
        """获取输入目录中的所有字幕文件"""
        try:
            files = file_service.list_input_subtitle_files()
            return {"files": files, "input_dir": file_service.input_dir}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取字幕文件列表失败: {str(e)}")

    @router.delete("/files/{filename}")
    async def delete_file(filename: str):
        """删除input目录中的文件"""
        try:
            if not file_service.delete_input_file(filename):
                raise HTTPException(status_code=404, detail="文件不存在")
            return {"success": True, "message": f"文件 {filename} 已删除"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}")

    @router.post("/files/open-input-folder")
    async def open_input_folder():
        """使用系统文件管理器打开input目录"""
        try:
            success = file_service.open_input_folder()
            if success:
                return {"success": True, "message": "已打开input目录"}
            else:
                raise HTTPException(status_code=500, detail="打开文件夹失败")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"打开文件夹失败: {str(e)}")

    return router