@echo off
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

title AnchorFlux v3.1.0 - One-Click Launcher

echo.
echo ========================================
echo   AnchorFlux v3.1.0 - One-Click Start
echo ========================================
echo.

REM ========================================
REM  路径配置
REM ========================================
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
set "TOOLS_DIR=%PROJECT_ROOT%\tools"
set "BACKEND_DIR=%PROJECT_ROOT%\backend"
set "FRONTEND_DIR=%PROJECT_ROOT%\frontend"
set "REQ_FILE=%PROJECT_ROOT%\requirements.txt"
set "MARKER_FILE=%PROJECT_ROOT%\.env_installed"
set "ENV_FILE=%PROJECT_ROOT%\.env"
set "KMP_DUPLICATE_LIB_OK=TRUE"

REM 从 .env 文件读取 PyPI 镜像源配置
set "PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"
set "HF_ENDPOINT=https://hf-mirror.com"

if exist "%ENV_FILE%" (
    for /f "usebackq tokens=1,* delims==" %%a in ("%ENV_FILE%") do (
        set "LINE=%%a"
        setlocal enabledelayedexpansion
        REM 去除前导空格
        for /f "tokens=* delims= " %%i in ("!LINE!") do set "LINE=%%i"
        REM 跳过注释行和空行
        if not "!LINE:~0,1!"=="#" if not "!LINE!"=="" (
            endlocal
            if "%%a"=="PYPI_MIRROR" set "PYPI_MIRROR=%%b"
            if "%%a"=="USE_HF_MIRROR" (
                if "%%b"=="true" (
                    set "HF_ENDPOINT=https://hf-mirror.com"
                ) else (
                    set "HF_ENDPOINT="
                )
            )
        ) else (
            endlocal
        )
    )
    echo [Config] Loaded configuration from .env
) else (
    echo [Config] .env file not found, using default settings
)

REM 如果 PYPI_MIRROR 为空,则使用官方源
if not defined PYPI_MIRROR (
    echo [Config] PyPI Mirror: Official PyPI
) else (
    echo [Config] PyPI Mirror: %PYPI_MIRROR%
)

REM ========================================
REM  Python 路径配置 (双模式: 嵌入式优先, 回退venv)
REM ========================================
set "EMBED_PYTHON=%TOOLS_DIR%\python\python.exe"
set "EMBED_SITE_PACKAGES=%TOOLS_DIR%\python\Lib\site-packages"
set "VENV_ROOT=%PROJECT_ROOT%\.venv"
set "VENV_PYTHON=%VENV_ROOT%\Scripts\python.exe"
set "VENV_SITE_PACKAGES=%VENV_ROOT%\Lib\site-packages"

REM 检测Python模式: 优先嵌入式, 回退venv
if exist "%EMBED_PYTHON%" (
    set "PYTHON_EXEC=%EMBED_PYTHON%"
    set "SITE_PACKAGES=%EMBED_SITE_PACKAGES%"
    set "PYTHON_MODE=embedded"
    goto python_found
)

if exist "%VENV_PYTHON%" (
    set "PYTHON_EXEC=%VENV_PYTHON%"
    set "SITE_PACKAGES=%VENV_SITE_PACKAGES%"
    set "PYTHON_MODE=venv"
    goto python_found
)

REM 两者都不存在, 尝试用系统Python创建venv
echo [INFO] No Python environment found, trying to create venv...
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo [INFO] Please install Python 3.10+ or place embedded Python in tools\python\
    pause
    exit /b 1
)

echo [INFO] Creating virtual environment...
python -m venv "%VENV_ROOT%"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
set "PYTHON_EXEC=%VENV_PYTHON%"
set "SITE_PACKAGES=%VENV_SITE_PACKAGES%"
set "PYTHON_MODE=venv"
echo [OK] Virtual environment created

:python_found
echo.
echo [Config] Project: %PROJECT_ROOT%
echo [Config] Python Mode: %PYTHON_MODE%
echo [Config] Python: %PYTHON_EXEC%
echo.

REM ========================================
REM  Step 1: 检查并安装 Python 依赖
REM ========================================
echo [Step 1/4] Checking Python dependencies...
if exist "%MARKER_FILE%" (
    echo [OK] Dependencies already installed, skipping...
    goto deps_done
)

echo [INFO] First time setup - Installing dependencies...
echo [INFO] This may take 10-30 minutes, please wait...
echo.

REM 升级 pip
echo [1.1] Upgrading pip...
"%PYTHON_EXEC%" -m pip install --upgrade pip -i %PYPI_MIRROR% -q
if errorlevel 1 (
    echo [WARNING] Mirror source failed, trying official PyPI...
    "%PYTHON_EXEC%" -m pip install --upgrade pip -q
    if errorlevel 1 (
        echo [ERROR] Failed to upgrade pip
        pause
        exit /b 1
    )
    set "PYPI_MIRROR="
    echo [OK] pip upgraded using official source
) else (
    echo [OK] pip upgraded
)
echo.

REM 分步安装: 先安装 PyTorch (官方源, 增加超时和重试)
echo [1.2] Installing PyTorch (CUDA 11.8)...
echo [INFO] Downloading from PyTorch official source (may take a while)...
echo [INFO] torch and torchaudio are large packages (~2.5GB), please be patient...
set "PYTORCH_INDEX=https://download.pytorch.org/whl/cu118"
"%PYTHON_EXEC%" -m pip install torch==2.4.0+cu118 torchaudio==2.4.0+cu118 --index-url %PYTORCH_INDEX% --timeout 600 --retries 5
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch
    echo [INFO] Please check your network connection and try again
    echo [INFO] You can also manually run: pip install torch==2.4.0+cu118 torchaudio==2.4.0+cu118 --index-url %PYTORCH_INDEX%
    pause
    exit /b 1
)
echo [OK] PyTorch installed
echo.

REM 安装其他依赖 (不含 PyTorch)
echo [1.3] Installing other dependencies...
if defined PYPI_MIRROR (
    echo [INFO] Using mirror: %PYPI_MIRROR%
    echo.
    "%PYTHON_EXEC%" -m pip install -r "%REQ_FILE%" -i %PYPI_MIRROR% --timeout 120
    if errorlevel 1 (
        echo.
        echo [WARNING] Mirror source failed, trying official PyPI...
        "%PYTHON_EXEC%" -m pip install -r "%REQ_FILE%" --timeout 120
        if errorlevel 1 (
            echo.
            echo [ERROR] Failed to install dependencies
            echo [INFO] Common issues:
            echo        1. Network connection problem
            echo        2. Insufficient disk space
            echo        3. Version conflict
            echo.
            pause
            exit /b 1
        )
        echo [OK] Dependencies installed using official source
    )
) else (
    echo [INFO] Using official PyPI source
    echo.
    "%PYTHON_EXEC%" -m pip install -r "%REQ_FILE%" --timeout 120
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install dependencies
        echo [INFO] Common issues:
        echo        1. Network connection problem
        echo        2. Insufficient disk space
        echo        3. Version conflict
        echo.
        pause
        exit /b 1
    )
)
echo.
echo [OK] Dependencies installed
echo.

REM 修复 onnxruntime 版本冲突
echo [1.4] Fixing onnxruntime version conflict...
REM funasr-onnx 会自动安装 onnxruntime (CPU版), 但我们只需要 onnxruntime-gpu
"%PYTHON_EXEC%" -m pip uninstall onnxruntime -y >nul 2>&1
if exist "%SITE_PACKAGES%\onnxruntime" (
    echo [INFO] Cleaning residual onnxruntime directory...
    rmdir /s /q "%SITE_PACKAGES%\onnxruntime" >nul 2>&1
)
echo [INFO] Reinstalling onnxruntime-gpu to ensure integrity...
if defined PYPI_MIRROR (
    "%PYTHON_EXEC%" -m pip install --force-reinstall --no-deps onnxruntime-gpu==1.18.0 -i %PYPI_MIRROR% -q
    if errorlevel 1 (
        echo [WARNING] Mirror source failed, trying official PyPI...
        "%PYTHON_EXEC%" -m pip install --force-reinstall --no-deps onnxruntime-gpu==1.18.0 -q
        if errorlevel 1 (
            echo [WARNING] Failed to install onnxruntime-gpu
        )
    )
) else (
    "%PYTHON_EXEC%" -m pip install --force-reinstall --no-deps onnxruntime-gpu==1.18.0 -q
)
echo [OK] onnxruntime-gpu installed
echo.

REM 修复 PyTorch DLL 依赖
echo [1.5] Fixing PyTorch DLL dependencies...
if exist "%SITE_PACKAGES%\torch\lib\libiomp5md.dll" (
    if not exist "%SITE_PACKAGES%\torch\lib\libomp140.x86_64.dll" (
        echo [INFO] Copying libiomp5md.dll to libomp140.x86_64.dll...
        copy "%SITE_PACKAGES%\torch\lib\libiomp5md.dll" "%SITE_PACKAGES%\torch\lib\libomp140.x86_64.dll" >nul 2>&1
        echo [OK] DLL fix applied
    ) else (
        echo [OK] DLL already fixed
    )
) else (
    echo [INFO] PyTorch DLL not found, skipping fix
)
echo.

REM 创建安装标记文件
echo installed > "%MARKER_FILE%"
echo [OK] First time setup completed!
echo.

:deps_done
echo.

REM ========================================
REM  Step 2: 检查 FFmpeg
REM ========================================
echo [Step 2/4] Checking FFmpeg...
if exist "%TOOLS_DIR%\ffmpeg.exe" (
    echo [OK] FFmpeg found
) else (
    echo [WARNING] FFmpeg not found in tools folder
)
echo.

REM ========================================
REM  Step 3: 配置环境变量
REM ========================================
echo [Step 3/4] Configuring environment...

REM 根据 Python 模式设置不同的库路径
if "%PYTHON_MODE%"=="embedded" (
    set "TORCH_LIB=%EMBED_SITE_PACKAGES%\torch\lib"
    set "NVIDIA_CUDNN=%EMBED_SITE_PACKAGES%\nvidia\cudnn\bin"
    set "NVIDIA_CUBLAS=%EMBED_SITE_PACKAGES%\nvidia\cublas\bin"
) else (
    set "TORCH_LIB=%VENV_SITE_PACKAGES%\torch\lib"
    set "NVIDIA_CUDNN=%VENV_SITE_PACKAGES%\nvidia\cudnn\bin"
    set "NVIDIA_CUBLAS=%VENV_SITE_PACKAGES%\nvidia\cublas\bin"
)
set "PATH=%TORCH_LIB%;%NVIDIA_CUDNN%;%NVIDIA_CUBLAS%;%TOOLS_DIR%;%PATH%"
echo [OK] Environment configured
echo.

REM ========================================
REM  Step 4: 清理旧进程并启动服务
REM ========================================
echo [Step 4/4] Starting services...
echo.

REM 清理可能残留的旧进程
echo [Cleanup] Checking for old processes...

REM 清理 FFmpeg 进程
taskkill /F /IM ffmpeg.exe >nul 2>&1
if %ERRORLEVEL%==0 echo [Cleanup] FFmpeg processes terminated

REM 清理 ffprobe 进程
taskkill /F /IM ffprobe.exe >nul 2>&1

REM 检查并释放端口 8000（旧后端进程）（改进版本兼容性）
echo [Cleanup] Checking port 8000...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| find ":8000" ^| find "LISTENING"') do (
    if not "%%a"=="" (
        echo [Cleanup] Found process on port 8000 - PID: %%a
        taskkill /F /PID %%a >nul 2>&1
        if not ERRORLEVEL 1 echo [Cleanup] Killed process PID %%a
    )
)

REM 使用 PowerShell 查找并终止残留的 uvicorn 进程
echo [Cleanup] Checking for old uvicorn processes...
for /f "tokens=*" %%p in ('powershell -NoProfile -Command "Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like '*uvicorn*app.main*'} | Select-Object -ExpandProperty Id" 2^>nul') do (
    if not "%%p"=="" (
        echo [Cleanup] Found old uvicorn process - PID: %%p
        taskkill /F /PID %%p >nul 2>&1
    )
)

REM 等待进程完全退出
timeout /t 2 /nobreak >nul
echo [OK] Old processes cleanup completed
echo.

REM 启动后端服务（后端托管前端静态文件）
echo [Starting] Backend service on port 8000...
start "Video2SRT Backend" cmd /c "title Video2SRT Backend && cd /d %BACKEND_DIR% && set KMP_DUPLICATE_LIB_OK=TRUE && set HF_ENDPOINT=https://hf-mirror.com && set PATH=%TORCH_LIB%;%NVIDIA_CUDNN%;%NVIDIA_CUBLAS%;%TOOLS_DIR%;%PATH% && %PYTHON_EXEC% -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

echo [Waiting] Backend initializing...
timeout /t 5 /nobreak >nul

cd /d "%PROJECT_ROOT%"

timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo   Services Started Successfully!
echo ========================================
echo.
echo   Application: http://localhost:8000
echo   API Docs:    http://localhost:8000/docs
echo.
echo   [!] This window will close automatically
echo       when you click "Exit System" button.
echo.
echo   [!] Do NOT close this window manually!
echo.
echo ========================================

REM 循环检测后端进程是否还在运行
:wait_loop
timeout /t 3 /nobreak >nul

REM 检查端口 8000 是否还有进程监听（改进版本兼容性）
netstat -ano 2>nul | find ":8000" | find "LISTENING" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [INFO] Backend service has stopped. Closing...
    timeout /t 1 /nobreak >nul
    exit /b 0
)

goto wait_loop
