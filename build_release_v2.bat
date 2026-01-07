@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

title AnchorFlux - Build Release Package V2

echo.
echo ========================================
echo   AnchorFlux - Release Builder V2
echo ========================================
echo.

set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

REM ========================================
REM  版本配置
REM  V3.1.1+dev.20260106.04:
REM    - 新增版本号输入和模型选择
REM    - 自动修改.env配置
REM    - 移除README.txt/.gitkeep/temp目录/.pytest_cache
REM ========================================
set "DEFAULT_VERSION=3.1.1"
set "BUILD_DIR=%PROJECT_ROOT%\build"

echo [Config] Project: %PROJECT_ROOT%
echo.

REM ========================================
REM  Step 0a: 版本号输入
REM ========================================
echo ========================================
echo   请输入版本号:
echo ========================================
echo.
echo   默认版本号: %DEFAULT_VERSION%
echo   直接回车使用默认版本号
echo.
set /p VERSION_INPUT="请输入版本号: "

if "%VERSION_INPUT%"=="" (
    set "VERSION=%DEFAULT_VERSION%"
) else (
    set "VERSION=%VERSION_INPUT%"
)

echo.
echo [Config] Version: %VERSION%
echo.

REM ========================================
REM  Step 0b: 模型选择菜单
REM ========================================
echo ========================================
echo   请选择要打包的 Whisper 模型:
echo ========================================
echo.
echo   [0] 不打包任何根目录下的 models 中的模型 (默认使用 medium)
echo   [1] 打包 models--Systran--faster-whisper-medium
echo   [2] 打包 models--Systran--faster-whisper-large-v3
echo.
set /p MODEL_CHOICE="请输入选项 (0/1/2): "

set "MODEL_VALID=0"
if "!MODEL_CHOICE!"=="0" (
    set "MODEL_OPTION=none"
    set "MODEL_NAME=无"
    set "WHISPER_MODEL_VALUE=medium"
    set "RELEASE_NAME=AnchorFlux_v!VERSION!_Portable"
    set "MODEL_VALID=1"
    echo.
    echo [选择] 不打包任何模型 ^(配置文件将设为 medium^)
)
if "!MODEL_CHOICE!"=="1" (
    set "MODEL_OPTION=medium"
    set "MODEL_NAME=faster-whisper-medium"
    set "WHISPER_MODEL_VALUE=medium"
    set "RELEASE_NAME=AnchorFlux_v!VERSION!_Portable_Medium"
    set "MODEL_VALID=1"
    echo.
    echo [选择] 打包 faster-whisper-medium 模型
)
if "!MODEL_CHOICE!"=="2" (
    set "MODEL_OPTION=large"
    set "MODEL_NAME=faster-whisper-large-v3"
    set "WHISPER_MODEL_VALUE=large-v3"
    set "RELEASE_NAME=AnchorFlux_v!VERSION!_Portable_LargeV3"
    set "MODEL_VALID=1"
    echo.
    echo [选择] 打包 faster-whisper-large-v3 模型
)
if "!MODEL_VALID!"=="0" (
    echo [ERROR] 无效选项，请输入 0, 1 或 2
    pause
    exit /b 1
)

set "RELEASE_DIR=%BUILD_DIR%\%RELEASE_NAME%"
echo [Config] Output: %RELEASE_DIR%
echo [Config] Model: %MODEL_NAME%
echo [Config] WHISPER_MODEL: %WHISPER_MODEL_VALUE%
echo.

REM 确认继续
set /p CONFIRM="确认开始打包？(Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo [取消] 用户取消打包
    pause
    exit /b 0
)

echo.

REM ========================================
REM  Step 1: 构建前端
REM ========================================
echo [Step 1/8] Building frontend...

where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found! Cannot build frontend.
    echo [INFO] Please install Node.js 18+ and try again.
    pause
    exit /b 1
)

echo [INFO] Installing frontend dependencies...
cd /d "%PROJECT_ROOT%\frontend"
call npm install >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to install frontend dependencies
    pause
    exit /b 1
)

echo [INFO] Building frontend for production...
call npm run build
if errorlevel 1 (
    echo [ERROR] Failed to build frontend
    pause
    exit /b 1
)
cd /d "%PROJECT_ROOT%"

if not exist "%PROJECT_ROOT%\frontend\dist" (
    echo [ERROR] Frontend dist not found after build!
    pause
    exit /b 1
)

echo [OK] Frontend built successfully
echo.

REM ========================================
REM  Step 2: 清理旧的构建目录
REM ========================================
echo [Step 2/8] Cleaning old build directory...
if exist "%RELEASE_DIR%" (
    rmdir /s /q "%RELEASE_DIR%"
)
mkdir "%RELEASE_DIR%"
echo [OK] Build directory created
echo.

REM ========================================
REM  Step 3: 复制核心文件并修改.env
REM ========================================
echo [Step 3/8] Copying core files and configuring .env...

REM 启动脚本和配置
copy "%PROJECT_ROOT%\run.bat" "%RELEASE_DIR%\" >nul 2>&1
copy "%PROJECT_ROOT%\install_deps.bat" "%RELEASE_DIR%\" >nul 2>&1
copy "%PROJECT_ROOT%\bootloader.py" "%RELEASE_DIR%\" >nul 2>&1
copy "%PROJECT_ROOT%\requirements.txt" "%RELEASE_DIR%\" >nul 2>&1

REM 配置文件
if exist "%PROJECT_ROOT%\user_config.json" (
    copy "%PROJECT_ROOT%\user_config.json" "%RELEASE_DIR%\" >nul 2>&1
)
if exist "%PROJECT_ROOT%\pyproject.toml" (
    copy "%PROJECT_ROOT%\pyproject.toml" "%RELEASE_DIR%\" >nul 2>&1
)
if exist "%PROJECT_ROOT%\.python-version" (
    copy "%PROJECT_ROOT%\.python-version" "%RELEASE_DIR%\" >nul 2>&1
)

REM 额外要求的文件
echo [INFO] Copying additional required files...
if exist "%PROJECT_ROOT%\首次使用提示.txt" (
    copy "%PROJECT_ROOT%\首次使用提示.txt" "%RELEASE_DIR%\" >nul 2>&1
    echo   - 首次使用提示.txt
)
if exist "%PROJECT_ROOT%\AnchorFlux.exe" (
    copy "%PROJECT_ROOT%\AnchorFlux.exe" "%RELEASE_DIR%\" >nul 2>&1
    echo   - AnchorFlux.exe
)

REM 复制并修改.env文件
if exist "%PROJECT_ROOT%\.env" (
    echo [INFO] Copying and modifying .env file...
    copy "%PROJECT_ROOT%\.env" "%RELEASE_DIR%\.env" >nul 2>&1
)

REM 使用PowerShell修改.env文件中的DEV_MODE和WHISPER_MODEL
REM 注意: 必须在if块外部执行，否则延迟扩展变量可能不会正确展开
if exist "!RELEASE_DIR!\.env" (
    echo [INFO] Modifying .env: DEV_MODE=false, WHISPER_MODEL=!WHISPER_MODEL_VALUE!
    powershell -NoProfile -Command "$envPath = '!RELEASE_DIR!\.env'; $whisperModel = '!WHISPER_MODEL_VALUE!'; $content = Get-Content $envPath -Raw -Encoding UTF8; $content = $content -replace 'DEV_MODE=true', 'DEV_MODE=false'; $content = $content -replace 'DEV_MODE=True', 'DEV_MODE=false'; $content = $content -replace 'WHISPER_MODEL=[^\r\n]*', \"WHISPER_MODEL=$whisperModel\"; Set-Content $envPath -Value $content -Encoding UTF8 -NoNewline"
    echo   - .env: OK
)

echo [OK] Core files copied
echo.

REM ========================================
REM  Step 4: 复制后端代码 (排除test开头的文件和.pytest_cache)
REM ========================================
echo [Step 4/8] Copying backend code (excluding test files)...
xcopy "%PROJECT_ROOT%\backend" "%RELEASE_DIR%\backend\" /E /I /Q /Y >nul

REM 排除不需要的目录
if exist "%RELEASE_DIR%\backend\__pycache__" rmdir /s /q "%RELEASE_DIR%\backend\__pycache__"
if exist "%RELEASE_DIR%\backend\scripts" rmdir /s /q "%RELEASE_DIR%\backend\scripts"
if exist "%RELEASE_DIR%\backend\jobs" rmdir /s /q "%RELEASE_DIR%\backend\jobs"
if exist "%RELEASE_DIR%\backend\tests" rmdir /s /q "%RELEASE_DIR%\backend\tests"

REM 删除 backend\models\demucs 缓存目录
echo [INFO] Removing backend\models\demucs cache...
if exist "%RELEASE_DIR%\backend\models\demucs" rmdir /s /q "%RELEASE_DIR%\backend\models\demucs" 2>nul

REM 删除 backend\models\pretrained\sensevoice\model.onnx (FP32大模型)
echo [INFO] Removing sensevoice model.onnx (FP32)...
if exist "%RELEASE_DIR%\backend\models\pretrained\sensevoice\model.onnx" del /q "%RELEASE_DIR%\backend\models\pretrained\sensevoice\model.onnx" 2>nul

REM 删除 .pyc 文件
echo [INFO] Cleaning up cache files...
del /s /q "%RELEASE_DIR%\backend\*.pyc" >nul 2>&1

REM 确保 backend\models 和 backend\app\assets 存在
echo [INFO] Verifying backend\models and backend\app\assets...
if exist "%RELEASE_DIR%\backend\models" (
    echo   - backend\models: OK
) else (
    echo   - [WARN] backend\models not found
)
if exist "%RELEASE_DIR%\backend\app\assets" (
    echo   - backend\app\assets: OK
) else (
    echo   - [WARN] backend\app\assets not found
)

echo [OK] Backend code copied
echo.

REM ========================================
REM  Step 5: 复制前端代码
REM ========================================
echo [Step 5/8] Copying frontend code...
mkdir "%RELEASE_DIR%\frontend"

REM 复制已构建的dist目录
echo [INFO] Copying frontend dist...
xcopy "%PROJECT_ROOT%\frontend\dist" "%RELEASE_DIR%\frontend\dist\" /E /I /Q /Y >nul

REM 复制前端配置文件 (用于开发模式)
copy "%PROJECT_ROOT%\frontend\package.json" "%RELEASE_DIR%\frontend\" >nul
copy "%PROJECT_ROOT%\frontend\package-lock.json" "%RELEASE_DIR%\frontend\" >nul 2>&1
copy "%PROJECT_ROOT%\frontend\vite.config.js" "%RELEASE_DIR%\frontend\" >nul
copy "%PROJECT_ROOT%\frontend\index.html" "%RELEASE_DIR%\frontend\" >nul
if exist "%PROJECT_ROOT%\frontend\jsconfig.json" copy "%PROJECT_ROOT%\frontend\jsconfig.json" "%RELEASE_DIR%\frontend\" >nul

REM 复制前端源码 (用于开发模式)
xcopy "%PROJECT_ROOT%\frontend\src" "%RELEASE_DIR%\frontend\src\" /E /I /Q /Y >nul
xcopy "%PROJECT_ROOT%\frontend\public" "%RELEASE_DIR%\frontend\public\" /E /I /Q /Y >nul 2>&1

echo [OK] Frontend code copied
echo.

REM ========================================
REM  Step 6: 复制工具目录 (含嵌入式Python)
REM ========================================
echo [Step 6/8] Copying tools directory...
xcopy "%PROJECT_ROOT%\tools" "%RELEASE_DIR%\tools\" /E /I /Q /Y >nul

REM 清理嵌入式Python中不需要的文件
if exist "%RELEASE_DIR%\tools\python\Lib\site-packages" (
    echo [INFO] Cleaning embedded Python site-packages...
    for /d %%d in ("%RELEASE_DIR%\tools\python\Lib\site-packages\*") do (
        set "dirname=%%~nxd"
        if /i not "!dirname!"=="pip" (
        if /i not "!dirname!"=="pip-25.3.dist-info" (
        if /i not "!dirname!"=="setuptools" (
        if /i not "!dirname!"=="setuptools-80.9.0.dist-info" (
        if /i not "!dirname!"=="wheel" (
        if /i not "!dirname!"=="wheel-0.45.1.dist-info" (
        if /i not "!dirname!"=="pkg_resources" (
        if /i not "!dirname!"=="_distutils_hack" (
            rmdir /s /q "%%d" 2>nul
        ))))))))
    )
    del /q "%RELEASE_DIR%\tools\python\Lib\site-packages\distutils-precedence.pth" 2>nul
)

echo [OK] Tools directory copied
echo.

REM ========================================
REM  Step 7: 创建目录结构和打包模型 (不创建.gitkeep和temp)
REM ========================================
echo [Step 7/8] Creating directory structure and copying models...
mkdir "%RELEASE_DIR%\input" 2>nul
mkdir "%RELEASE_DIR%\output" 2>nul
mkdir "%RELEASE_DIR%\jobs" 2>nul
mkdir "%RELEASE_DIR%\logs" 2>nul
mkdir "%RELEASE_DIR%\models" 2>nul
mkdir "%RELEASE_DIR%\models\huggingface" 2>nul
mkdir "%RELEASE_DIR%\models\torch" 2>nul

REM 根据选项打包模型
REM HuggingFace 缓存结构: models--xxx/blobs/ + models--xxx/snapshots/hash/
REM snapshots 下的文件是指向 blobs 的符号链接
REM 策略: 只复制最新的一个 snapshot（解析符号链接），不复制 blobs（避免重复）
if "%MODEL_OPTION%"=="medium" (
    echo [INFO] Copying faster-whisper-medium model...
    set "MODEL_SRC=%PROJECT_ROOT%\models\huggingface\models--Systran--faster-whisper-medium"
    set "MODEL_DST=%RELEASE_DIR%\models\huggingface\models--Systran--faster-whisper-medium"
    if exist "!MODEL_SRC!" (
        echo [INFO] Finding latest snapshot...
        REM 获取最新的 snapshot 目录（按修改时间排序，取最新的一个）
        set "LATEST_SNAPSHOT="
        for /f "delims=" %%d in ('dir /b /ad /o-d "!MODEL_SRC!\snapshots" 2^>nul') do (
            if not defined LATEST_SNAPSHOT set "LATEST_SNAPSHOT=%%d"
        )
        if defined LATEST_SNAPSHOT (
            echo [INFO] Copying snapshot: !LATEST_SNAPSHOT!
            mkdir "!MODEL_DST!\snapshots\!LATEST_SNAPSHOT!" 2>nul
            REM 复制单个 snapshot 目录（解析符号链接为实际文件）
            robocopy "!MODEL_SRC!\snapshots\!LATEST_SNAPSHOT!" "!MODEL_DST!\snapshots\!LATEST_SNAPSHOT!" /E /COPY:DAT /R:1 /W:1 /NJH /NJS /NDL /NC /NS >nul 2>&1
            REM 复制 refs 目录（记录分支信息）
            if exist "!MODEL_SRC!\refs" (
                robocopy "!MODEL_SRC!\refs" "!MODEL_DST!\refs" /E /COPY:DAT /R:1 /W:1 /NJH /NJS /NDL /NC /NS >nul 2>&1
            )
            echo   - models--Systran--faster-whisper-medium: OK
        ) else (
            echo   - [WARN] No snapshots found in model directory
        )
    ) else (
        echo   - [WARN] models--Systran--faster-whisper-medium not found
    )
)

if "%MODEL_OPTION%"=="large" (
    echo [INFO] Copying faster-whisper-large-v3 model...
    set "MODEL_SRC=%PROJECT_ROOT%\models\huggingface\models--Systran--faster-whisper-large-v3"
    set "MODEL_DST=%RELEASE_DIR%\models\huggingface\models--Systran--faster-whisper-large-v3"
    if exist "!MODEL_SRC!" (
        echo [INFO] Finding latest snapshot...
        REM 获取最新的 snapshot 目录（按修改时间排序，取最新的一个）
        set "LATEST_SNAPSHOT="
        for /f "delims=" %%d in ('dir /b /ad /o-d "!MODEL_SRC!\snapshots" 2^>nul') do (
            if not defined LATEST_SNAPSHOT set "LATEST_SNAPSHOT=%%d"
        )
        if defined LATEST_SNAPSHOT (
            echo [INFO] Copying snapshot: !LATEST_SNAPSHOT!
            mkdir "!MODEL_DST!\snapshots\!LATEST_SNAPSHOT!" 2>nul
            REM 复制单个 snapshot 目录（解析符号链接为实际文件）
            robocopy "!MODEL_SRC!\snapshots\!LATEST_SNAPSHOT!" "!MODEL_DST!\snapshots\!LATEST_SNAPSHOT!" /E /COPY:DAT /R:1 /W:1 /NJH /NJS /NDL /NC /NS >nul 2>&1
            REM 复制 refs 目录（记录分支信息）
            if exist "!MODEL_SRC!\refs" (
                robocopy "!MODEL_SRC!\refs" "!MODEL_DST!\refs" /E /COPY:DAT /R:1 /W:1 /NJH /NJS /NDL /NC /NS >nul 2>&1
            )
            echo   - models--Systran--faster-whisper-large-v3: OK
        ) else (
            echo   - [WARN] No snapshots found in model directory
        )
    ) else (
        echo   - [WARN] models--Systran--faster-whisper-large-v3 not found
    )
)

if "%MODEL_OPTION%"=="none" (
    echo [INFO] No models will be packaged
)

echo [OK] Directory structure created
echo.

REM ========================================
REM  Step 8: 完成统计
REM ========================================
echo [Step 8/8] Calculating package size...

REM 计算目录大小
for /f "tokens=3" %%a in ('dir "%RELEASE_DIR%" /s /-c 2^>nul ^| findstr "File(s)"') do set SIZE=%%a

echo.
echo ========================================
echo   Build Complete!
echo ========================================
echo.
echo   Output: %RELEASE_DIR%
echo   Version: %VERSION%
echo   Model: %MODEL_NAME%
echo   WHISPER_MODEL: %WHISPER_MODEL_VALUE%
echo   DEV_MODE: false
echo   Size: %SIZE% bytes
echo.
echo   Next steps:
echo   1. Compress %RELEASE_NAME% folder to .7z or .zip
echo   2. Recommended: 7z a -t7z -mx=9 %RELEASE_NAME%.7z %RELEASE_NAME%
echo.
echo ========================================

pause
endlocal
