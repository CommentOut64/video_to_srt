!include "MUI2.nsh"

!ifndef SOURCE_DIR
  !error "SOURCE_DIR 未定义"
!endif
!ifndef OUT_FILE
  !error "OUT_FILE 未定义"
!endif

!define APP_NAME "AnchorFlux"
!define MUI_FINISHPAGE_RUN "$INSTDIR\AnchorFlux.exe"
!define MUI_FINISHPAGE_RUN_TEXT "安装完成后启动 AnchorFlux"
!define MUI_FINISHPAGE_RUN_NOTCHECKED

Name "${APP_NAME}"
OutFile "${OUT_FILE}"
InstallDir "$PROGRAMFILES64\AnchorFlux"
RequestExecutionLevel admin
Unicode true

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "SimpChinese"

Section "主程序（必选）" SecMain
  SectionIn RO
  IfFileExists "$INSTDIR\AnchorFlux.exe" 0 +3
  DetailPrint "检测到旧版本，执行覆盖安装（保留用户数据目录）..."
  Call CleanupProgramFilesOnly

  SetOutPath "$INSTDIR"
  File /r "${SOURCE_DIR}\*.*"
  ; 用户数据目录统一保留在安装目录，不随覆盖安装删除
  CreateDirectory "$INSTDIR\jobs"
  CreateDirectory "$INSTDIR\models"
  CreateDirectory "$INSTDIR\input"
  CreateDirectory "$INSTDIR\output"
  CreateDirectory "$INSTDIR\logs"
  CreateDirectory "$INSTDIR\data"
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  CreateDirectory "$SMPROGRAMS\AnchorFlux"
  CreateShortcut "$SMPROGRAMS\AnchorFlux\AnchorFlux.lnk" "$INSTDIR\AnchorFlux.exe"
  ; 桌面快捷方式改为可选组件，默认不创建
  Delete "$DESKTOP\AnchorFlux.lnk"
SectionEnd

Section /o "创建桌面快捷方式" SecDesktop
  CreateShortcut "$DESKTOP\AnchorFlux.lnk" "$INSTDIR\AnchorFlux.exe"
SectionEnd

Section "Uninstall"
  ; 卸载程序文件，但保留用户数据目录（jobs/models/input/output/logs/data）
  Delete "$DESKTOP\AnchorFlux.lnk"
  Delete "$SMPROGRAMS\AnchorFlux\AnchorFlux.lnk"
  RMDir "$SMPROGRAMS\AnchorFlux"
  Call un.CleanupProgramFilesOnly
  RMDir "$INSTDIR"
SectionEnd

Function CleanupProgramFilesOnly
  Delete "$INSTDIR\AnchorFlux.exe"
  Delete "$INSTDIR\pyproject.toml"
  Delete "$INSTDIR\uv.lock"
  Delete "$INSTDIR\user_config.json"
  Delete "$INSTDIR\model_runtime_config.json"
  Delete "$INSTDIR\.env"
  Delete "$INSTDIR\Uninstall.exe"

  RMDir /r "$INSTDIR\backend"
  RMDir /r "$INSTDIR\frontend"
  RMDir /r "$INSTDIR\launcher"
  RMDir /r "$INSTDIR\core"
  RMDir /r "$INSTDIR\.venv"
  RMDir /r "$INSTDIR\_vendor"
  RMDir /r "$INSTDIR\tools"
FunctionEnd

Function un.CleanupProgramFilesOnly
  Delete "$INSTDIR\AnchorFlux.exe"
  Delete "$INSTDIR\pyproject.toml"
  Delete "$INSTDIR\uv.lock"
  Delete "$INSTDIR\user_config.json"
  Delete "$INSTDIR\model_runtime_config.json"
  Delete "$INSTDIR\.env"
  Delete "$INSTDIR\Uninstall.exe"

  RMDir /r "$INSTDIR\backend"
  RMDir /r "$INSTDIR\frontend"
  RMDir /r "$INSTDIR\launcher"
  RMDir /r "$INSTDIR\core"
  RMDir /r "$INSTDIR\.venv"
  RMDir /r "$INSTDIR\_vendor"
  RMDir /r "$INSTDIR\tools"
FunctionEnd
