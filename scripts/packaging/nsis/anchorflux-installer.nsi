!include "MUI2.nsh"
!include "LogicLib.nsh"

!ifndef SOURCE_DIR
  !error "SOURCE_DIR 未定义"
!endif
!ifndef OUT_FILE
  !error "OUT_FILE 未定义"
!endif

!define APP_NAME "AnchorFlux"
!define APP_REG_KEY "Software\AnchorFlux"
!define APP_UNINSTALL_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\AnchorFlux"
!define MUI_FINISHPAGE_RUN "$INSTDIR\AnchorFlux.exe"
!define MUI_FINISHPAGE_RUN_TEXT "安装完成后启动 AnchorFlux"
!define MUI_FINISHPAGE_RUN_NOTCHECKED
!define MUI_FINISHPAGE_SHOWREADME
!define MUI_FINISHPAGE_SHOWREADME_TEXT "创建桌面快捷方式"
!define MUI_FINISHPAGE_SHOWREADME_FUNCTION CreateDesktopShortcut
!define MUI_FINISHPAGE_SHOWREADME_NOTCHECKED

Name "${APP_NAME}"
OutFile "${OUT_FILE}"
InstallDir "$PROGRAMFILES64\AnchorFlux"
InstallDirRegKey HKLM "${APP_REG_KEY}" "InstallDir"
RequestExecutionLevel admin
Unicode true

!insertmacro MUI_PAGE_WELCOME
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
  WriteRegStr HKLM "${APP_REG_KEY}" "InstallDir" "$INSTDIR"
  WriteRegStr HKLM "${APP_UNINSTALL_KEY}" "DisplayName" "${APP_NAME}"
  !ifdef APP_VERSION
    WriteRegStr HKLM "${APP_UNINSTALL_KEY}" "DisplayVersion" "${APP_VERSION}"
  !endif
  WriteRegStr HKLM "${APP_UNINSTALL_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "${APP_UNINSTALL_KEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegDWORD HKLM "${APP_UNINSTALL_KEY}" "NoModify" 1
  WriteRegDWORD HKLM "${APP_UNINSTALL_KEY}" "NoRepair" 1
  CreateDirectory "$SMPROGRAMS\AnchorFlux"
  CreateShortcut "$SMPROGRAMS\AnchorFlux\AnchorFlux.lnk" "$INSTDIR\AnchorFlux.exe"
SectionEnd

Section "Uninstall"
  ; 卸载程序文件，但保留用户数据目录（jobs/models/input/output/logs/data）
  Delete "$DESKTOP\AnchorFlux.lnk"
  Delete "$SMPROGRAMS\AnchorFlux\AnchorFlux.lnk"
  RMDir "$SMPROGRAMS\AnchorFlux"
  Call un.CleanupProgramFilesOnly
  DeleteRegKey HKLM "${APP_UNINSTALL_KEY}"
  DeleteRegKey HKLM "${APP_REG_KEY}"
  RMDir "$INSTDIR"
SectionEnd

Function .onInit
  Call DetectExistingInstallDir
FunctionEnd

Function DetectExistingInstallDir
  ReadRegStr $0 HKLM "${APP_REG_KEY}" "InstallDir"
  ${If} $0 != ""
    IfFileExists "$0\AnchorFlux.exe" 0 +2
      StrCpy $INSTDIR "$0"
  ${EndIf}

  IfFileExists "$INSTDIR\AnchorFlux.exe" 0 +2
    Return

  ReadRegStr $1 HKLM "${APP_UNINSTALL_KEY}" "InstallLocation"
  ${If} $1 != ""
    IfFileExists "$1\AnchorFlux.exe" 0 +2
      StrCpy $INSTDIR "$1"
  ${EndIf}

  IfFileExists "$INSTDIR\AnchorFlux.exe" 0 +2
    Return

  IfFileExists "$PROGRAMFILES64\AnchorFlux\AnchorFlux.exe" 0 +2
    StrCpy $INSTDIR "$PROGRAMFILES64\AnchorFlux"
FunctionEnd

Function CreateDesktopShortcut
  CreateShortcut "$DESKTOP\AnchorFlux.lnk" "$INSTDIR\AnchorFlux.exe"
FunctionEnd

Function CleanupProgramFilesOnly
  Delete "$INSTDIR\AnchorFlux.exe"
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
