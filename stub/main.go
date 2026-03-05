// AnchorFlux Stub Launcher
// Go 入口点，替代 bootloader.py
// V3.2.4+dev.20260306.03
//
// 功能：启动嵌入式 Python 运行 launcher 模块
// 和 bootloader.py 行为完全一致，只是入口从 Python 脚本变成 exe

package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const (
	embeddedPythonPath = "tools\\python\\python.exe"
	embeddedPythonwPath = "tools\\python\\pythonw.exe"
	pythonExeRelPath  = ".venv\\Scripts\\python.exe"
	pythonwExeRelPath = ".venv\\Scripts\\pythonw.exe"
	version           = "3.2.4"
)

func main() {
	// 获取 exe 所在目录
	exePath, err := os.Executable()
	if err != nil {
		fatal("无法获取程序路径: %v", err)
	}
	baseDir := filepath.Dir(exePath)

	// 解析 Python 入口：
	// V3.2.4+dev.20260306.02: 生产模式仅使用嵌入式 Python（tools/python/pythonw.exe）
	// - 开发模式：使用 .venv/Scripts/python.exe
	// - 生产模式：仅使用 tools/python/pythonw.exe，必要时回退 tools/python/python.exe
	// - 通过 ANCHORFLUX_STUB_CONSOLE=true 可强制使用 python.exe（便于调试）
	pythonPath, useConsole := resolvePythonPath(baseDir, os.Args[1:])
	if pythonPath == "" {
		fatal(
			"Python 环境不可用（嵌入式运行时缺失或开发环境未准备）\n\n候选路径:\n- %s\n- %s\n- %s\n- %s",
			filepath.Join(baseDir, embeddedPythonwPath),
			filepath.Join(baseDir, embeddedPythonPath),
			filepath.Join(baseDir, pythonwExeRelPath),
			filepath.Join(baseDir, pythonExeRelPath),
		)
	}

	// 构建命令参数
	args := buildLauncherArgs(baseDir, pythonPath, os.Args[1:])

	// 启动 Python launcher
	cmd := exec.Command(pythonPath, args...)
	cmd.Dir = baseDir
	cmd.Env = buildPythonEnv(baseDir)
	if useConsole {
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	applyWindowlessProcessAttr(cmd, !useConsole)

	// 运行并等待退出
	if err := cmd.Run(); err != nil {
		// 非零退出码时显示
		if exitErr, ok := err.(*exec.ExitError); ok {
			os.Exit(exitErr.ExitCode())
		}
		if useConsole {
			fmt.Fprintf(os.Stderr, "启动失败: %v\n", err)
			waitEnter()
		}
		os.Exit(1)
	}
}

func buildPythonEnv(baseDir string) []string {
	env := os.Environ()
	pyPath := os.Getenv("PYTHONPATH")
	if pyPath == "" {
		env = append(env, "PYTHONPATH="+baseDir)
		return env
	}
	env = append(env, "PYTHONPATH="+baseDir+string(os.PathListSeparator)+pyPath)
	return env
}

func buildLauncherArgs(baseDir string, pythonPath string, userArgs []string) []string {
	if isEmbeddedPython(baseDir, pythonPath) {
		// 嵌入式 Python 启用 _pth 隔离后，PYTHONPATH 不一定生效。
		// 这里强制把应用根目录注入 sys.path，确保 launcher 可被导入。
		bootstrap := fmt.Sprintf(
			"import runpy, sys; sys.path.insert(0, %q); sys.argv=['launcher.main'] + sys.argv[1:]; runpy.run_module('launcher.main', run_name='__main__')",
			baseDir,
		)
		args := []string{"-c", bootstrap}
		return append(args, userArgs...)
	}

	args := []string{"-m", "launcher.main"}
	return append(args, userArgs...)
}

func isEmbeddedPython(baseDir string, pythonPath string) bool {
	normalized := filepath.Clean(pythonPath)
	embeddedPython := filepath.Clean(filepath.Join(baseDir, embeddedPythonPath))
	embeddedPythonw := filepath.Clean(filepath.Join(baseDir, embeddedPythonwPath))
	return strings.EqualFold(normalized, embeddedPython) || strings.EqualFold(normalized, embeddedPythonw)
}

// resolvePythonPath 返回可用 python 路径及是否需要控制台模式。
// V3.2.4+dev.20260306.02: 生产模式仅使用嵌入式 Python
func resolvePythonPath(baseDir string, userArgs []string) (string, bool) {
	useConsole := isTrue(os.Getenv("ANCHORFLUX_STUB_CONSOLE"))
	isDevMode := detectDevMode(baseDir, userArgs)

	// 生产模式：仅使用嵌入式 Python
	if !isDevMode {
		if !useConsole {
			embeddedPythonw := filepath.Join(baseDir, embeddedPythonwPath)
			if fileExists(embeddedPythonw) {
				return embeddedPythonw, false
			}
		}
		embeddedPython := filepath.Join(baseDir, embeddedPythonPath)
		if fileExists(embeddedPython) {
			return embeddedPython, useConsole
		}
		return "", false
	}

	// 开发模式：默认使用 python.exe，确保错误堆栈可见，避免 pythonw 隐式挂起。
	python := filepath.Join(baseDir, pythonExeRelPath)
	if fileExists(python) {
		return python, true
	}

	if !useConsole {
		pythonw := filepath.Join(baseDir, pythonwExeRelPath)
		if fileExists(pythonw) {
			return pythonw, false
		}
	}
	return "", false
}

func detectDevMode(baseDir string, userArgs []string) bool {
	for _, arg := range userArgs {
		if strings.EqualFold(strings.TrimSpace(arg), "--dev") {
			return true
		}
	}

	if isTrue(os.Getenv("DEV_MODE")) {
		return true
	}

	envPath := filepath.Join(baseDir, ".env")
	file, err := os.Open(envPath)
	if err != nil {
		return false
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if !strings.Contains(line, "=") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		key := strings.TrimSpace(parts[0])
		value := strings.Trim(strings.TrimSpace(parts[1]), "\"'")
		if strings.EqualFold(key, "DEV_MODE") && isTrue(value) {
			return true
		}
	}

	return false
}

func isTrue(raw string) bool {
	value := strings.TrimSpace(strings.ToLower(raw))
	switch value {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// fileExists 检查文件是否存在
func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

// fatal 打印错误并退出
func fatal(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "[错误] "+format+"\n", args...)
	waitEnter()
	os.Exit(1)
}

// waitEnter 等待用户按 Enter
func waitEnter() {
	fmt.Println("\n按 Enter 键退出...")
	fmt.Scanln()
}
