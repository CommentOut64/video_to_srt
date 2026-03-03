// AnchorFlux Stub Launcher
// Go 入口点，替代 bootloader.py
// V3.2.0+dev.20260209.05
//
// 功能：启动 .venv 中的 Python 运行 launcher 模块
// 和 bootloader.py 行为完全一致，只是入口从 Python 脚本变成 exe

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const (
	pythonExeRelPath  = ".venv\\Scripts\\python.exe"
	pythonwExeRelPath = ".venv\\Scripts\\pythonw.exe"
	version           = "3.2.0"
)

func main() {
	// 获取 exe 所在目录
	exePath, err := os.Executable()
	if err != nil {
		fatal("无法获取程序路径: %v", err)
	}
	baseDir := filepath.Dir(exePath)

	// 解析 Python 入口：
	// - 生产默认优先 pythonw.exe（避免弹出控制台窗口）；
	// - 通过 ANCHORFLUX_STUB_CONSOLE=true 可强制使用 python.exe（便于调试）。
	pythonPath, useWindowless := resolvePythonPath(baseDir)
	if pythonPath == "" {
		fatal(
			"Python 环境未安装\n\n请先运行: uv sync\n\n候选路径:\n- %s\n- %s",
			filepath.Join(baseDir, pythonwExeRelPath),
			filepath.Join(baseDir, pythonExeRelPath),
		)
	}

	// 构建命令参数
	args := []string{"-m", "launcher.main"}
	args = append(args, os.Args[1:]...)

	// 启动 Python launcher
	cmd := exec.Command(pythonPath, args...)
	cmd.Dir = baseDir
	if !useWindowless {
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}

	// 运行并等待退出
	if err := cmd.Run(); err != nil {
		// 非零退出码时显示
		if exitErr, ok := err.(*exec.ExitError); ok {
			os.Exit(exitErr.ExitCode())
		}
		if !useWindowless {
			fmt.Fprintf(os.Stderr, "启动失败: %v\n", err)
			waitEnter()
		}
		os.Exit(1)
	}
}

// resolvePythonPath 返回可用 python 路径及是否为无窗口模式。
func resolvePythonPath(baseDir string) (string, bool) {
	useConsole := isTrue(os.Getenv("ANCHORFLUX_STUB_CONSOLE"))
	if !useConsole {
		pythonw := filepath.Join(baseDir, pythonwExeRelPath)
		if fileExists(pythonw) {
			return pythonw, true
		}
	}

	python := filepath.Join(baseDir, pythonExeRelPath)
	if fileExists(python) {
		return python, false
	}
	return "", false
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
