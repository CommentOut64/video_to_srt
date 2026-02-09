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
)

const (
	pythonRelPath = ".venv\\Scripts\\python.exe"
	version       = "3.2.0"
)

func main() {
	// 获取 exe 所在目录
	exePath, err := os.Executable()
	if err != nil {
		fatal("无法获取程序路径: %v", err)
	}
	baseDir := filepath.Dir(exePath)

	// 检查 Python 环境
	pythonPath := filepath.Join(baseDir, pythonRelPath)
	if !fileExists(pythonPath) {
		fatal("Python 环境未安装\n\n请先运行: uv sync\n\n路径: %s", pythonPath)
	}

	// 构建命令参数
	args := []string{"-m", "launcher.main"}
	args = append(args, os.Args[1:]...)

	// 启动 Python launcher
	cmd := exec.Command(pythonPath, args...)
	cmd.Dir = baseDir
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// 运行并等待退出
	if err := cmd.Run(); err != nil {
		// 非零退出码时显示
		if exitErr, ok := err.(*exec.ExitError); ok {
			os.Exit(exitErr.ExitCode())
		}
		fmt.Fprintf(os.Stderr, "启动失败: %v\n", err)
		waitEnter()
		os.Exit(1)
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
