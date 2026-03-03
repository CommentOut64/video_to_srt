//go:build windows

package main

import (
	"os/exec"
	"syscall"
)

// applyWindowlessProcessAttr 在 Windows 下为子进程设置无窗口属性。
func applyWindowlessProcessAttr(cmd *exec.Cmd, isSilent bool) {
	if !isSilent || cmd == nil {
		return
	}
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
}

