//go:build !windows

package main

import "os/exec"

// applyWindowlessProcessAttr 非 Windows 平台无需处理。
func applyWindowlessProcessAttr(cmd *exec.Cmd, isSilent bool) {}

