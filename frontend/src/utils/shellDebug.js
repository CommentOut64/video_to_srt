function readAnchorfluxShell() {
  if (typeof window === 'undefined') {
    return null
  }
  return window.anchorfluxShell || null
}

export function getShellDebugFlags() {
  const shell = readAnchorfluxShell()
  const flags = shell?.getDebugFlags?.() || shell?.debugFlags || {}

  return {
    nativeContextMenuEnabled: flags.nativeContextMenuEnabled === true,
    devToolsEnabled: flags.devToolsEnabled === true,
    rendererProfilingEnabled: flags.rendererProfilingEnabled === true,
  }
}

export function shouldUseNativeContextMenu() {
  return getShellDebugFlags().nativeContextMenuEnabled
}

export function canUseShellDevTools() {
  return getShellDebugFlags().devToolsEnabled
}

export async function openShellDevTools() {
  const shell = readAnchorfluxShell()
  if (typeof shell?.openDevTools !== 'function') {
    return { success: false, reason: 'shell-api-unavailable' }
  }
  return shell.openDevTools()
}
