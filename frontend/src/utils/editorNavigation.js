/**
 * 编辑器路由导航工具。
 *
 * 目标：
 * - 统一 Full/Lite 入口到 `project_id` 语义。
 * - 保留 `/editor/:jobId` 兼容回退。
 */

import { legacyApi } from '@/services/api'

function normalizeId(value) {
  if (value === null || value === undefined) {
    return ''
  }
  return String(value).trim()
}

export async function resolveProjectIdByJobId(jobId) {
  const normalizedJobId = normalizeId(jobId)
  if (!normalizedJobId) {
    return null
  }

  try {
    const result = await legacyApi.resolveTask(normalizedJobId)
    const projectId = normalizeId(result?.project_id)
    return projectId || null
  } catch (error) {
    console.warn('[editorNavigation] legacy resolve 失败，保留 job 路径兼容:', error)
    return null
  }
}

export async function navigateToEditor(router, { projectId = null, jobId = null, replace = false } = {}) {
  const normalizedProjectId = normalizeId(projectId)
  const normalizedJobId = normalizeId(jobId)

  let targetPath = ''
  if (normalizedProjectId) {
    targetPath = `/editor/project/${normalizedProjectId}`
  } else if (normalizedJobId) {
    const resolvedProjectId = await resolveProjectIdByJobId(normalizedJobId)
    targetPath = resolvedProjectId
      ? `/editor/project/${resolvedProjectId}`
      : `/editor/${normalizedJobId}`
  } else {
    return false
  }

  if (router.currentRoute.value.path === targetPath) {
    return true
  }

  if (replace) {
    await router.replace(targetPath)
  } else {
    await router.push(targetPath)
  }
  return true
}
