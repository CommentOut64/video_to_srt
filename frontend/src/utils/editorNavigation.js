/**
 * 编辑器路由导航工具。
 *
 * 目标：
 * - 统一 Full/Lite 入口到 `project_id` 语义。
 * - `/editor/:jobId` 仅用于兼容跳板，必须转换为 `/editor/project/:projectId`。
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
    throw new Error('job_id 不能为空，无法转换为 project_id')
  }

  try {
    const result = await legacyApi.resolveTask(normalizedJobId)
    const projectId = normalizeId(result?.project_id)
    if (!projectId) {
      throw new Error(`未返回 project_id（job_id=${normalizedJobId}）`)
    }
    return projectId
  } catch (error) {
    const reason = error?.message || '未知错误'
    throw new Error(`job_id 转 project_id 失败（job_id=${normalizedJobId}）：${reason}`)
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
    targetPath = `/editor/project/${resolvedProjectId}`
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
