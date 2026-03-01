/**
 * Vue Router 配置。
 */
import { createRouter, createWebHistory } from 'vue-router'
import { resolveProjectIdByJobId } from '@/utils/editorNavigation'

const routes = [
  {
    path: '/',
    redirect: '/tasks',
  },
  {
    path: '/tasks',
    name: 'TaskList',
    component: () => import('@/views/TaskListView.vue'),
    meta: { title: '任务列表' },
  },
  {
    // 兼容旧 /import 路由，重定向到 /tasks?action=import
    path: '/import',
    redirect: { path: '/tasks', query: { action: 'import' } },
  },
  {
    path: '/editor/project/:projectId',
    name: 'ProjectEditor',
    component: () => import('@/views/EditorView.vue'),
    props: (route) => ({
      projectId: String(route.params.projectId || ''),
      jobId: null,
    }),
    meta: { title: '字幕编辑' },
  },
  {
    path: '/editor/:jobId',
    name: 'Editor',
    component: () => import('@/views/EditorView.vue'),
    props: (route) => ({
      projectId: null,
      jobId: String(route.params.jobId || ''),
    }),
    meta: { title: '字幕编辑' },
    beforeEnter: async (to, _, next) => {
      const jobId = String(to.params.jobId || '').trim()
      if (!jobId) {
        next('/tasks')
        return
      }

      try {
        const projectId = await resolveProjectIdByJobId(jobId)
        next(`/editor/project/${projectId}`)
        return
      } catch (error) {
        console.error('[router] job 路径转换 project 失败:', error)
        next('/tasks')
      }
    },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.beforeEach((to, from, next) => {
  document.title = to.meta.title ? `${to.meta.title} - AnchorFlux` : 'AnchorFlux'
  next()
})

export default router
