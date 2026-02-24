<template>
  <div class="tw-min-h-screen tw-bg-bg-base tw-text-text-normal">
    <main class="tw-mx-auto tw-w-full tw-max-w-5xl tw-px-6 tw-py-10">
      <header class="tw-mb-8">
        <p class="tw-text-xs tw-uppercase tw-tracking-[0.16em] tw-text-text-muted">Lite Import</p>
        <h1 class="tw-mt-2 tw-text-3xl tw-font-semibold tw-text-text-primary">字幕导入工作台</h1>
        <p class="tw-mt-2 tw-text-sm tw-text-text-secondary">
          导入字幕后直接进入编辑器，可附带音频或视频文件用于波形与预览。
        </p>
      </header>

      <section class="tw-grid tw-gap-8 lg:tw-grid-cols-[1.1fr_0.9fr]">
        <div class="tw-rounded-lg tw-border tw-border-border tw-bg-bg-primary tw-p-5">
          <div class="tw-space-y-4">
            <label class="tw-block">
              <span class="tw-mb-2 tw-block tw-text-sm tw-font-medium tw-text-text-primary">字幕文件</span>
              <input
                class="tw-w-full tw-rounded-md tw-border tw-border-border tw-bg-bg-secondary tw-px-3 tw-py-2 tw-text-sm"
                type="file"
                accept=".srt,.ass,.vtt"
                @change="onSubtitleFileChange"
              />
            </label>

            <label class="tw-block">
              <span class="tw-mb-2 tw-block tw-text-sm tw-font-medium tw-text-text-primary">媒体文件（可选）</span>
              <input
                class="tw-w-full tw-rounded-md tw-border tw-border-border tw-bg-bg-secondary tw-px-3 tw-py-2 tw-text-sm"
                type="file"
                accept="video/*,audio/*"
                @change="onVideoFileChange"
              />
            </label>

            <label class="tw-block">
              <span class="tw-mb-2 tw-block tw-text-sm tw-font-medium tw-text-text-primary">项目名称</span>
              <input
                v-model.trim="projectTitle"
                class="tw-w-full tw-rounded-md tw-border tw-border-border tw-bg-bg-secondary tw-px-3 tw-py-2 tw-text-sm"
                type="text"
                placeholder="我的字幕项目"
              />
            </label>
          </div>

          <div class="tw-mt-6 tw-flex tw-items-center tw-gap-3">
            <button
              class="tw-rounded-md tw-bg-accent-primary tw-px-4 tw-py-2 tw-text-sm tw-font-medium tw-text-white disabled:tw-cursor-not-allowed disabled:tw-opacity-60"
              :disabled="!subtitleFile || isImporting"
              @click="handleImport"
            >
              {{ isImporting ? '导入中...' : '开始导入' }}
            </button>
            <span v-if="subtitleFile" class="tw-text-xs tw-text-text-muted">
              已选择：{{ subtitleFile.name }}
            </span>
          </div>
        </div>

        <aside class="tw-rounded-lg tw-border tw-border-border tw-bg-bg-primary tw-p-5">
          <div class="tw-mb-3 tw-flex tw-items-center tw-justify-between">
            <h2 class="tw-text-base tw-font-semibold tw-text-text-primary">已有项目</h2>
            <button
              class="tw-text-xs tw-text-accent-primary hover:tw-underline"
              :disabled="isProjectLoading"
              @click="loadProjects"
            >
              刷新
            </button>
          </div>

          <div v-if="isProjectLoading" class="tw-py-8 tw-text-center tw-text-sm tw-text-text-muted">
            正在加载...
          </div>

          <ul v-else-if="projects.length > 0" class="tw-space-y-2">
            <li
              v-for="project in projects"
              :key="project.project_id"
              class="tw-cursor-pointer tw-rounded-md tw-border tw-border-border tw-bg-bg-secondary tw-p-3 hover:tw-border-accent-primary"
              @click="openProject(project.project_id)"
            >
              <div class="tw-text-sm tw-font-medium tw-text-text-primary">{{ project.title || project.project_id }}</div>
              <div class="tw-mt-1 tw-text-xs tw-text-text-muted">
                {{ project.subtitle_doc?.segment_count ?? 0 }} 条字幕
              </div>
            </li>
          </ul>

          <div v-else class="tw-py-8 tw-text-center tw-text-sm tw-text-text-muted">暂无项目</div>
        </aside>
      </section>
    </main>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { projectApi } from '@/services/api'

const router = useRouter()

const subtitleFile = ref(null)
const videoFile = ref(null)
const projectTitle = ref('')
const projects = ref([])
const isImporting = ref(false)
const isProjectLoading = ref(false)

function detectFormat(fileName) {
  const ext = String(fileName || '')
    .split('.')
    .pop()
    .toLowerCase()
  if (ext === 'ass' || ext === 'vtt') {
    return ext
  }
  return 'srt'
}

function onSubtitleFileChange(event) {
  const [file] = event.target.files || []
  subtitleFile.value = file || null
  if (file && !projectTitle.value) {
    projectTitle.value = file.name.replace(/\.(srt|ass|vtt)$/i, '')
  }
}

function onVideoFileChange(event) {
  const [file] = event.target.files || []
  videoFile.value = file || null
}

async function loadProjects() {
  isProjectLoading.value = true
  try {
    projects.value = await projectApi.listProjects()
  } catch (error) {
    ElMessage.error(`加载项目失败：${error.message || '未知错误'}`)
  } finally {
    isProjectLoading.value = false
  }
}

function openProject(projectId) {
  router.push(`/editor/project/${projectId}`)
}

async function handleImport() {
  if (!subtitleFile.value) {
    ElMessage.warning('请先选择字幕文件')
    return
  }
  isImporting.value = true
  try {
    const format = detectFormat(subtitleFile.value.name)
    const project = await projectApi.importProject(
      subtitleFile.value,
      format,
      videoFile.value,
      projectTitle.value
    )
    const projectId = project?.project_id
    if (!projectId) {
      throw new Error('后端未返回 project_id')
    }
    ElMessage.success('导入成功')
    router.push(`/editor/project/${projectId}`)
  } catch (error) {
    ElMessage.error(`导入失败：${error.message || '未知错误'}`)
  } finally {
    isImporting.value = false
  }
}

onMounted(() => {
  loadProjects()
})
</script>
