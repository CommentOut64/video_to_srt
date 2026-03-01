import { ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import fileApi from '@/services/api/fileApi'
import projectApi from '@/services/api/projectApi'

/**
 * 导入字幕对话框逻辑管理
 *
 * 设计说明：
 * - 支持「直接上传」和「从本地目录选择」双模式
 * - 字幕文件为必选，媒体文件为可选
 * - 每次导入一个字幕文件，创建一个项目
 */
export function useTaskImport() {
  // === 共享状态 ===
  const importMode = ref('upload') // 'upload' | 'select'
  const projectTitle = ref('')
  const isImporting = ref(false)

  // === 直接上传模式 ===
  const subtitleFile = ref(null) // File 对象
  const mediaFile = ref(null) // File 对象

  // === 本地选择模式 ===
  const subtitleFiles = ref([]) // input 目录中的字幕文件列表
  const mediaFiles = ref([]) // input 目录中的媒体文件列表
  const selectedSubtitle = ref(null) // 选中的字幕文件名 (string)
  const selectedMedia = ref(null) // 选中的媒体文件名 (string)
  const loadingSubtitles = ref(false)
  const loadingMedias = ref(false)

  // === 切换 Tab 时自动加载文件列表 ===
  watch(importMode, (mode) => {
    if (mode === 'select') {
      loadSubtitleFiles()
      loadMediaFiles()
    }
  })

  // === 辅助函数 ===

  /** 从文件名提取项目名称（去掉扩展名） */
  function extractProjectName(filename) {
    if (!filename) return ''
    const lastDot = filename.lastIndexOf('.')
    return lastDot > 0 ? filename.substring(0, lastDot) : filename
  }

  /** 根据文件扩展名检测字幕格式 */
  function detectFormat(fileName) {
    const ext = String(fileName || '')
      .split('.')
      .pop()
      .toLowerCase()
    if (ext === 'ass' || ext === 'vtt') return ext
    return 'srt'
  }

  function formatFileSize(bytes) {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`
  }

  // === 直接上传模式方法 ===

  function handleSubtitleFileChange(uploadFile) {
    const file = uploadFile.raw || uploadFile
    subtitleFile.value = file
    // 自动从文件名提取项目名称
    projectTitle.value = extractProjectName(file.name)
  }

  function removeSubtitleFile() {
    subtitleFile.value = null
    projectTitle.value = ''
  }

  function handleMediaFileChange(event) {
    const file = event.target.files?.[0]
    if (file) {
      mediaFile.value = file
    }
    // 重置 input 以允许重复选择同一文件
    event.target.value = ''
  }

  function removeMediaFile() {
    mediaFile.value = null
  }

  // === 本地选择模式方法 ===

  async function loadSubtitleFiles() {
    loadingSubtitles.value = true
    try {
      const { files } = await fileApi.listSubtitleFiles()
      subtitleFiles.value = files || []
    } catch (error) {
      console.error('加载字幕文件列表失败:', error)
      ElMessage.error(`加载字幕文件列表失败: ${error.message || '未知错误'}`)
      subtitleFiles.value = []
    } finally {
      loadingSubtitles.value = false
    }
  }

  async function loadMediaFiles() {
    loadingMedias.value = true
    try {
      const { files } = await fileApi.listFiles()
      mediaFiles.value = files || []
    } catch (error) {
      console.error('加载媒体文件列表失败:', error)
      mediaFiles.value = []
    } finally {
      loadingMedias.value = false
    }
  }

  async function handleOpenInputFolder() {
    try {
      await fileApi.openInputFolder()
      ElMessage.success('已打开 input 文件夹')
    } catch (error) {
      console.error('打开文件夹失败:', error)
      ElMessage.error(`打开文件夹失败: ${error.message || '未知错误'}`)
    }
  }

  function handleSubtitleSelect(row) {
    if (!row) {
      selectedSubtitle.value = null
      return
    }
    selectedSubtitle.value = row.name
    // 自动从文件名提取项目名称
    projectTitle.value = extractProjectName(row.name)
  }

  function handleMediaSelect(row) {
    if (!row) {
      selectedMedia.value = null
      return
    }
    // 点击同一行取消选中（媒体是可选的）
    if (selectedMedia.value === row.name) {
      selectedMedia.value = null
      return
    }
    selectedMedia.value = row.name
  }

  /** 刷新本地文件列表 */
  async function refreshLocalFiles() {
    await Promise.all([loadSubtitleFiles(), loadMediaFiles()])
  }

  // === 导入操作 ===

  async function handleImport() {
    isImporting.value = true
    try {
      let project
      if (importMode.value === 'upload') {
        if (!subtitleFile.value) {
          ElMessage.warning('请先选择字幕文件')
          return null
        }
        const format = detectFormat(subtitleFile.value.name)
        project = await projectApi.importProject(
          subtitleFile.value,
          format,
          mediaFile.value,
          projectTitle.value,
        )
      } else {
        if (!selectedSubtitle.value) {
          ElMessage.warning('请先选择字幕文件')
          return null
        }
        project = await projectApi.importProjectLocal(
          selectedSubtitle.value,
          selectedMedia.value,
          projectTitle.value,
        )
      }

      if (project?.project_id) {
        ElMessage.success('导入成功')
        return project.project_id
      }
      ElMessage.error('导入失败：未返回项目信息')
      return null
    } catch (error) {
      console.error('导入失败:', error)
      ElMessage.error(`导入失败: ${error.message || '未知错误'}`)
      return null
    } finally {
      isImporting.value = false
    }
  }

  /** 重置所有状态 */
  function resetState() {
    importMode.value = 'upload'
    projectTitle.value = ''
    isImporting.value = false
    subtitleFile.value = null
    mediaFile.value = null
    subtitleFiles.value = []
    mediaFiles.value = []
    selectedSubtitle.value = null
    selectedMedia.value = null
    loadingSubtitles.value = false
    loadingMedias.value = false
  }

  return {
    // 共享状态
    importMode,
    isImporting,
    // 直接上传模式
    subtitleFile,
    mediaFile,
    handleSubtitleFileChange,
    removeSubtitleFile,
    handleMediaFileChange,
    removeMediaFile,
    // 本地选择模式
    subtitleFiles,
    mediaFiles,
    selectedSubtitle,
    selectedMedia,
    loadingSubtitles,
    loadingMedias,
    loadSubtitleFiles,
    loadMediaFiles,
    handleOpenInputFolder,
    handleSubtitleSelect,
    handleMediaSelect,
    refreshLocalFiles,
    // 工具
    formatFileSize,
    // 操作
    handleImport,
    resetState,
  }
}
