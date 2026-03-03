<template>
  <el-dialog
    v-model="visible"
    title="关于"
    width="320px"
    :lock-scroll="false"
    :close-on-click-modal="true"
    :close-on-press-escape="true"
    class="about-dialog"
  >
    <div class="about-content">
      <!-- 图标占位 -->
      <div class="app-icon">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <path
    fill="currentColor"
    d="M21 3H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H3V5h18v14zM5 10h9v2H5zm0-3h9v2H5zm0 6h6v2H5z"
  />
</svg>
      </div>

      <!-- 应用名称 -->
      <h2 class="app-name">AnchorFlux</h2>

      <!-- 版本号 -->
      <div class="version">版本 {{ version }}</div>

      <!-- 检查更新按钮 V3.1.1+dev.20260105.01 -->
      <el-button
        class="check-update-btn"
        :loading="isCheckingUpdate"
        @click="handleCheckUpdate"
        size="small"
        plain
      >
        {{ isCheckingUpdate ? '检查中...' : '检查更新' }}
      </el-button>

      <!-- 链接 -->
      <div class="links">
        <a
          href="https://github.com/CommentOut64/anchor-flux"
          target="_blank"
          class="link-item"
        >
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
          </svg>
          <span>GitHub</span>
        </a>

        <a
          href="https://space.bilibili.com/515407408"
          target="_blank"
          class="link-item"
        >
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M17.813 4.653h.854c1.51.054 2.769.578 3.773 1.574 1.004.995 1.524 2.249 1.56 3.76v7.36c-.036 1.51-.556 2.769-1.56 3.773s-2.262 1.524-3.773 1.56H5.333c-1.51-.036-2.769-.556-3.773-1.56S.036 18.858 0 17.347v-7.36c.036-1.511.556-2.765 1.56-3.76 1.004-.996 2.262-1.52 3.773-1.574h.774l-1.174-1.12a1.234 1.234 0 0 1-.373-.906c0-.356.124-.658.373-.907l.027-.027c.267-.249.573-.373.92-.373.347 0 .653.124.92.373L9.653 4.44c.071.071.134.142.187.213h4.267a.836.836 0 0 1 .16-.213l2.853-2.747c.267-.249.573-.373.92-.373.347 0 .662.151.929.4.267.249.391.551.391.907 0 .355-.124.657-.373.906zM5.333 7.24c-.746.018-1.373.276-1.88.773-.506.498-.769 1.13-.786 1.894v7.52c.017.764.28 1.395.786 1.893.507.498 1.134.756 1.88.773h13.334c.746-.017 1.373-.275 1.88-.773.506-.498.769-1.129.786-1.893v-7.52c-.017-.765-.28-1.396-.786-1.894-.507-.497-1.134-.755-1.88-.773zM8 11.107c.373 0 .684.124.933.373.25.249.383.569.4.96v1.173c-.017.391-.15.711-.4.96-.249.25-.56.374-.933.374s-.684-.125-.933-.374c-.25-.249-.383-.569-.4-.96V12.44c0-.373.129-.689.386-.947.258-.257.574-.386.947-.386zm8 0c.373 0 .684.124.933.373.25.249.383.569.4.96v1.173c-.017.391-.15.711-.4.96-.249.25-.56.374-.933.374s-.684-.125-.933-.374c-.25-.249-.383-.569-.4-.96V12.44c.017-.391.15-.711.4-.96.249-.249.56-.373.933-.373Z"/>
          </svg>
          <span>Bilibili</span>
        </a>
      </div>

      <!-- 日志级别设置（已禁用：控制台固定INFO，文件固定DEBUG）
      <div class="log-level-section">
        <div class="log-level-label">日志级别</div>
        <el-select
          v-model="logLevel"
          placeholder="选择日志级别"
          size="small"
          @change="handleLogLevelChange"
          class="log-level-select"
        >
          <el-option label="DEBUG" value="DEBUG" />
          <el-option label="INFO" value="INFO" />
          <el-option label="WARNING" value="WARNING" />
          <el-option label="ERROR" value="ERROR" />
        </el-select>
        <div class="log-level-hint">修改后需要重启系统生效</div>
      </div>
      -->
    </div>

    <!-- 更新窗口 V3.1.1+dev.20260105.01 -->
    <UpdateDialog
      v-model="showUpdateDialog"
      :update-info="pendingUpdateInfo"
      @ignore="handleUpdateIgnored"
    />
  </el-dialog>
</template>

<script setup>
import { ref, watch, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { useAppUpdateStore } from '@/stores/appUpdateStore'
import systemApi from '@/services/api/systemApi'
import UpdateDialog from './UpdateDialog.vue'

const visible = defineModel({ type: Boolean, default: false })

// 版本号
const version = ref('...')

// V3.1.1+dev.20260105.01: 更新相关状态
const isCheckingUpdate = ref(false)
const showUpdateDialog = ref(false)
const pendingUpdateInfo = ref(null)

const appUpdateStore = useAppUpdateStore()
appUpdateStore.initialize()
const checkForUpdate = appUpdateStore.checkForUpdate

// 获取版本号
async function fetchVersion() {
  try {
    const response = await systemApi.getVersion()
    if (response?.version) {
      version.value = response.version
    }
  } catch (e) {
    console.warn('[About] Failed to fetch version:', e)
    version.value = '未知'
  }
}

// V3.1.1+dev.20260105.01: 检查更新
async function handleCheckUpdate() {
  isCheckingUpdate.value = true

  try {
    // 手动检查时忽略已忽略的版本
    const result = await checkForUpdate(true)

    if (result.hasUpdate && result.updateInfo) {
      pendingUpdateInfo.value = result.updateInfo
      showUpdateDialog.value = true
    } else if (result.error) {
      ElMessage.error('检查更新失败: ' + result.error)
    } else {
      ElMessage.success('当前已是最新版本')
    }
  } catch (e) {
    ElMessage.error('检查更新失败: ' + (e.message || '网络错误'))
  } finally {
    isCheckingUpdate.value = false
  }
}

// 更新被忽略
function handleUpdateIgnored() {
  pendingUpdateInfo.value = null
}

// 对话框打开时获取版本号
watch(visible, (newVal) => {
  if (newVal) {
    fetchVersion()
  }
})

// 初始化时也获取版本号
onMounted(() => {
  fetchVersion()
})
</script>

<style scoped>

.about-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 14px;
}

.app-icon {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 80px;
  height: 80px;
  color: var(--af-accent-primary);
}

.app-icon .icon-placeholder {
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, var(--af-accent-primary) 0%, var(--af-accent-primary) 100%);
  border-radius: 16px;
  opacity: 0.8;
}

.app-name {
  margin: 0;
  color: var(--af-text-primary);
  font-size: 22px;
  font-weight: 600;
}

.version {
  color: var(--af-text-secondary);
  font-size: 13px;
}

.check-update-btn {
  margin-top: 4px;
}

.links {
  display: flex;
  gap: 16px;
  margin-top: 8px;
}

.link-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  background: var(--af-bg-tertiary);
  border: 1px solid var(--af-border-default);
  border-radius: 6px;
  color: var(--af-text-normal);
  text-decoration: none;
  transition: all 0.2s;
}

.link-item svg {
  width: 18px;
  height: 18px;
}

.link-item span {
  font-size: 13px;
}

.link-item:hover {
  background: var(--af-bg-elevated);
  border-color: var(--af-accent-primary);
  color: var(--af-accent-primary);
}

.log-level-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid var(--af-border-default);
}

.log-level-label {
  color: var(--af-text-normal);
  font-size: 13px;
  font-weight: 500;
}

.log-level-select {
  width: 100%;
}

.log-level-hint {
  color: var(--af-text-secondary);
  font-size: 12px;
  text-align: center;
}
</style>
