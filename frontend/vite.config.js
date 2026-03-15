import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

function resolveVendorChunk(id) {
  if (!id.includes('node_modules')) {
    return null
  }

  // 按依赖边界做稳定拆包，避免单一 vendor 主包持续膨胀。
  if (id.includes('/@element-plus/icons-vue/')) {
    return 'element-plus-icons'
  }
  if (id.includes('/element-plus/')) {
    return resolveElementPlusChunk(id)
  }
  if (
    id.includes('/vue/') ||
    id.includes('/vue-router/') ||
    id.includes('/pinia/') ||
    id.includes('/@vueuse/')
  ) {
    return 'vue-core'
  }
  if (
    id.includes('/wavesurfer.js/') ||
    id.includes('/vue-virtual-scroller/') ||
    id.includes('/splitpanes/') ||
    id.includes('/vuedraggable/')
  ) {
    return 'editor-ui-vendor'
  }

  return 'vendor'
}

function resolveElementPlusChunk(id) {
  const normalizedId = id.replace(/\\/g, '/')
  const componentMatch = normalizedId.match(/\/element-plus\/(?:es|lib)\/components\/([^/]+)\//)
  const componentName = componentMatch?.[1]

  if (!componentName) {
    if (
      normalizedId.includes('/element-plus/es/hooks/') ||
      normalizedId.includes('/element-plus/lib/hooks/') ||
      normalizedId.includes('/element-plus/es/utils/') ||
      normalizedId.includes('/element-plus/lib/utils/') ||
      normalizedId.includes('/element-plus/es/tokens/') ||
      normalizedId.includes('/element-plus/lib/tokens/') ||
      normalizedId.includes('/element-plus/es/directives/') ||
      normalizedId.includes('/element-plus/lib/directives/')
    ) {
      return 'element-plus-shared'
    }
    if (
      normalizedId.includes('/element-plus/es/locale/') ||
      normalizedId.includes('/element-plus/lib/locale/')
    ) {
      return 'element-plus-locale'
    }
    return 'element-plus-core'
  }

  const formComponents = new Set([
    'form', 'input', 'input-number', 'select', 'option', 'option-group',
    'checkbox', 'checkbox-button', 'checkbox-group',
    'radio', 'radio-button', 'radio-group',
    'switch', 'slider', 'autocomplete', 'cascader', 'date-picker',
    'time-picker', 'time-select', 'color-picker', 'rate', 'upload', 'mention'
  ])
  if (formComponents.has(componentName)) {
    return 'element-plus-form'
  }

  const overlayComponents = new Set([
    'dialog', 'drawer', 'popover', 'popconfirm', 'tooltip', 'tour', 'image-viewer'
  ])
  if (overlayComponents.has(componentName)) {
    return 'element-plus-overlay'
  }

  const feedbackComponents = new Set([
    'message', 'message-box', 'notification', 'loading', 'progress', 'result',
    'skeleton', 'empty', 'alert'
  ])
  if (feedbackComponents.has(componentName)) {
    return 'element-plus-feedback'
  }

  const dataComponents = new Set([
    'table', 'table-v2', 'tree', 'tree-select', 'pagination', 'calendar',
    'descriptions', 'statistic', 'virtual-list'
  ])
  if (dataComponents.has(componentName)) {
    return 'element-plus-data'
  }

  const navigationComponents = new Set([
    'menu', 'tabs', 'tab-pane', 'breadcrumb', 'breadcrumb-item', 'steps',
    'step', 'dropdown', 'dropdown-item', 'dropdown-menu', 'anchor',
    'page-header', 'backtop', 'affix'
  ])
  if (navigationComponents.has(componentName)) {
    return 'element-plus-navigation'
  }

  const layoutComponents = new Set([
    'container', 'aside', 'header', 'main', 'footer', 'row', 'col',
    'divider', 'space', 'scrollbar'
  ])
  if (layoutComponents.has(componentName)) {
    return 'element-plus-layout'
  }

  return 'element-plus-misc'
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  define: {
    // Task6-Lite: 前端 flavor 编译时常量，支持 tree-shaking
    __APP_FLAVOR__: JSON.stringify(process.env.VITE_APP_FLAVOR || 'full'),
    __IS_LITE__:
      process.env.VITE_LITE_MODE === 'true' || process.env.VITE_APP_FLAVOR === 'lite',
  },
  // 生产环境构建配置 - 移除 console 和 debugger
  esbuild: {
    drop: ['console', 'debugger'],
    pure: ['console.log', 'console.info', 'console.debug', 'console.warn']
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          return resolveVendorChunk(id)
        }
      }
    }
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src')
    }
  },
  css: {
    preprocessorOptions: {
      scss: {
        // 使用新版 Sass API
        // 注意：已移除全局 SCSS 变量注入，改用 CSS Variables（主题系统）
        api: 'modern-compiler',
        // additionalData: `@use "@/styles/_variables" as *; @use "@/styles/_mixins" as *;`
      }
    }
  },
  server: {
    port: 5173,
    proxy: {
      // 后端 API 代理
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      },
      // 媒体资源代理
      '/media': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      }
    }
  }
})
