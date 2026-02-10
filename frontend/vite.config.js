import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  // 生产环境构建配置 - 移除 console 和 debugger
  esbuild: {
    drop: ['console', 'debugger'],
    pure: ['console.log', 'console.info', 'console.debug', 'console.warn']
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
