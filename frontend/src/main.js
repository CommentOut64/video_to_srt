/**
 * 应用入口
 */
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import 'element-plus/dist/index.css'
import 'vue-virtual-scroller/dist/vue-virtual-scroller.css'
// 注意: 需要在 element-plus 之后导入自定义样式以覆盖默认样式
import './styles/tailwind.css'
import './styles/base.css'
import './styles/element-override.css'
// 注意：main.scss 已废弃，所有样式已迁移到 base.css 和 element-override.css
// import './styles/main.scss'
import App from './App.vue'
import router from './router'
import { useAppPreferenceStore } from './stores/appPreferenceStore'

// 创建 Pinia 并初始化全局偏好
const pinia = createPinia()
const preferenceStore = useAppPreferenceStore(pinia)
preferenceStore.initialize()

// 创建应用实例
const app = createApp(App)

// 注册 Pinia 状态管理
app.use(pinia)

// 注册路由
app.use(router)

// 注册 Element Plus
app.use(ElementPlus)

// 注册 Element Plus 图标
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

// 挂载应用
app.mount('#app')
