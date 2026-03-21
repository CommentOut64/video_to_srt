<template>
  <Teleport to="body">
    <Transition name="context-menu">
      <div
        v-if="visible"
        ref="menuRef"
        class="context-menu"
        :style="menuStyle"
        @click.stop
        @contextmenu.prevent
      >
        <div
          v-for="item in items"
          :key="item.key"
          class="context-menu-item"
          :class="{ disabled: item.disabled }"
          @click="handleClick(item)"
        >
          <span class="item-icon" v-if="item.icon">{{ item.icon }}</span>
          <span class="item-label">{{ item.label }}</span>
          <span class="item-shortcut" v-if="item.shortcut">{{
            item.shortcut
          }}</span>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script>
// 模块级单例：确保全局只有一个右键菜单同时可见
// 必须放在 <script>（非 setup）中，否则每个实例各持一份副本，无法互斥
let activeMenuController = null;
</script>

<script setup>
/**
 * ContextMenu - 通用右键菜单组件
 *
 * 用于字幕切分等右键操作
 */
import { ref, computed, onMounted, onUnmounted, nextTick } from "vue";

const props = defineProps({
  // 菜单项列表
  items: {
    type: Array,
    default: () => [],
    // 每项格式: { key: string, label: string, icon?: string, shortcut?: string, disabled?: boolean }
  },
});

const emit = defineEmits(["select", "close"]);

const visible = ref(false);
const position = ref({ x: 0, y: 0 });
const menuRef = ref(null);

// 计算菜单位置样式，确保不超出视口
const menuStyle = computed(() => {
  return {
    left: `${position.value.x}px`,
    top: `${position.value.y}px`,
  };
});

// 显示菜单
function show(x, y) {
  if (activeMenuController && activeMenuController !== menuController) {
    activeMenuController.hide();
  }

  position.value = { x, y };
  visible.value = true;
  activeMenuController = menuController;

  // 下一帧调整位置，防止超出视口
  nextTick(() => {
    if (menuRef.value) {
      const rect = menuRef.value.getBoundingClientRect();
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;

      let newX = x;
      let newY = y;

      if (x + rect.width > viewportWidth) {
        newX = viewportWidth - rect.width - 8;
      }
      if (y + rect.height > viewportHeight) {
        newY = viewportHeight - rect.height - 8;
      }

      position.value = { x: newX, y: newY };
    }
  });
}

// 隐藏菜单
function hide(closeMeta = null) {
  if (!visible.value) {
    if (activeMenuController === menuController) {
      activeMenuController = null;
    }
    return;
  }
  visible.value = false;
  if (activeMenuController === menuController) {
    activeMenuController = null;
  }
  emit("close", closeMeta);
}

// 处理菜单项点击
function handleClick(item) {
  if (item.disabled) {
    return;
  }
  hide({ reason: 'selection', key: item.key });
  emit("select", item.key);
}

// 点击外部关闭菜单
function handleClickOutside(e) {
  if (visible.value && menuRef.value && !menuRef.value.contains(e.target)) {
    hide();
  }
}

// ESC 键关闭菜单
function handleKeydown(e) {
  if (e.key === "Escape" && visible.value) {
    hide();
  }
}

onMounted(() => {
  document.addEventListener("click", handleClickOutside);
  document.addEventListener("contextmenu", handleClickOutside);
  document.addEventListener("keydown", handleKeydown);
});

onUnmounted(() => {
  document.removeEventListener("click", handleClickOutside);
  document.removeEventListener("contextmenu", handleClickOutside);
  document.removeEventListener("keydown", handleKeydown);
  if (activeMenuController === menuController) {
    activeMenuController = null;
  }
});

const menuController = {
  show,
  hide,
};

// 暴露方法给父组件
defineExpose({ show, hide });
</script>

<style scoped>
.context-menu {
  position: fixed;
  z-index: 9999;
  padding: 0;
  background: var(--bg-secondary, #2d2d2d);
  border: 1px solid var(--border-color, #404040);
  border-radius: 6px;
  min-width: 160px;
  box-shadow: 0 4px 12px rgb(0 0 0 / 30%);
  user-select: none;
  overflow: hidden;
}

.context-menu-item {
  display: flex;
  align-items: center;
  width: 100%;
  padding: 8px 12px;
  margin: 0;
  cursor: pointer;
  color: var(--text-primary, #e0e0e0);
  font-size: 13px;
  transition: background-color 0.15s;
  box-sizing: border-box;
}

.context-menu-item:hover:not(.disabled) {
  background: var(--bg-hover, #3d3d3d);
}

.context-menu-item.disabled {
  color: var(--text-disabled, #666);
  cursor: not-allowed;
}

.item-icon {
  margin-right: 8px;
  width: 16px;
  text-align: center;
}

.item-label {
  flex: 1;
}

.item-shortcut {
  margin-left: 16px;
  color: var(--text-secondary, #888);
  font-size: 12px;
}

/* 动画 */
.context-menu-enter-active,
.context-menu-leave-active {
  transition: opacity 0.15s, transform 0.15s;
}

.context-menu-enter-from,
.context-menu-leave-to {
  opacity: 0;
  transform: scale(0.95);
}
</style>
