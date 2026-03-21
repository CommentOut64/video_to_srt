import { ref } from 'vue'

function queueSelectionDispatch(callback) {
  if (typeof queueMicrotask === 'function') {
    queueMicrotask(callback)
    return
  }

  Promise.resolve().then(callback)
}

export function useSharedContextMenu(options = {}) {
  const isOpen = ref(false)
  const items = ref([])
  const context = ref(null)
  const lastClosedContext = ref(null)

  let activeCloseHandler = null

  function openMenu(payload = {}) {
    const {
      items: nextItems = [],
      context: nextContext = null,
      onClose = null,
    } = payload

    isOpen.value = true
    items.value = Array.isArray(nextItems) ? nextItems : []
    context.value = nextContext
    lastClosedContext.value = null
    activeCloseHandler = typeof onClose === 'function' ? onClose : null
  }

  function handleMenuClose(closeMeta = null) {
    const closeHandler = activeCloseHandler
    const effectiveContext = context.value ?? lastClosedContext.value ?? null
    const normalizedMeta = {
      reason: closeMeta?.reason || 'dismiss',
      key: closeMeta?.key ?? null,
      context: closeMeta?.context ?? effectiveContext,
    }

    lastClosedContext.value = effectiveContext
    isOpen.value = false
    items.value = []
    context.value = null
    activeCloseHandler = null

    closeHandler?.(normalizedMeta)
  }

  function selectMenuItem(key) {
    const selectionContext = context.value ?? lastClosedContext.value

    if (isOpen.value) {
      handleMenuClose({
        reason: 'selection',
        key,
        context: selectionContext,
      })
    }

    if (!selectionContext) {
      return
    }

    queueSelectionDispatch(() => {
      options.onSelect?.({
        key,
        context: selectionContext,
      })
      lastClosedContext.value = null
    })
  }

  return {
    context,
    handleMenuClose,
    isOpen,
    items,
    openMenu,
    selectMenuItem,
  }
}
