export function resolveWaveformDomBindings(options = {}) {
  const wrapper = options.wrapper ?? null
  const fallbackScrollContainer = options.fallbackScrollContainer ?? null
  const fallbackContentElement = options.fallbackContentElement ?? null
  const scrollContainer = wrapper?.parentElement ?? fallbackScrollContainer
  const contentElement = wrapper ?? fallbackContentElement ?? fallbackScrollContainer

  return {
    scrollContainer,
    contentElement,
    overlayMountTarget: wrapper,
  }
}
