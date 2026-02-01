/**
 * 深色主题 - GitHub Dark Dimmed 风格
 * 提供完整的深色模式配色方案
 */
export const darkTheme = {
  name: 'dark',
  displayName: '深色模式',

  colors: {
    // 背景色
    bg: {
      base: '#0d1117',
      primary: '#161b22',
      secondary: '#21262d',
      tertiary: '#30363d',
      elevated: '#2d333b',
      overlay: 'rgba(0, 0, 0, 0.5)',
    },

    // 文本色
    text: {
      primary: '#e6edf3',
      normal: '#c9d1d9',
      secondary: '#8b949e',
      secondaryRgb: '139, 148, 158',
      muted: '#6e7681',
      mutedRgb: '110, 118, 129',
      disabled: '#484f58',
      inverse: '#ffffff',
      onDark: '#ffffff',
      onDarkRgb: '255, 255, 255',
    },

    // 强调色
    accent: {
      primary: '#58a6ff',
      primaryHover: '#79c0ff',
      primaryActive: '#388bfd',
      primaryRgb: '88, 166, 255',
      success: '#3fb950',
      successDim: '#238636',
      successRgb: '63, 185, 80',
      warning: '#d29922',
      warningDim: '#9e6a03',
      warningRgb: '210, 153, 34',
      danger: '#f85149',
      dangerDim: '#da3633',
      dangerRgb: '248, 81, 73',
      purple: '#a371f7',
      pink: '#db61a2',
    },

    // 边框
    border: {
      default: '#30363d',
      muted: '#21262d',
      subtle: 'rgba(240, 246, 252, 0.1)',
    },
  },

  // 功能性颜色
  functional: {
    // 草稿状态颜色
    draft: {
      bg: '#6b7280',
      bgRgb: '107, 114, 128',
      text: '#9ca3af',
    },

    // 阶段标记颜色
    stage: {
      sensevoice: '#58a6ff',
      whisper: '#3fb950',
      llm: '#a371f7',
    },

    // 状态颜色
    status: {
      processing: '#028AC5',
      processingRgb: '2, 138, 197',
      warning: '#e67700',
      warningRgb: '230, 119, 0',
      error: '#f85149',
      errorRgb: '248, 81, 73',
    },

    // 字级置信度高亮颜色
    confidence: {
      warning: '#ffc107',
      warningRgb: '255, 193, 7',
      critical: '#f44336',
      criticalRgb: '244, 67, 54',
    },

    // 波形图专用颜色
    waveform: {
      color: '#58a6ff',
      progress: '#238636',
      cursor: '#f85149',
      regionDefault: 'rgba(88, 166, 255, 0.25)',
      regionActive: 'rgba(88, 166, 255, 0.4)',
      regionSelected: 'rgba(163, 113, 247, 0.35)',
    },

    // 视频播放器专用颜色
    video: {
      bg: '#000000',
      text: '#ffffff',
      overlay: 'rgba(0, 0, 0, 0.6)',
      overlayMedium: 'rgba(0, 0, 0, 0.75)',
      overlayDark: 'rgba(0, 0, 0, 0.85)',
      overlayDarker: 'rgba(0, 0, 0, 0.9)',
      overlayHint: 'rgba(0, 0, 0, 0.7)',
      controlBg: 'rgba(0, 0, 0, 0.85)',
      controlBgHover: 'rgba(0, 0, 0, 0.95)',
      controlBorder: 'rgba(255, 255, 255, 0.2)',
      controlBorderHover: 'rgba(255, 255, 255, 0.4)',
      controlText: 'rgba(255, 255, 255, 0.7)',
      textShadow: '1px 1px 2px rgba(0, 0, 0, 0.8)',
      textShadowSm: '0 1px 2px rgba(0, 0, 0, 0.5)',
      scrollbarThumb: 'rgba(255, 255, 255, 0.3)',
      shadowLg: '0 4px 12px rgba(0, 0, 0, 0.5)',
    },

    // 状态标志颜色（分辨率标志等）
    statusBadge: {
      warningBg: 'rgba(255, 193, 7, 0.9)',
      warningText: '#000',
      successBg: 'rgba(76, 175, 80, 0.9)',
      successText: '#fff',
      infoBg: 'rgba(33, 150, 243, 0.9)',
      infoText: '#fff',
      premiumBg: 'rgba(186, 104, 200, 0.85)',
      premiumText: '#fff',
    },
  },
}
