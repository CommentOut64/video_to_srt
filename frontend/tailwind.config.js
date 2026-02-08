/** @type {import('tailwindcss').Config} */
export default {
  prefix: 'tw-',
  content: ['./index.html', './src/**/*.{vue,js}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'bg': {
          base: 'var(--af-bg-base)',
          primary: 'var(--af-bg-primary)',
          secondary: 'var(--af-bg-secondary)',
          tertiary: 'var(--af-bg-tertiary)',
          elevated: 'var(--af-bg-elevated)',
        },
        'text': {
          primary: 'var(--af-text-primary)',
          normal: 'var(--af-text-normal)',
          secondary: 'var(--af-text-secondary)',
          muted: 'var(--af-text-muted)',
          disabled: 'var(--af-text-disabled)',
        },
        'accent': {
          primary: 'var(--af-accent-primary)',
          'primary-hover': 'var(--af-accent-primary-hover)',
          'primary-active': 'var(--af-accent-primary-active)',
          success: 'var(--af-accent-success)',
          'success-dim': 'var(--af-accent-success-dim)',
          warning: 'var(--af-accent-warning)',
          'warning-dim': 'var(--af-accent-warning-dim)',
          danger: 'var(--af-accent-danger)',
          'danger-dim': 'var(--af-accent-danger-dim)',
          purple: 'var(--af-accent-purple)',
          pink: 'var(--af-accent-pink)',
        },
        'border': {
          DEFAULT: 'var(--af-border-default)',
          muted: 'var(--af-border-muted)',
          subtle: 'var(--af-border-subtle)',
        },
      },
      borderRadius: {
        'xs': 'var(--af-radius-xs)',
        'sm': 'var(--af-radius-sm)',
        'md': 'var(--af-radius-md)',
        'lg': 'var(--af-radius-lg)',
        'xl': 'var(--af-radius-xl)',
      },
      boxShadow: {
        'sm': 'var(--af-shadow-sm)',
        'md': 'var(--af-shadow-md)',
        'lg': 'var(--af-shadow-lg)',
        'xl': 'var(--af-shadow-xl)',
        'glow': 'var(--af-shadow-glow)',
      },
      transitionDuration: {
        'fast': '100ms',
        'normal': '200ms',
        'slow': '300ms',
      },
      zIndex: {
        'dropdown': '100',
        'sticky': '200',
        'fixed': '300',
        'modal-backdrop': '400',
        'modal': '500',
        'popover': '600',
        'tooltip': '700',
        'notification': '800',
      },
    },
  },
  corePlugins: {
    preflight: false, // 禁用 Tailwind 的样式重置，避免与 Element Plus 冲突
  },
  plugins: [],
}
