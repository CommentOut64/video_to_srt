import { defineConfig, mergeConfig } from 'vitest/config'
import viteConfig from './vite.config.js'

export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      globals: true,
      environment: 'node',
      setupFiles: ['./tests/setup/vitest.setup.js'],
      include: ['tests/**/*.spec.{js,ts}'],
      clearMocks: true,
      restoreMocks: true,
    },
  })
)
