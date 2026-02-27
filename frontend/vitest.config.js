import { defineConfig, mergeConfig } from 'vitest/config'
import viteConfig from './vite.config.js'

export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      globals: true,
      environment: 'node',
      setupFiles: ['./src/stores/__tests__/setup/vitest.setup.js'],
      include: ['src/**/*.{spec,test}.{js,ts}'],
      clearMocks: true,
      restoreMocks: true,
    },
  })
)
