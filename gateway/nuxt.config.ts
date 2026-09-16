export default defineNuxtConfig({
  compatibilityDate: '2026-09-15',
  devtools: { enabled: false },
  runtimeConfig: { recognitionUrl: 'http://127.0.0.1:8080' },
  nitro: { preset: 'node-server' }
})
