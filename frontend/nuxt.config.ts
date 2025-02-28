// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: true },
  modules: [
    "@nuxtjs/tailwindcss",
    "nuxt-headlessui"
  ],
  app: {
    pageTransition: { name: 'page', mode: 'out-in' }
  },
  nitro: {
    routeRules: {
      '/': { redirect: '/about' }
    }
  }
})
