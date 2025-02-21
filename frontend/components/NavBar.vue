<template>
  <div class="fixed z-50 w-full border-b-2 border-amber-500 bg-white shadow-md pt-4">
    <div class="mx-auto flex max-w-7xl items-center justify-between px-4 py-6">
      <NuxtLink to="/" class="flex items-center">
        <Logo class="h-12 sm:h-14 w-auto" />
      </NuxtLink>
      <div class="flex items-center space-x-8">
        <NuxtLink
          class="text-xl font-medium text-gray-800 hover:text-amber-500"
          to="/about"
        >
          About
        </NuxtLink>
        <NuxtLink
          class="text-xl font-medium text-gray-800 hover:text-amber-500"
          to="/team"
        >
          Team
        </NuxtLink>
        <NuxtLink
          class="text-xl font-medium text-gray-800 hover:text-amber-500"
          to="/demo"
        >
          Demo
        </NuxtLink>
        </div>
      </div>
    </div>
  </div>
</template>
<script setup lang="ts">
defineProps<{
  showPageLinks?: boolean;
}>();
</script>
