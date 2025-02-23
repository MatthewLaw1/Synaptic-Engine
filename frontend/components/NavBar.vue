<template>
  <div class="fixed top-0 left-0 right-0 z-50 m-0">
    <div class="w-full border-t-[16px] border-t-gray-300 border-b-[3px] border-b-amber-500 bg-white shadow-2xl overflow-hidden h-20">
      <div class="mx-auto flex max-w-7xl items-center justify-between px-4 h-full relative">
        <NuxtLink to="/" class="flex items-center h-full pt-1 pb-4">
          <DecoupledLogo class="h-12 w-auto" />
        </NuxtLink>
        <div class="flex items-center bg-gray-100 rounded-lg p-1">
          <NuxtLink
            class="text-xl font-medium text-gray-800 hover:text-amber-500 px-4 py-2 rounded bg-gray-200 hover:bg-gray-300 mx-1"
            to="/past-work"
          >
            Past Work
          </NuxtLink>
          <NuxtLink
            class="text-xl font-medium text-gray-800 hover:text-amber-500 px-4 py-2 rounded bg-gray-200 hover:bg-gray-300 mx-1"
            to="/research"
          >
            Research
          </NuxtLink>
          <NuxtLink
            class="text-xl font-medium text-gray-800 hover:text-amber-500 px-4 py-2 rounded bg-gray-200 hover:bg-gray-300 mx-1"
            to="/demo"
          >
            Demo
          </NuxtLink>
          <NuxtLink
            class="text-xl font-medium text-gray-800 hover:text-amber-500 px-4 py-2 rounded bg-gray-200 hover:bg-gray-300 mx-1"
            to="/team"
          >
            Team
          </NuxtLink>
       </div>
      </div>
    </div>
  </div>
</template>
<script setup lang="ts">
import DecoupledLogo from './DecoupledLogo.vue'

defineProps<{
  showPageLinks?: boolean;
}>();
</script>
