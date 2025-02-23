<template>
  <div class="flex flex-col items-center p-6 rounded-lg shadow-2xl border border-gray-200 m-4 hover:shadow-3xl transition-shadow duration-300"
       :class="isLeadership ? 'bg-white' : 'bg-gray-50/80 backdrop-blur-sm'">
    <div class="w-48 h-48 mb-4 overflow-hidden rounded-full">
      <img 
        :src="imageUrl || '/images/placeholder-profile.jpg'" 
        :alt="name"
        class="w-full h-full object-cover"
      >
    </div>
    <h3 class="text-xl font-bold mb-2">{{ name }}</h3>
    <p class="text-lg text-gray-600 mb-4">{{ role }}</p>
    <a 
      v-if="linkedinUrl"
      :href="linkedinUrl"
      target="_blank"
      rel="noopener noreferrer"
      class="flex items-center gap-2 text-blue-600 hover:text-blue-800 transition-colors"
    >
      <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
      </svg>
      <span>{{ isLeadership ? 'Connect on LinkedIn' : 'LinkedIn' }}</span>
    </a>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  name: string;
  role: string;
  imageUrl?: string;
  linkedinUrl?: string;
  isLeadership?: boolean;
}>();
</script>

<style>
.hover\:shadow-3xl:hover {
  --tw-shadow: 0 35px 60px -15px rgba(0, 0, 0, 0.3);
  box-shadow: var(--tw-ring-offset-shadow, 0 0 #0000), var(--tw-ring-shadow, 0 0 #0000), var(--tw-shadow);
}
</style>