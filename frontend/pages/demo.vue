<template>
  <div class="max-w-7xl mx-auto h-56 w-full p-4">
    <div class="flex w-full space-x-2">
      <StyledDiv class="w-full grow">
        <template #title>
          EEG Stream
        </template>
        <div class="w-full h-full flex items-center justify-center">
          <h3 class="text-lg font-medium text-gray-300 uppercase tracking-wide">
            Place EEG terminal feed here
          </h3>
        </div>
      </StyledDiv>
      <StyledDiv center class="max-w-96 w-full shrink-0">
        <template #title>
          Stress Levels
        </template>
        <div class="space-y-6 py-4">
          <PercentCircle :percent="sample == null ? 0 : sample.level / 100" class="h-44 w-44">
            <div class="flex space-x-0.5 items-end">
              <div v-if="sample == null" class="bg-gray-300 animate-pulse h-4 mb-1.5 w-6 rounded-md" />
              <h2 v-else class="text-5xl font-semibold text-gray-800 tabular-nums">{{ sample.level }}</h2>
              <h3 class="text-xl text-gray-500">/ 100</h3>
            </div>
          </PercentCircle>
          <div class="flex flex-col items-center">
            <p class="text-sm tracking-wide uppercase font-medium text-gray-400 mb-1">
              Updates in <span class="tabular-nums">{{ secToUpdate }}</span> SEC
            </p>
            <div class="w-36 rounded-b rounded-t-sm h-1 bg-gray-100 overflow-hidden">
              <div class="bg-gray-300 h-full transition-[width] duration-2000" :class="startAnimation ? 'animate-expand' : 'w-0'" />
            </div>
          </div>
        </div>
      </StyledDiv>
    </div>
    <button class="flex items-center space-x-2 py-2 w-full rounded-md justify-center bg-amber-500 text-white hover:bg-amber-600" @click="interpret" :disabled="pending" :class="pending && 'animate-pulse'">
      <svg v-if="pending" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="animate-spin"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M12 3a9 9 0 1 0 9 9" /></svg>
      <svg v-else xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="size-5">
        <path d="M11.983 1.907a.75.75 0 0 0-1.292-.657l-8.5 9.5A.75.75 0 0 0 2.75 12h6.572l-1.305 6.093a.75.75 0 0 0 1.292.657l8.5-9.5A.75.75 0 0 0 17.25 8h-6.572l1.305-6.093Z" />
      </svg>
      <span>
        Interpret Brain Activity
      </span>
    </button>
    <StyledDiv>
      <template #title>
        Generated Prompt
      </template>
      <p v-if="prompt == null" class="text-gray-400 italic font-medium" :class="pending && 'animate-pulse'">
        Interpret brain activity to put your thoughts into words.
      </p>
      <p v-else class="text-gray-800">{{ prompt }}</p>
    </StyledDiv>
    <StyledDiv>
      <template #title>
        Generated Images
      </template>
      <p v-if="video == null" class="text-gray-400 italic font-medium" :class="pending && 'animate-pulse'">
        Interpret brain activity to see a visualization of your thoughts.
      </p>
      <div v-else class="aspect-video w-full">
        <video :src="video" controls class="w-full h-full object-cover rounded-md"></video>
      </div>
    </StyledDiv>
  </div>
</template>

<script setup lang="ts">
// Using existing Terra information from last night, simulate real-time 
// data stream as proof of concept (updates every 15 seconds)
const { data } = await useFetch("/api/terra/stress", {
  query: {
    start_date: "2025-02-15"
  }
})

const sampleIdx = ref(-1);
const secToUpdate = ref(15);
const startAnimation = ref(false);
const sample = computed(() => sampleIdx.value < 0 ? null : data.value.samples[sampleIdx.value]);

function initialize() {
  if (sampleIdx.value < 0 || sampleIdx.value > data.value.samples.length) {
    sampleIdx.value = (data.value!.samples as IStressSample[]).findIndex(
      (sample) => sample.timestamp === "2025-02-15T22:50:00.000000-07:55"
    )
  }
  secToUpdate.value = 15;
  startAnimation.value = false;
  setTimeout(() => startAnimation.value = true, 500)
}

onMounted(() => {
  initialize();
  setInterval(() => {
    if (secToUpdate.value === 1) {
      initialize();
      sampleIdx.value++;
    } else {
      secToUpdate.value = secToUpdate.value - 1;
    }
  }, 1000);
})

const pending = ref(false);
const prompt = ref(null);
const video = ref(null);

async function interpret() {
  try {
    pending.value = true;
    
    // Get latest thought interpretation
    const thoughtResponse = await fetch('http://localhost:8000/api/latest-thought');
    if (!thoughtResponse.ok) {
      throw new Error('Failed to get thought interpretation');
    }
    const thoughtData = await thoughtResponse.json();
    prompt.value = thoughtData.thought;
    
    // Generate video
    const videoResponse = await fetch('http://localhost:8000/api/generate-video', {
      method: 'POST'
    });
    if (!videoResponse.ok) {
      throw new Error('Failed to generate video');
    }
    const videoData = await videoResponse.json();
    video.value = videoData.video_url;
    
  } catch (error) {
    console.error('Error:', error);
    // You might want to show an error message to the user here
  } finally {
    pending.value = false;
  }
}
</script>
