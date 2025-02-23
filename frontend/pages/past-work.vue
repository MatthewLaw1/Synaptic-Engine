<template>
  <div class="mx-auto max-w-7xl py-12 px-4">
    <h1 class="text-4xl font-bold text-center mb-16">Past Work</h1>

    <!-- Project Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
      <!-- Placeholder Project Cards -->
      <div class="bg-white rounded-xl border border-gray-100 overflow-hidden shadow-lg hover:shadow-xl transition-shadow duration-300">
        <div class="aspect-video bg-gray-100"></div>
        <div class="p-6">
          <h3 class="text-xl font-bold mb-2">Initial Research Phase</h3>
          <p class="text-gray-600">Early development and testing of neural signal processing algorithms for aphasia communication.</p>
        </div>
      </div>

      <div class="bg-white rounded-xl border border-gray-100 overflow-hidden shadow-lg hover:shadow-xl transition-shadow duration-300">
        <div class="aspect-video bg-gray-100"></div>
        <div class="p-6">
          <h3 class="text-xl font-bold mb-2">Prototype Development</h3>
          <p class="text-gray-600">Creation and testing of the first functional prototype for neural-visual communication.</p>
        </div>
      </div>

      <div class="bg-white rounded-xl border border-gray-100 overflow-hidden shadow-lg hover:shadow-xl transition-shadow duration-300">
        <div class="aspect-video bg-gray-100"></div>
        <div class="p-6">
          <h3 class="text-xl font-bold mb-2">Clinical Trials</h3>
          <p class="text-gray-600">Preliminary testing with aphasia patients to validate system effectiveness.</p>
        </div>
      </div>
    </div>

    <!-- Timeline Section -->
    <div class="mt-24">
      <h2 class="text-3xl font-bold text-center mb-12">Development Timeline</h2>
      <div class="max-w-3xl mx-auto space-y-8">
        <div class="flex items-start">
          <div class="flex-shrink-0 w-24 text-right pr-6">
            <span class="text-amber-600 font-semibold">2024</span>
          </div>
          <div class="flex-grow pb-8 border-l-2 border-gray-200 pl-6">
            <h3 class="text-xl font-bold mb-2">Project Inception</h3>
            <p class="text-gray-600">Initial research and development of core neural processing algorithms.</p>
          </div>
        </div>

        <div class="flex items-start">
          <div class="flex-shrink-0 w-24 text-right pr-6">
            <span class="text-amber-600 font-semibold">2025</span>
          </div>
          <div class="flex-grow pb-8 border-l-2 border-gray-200 pl-6">
            <h3 class="text-xl font-bold mb-2">Prototype Testing</h3>
            <p class="text-gray-600">Development and validation of the first working prototype.</p>
          </div>
        </div>

        <div class="flex items-start">
          <div class="flex-shrink-0 w-24 text-right pr-6">
            <span class="text-amber-600 font-semibold">Future</span>
          </div>
          <div class="flex-grow border-l-2 border-gray-200 pl-6">
            <h3 class="text-xl font-bold mb-2">Clinical Implementation</h3>
            <p class="text-gray-600">Planned deployment in clinical settings and continued development.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>