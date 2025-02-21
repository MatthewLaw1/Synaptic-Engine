<template>
  <div class="container mx-auto px-4 py-12">
    <h1 class="text-5xl font-bold text-center mb-16">Our Team</h1>
    
    <!-- Leadership Tier -->
    <div class="mb-32">
      <h2 class="text-3xl font-bold text-center mb-12 text-gray-800 relative">
        <span class="inline-block pb-4 border-b-4 border-amber-500">Leadership</span>
      </h2>
      <div class="flex justify-center">
        <div class="transform scale-125">
          <TeamMember
            v-for="member in leadershipTeam"
            :key="member.name"
            v-bind="member"
            :isLeadership="true"
            class="leadership-member"
          />
        </div>
      </div>
    </div>

    <!-- Product & Data Science Tier -->
    <div class="mb-24">
      <h2 class="text-2xl font-semibold text-center mb-10 text-gray-800">Product & Data Science</h2>
      <div class="flex flex-wrap justify-center gap-16">
        <div class="transform scale-120" v-for="member in productDataTeam" :key="member.name">
          <TeamMember
            v-bind="member"
            :isLeadership="false"
            class="product-data-member"
          />
        </div>
      </div>
    </div>

    <!-- Research Team Tier -->
    <div class="mb-16">
      <h2 class="text-2xl font-semibold text-center mb-10 text-gray-800">Research Team</h2>
      <div class="flex flex-wrap justify-center gap-16">
        <div class="transform scale-120" v-for="member in researchTeam" :key="member.name">
          <TeamMember
            v-bind="member"
            :isLeadership="false"
            class="research-member"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const leadershipTeam = [
  {
    name: "Matthew Law",
    role: "Principal Research Director, Lead AI Researcher",
    imageUrl: "/images/matthewlaw.png",
    linkedinUrl: "https://www.linkedin.com/in/matthew-law-0x251/"
  }
];

const productDataTeam = [
  {
    name: "Angad Bhargav",
    role: "Solutions Architect",
    imageUrl: "/images/angadbhargav.png",
    linkedinUrl: "https://www.linkedin.com/in/angad-bhargav/"
  },
  {
    name: "Vihaan Mathur",
    role: "Data Scientist",
    imageUrl: "/images/vihaanmathur.png",
    linkedinUrl: "https://www.linkedin.com/in/vihaan-mathur-a952ab207/"
  }
];

const researchTeam = [
  {
    name: "Girvin Djapardi",
    role: "Research Assistant",
    imageUrl: "/images/girvindjapardi.png",
    linkedinUrl: "https://www.linkedin.com/in/girvin-djapardi/"
  },
  {
    name: "Shivam Kumar",
    role: "Research Assistant",
    imageUrl: "/images/shivamkumar.png",
    linkedinUrl: "https://www.linkedin.com/in/shivam-kumar-ent/"
  },
  {
    name: "Pranav Manikonda",
    role: "Research Assistant",
    imageUrl: "/images/pranav.png",
    linkedinUrl: "https://www.linkedin.com/in/pranav-manikonda/"
  }
];
</script>

<style scoped>
.leadership-member {
  @apply shadow-2xl;
}

.product-data-member {
  @apply shadow-xl;
}

.research-member {
  @apply shadow-lg;
}

/* Custom scale class since Tailwind doesn't have scale-120 by default */
.scale-120 {
  --tw-scale-x: 1.2;
  --tw-scale-y: 1.2;
  transform: translate(var(--tw-translate-x), var(--tw-translate-y)) rotate(var(--tw-rotate)) skewX(var(--tw-skew-x)) skewY(var(--tw-skew-y)) scaleX(var(--tw-scale-x)) scaleY(var(--tw-scale-y));
}
</style>