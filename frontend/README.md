# Synaptic Engine Frontend

The frontend application for Synaptic Engine, built with Nuxt 3 and TailwindCSS. This web application provides a modern, responsive interface for interacting with the Synaptic Engine backend.

## Technologies

- [Nuxt 3](https://nuxt.com/) - The Vue.js Framework
- [TailwindCSS](https://tailwindcss.com/) - Utility-first CSS framework
- [HeadlessUI](https://headlessui.com/) - Unstyled, fully accessible UI components
- [Chart.js](https://www.chartjs.org/) - Data visualization
- [Muse.js](https://github.com/urish/muse-js) - Brain-Computer Interface integration

## Setup

1. Install dependencies:
```bash
npm install
```

2. Create environment file:
```bash
cp example.env .env
```
Edit `.env` with your configuration values.

## Development

Start the development server on `http://localhost:3000`:

```bash
npx nuxt dev
```

The application will be available at `http://localhost:3000`. The development server includes:
- Hot Module Replacement (HMR)
- Vue DevTools integration (press Shift + Option + D in browser)
- TailwindCSS class autocomplete
- TypeScript support

## Production

Build the application for production:

```bash
npm run build
```

Preview the production build locally:

```bash
npm run preview
```

For deployment information, see the [Nuxt deployment documentation](https://nuxt.com/docs/getting-started/deployment).

## Project Structure

- `components/` - Reusable Vue components
- `pages/` - Application routes and page components
- `public/` - Static assets (images, etc.)
- `server/` - API routes and server middleware
- `types/` - TypeScript type definitions
