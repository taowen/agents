import { cloudflare } from "@cloudflare/vite-plugin";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { sentryVitePlugin } from "@sentry/vite-plugin";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [
    react(),
    cloudflare(),
    tailwindcss(),
    sentryVitePlugin({
      authToken: process.env.SENTRY_SOURCEMAP_AUTH_TOKEN,
      org: "txom",
      project: "cloudflare-worker",
      url: "https://us.sentry.io"
    })
  ],
  build: {
    sourcemap: true
  }
});
