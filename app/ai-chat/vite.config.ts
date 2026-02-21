import { cloudflare } from "@cloudflare/vite-plugin";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { sentryVitePlugin } from "@sentry/vite-plugin";
import { defineConfig } from "vite";
import dotenv from "dotenv";

dotenv.config({ path: ".env.sentry" });

export default defineConfig({
  define: {
    __SENTRY_DSN__: JSON.stringify(process.env.SENTRY_DSN || "")
  },
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
