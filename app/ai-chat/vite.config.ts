import { cloudflare } from "@cloudflare/vite-plugin";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { sentryVitePlugin } from "@sentry/vite-plugin";
import { defineConfig } from "vite";
import { readFileSync, existsSync } from "node:fs";

const envSentryPath = new URL(".env.sentry", import.meta.url);
if (existsSync(envSentryPath)) {
  for (const line of readFileSync(envSentryPath, "utf-8").split("\n")) {
    const match = line.match(/^\s*([^#=]+?)\s*=\s*(.*?)\s*$/);
    if (match) process.env[match[1]] = match[2];
  }
}

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
      project: "cloudflare-worker"
    })
  ],
  build: {
    sourcemap: true
  }
});
