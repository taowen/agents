import { defineWorkersConfig } from "@cloudflare/vitest-pool-workers/config";

export default defineWorkersConfig({
  test: {
    name: "app-tests",
    deps: {
      optimizer: {
        ssr: {
          include: ["ajv"]
        }
      }
    },
    poolOptions: {
      workers: {
        isolatedStorage: false,
        singleWorker: true,
        wrangler: {
          configPath: "./wrangler.jsonc"
        }
      }
    }
  }
});
