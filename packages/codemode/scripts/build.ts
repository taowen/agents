import { build } from "tsdown";

async function main() {
  await build({
    clean: true,
    dts: true,
    entry: ["src/index.ts", "src/ai.ts"],
    external: ["cloudflare:workers", "agents"],
    format: "esm",
    sourcemap: true
  });

  process.exit(0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
