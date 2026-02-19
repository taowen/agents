#!/usr/bin/env node
/**
 * Bundle the Electron main process with esbuild, then copy assets
 * needed at runtime (preload script, PowerShell scripts).
 */
import { build } from "esbuild";
import { cpSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const DIST = join(ROOT, "dist-electron");

// 1. Clean
rmSync(DIST, { recursive: true, force: true });
mkdirSync(DIST, { recursive: true });

// 2. Bundle main process
await build({
  entryPoints: [join(ROOT, "electron/main.ts")],
  bundle: true,
  platform: "node",
  format: "esm",
  outfile: join(DIST, "main.mjs"),
  external: ["electron"]
});

// 3. Copy preload script
cpSync(join(ROOT, "electron/preload.cjs"), join(DIST, "preload.cjs"));

// 4. Copy PowerShell scripts
cpSync(join(ROOT, "electron/scripts"), join(DIST, "scripts"), {
  recursive: true
});

// 5. Write minimal package.json for electron-builder
writeFileSync(
  join(DIST, "package.json"),
  JSON.stringify(
    {
      name: "windows-agent",
      version: "1.0.0",
      description: "Windows Agent desktop app",
      author: "Connect Screen",
      private: true,
      type: "module",
      main: "main.mjs"
    },
    null,
    2
  )
);

console.log("Electron build complete → dist-electron/");
