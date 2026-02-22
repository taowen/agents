import { execSync } from "child_process";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = __dirname;
const aiChatIndex = resolve(__dirname, "../ai-chat/src/server/index.ts");

// 1. Get short git commit hash
const hash = execSync("git rev-parse --short HEAD", {
  encoding: "utf-8"
}).trim();
const zipName = `connect-screen-win-${hash}.zip`;
const zipPath = resolve(projectRoot, zipName);
console.log(`Packaging: ${zipName}`);

// 2. Create zip with all PowerShell scripts
console.log("Creating zip archive...");
execSync(`zip -j ${zipPath} connect.ps1 functions.ps1`, {
  cwd: projectRoot,
  stdio: "inherit"
});
// Add scripts/ subdirectory preserving structure
execSync(`zip ${zipPath} scripts/*.ps1`, {
  cwd: projectRoot,
  stdio: "inherit"
});

if (!existsSync(zipPath)) {
  console.error(`Zip not found at ${zipPath}`);
  process.exit(1);
}
console.log("Zip created successfully.");

// 3. Upload to R2
console.log(`Uploading to R2 as ${zipName}...`);
execSync(
  `npx wrangler r2 object put ai-chat-public/${zipName} --file=${zipPath} --content-type=application/zip --remote`,
  { stdio: "inherit" }
);

// 4. Clean up local zip
execSync(`rm ${zipPath}`);

// 5. Update winZipUrl in ai-chat server
const indexSrc = readFileSync(aiChatIndex, "utf-8");
const updated = indexSrc.replace(
  /const winZipUrl\s*=\s*\n?\s*"https:\/\/ai\.connect-screen\.com\/api\/public\/connect-screen-win[^"]*\.zip"/,
  `const winZipUrl =\n    "https://ai.connect-screen.com/api/public/${zipName}"`
);
if (updated === indexSrc) {
  console.error(
    "Failed to update winZipUrl in ai-chat/src/server/index.ts — pattern not found."
  );
  process.exit(1);
}
writeFileSync(aiChatIndex, updated);
console.log(`Updated winZipUrl in ai-chat/src/server/index.ts → ${zipName}`);

// 6. Summary
console.log(
  `\nDone! Zip: ${zipName} | Download: https://ai.connect-screen.com/download`
);
console.log("Remember to deploy ai-chat for the link update to take effect:");
console.log("  npm run deploy -w app/ai-chat");
