import { execSync } from "child_process";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, "..");
const aiChatIndex = resolve(__dirname, "../../ai-chat/src/server/index.ts");

// 1. Get short git commit hash
const hash = execSync("git rev-parse --short HEAD", {
  encoding: "utf-8"
}).trim();
const apkName = `connect-screen-${hash}.apk`;
console.log(`Building APK: ${apkName}`);

// 2. Build release APK
console.log("Running assembleRelease...");
execSync("./gradlew assembleRelease --quiet", {
  cwd: resolve(projectRoot, "android"),
  stdio: "inherit"
});

// 3. Verify APK exists
const apkPath = resolve(
  projectRoot,
  "android/app/build/outputs/apk/release/app-release.apk"
);
if (!existsSync(apkPath)) {
  console.error(`APK not found at ${apkPath}`);
  process.exit(1);
}
console.log("APK built successfully.");

// 4. Upload to R2
console.log(`Uploading to R2 as ${apkName}...`);
execSync(
  `npx wrangler r2 object put ai-chat-public/${apkName} --file=${apkPath} --content-type=application/vnd.android.package-archive`,
  { stdio: "inherit" }
);

// 5. Update apkUrl in ai-chat server
const indexSrc = readFileSync(aiChatIndex, "utf-8");
const updated = indexSrc.replace(
  /const apkUrl\s*=\s*\n?\s*"https:\/\/ai\.connect-screen\.com\/api\/public\/connect-screen[^"]*\.apk"/,
  `const apkUrl =\n    "https://ai.connect-screen.com/api/public/${apkName}"`
);
if (updated === indexSrc) {
  console.error(
    "Failed to update apkUrl in ai-chat/src/server/index.ts — pattern not found."
  );
  process.exit(1);
}
writeFileSync(aiChatIndex, updated);
console.log(`Updated apkUrl in ai-chat/src/server/index.ts → ${apkName}`);

// 6. Summary
console.log(
  `\nDone! APK: ${apkName} | Download: https://ai.connect-screen.com/download`
);
console.log("Remember to deploy ai-chat for the link update to take effect:");
console.log("  npm run deploy -w app/ai-chat");
