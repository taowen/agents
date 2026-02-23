/**
 * Upload pre-quantized ASR model (.qmodel) to R2 public bucket.
 *
 * Uses the S3-compatible API with @aws-sdk/lib-storage for multipart upload.
 * Small files (<300MB) use wrangler; large files use the S3 API.
 *
 * Required env vars for large files:
 *   R2_ACCESS_KEY_ID     — R2 S3 API token Access Key ID
 *   R2_SECRET_ACCESS_KEY — R2 S3 API token Secret Access Key
 *
 * Create these at: https://dash.cloudflare.com/<account>/r2/api-tokens
 *
 * Usage:
 *   npx tsx scripts/upload-asr-model.ts /path/to/model-dir
 */

import { readFileSync, statSync, createReadStream } from "fs";
import { resolve } from "path";
import { execSync } from "child_process";
import { S3Client } from "@aws-sdk/client-s3";
import { Upload } from "@aws-sdk/lib-storage";

const ACCOUNT_ID = "488e1103da07b07978ee032a9cac809e";
const BUCKET = "ai-chat-public";
const PREFIX = "qwen3-asr-0.6b";
const S3_ENDPOINT = `https://${ACCOUNT_ID}.r2.cloudflarestorage.com`;

function uploadSmallWithWrangler(
  key: string,
  filePath: string,
  contentType: string
) {
  console.log(`  wrangler put ${key}...`);
  execSync(
    `npx wrangler r2 object put "${BUCKET}/${key}" --file="${filePath}" --content-type="${contentType}" --remote`,
    { stdio: "inherit" }
  );
  console.log(`  ✓ ${key}`);
}

async function uploadLargeWithS3(
  key: string,
  filePath: string,
  contentType: string
) {
  const accessKeyId = process.env.R2_ACCESS_KEY_ID;
  const secretAccessKey = process.env.R2_SECRET_ACCESS_KEY;

  if (!accessKeyId || !secretAccessKey) {
    console.error("\nError: Large file upload requires R2 S3 credentials.");
    console.error("Set these environment variables:");
    console.error("  export R2_ACCESS_KEY_ID=...");
    console.error("  export R2_SECRET_ACCESS_KEY=...");
    console.error(
      "\nCreate them at: https://dash.cloudflare.com/" +
        ACCOUNT_ID +
        "/r2/api-tokens"
    );
    process.exit(1);
  }

  const client = new S3Client({
    region: "auto",
    endpoint: S3_ENDPOINT,
    credentials: { accessKeyId, secretAccessKey }
  });

  const fileSize = statSync(filePath).size;
  const sizeMB = (fileSize / 1024 / 1024).toFixed(0);

  console.log(`  S3 multipart upload: ${key} (${sizeMB} MB)...`);
  const upload = new Upload({
    client,
    params: {
      Bucket: BUCKET,
      Key: key,
      Body: createReadStream(filePath),
      ContentType: contentType
    },
    queueSize: 1,
    partSize: 100 * 1024 * 1024 // 100 MiB
  });

  upload.on("httpUploadProgress", (progress) => {
    if (progress.loaded) {
      const pct = progress.total
        ? ((progress.loaded / progress.total) * 100).toFixed(0)
        : "?";
      const loadedMB = (progress.loaded / 1024 / 1024).toFixed(0);
      process.stdout.write(`\r  Progress: ${loadedMB}/${sizeMB} MB (${pct}%)`);
    }
  });

  await upload.done();
  process.stdout.write("\n");
  console.log(`  ✓ ${key}`);
}

async function main() {
  const modelDir = process.argv[2];
  if (!modelDir) {
    console.error("Usage: npx tsx scripts/upload-asr-model.ts <model-dir>");
    console.error("  model-dir should contain vocab.json and model.qmodel");
    process.exit(1);
  }

  console.log(`Uploading ASR model to R2: ${BUCKET}/${PREFIX}/\n`);

  // Upload vocab.json (small — use wrangler)
  const vocabPath = resolve(modelDir, "vocab.json");
  uploadSmallWithWrangler(
    `${PREFIX}/vocab.json`,
    vocabPath,
    "application/json"
  );

  // Upload model.qmodel (large — use S3 multipart)
  const qmodelPath = resolve(modelDir, "model.qmodel");
  await uploadLargeWithS3(
    `${PREFIX}/model.qmodel`,
    qmodelPath,
    "application/octet-stream"
  );

  console.log(`\nDone! Files uploaded to ${BUCKET}/${PREFIX}/`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
