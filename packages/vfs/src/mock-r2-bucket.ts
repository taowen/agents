/**
 * MockR2Bucket — in-memory implementation of the Cloudflare R2Bucket interface.
 *
 * Used in tests to avoid a real R2 binding. Implements the subset of R2Bucket
 * methods that R2FsAdapter actually uses: head, get, put, delete, list.
 */

interface StoredObject {
  key: string;
  body: Uint8Array;
  customMetadata?: Record<string, string>;
  uploaded: Date;
  size: number;
}

export class MockR2Bucket {
  private objects = new Map<string, StoredObject>();

  async head(key: string): Promise<R2Object | null> {
    const obj = this.objects.get(key);
    if (!obj) return null;
    return this.toR2Object(obj);
  }

  async get(key: string): Promise<R2ObjectBody | null> {
    const obj = this.objects.get(key);
    if (!obj) return null;
    return this.toR2ObjectBody(obj);
  }

  async put(
    key: string,
    value:
      | ReadableStream
      | ArrayBuffer
      | ArrayBufferView
      | string
      | null
      | Blob,
    options?: R2PutOptions
  ): Promise<R2Object> {
    let body: Uint8Array;
    if (value === null) {
      body = new Uint8Array(0);
    } else if (typeof value === "string") {
      body = new TextEncoder().encode(value);
    } else if (value instanceof Uint8Array) {
      body = new Uint8Array(value);
    } else if (value instanceof ArrayBuffer) {
      body = new Uint8Array(value);
    } else if (ArrayBuffer.isView(value)) {
      body = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
    } else {
      // ReadableStream or Blob — simplify for tests
      body = new Uint8Array(0);
    }

    const stored: StoredObject = {
      key,
      body,
      customMetadata: options?.customMetadata,
      uploaded: new Date(),
      size: body.length
    };
    this.objects.set(key, stored);
    return this.toR2Object(stored);
  }

  async delete(keys: string | string[]): Promise<void> {
    const keyList = Array.isArray(keys) ? keys : [keys];
    for (const k of keyList) {
      this.objects.delete(k);
    }
  }

  async list(options?: R2ListOptions): Promise<R2Objects> {
    const prefix = options?.prefix || "";
    const delimiter = options?.delimiter;
    const limit = options?.limit ?? 1000;
    const cursor = options?.cursor;

    // Get all matching keys, sorted
    let allKeys = [...this.objects.keys()]
      .filter((k) => k.startsWith(prefix))
      .sort();

    // Handle cursor (cursor = key to start after)
    if (cursor) {
      const idx = allKeys.indexOf(cursor);
      if (idx >= 0) {
        allKeys = allKeys.slice(idx + 1);
      }
    }

    const objects: R2Object[] = [];
    const delimitedPrefixes: string[] = [];
    const seenPrefixes = new Set<string>();

    for (const key of allKeys) {
      if (objects.length + delimitedPrefixes.length >= limit) break;

      if (delimiter) {
        const rest = key.slice(prefix.length);
        const delimIdx = rest.indexOf(delimiter);
        if (delimIdx >= 0) {
          const dp = prefix + rest.slice(0, delimIdx + delimiter.length);
          if (!seenPrefixes.has(dp)) {
            seenPrefixes.add(dp);
            delimitedPrefixes.push(dp);
          }
          continue;
        }
      }

      objects.push(this.toR2Object(this.objects.get(key)!));
    }

    const truncated = allKeys.length > limit;
    const lastKey =
      objects.length > 0 ? objects[objects.length - 1].key : undefined;

    return {
      objects,
      delimitedPrefixes,
      truncated,
      cursor: truncated ? lastKey : undefined
    } as unknown as R2Objects;
  }

  // ---- Helpers ----

  private toR2Object(stored: StoredObject): R2Object {
    return {
      key: stored.key,
      size: stored.size,
      uploaded: stored.uploaded,
      customMetadata: stored.customMetadata || {},
      httpMetadata: {},
      etag: "mock-etag",
      version: "mock-version",
      httpEtag: '"mock-etag"',
      checksums: { toJSON: () => ({}) },
      storageClass: "Standard",
      writeHttpMetadata: () => {}
    } as unknown as R2Object;
  }

  private toR2ObjectBody(stored: StoredObject): R2ObjectBody {
    const r2Obj = this.toR2Object(stored);
    const bodyData = stored.body;
    return {
      ...r2Obj,
      body: new ReadableStream({
        start(controller) {
          controller.enqueue(bodyData);
          controller.close();
        }
      }),
      bodyUsed: false,
      arrayBuffer: async () =>
        bodyData.buffer.slice(
          bodyData.byteOffset,
          bodyData.byteOffset + bodyData.byteLength
        ),
      text: async () => new TextDecoder().decode(bodyData),
      json: async () => JSON.parse(new TextDecoder().decode(bodyData)),
      blob: async () => new Blob([bodyData])
    } as unknown as R2ObjectBody;
  }
}
