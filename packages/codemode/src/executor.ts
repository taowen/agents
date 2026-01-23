interface ExecutorEntrypoint {
  evaluate(): Promise<{ result: unknown; err?: string; stack?: string }>;
}

export interface CodeExecutorOptions {
  loader: WorkerLoader;
  /** globalOutbound that handles tool calls and optionally other requests */
  globalOutbound: Fetcher;
}

/**
 * Create a sandboxed code executor that runs user-generated code
 * in an isolated Worker. Tools are called via fetch to codemode:// URLs.
 */
export function createCodeExecutor(options: CodeExecutorOptions) {
  return async (code: string): Promise<unknown> => {
    const worker = options.loader.get(
      `codemode-${crypto.randomUUID()}`,
      () => ({
        compatibilityDate: "2025-06-01",
        compatibilityFlags: ["nodejs_compat"],
        mainModule: "executor.js",
        modules: {
          "executor.js": `
import { WorkerEntrypoint } from "cloudflare:workers";

export default class CodeExecutor extends WorkerEntrypoint {
  async evaluate() {
    // Create codemode proxy that calls tools via fetch
    const codemode = new Proxy({}, {
      get: (_, toolName) => async (args) => {
        const res = await fetch(\`codemode://\${String(toolName)}\`, {
          method: "POST",
          body: JSON.stringify(args ?? {})
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        return data.result;
      }
    });

    try {
      const result = await (${code})();
      return { result };
    } catch (err) {
      return { result: undefined, err: err.message, stack: err.stack };
    }
  }
}
        `
        },
        globalOutbound: options.globalOutbound
      })
    );

    const entrypoint = worker.getEntrypoint() as unknown as ExecutorEntrypoint;
    const response = await entrypoint.evaluate();

    if (response.err) {
      throw new Error(response.err);
    }

    return response.result;
  };
}
