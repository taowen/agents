import type { CodeModeProxy } from "./proxy";

interface ExecutorEntrypoint {
  evaluate(): Promise<{ result: unknown; err?: string; stack?: string }>;
}

export interface CodeExecutorOptions {
  loader: WorkerLoader;
  /** Service binding to CodeModeProxy for tool RPC */
  proxy: Fetcher<CodeModeProxy>;
  /** Optional outbound fetch handler to filter requests. Set to null to block all outbound. */
  globalOutbound?: Fetcher | null;
}

/**
 * Create a sandboxed code executor that runs user-generated code
 * in an isolated Worker. Tools are called via the CodeModeProxy binding.
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
    const { CodeModeProxy } = this.env;

    // Create codemode proxy that routes tool calls through CodeModeProxy
    const codemode = new Proxy({}, {
      get: (_, toolName) => async (args) => {
        return CodeModeProxy.callFunction({
          functionName: String(toolName),
          args: args ?? {}
        });
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
        env: {
          CodeModeProxy: options.proxy
        },
        // null blocks all outbound, undefined allows all (if user wants filtering, they pass a Fetcher)
        globalOutbound: options.globalOutbound ?? null
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
