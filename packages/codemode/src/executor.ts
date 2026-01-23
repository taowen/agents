import type { CodeModeProxy } from "./proxy";

interface ExecutorEntrypoint {
  evaluate(): Promise<{ result: unknown; err?: string; stack?: string }>;
}

export interface CodeExecutorOptions {
  loader: WorkerLoader;
  proxy: Fetcher<CodeModeProxy>;
}

/**
 * Create a sandboxed code executor that runs user-generated code
 * in an isolated Worker with access to tools via the proxy.
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

    const codemode = new Proxy({}, {
      get: (_, toolName) => {
        return (args) => CodeModeProxy.callFunction({
          functionName: toolName,
          args
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
        }
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
