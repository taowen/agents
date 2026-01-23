import { getAgentByName } from "agents";
import { env, WorkerEntrypoint } from "cloudflare:workers";

export interface CodeModeProxyProps {
  /** The Durable Object binding name (e.g., "MyAgent") */
  binding: string;
  /** The agent instance name */
  name: string;
  /** The method name on the agent to call for tool execution */
  callback: string;
}

/**
 * WorkerEntrypoint that proxies tool calls from the sandbox back to the Agent.
 *
 * Usage:
 * 1. Re-export from your module: `export { CodeModeProxy } from "@cloudflare/codemode"`
 * 2. Add service binding in wrangler.jsonc
 * 3. Define a callTool method on your Agent
 * 4. Pass to createCodeTool via ctx.exports.CodeModeProxy({props: {...}})
 */
export class CodeModeProxy extends WorkerEntrypoint<
  Cloudflare.Env,
  CodeModeProxyProps
> {
  async callFunction(options: { functionName: string; args: unknown }) {
    const stub = (await getAgentByName(
      // @ts-expect-error - dynamic binding access
      env[this.ctx.props.binding],
      this.ctx.props.name
    )) as DurableObjectStub;

    // @ts-expect-error - dynamic method call
    return stub[this.ctx.props.callback](options.functionName, options.args);
  }
}
