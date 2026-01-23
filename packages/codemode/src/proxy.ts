import { getAgentByName } from "agents";
import { env, WorkerEntrypoint } from "cloudflare:workers";

export interface CodeModeProxyProps {
  binding: string;
  name: string;
  callback: string;
}

/**
 * WorkerEntrypoint that proxies tool calls from the sandboxed executor
 * back to your Agent's tools.
 */
export class CodeModeProxy extends WorkerEntrypoint<
  Cloudflare.Env,
  CodeModeProxyProps
> {
  async callFunction(options: { functionName: string; args: unknown }) {
    const stub = await getAgentByName(
      // @ts-expect-error - dynamic env access
      env[this.ctx.props.binding] as DurableObjectNamespace,
      this.ctx.props.name
    );
    // @ts-expect-error - dynamic method call
    return stub[this.ctx.props.callback](options.functionName, options.args);
  }
}
