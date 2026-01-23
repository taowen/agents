/* eslint-disable */
declare namespace Cloudflare {
  interface GlobalProps {
    mainModule: typeof import("./src/server");
    durableNamespaces: "Codemode";
  }
  interface Env {
    Codemode: DurableObjectNamespace;
    LOADER: WorkerLoader;
    CodeModeProxy: Service<typeof import("./src/server").CodeModeProxy>;
    globalOutbound: Service<typeof import("./src/server").globalOutbound>;
  }
}
interface Env extends Cloudflare.Env {}
