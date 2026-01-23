/* eslint-disable */
declare namespace Cloudflare {
  interface GlobalProps {
    mainModule: typeof import("./src/server");
    durableNamespaces: "Codemode";
  }
  interface Env {
    Codemode: DurableObjectNamespace;
    LOADER: WorkerLoader;
  }
}
interface Env extends Cloudflare.Env {}
