import { AsyncLocalStorage } from "node:async_hooks";
import type { Connection } from "partyserver";

export type AgentContextStore = {
  // Using unknown to avoid circular dependency with Agent
  agent: unknown;
  connection: Connection | undefined;
  request: Request | undefined;
  email: unknown;
};

/**
 * @internal — This is an internal implementation detail.
 * Importing or relying on this symbol **will** break your code in a future release.
 */
export const __DO_NOT_USE_WILL_BREAK__agentContext =
  new AsyncLocalStorage<AgentContextStore>();
