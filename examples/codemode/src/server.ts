import { routeAgentRequest, Agent, callable, type Connection } from "agents";
import { getSchedulePrompt } from "agents/schedule";
import { experimental_createCodeTool } from "@cloudflare/codemode";
import {
  streamText,
  type UIMessage,
  stepCountIs,
  convertToModelMessages,
  readUIMessageStream,
  generateId
} from "ai";
import { openai } from "@ai-sdk/openai";
import { tools } from "./tools";

// inline this until enable_ctx_exports is supported by default
declare global {
  interface ExecutionContext<Props = unknown> {
    readonly exports: Cloudflare.Exports;
    readonly props: Props;
  }

  interface DurableObjectState<Props = unknown> {
    readonly exports: Cloudflare.Exports;
  }
}

const model = openai("gpt-5");

type State = {
  messages: UIMessage[];
  loading: boolean;
};

export class Codemode extends Agent<Env, State> {
  observability = undefined;
  lastMessageRepliedTo: string | undefined;

  initialState: State = {
    messages: [],
    loading: false
  };

  async onStart() {
    this.lastMessageRepliedTo =
      this.state.messages[this.state.messages.length - 1]?.id;
  }

  @callable({ description: "Add an MCP server to the agent" })
  addMcp({ name, url }: { name: string; url: string }) {
    void this.addMcpServer(name, url, "http://localhost:5173")
      .then(() => console.log("mcpServer added", name, url))
      .catch((error) => console.error("mcpServer addition failed", error));
  }

  @callable({ description: "Remove an MCP server from the agent" })
  removeMcp(id: string) {
    void this.removeMcpServer(id);
  }

  async onStateUpdate(state: State, source: Connection | "server") {
    if (source === "server") return;

    const lastMessage = state.messages[state.messages.length - 1];
    if (
      state.messages.length > 0 &&
      this.lastMessageRepliedTo !== lastMessage?.id
    ) {
      await this.onChatMessage();
      this.lastMessageRepliedTo = lastMessage?.id;
    }
  }

  async onChatMessage() {
    this.setState({ messages: this.state.messages, loading: true });

    const codemode = experimental_createCodeTool({
      tools,
      loader: this.env.LOADER,
      // Optional: allow specific outbound requests
      onFetch: async (request) => {
        const url = new URL(request.url);
        // Block requests to example.com/sub-path
        if (url.hostname === "example.com" && url.pathname === "/sub-path") {
          return null; // Block
        }
        return fetch(request); // Allow
      }
    });

    const result = streamText({
      system: `You are a helpful assistant that can execute code to achieve goals.

When you need to perform multiple operations or complex logic, use the codemode tool.
The codemode tool lets you write JavaScript code that calls available functions.

${getSchedulePrompt({ date: new Date() })}
`,
      messages: await convertToModelMessages(this.state.messages),
      model,
      tools: { codemode },
      onError: (error) => console.error("error", error),
      stopWhen: stepCountIs(10)
    });

    for await (const uiMessage of readUIMessageStream<UIMessage>({
      stream: result.toUIMessageStream({ generateMessageId: generateId }),
      onError: (error) => console.error("error", error)
    })) {
      this.setState({
        messages: updateMessages(this.state.messages, uiMessage),
        loading: this.state.loading
      });
    }

    this.setState({ messages: this.state.messages, loading: false });
  }
}

function updateMessages(messages: UIMessage[], newMessage: UIMessage) {
  const index = messages.findIndex((m) => m.id === newMessage.id);
  if (index >= 0) {
    return [
      ...messages.slice(0, index),
      newMessage,
      ...messages.slice(index + 1)
    ];
  }
  return [...messages, newMessage];
}

export default {
  async fetch(request: Request, env: Env, _ctx: ExecutionContext) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
