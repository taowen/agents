/**
 * Tests for the proxy-based codemode implementation.
 *
 * This tests that tool execute functions are called correctly through
 * the CodeModeProxy service binding.
 */
import { describe, it, expect, vi } from "vitest";
import { experimental_createCodeTool } from "../tool";
import { z } from "zod";
import type { ToolDescriptors } from "../types";

// Helper to create a mock proxy that routes to tools
function createMockProxy(tools: ToolDescriptors) {
  return {
    callFunction: async (options: { functionName: string; args: unknown }) => {
      const tool = tools[options.functionName];
      if (!tool?.execute) {
        throw new Error(`Tool ${options.functionName} not found`);
      }
      return tool.execute(options.args);
    }
  };
}

// Mock WorkerLoader that simulates sandbox execution with proxy
function createMockLoader(mockProxy: ReturnType<typeof createMockProxy>) {
  return {
    get: vi.fn((name: string, factory: () => any) => {
      const config = factory();

      return {
        getEntrypoint: () => ({
          evaluate: async () => {
            // In a real sandbox, the code would call CodeModeProxy.callFunction
            // Here we simulate that by making the codemode proxy available
            const codemode = new Proxy(
              {},
              {
                get: (_, toolName) => async (args: unknown) => {
                  // Route through the mock proxy (simulates env.CodeModeProxy)
                  return mockProxy.callFunction({
                    functionName: String(toolName),
                    args: args ?? {}
                  });
                }
              }
            );

            // Return the config for inspection (in real execution, code runs)
            return {
              result: { codemode, config }
            };
          }
        })
      };
    })
  };
}

describe("proxy-based codemode", () => {
  it("should create a tool with correct structure", () => {
    const tools: ToolDescriptors = {
      testTool: {
        description: "A test tool",
        inputSchema: z.object({ value: z.string() }),
        execute: async (args: any) => ({ result: args.value.toUpperCase() })
      }
    };

    const mockProxy = createMockProxy(tools);
    const mockLoader = createMockLoader(mockProxy);

    const codemode = experimental_createCodeTool({
      tools,
      loader: mockLoader as any,
      proxy: mockProxy as any
    });

    expect(codemode).toBeDefined();
    expect(codemode.description).toContain("testTool");
  });

  it("should handle tool calls through proxy", async () => {
    const tools: ToolDescriptors = {
      greet: {
        description: "Greet someone",
        inputSchema: z.object({ name: z.string() }),
        execute: async (args: any) => ({ greeting: `Hello, ${args.name}!` })
      }
    };

    const mockProxy = createMockProxy(tools);

    // Verify proxy routes to tool correctly
    const result = await mockProxy.callFunction({
      functionName: "greet",
      args: { name: "World" }
    });

    expect(result).toEqual({ greeting: "Hello, World!" });
  });

  it("should preserve closures in execute functions", async () => {
    // This is the key test - closures should work
    const secret = "secret-value-123";

    const tools: ToolDescriptors = {
      getSecret: {
        description: "Get a secret value",
        inputSchema: z.object({}),
        // This execute function closes over `secret`
        execute: async () => ({ secret })
      }
    };

    const mockProxy = createMockProxy(tools);

    const result = await mockProxy.callFunction({
      functionName: "getSecret",
      args: {}
    });

    expect(result).toEqual({ secret: "secret-value-123" });
  });

  it("should handle errors in tool execution", async () => {
    const tools: ToolDescriptors = {
      failingTool: {
        description: "A tool that fails",
        inputSchema: z.object({}),
        execute: async () => {
          throw new Error("Intentional failure");
        }
      }
    };

    const mockProxy = createMockProxy(tools);

    await expect(
      mockProxy.callFunction({
        functionName: "failingTool",
        args: {}
      })
    ).rejects.toThrow("Intentional failure");
  });

  it("should handle missing tools", async () => {
    const tools: ToolDescriptors = {};
    const mockProxy = createMockProxy(tools);

    await expect(
      mockProxy.callFunction({
        functionName: "nonexistent",
        args: {}
      })
    ).rejects.toThrow("Tool nonexistent not found");
  });

  it("should pass globalOutbound to worker config when provided", async () => {
    const tools: ToolDescriptors = {
      dummy: {
        description: "Dummy",
        inputSchema: z.object({}),
        execute: async () => ({})
      }
    };

    const userGlobalOutbound = {
      fetch: async (input: any, init: any) => {
        return new Response(JSON.stringify({ proxied: true }));
      }
    };

    let capturedConfig: any = null;
    const mockLoader = {
      get: vi.fn((name: string, factory: () => any) => {
        capturedConfig = factory();
        return {
          getEntrypoint: () => ({
            evaluate: async () => ({ result: "test" })
          })
        };
      })
    };

    const codemode = experimental_createCodeTool({
      tools,
      loader: mockLoader as any,
      proxy: createMockProxy(tools) as any,
      globalOutbound: userGlobalOutbound as any
    });

    await codemode.execute?.({ code: "async () => {}" }, {} as any);

    // globalOutbound should be passed through
    expect(capturedConfig.globalOutbound).toBe(userGlobalOutbound);
  });

  it("should default globalOutbound to null when not provided", async () => {
    const tools: ToolDescriptors = {
      dummy: {
        description: "Dummy",
        inputSchema: z.object({}),
        execute: async () => ({})
      }
    };

    let capturedConfig: any = null;
    const mockLoader = {
      get: vi.fn((name: string, factory: () => any) => {
        capturedConfig = factory();
        return {
          getEntrypoint: () => ({
            evaluate: async () => ({ result: "test" })
          })
        };
      })
    };

    const codemode = experimental_createCodeTool({
      tools,
      loader: mockLoader as any,
      proxy: createMockProxy(tools) as any
      // No globalOutbound provided
    });

    await codemode.execute?.({ code: "async () => {}" }, {} as any);

    // globalOutbound should be null (blocks all outbound)
    expect(capturedConfig.globalOutbound).toBeNull();
  });

  it("should generate correct sandbox code structure", async () => {
    const tools: ToolDescriptors = {
      testTool: {
        description: "Test",
        inputSchema: z.object({}),
        execute: async () => ({})
      }
    };

    let capturedConfig: any = null;
    const mockLoader = {
      get: vi.fn((name: string, factory: () => any) => {
        capturedConfig = factory();
        return {
          getEntrypoint: () => ({
            evaluate: async () => ({ result: null })
          })
        };
      })
    };

    const codemode = experimental_createCodeTool({
      tools,
      loader: mockLoader as any,
      proxy: createMockProxy(tools) as any
    });

    await codemode.execute?.({ code: "async () => null" }, {} as any);

    const config = capturedConfig;
    const executorCode = config.modules["executor.js"];

    // Should use CodeModeProxy for tool calls
    expect(executorCode).toContain("CodeModeProxy");
    expect(executorCode).toContain("callFunction");
    // Should NOT use the old codemode:// protocol
    expect(executorCode).not.toContain("codemode://");
  });

  it("should pass CodeModeProxy in env", async () => {
    const tools: ToolDescriptors = {
      dummy: {
        description: "Dummy",
        inputSchema: z.object({}),
        execute: async () => ({})
      }
    };

    const mockProxy = createMockProxy(tools);

    let capturedConfig: any = null;
    const mockLoader = {
      get: vi.fn((name: string, factory: () => any) => {
        capturedConfig = factory();
        return {
          getEntrypoint: () => ({
            evaluate: async () => ({ result: null })
          })
        };
      })
    };

    const codemode = experimental_createCodeTool({
      tools,
      loader: mockLoader as any,
      proxy: mockProxy as any
    });

    await codemode.execute?.({ code: "async () => null" }, {} as any);

    // CodeModeProxy should be in env
    expect(capturedConfig.env.CodeModeProxy).toBe(mockProxy);
  });
});
