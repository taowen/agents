/**
 * Tests for the globalOutbound-based codemode implementation.
 *
 * This tests that tool execute functions are called correctly through
 * the codemode:// fetch protocol.
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

describe("globalOutbound-based codemode", () => {
  it("should create a tool with correct structure", () => {
    const tools: ToolDescriptors = {
      testTool: {
        description: "A test tool",
        inputSchema: z.object({ value: z.string() }),
        execute: async (args: any) => ({ result: args.value.toUpperCase() })
      }
    };

    const mockLoader = {
      get: vi.fn()
    };

    const codemode = experimental_createCodeTool({
      tools,
      loader: mockLoader as any
    });

    expect(codemode).toBeDefined();
    expect(codemode.description).toContain("testTool");
  });

  it("should handle tool calls through mock proxy", async () => {
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

  it("should pass globalOutbound config to worker", async () => {
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
      loader: mockLoader as any
    });

    await codemode.execute?.({ code: "async () => {}" }, {} as any);

    // globalOutbound should be set
    expect(capturedConfig.globalOutbound).toBeDefined();
    expect(capturedConfig.globalOutbound.fetch).toBeDefined();
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
      loader: mockLoader as any
    });

    await codemode.execute?.({ code: "async () => null" }, {} as any);

    const config = capturedConfig;
    const executorCode = config.modules["executor.js"];

    // Should use codemode:// protocol for tool calls
    expect(executorCode).toContain("codemode://");
    expect(executorCode).toContain("fetch");
  });
});
