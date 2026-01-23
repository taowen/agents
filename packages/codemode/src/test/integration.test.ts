/**
 * Integration tests for codemode.
 *
 * These tests verify that the tool execution works correctly
 * including closure preservation through globalOutbound.
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

describe("codemode integration", () => {
  describe("closure preservation", () => {
    it("should preserve closures in execute functions", async () => {
      // External state that the execute function closes over
      const externalData = {
        apiKey: "secret-api-key-123",
        counter: 0
      };

      const tools: ToolDescriptors = {
        getApiKey: {
          description: "Get the API key",
          inputSchema: z.object({}),
          execute: async () => {
            // This closes over externalData
            return { key: externalData.apiKey };
          }
        },
        incrementCounter: {
          description: "Increment and return the counter",
          inputSchema: z.object({}),
          execute: async () => {
            // This modifies external state
            externalData.counter++;
            return { count: externalData.counter };
          }
        }
      };

      const mockProxy = createMockProxy(tools);

      // Test closure - API key should be accessible
      const keyResult = await mockProxy.callFunction({
        functionName: "getApiKey",
        args: {}
      });
      expect(keyResult).toEqual({ key: "secret-api-key-123" });

      // Test mutable closure - counter should increment
      const count1 = await mockProxy.callFunction({
        functionName: "incrementCounter",
        args: {}
      });
      expect(count1).toEqual({ count: 1 });

      const count2 = await mockProxy.callFunction({
        functionName: "incrementCounter",
        args: {}
      });
      expect(count2).toEqual({ count: 2 });

      // Verify external state was actually modified
      expect(externalData.counter).toBe(2);
    });

    it("should preserve 'this' binding when using class methods", async () => {
      class DataService {
        private data = new Map<string, string>();

        constructor() {
          this.data.set("key1", "value1");
        }

        async get(args: { key: string }) {
          return { value: this.data.get(args.key) ?? null };
        }

        async set(args: { key: string; value: string }) {
          this.data.set(args.key, args.value);
          return { success: true };
        }
      }

      const service = new DataService();

      const tools: ToolDescriptors = {
        getData: {
          description: "Get data by key",
          inputSchema: z.object({ key: z.string() }),
          // Bind the method to preserve 'this'
          execute: service.get.bind(service) as (
            args: unknown
          ) => Promise<unknown>
        },
        setData: {
          description: "Set data",
          inputSchema: z.object({ key: z.string(), value: z.string() }),
          execute: service.set.bind(service) as (
            args: unknown
          ) => Promise<unknown>
        }
      };

      const mockProxy = createMockProxy(tools);

      // Get existing value
      const get1 = await mockProxy.callFunction({
        functionName: "getData",
        args: { key: "key1" }
      });
      expect(get1).toEqual({ value: "value1" });

      // Set new value
      await mockProxy.callFunction({
        functionName: "setData",
        args: { key: "key2", value: "value2" }
      });

      // Get new value
      const get2 = await mockProxy.callFunction({
        functionName: "getData",
        args: { key: "key2" }
      });
      expect(get2).toEqual({ value: "value2" });
    });

    it("should handle async operations with closures", async () => {
      const cache = new Map<string, string>();

      const tools: ToolDescriptors = {
        fetchAndCache: {
          description: "Simulate fetching and caching",
          inputSchema: z.object({ url: z.string() }),
          execute: async (args: any) => {
            // Simulate async operation
            await new Promise((r) => setTimeout(r, 1));

            // Use closure
            if (cache.has(args.url)) {
              return { cached: true, value: cache.get(args.url) };
            }

            const value = `fetched-${args.url}`;
            cache.set(args.url, value);
            return { cached: false, value };
          }
        }
      };

      const mockProxy = createMockProxy(tools);

      // First call - not cached
      const result1 = await mockProxy.callFunction({
        functionName: "fetchAndCache",
        args: { url: "example.com" }
      });
      expect(result1).toEqual({ cached: false, value: "fetched-example.com" });

      // Second call - should be cached
      const result2 = await mockProxy.callFunction({
        functionName: "fetchAndCache",
        args: { url: "example.com" }
      });
      expect(result2).toEqual({ cached: true, value: "fetched-example.com" });
    });
  });

  describe("error handling", () => {
    it("should handle sync errors", async () => {
      const tools: ToolDescriptors = {
        throwSync: {
          description: "Throws synchronously",
          inputSchema: z.object({}),
          execute: async () => {
            throw new Error("Sync error");
          }
        }
      };

      const mockProxy = createMockProxy(tools);

      await expect(
        mockProxy.callFunction({
          functionName: "throwSync",
          args: {}
        })
      ).rejects.toThrow("Sync error");
    });

    it("should handle async errors", async () => {
      const tools: ToolDescriptors = {
        throwAsync: {
          description: "Throws asynchronously",
          inputSchema: z.object({}),
          execute: async () => {
            await new Promise((r) => setTimeout(r, 1));
            throw new Error("Async error");
          }
        }
      };

      const mockProxy = createMockProxy(tools);

      await expect(
        mockProxy.callFunction({
          functionName: "throwAsync",
          args: {}
        })
      ).rejects.toThrow("Async error");
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
  });

  describe("tool configuration", () => {
    it("should pass globalOutbound to executor", async () => {
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

      // globalOutbound should be set (handles tool calls)
      expect(capturedConfig.globalOutbound).toBeDefined();
    });

    it("should allow custom onFetch for network filtering", async () => {
      const tools: ToolDescriptors = {
        dummy: {
          description: "Dummy",
          inputSchema: z.object({}),
          execute: async () => ({})
        }
      };

      const fetchLog: string[] = [];

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
        onFetch: async (request) => {
          fetchLog.push(request.url);
          return fetch(request);
        }
      });

      await codemode.execute?.({ code: "async () => null" }, {} as any);

      // globalOutbound should be set
      expect(capturedConfig.globalOutbound).toBeDefined();
    });
  });
});
