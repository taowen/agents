import { z } from "zod";
import type { ToolDescriptors } from "@cloudflare/codemode";

export const tools: ToolDescriptors = {
  getWeather: {
    description: "Get the current weather for a location",
    inputSchema: z.object({
      location: z.string().describe("The city name")
    }),
    execute: async () => ({ temperature: 72, condition: "sunny" })
  },
  searchWeb: {
    description: "Search the web for information",
    inputSchema: z.object({
      query: z.string().describe("The search query")
    }),
    execute: async () => ({
      results: [
        { title: "Example", url: "https://example.com", snippet: "An example" }
      ]
    })
  }
};
