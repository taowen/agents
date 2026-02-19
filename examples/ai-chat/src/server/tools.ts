import * as Sentry from "@sentry/cloudflare";
import { tool, type ToolSet } from "ai";
import { z } from "zod";
import type { Bash } from "just-bash";
import { createBrowserTool, type BrowserState } from "./browser-tool";

export function createBashTool(bash: Bash, ensureMounted: () => Promise<void>) {
  return tool({
    description:
      "Execute a bash command in a sandboxed virtual filesystem. " +
      "Supports ls, grep, awk, sed, find, cat, echo, mkdir, cp, mv, sort, uniq, wc, head, tail, curl, and more. " +
      "Use curl to fetch content from URLs. Files persist across commands within the session.",
    inputSchema: z.object({
      command: z.string().describe("The bash command to execute")
    }),
    execute: async ({ command }) => {
      return Sentry.startSpan(
        { name: `bash: ${command.slice(0, 80)}`, op: "tool.bash" },
        async () => {
          await Sentry.startSpan({ name: "ensureMounted", op: "mount" }, () =>
            ensureMounted()
          );
          return await Sentry.startSpan({ name: "bash.exec", op: "exec" }, () =>
            bash.exec(command)
          );
        }
      );
    }
  });
}

/**
 * Fetch the list of connected remote desktop devices from BridgeManager DO.
 */
export type BridgeDevicesCache = {
  data: { deviceName: string }[];
  fetchedAt: number;
} | null;

const BRIDGE_DEVICES_TTL = 30_000; // 30 seconds

export async function getAvailableBridgeDevices(
  env: Env,
  userId: string,
  cache: BridgeDevicesCache
): Promise<{ data: { deviceName: string }[]; cache: BridgeDevicesCache }> {
  const now = Date.now();
  if (cache && now - cache.fetchedAt < BRIDGE_DEVICES_TTL) {
    return { data: cache.data, cache };
  }
  try {
    const id = env.BridgeManager.idFromName(userId);
    const stub = env.BridgeManager.get(id);
    const resp = await stub.fetch(
      new Request("http://bridge/devices", {
        method: "GET",
        headers: { "x-partykit-room": userId }
      })
    );
    const data = (await resp.json()) as { deviceName: string }[];
    const newCache = { data, fetchedAt: now };
    return { data, cache: newCache };
  } catch (e) {
    console.error("getAvailableBridgeDevices:", e);
    Sentry.captureException(e);
    return { data: [], cache };
  }
}

/**
 * Send a message to a remote desktop agent via BridgeManager and wait for the response.
 */
export async function sendToRemoteDesktop(
  env: Env,
  userId: string,
  deviceName: string,
  content: string
): Promise<string> {
  const id = env.BridgeManager.idFromName(userId);
  const stub = env.BridgeManager.get(id);
  const resp = await stub.fetch(
    new Request("http://bridge/message", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-partykit-room": userId
      },
      body: JSON.stringify({ deviceName, content })
    })
  );
  const data = (await resp.json()) as { response?: string; error?: string };
  if (data.error) {
    return `[Error] ${data.error}`;
  }
  return data.response || "[No response from remote desktop]";
}

export interface CreateToolsDeps {
  bashTool: ReturnType<typeof createBashTool>;
  browserState: BrowserState;
  mybrowser: Fetcher;
  schedule: (
    when: Date | number | string,
    method: any,
    payload: any
  ) => Promise<{ id: string; time: number }>;
  getSchedules: () => Array<{
    id: string;
    type: string;
    payload: any;
    time: number;
    cron?: string;
  }>;
  cancelSchedule: (id: string) => Promise<boolean>;
  getTimezone: () => Promise<string>;
  bridgeDevices: { deviceName: string }[];
  env: Env;
  userId: string;
}

export function createTools(deps: CreateToolsDeps): ToolSet {
  const tools: ToolSet = {
    bash: deps.bashTool,
    browser: createBrowserTool(deps.browserState, deps.mybrowser),
    schedule_task: tool({
      description:
        "Schedule a one-time task. Provide EITHER delaySeconds (e.g. 60 for 1 minute) " +
        "OR scheduledAt (ISO 8601 datetime). Prefer delaySeconds for relative times like 'in 5 minutes'.",
      inputSchema: z.object({
        description: z.string().describe("Brief description of the task"),
        prompt: z
          .string()
          .describe(
            "The detailed prompt/instruction for the AI to execute when the task fires"
          ),
        delaySeconds: z
          .number()
          .optional()
          .describe(
            "Delay in seconds from now. e.g. 60 = 1 minute, 3600 = 1 hour. Use this for relative times."
          ),
        scheduledAt: z
          .string()
          .optional()
          .describe(
            "ISO 8601 datetime string for absolute time, e.g. '2025-06-01T09:00:00Z'"
          )
      }),
      execute: async ({ description, prompt, delaySeconds, scheduledAt }) => {
        let when: Date | number;
        if (delaySeconds != null) {
          if (delaySeconds <= 0)
            return { error: "delaySeconds must be positive" };
          when = delaySeconds;
        } else if (scheduledAt) {
          when = new Date(scheduledAt);
          if (when.getTime() <= Date.now())
            return { error: "Scheduled time must be in the future" };
        } else {
          return { error: "Provide either delaySeconds or scheduledAt" };
        }
        const tz = await deps.getTimezone();
        const s = await deps.schedule(when, "executeScheduledTask" as any, {
          description,
          prompt,
          timezone: tz
        });
        return {
          success: true,
          id: s.id,
          scheduledAt: new Date(s.time * 1000).toISOString(),
          description
        };
      }
    }),
    schedule_recurring: tool({
      description:
        "Schedule a recurring task using a cron expression. " +
        "Examples: '0 9 * * *' = daily at 9am UTC, '0 */2 * * *' = every 2 hours, '0 9 * * 1-5' = weekdays at 9am UTC.",
      inputSchema: z.object({
        description: z
          .string()
          .describe("Brief description of the recurring task"),
        prompt: z
          .string()
          .describe(
            "The detailed prompt/instruction for the AI to execute each time"
          ),
        cron: z
          .string()
          .describe(
            "Cron expression (5 fields: minute hour day-of-month month day-of-week)"
          )
      }),
      execute: async ({ description, prompt, cron }) => {
        const tz = await deps.getTimezone();
        const s = await deps.schedule(cron, "executeScheduledTask" as any, {
          description,
          prompt,
          timezone: tz
        });
        return {
          success: true,
          id: s.id,
          cron,
          description,
          nextRun: new Date(s.time * 1000).toISOString()
        };
      }
    }),
    manage_tasks: tool({
      description: "List all scheduled tasks, or cancel a specific task by ID.",
      inputSchema: z.object({
        action: z.enum(["list", "cancel"]).describe("Action to perform"),
        taskId: z
          .string()
          .optional()
          .describe("Task ID to cancel (required for cancel action)")
      }),
      execute: async ({ action, taskId }) => {
        if (action === "list") {
          const schedules = deps.getSchedules();
          return schedules.map((s) => {
            let description = "";
            try {
              const p =
                typeof s.payload === "string"
                  ? JSON.parse(s.payload)
                  : s.payload;
              description = p.description || "";
            } catch {}
            return {
              id: s.id,
              type: s.type,
              description,
              nextRun: new Date(s.time * 1000).toISOString(),
              ...(s.type === "cron" ? { cron: (s as any).cron } : {})
            };
          });
        }
        if (action === "cancel" && taskId) {
          const ok = await deps.cancelSchedule(taskId);
          return ok
            ? { success: true, cancelled: taskId }
            : { error: "Task not found" };
        }
        return { error: "Invalid action or missing taskId" };
      }
    })
  };

  // Add remote_desktop tool when bridge devices are available
  if (deps.bridgeDevices.length > 0) {
    tools.remote_desktop = tool({
      description:
        "Send a message to a connected remote desktop agent. " +
        "The remote agent can see the screen, control mouse/keyboard, and execute PowerShell commands. " +
        "It has a cloud:\\ drive that maps to your persistent filesystem (cloud:\\home\\user = /home/user, cloud:\\data = /data). " +
        "Use it to transfer files between cloud and Windows, or to run Windows-native tasks. " +
        "It maintains conversation context across calls — you can give follow-up instructions. " +
        "Describe what you want done in natural language. Returns the agent's text response.",
      inputSchema: z.object({
        message: z
          .string()
          .describe("What to do on the remote desktop, in natural language"),
        device: z
          .string()
          .optional()
          .describe("Device name (omit if only one device)")
      }),
      execute: async ({ message, device }) => {
        const targetDevice =
          device ||
          (deps.bridgeDevices.length === 1
            ? deps.bridgeDevices[0].deviceName
            : "default");
        return sendToRemoteDesktop(
          deps.env,
          deps.userId,
          targetDevice,
          message
        );
      }
    });
  }

  return tools;
}
