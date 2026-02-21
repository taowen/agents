import * as Sentry from "@sentry/cloudflare";
import { tool, type ToolSet } from "ai";
import { z } from "zod";
import type { Bash } from "just-bash";

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

// ---- Device tools ----

/** Check liveness of device sessions by querying each DO's /status endpoint. */
async function checkDeviceLiveness(
  env: Env,
  userId: string,
  sessions: { id: string; title: string }[]
): Promise<{ id: string; title: string; online: boolean }[]> {
  const results = await Promise.allSettled(
    sessions.map(async (s) => {
      const stub = env.ChatAgent.get(
        env.ChatAgent.idFromName(encodeURIComponent(`${userId}:${s.id}`))
      );
      const res = await stub.fetch(new Request("http://agent/status"));
      const body = (await res.json()) as { online: boolean };
      return { ...s, online: body.online };
    })
  );
  return results.map((r, i) =>
    r.status === "fulfilled" ? r.value : { ...sessions[i], online: false }
  );
}

export function createDeviceTools(env: Env, userId: string): ToolSet {
  return {
    list_devices: tool({
      description:
        "List online devices connected to this user's account. Returns an array of devices with deviceName and sessionId.",
      inputSchema: z.object({}),
      execute: async () => {
        const rows = await env.DB.prepare(
          "SELECT id, title FROM sessions WHERE user_id = ? AND id LIKE 'device-%'"
        )
          .bind(userId)
          .all<{ id: string; title: string }>();
        const withLiveness = await checkDeviceLiveness(
          env,
          userId,
          rows.results
        );
        return withLiveness
          .filter((r) => r.online)
          .map((r) => ({
            deviceName: r.id.replace("device-", ""),
            sessionId: r.id,
            title: r.title
          }));
      }
    }),
    device_agent: tool({
      description:
        "Dispatch a task to a connected Android device for execution. " +
        "The device has an AI agent that can interact with the phone UI (tap, scroll, type, read screen, open apps, etc.). " +
        "Use this when the user asks to do something on their phone/device, such as opening an app, searching, sending messages, etc. " +
        "The task description should be a clear natural language instruction of what to do on the device.",
      inputSchema: z.object({
        task: z
          .string()
          .describe(
            "Natural language description of the task to execute on the device"
          ),
        deviceName: z
          .string()
          .optional()
          .describe(
            "Specific device name to target. If omitted, the first available device is used."
          )
      }),
      execute: async ({ task, deviceName }) => {
        // Find the target device session
        let sessionId: string;
        if (deviceName) {
          sessionId = `device-${deviceName}`;
        } else {
          // Find first online device by checking DO liveness
          const rows = await env.DB.prepare(
            "SELECT id, title FROM sessions WHERE user_id = ? AND id LIKE 'device-%'"
          )
            .bind(userId)
            .all<{ id: string; title: string }>();
          const withLiveness = await checkDeviceLiveness(
            env,
            userId,
            rows.results
          );
          const onlineDevice = withLiveness.find((r) => r.online);
          if (!onlineDevice) {
            return { error: "No device available" };
          }
          sessionId = onlineDevice.id;
        }

        // Dispatch task directly to the device's ChatAgent session
        const isolatedName = encodeURIComponent(`${userId}:${sessionId}`);
        const doId = env.ChatAgent.idFromName(isolatedName);
        const stub = env.ChatAgent.get(doId);
        const res = await stub.fetch(
          new Request("http://agent/dispatch", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "x-user-id": userId
            },
            body: JSON.stringify({ task })
          })
        );
        return await res.json();
      }
    })
  };
}

export interface CreateToolsDeps {
  bashTool: ReturnType<typeof createBashTool>;
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
}

export function createTools(deps: CreateToolsDeps): ToolSet {
  const tools: ToolSet = {
    bash: deps.bashTool,
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

  return tools;
}
