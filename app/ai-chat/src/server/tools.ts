import * as Sentry from "@sentry/cloudflare";
import { tool, type ToolSet } from "ai";
import { z } from "zod";
import type { Bash } from "just-bash";
import type { DeviceHub } from "./device-hub";

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

// ---- Device exec tool (for device sessions using streamText) ----

/**
 * Create the execute_js tool that sends JS code to the connected device.
 * The tool description is dynamically built from the device-reported prompt.
 */
export function createDeviceExecTool(
  deviceHub: DeviceHub,
  toolDescription?: string
): ToolSet {
  return {
    execute_js: tool({
      description:
        toolDescription ||
        "Execute JavaScript code on the connected Android device. " +
          "The code runs in a Hermes runtime with access to accessibility APIs.",
      inputSchema: z.object({
        code: z
          .string()
          .describe(
            "JavaScript code to execute. All screen automation functions are available as globals."
          )
      }),
      execute: async ({ code }) => {
        const { result, screenshots } = await deviceHub.execOnDevice(code);
        return { result, screenshots };
      },
      toModelOutput: ({ output }) => {
        const parts: Array<
          | { type: "text"; text: string }
          | { type: "file-data"; data: string; mediaType: string }
        > = [{ type: "text", text: output.result }];
        for (const s of output.screenshots ?? []) {
          parts.push({ type: "file-data", data: s, mediaType: "image/jpeg" });
        }
        return { type: "content", value: parts };
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
