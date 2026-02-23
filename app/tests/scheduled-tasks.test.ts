/**
 * Scheduled tasks tests.
 *
 * Tests the scheduling tool execute functions from tools.ts.
 * Uses dependency injection (CreateToolsDeps) — no LLM needed.
 */
import { describe, it, expect } from "vitest";
import { createTools, type CreateToolsDeps } from "../ai-chat/src/server/tools";

function createMockDeps(overrides?: Partial<CreateToolsDeps>): CreateToolsDeps {
  return {
    bashTool: {} as ReturnType<
      typeof import("../ai-chat/src/server/tools").createBashTool
    >,
    schedule:
      overrides?.schedule ??
      (async (when, _method, _payload) => {
        const time =
          typeof when === "number"
            ? Math.floor(Date.now() / 1000) + when
            : typeof when === "string"
              ? Math.floor(Date.now() / 1000) + 60
              : Math.floor(when.getTime() / 1000);
        return {
          id: `sched-${crypto.randomUUID().slice(0, 8)}`,
          time
        };
      }),
    getSchedules: overrides?.getSchedules ?? (() => []),
    cancelSchedule: overrides?.cancelSchedule ?? (async () => false),
    getTimezone: overrides?.getTimezone ?? (async () => "UTC")
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function executeTool(
  tools: Record<string, any>,
  name: string,
  input: Record<string, unknown>
): Promise<unknown> {
  const tool = tools[name];
  if (!tool?.execute)
    throw new Error(`Tool ${name} not found or has no execute`);
  return tool.execute(input);
}

describe("Scheduled tasks", () => {
  it("schedule one-time task with delaySeconds", async () => {
    const tools = createTools(createMockDeps());
    const result = (await executeTool(tools, "schedule_task", {
      description: "Remind me",
      prompt: "Send a reminder",
      delaySeconds: 60
    })) as {
      success: boolean;
      id: string;
      scheduledAt: string;
      description: string;
    };

    expect(result.success).toBe(true);
    expect(result.id).toBeDefined();
    expect(result.scheduledAt).toBeDefined();
    expect(result.description).toBe("Remind me");
  });

  it("schedule one-time task with absolute scheduledAt", async () => {
    const futureDate = new Date(Date.now() + 3600_000).toISOString();
    const tools = createTools(
      createMockDeps({
        schedule: async (when, _method, _payload) => {
          const time =
            when instanceof Date ? Math.floor(when.getTime() / 1000) : 0;
          return { id: "sched-abs", time };
        }
      })
    );

    const result = (await executeTool(tools, "schedule_task", {
      description: "Future task",
      prompt: "Do something later",
      scheduledAt: futureDate
    })) as { success: boolean; id: string; scheduledAt: string };

    expect(result.success).toBe(true);
    expect(result.id).toBe("sched-abs");
    expect(result.scheduledAt).toBeDefined();
  });

  it("schedule recurring task with cron", async () => {
    const tools = createTools(createMockDeps());
    const result = (await executeTool(tools, "schedule_recurring", {
      description: "Daily standup",
      prompt: "Send standup reminder",
      cron: "0 9 * * *"
    })) as { success: boolean; id: string; cron: string; nextRun: string };

    expect(result.success).toBe(true);
    expect(result.id).toBeDefined();
    expect(result.cron).toBe("0 9 * * *");
    expect(result.nextRun).toBeDefined();
  });

  it("list tasks via manage_tasks", async () => {
    const tools = createTools(
      createMockDeps({
        getSchedules: () => [
          {
            id: "task-1",
            type: "scheduled",
            payload: JSON.stringify({
              description: "Reminder",
              prompt: "remind"
            }),
            time: Math.floor(Date.now() / 1000) + 60
          },
          {
            id: "task-2",
            type: "cron",
            payload: { description: "Daily check", prompt: "check" },
            time: Math.floor(Date.now() / 1000) + 120,
            cron: "0 9 * * *"
          }
        ]
      })
    );

    const result = (await executeTool(tools, "manage_tasks", {
      action: "list"
    })) as Array<{ id: string; description: string; cron?: string }>;

    expect(result).toHaveLength(2);
    expect(result[0].id).toBe("task-1");
    expect(result[0].description).toBe("Reminder");
    expect(result[1].cron).toBe("0 9 * * *");
  });

  it("cancel existing task", async () => {
    const tools = createTools(
      createMockDeps({
        cancelSchedule: async (id) => id === "task-1"
      })
    );

    const result = (await executeTool(tools, "manage_tasks", {
      action: "cancel",
      taskId: "task-1"
    })) as { success: boolean; cancelled: string };

    expect(result.success).toBe(true);
    expect(result.cancelled).toBe("task-1");
  });

  it("cancel non-existent task returns error", async () => {
    const tools = createTools(
      createMockDeps({
        cancelSchedule: async () => false
      })
    );

    const result = (await executeTool(tools, "manage_tasks", {
      action: "cancel",
      taskId: "nonexistent"
    })) as { error: string };

    expect(result.error).toBe("Task not found");
  });

  it("validation: negative delaySeconds returns error", async () => {
    const tools = createTools(createMockDeps());
    const result = (await executeTool(tools, "schedule_task", {
      description: "Bad task",
      prompt: "fail",
      delaySeconds: -10
    })) as { error: string };

    expect(result.error).toBe("delaySeconds must be positive");
  });

  it("validation: past scheduledAt returns error", async () => {
    const tools = createTools(createMockDeps());
    const result = (await executeTool(tools, "schedule_task", {
      description: "Past task",
      prompt: "fail",
      scheduledAt: "2020-01-01T00:00:00Z"
    })) as { error: string };

    expect(result.error).toBe("Scheduled time must be in the future");
  });
});
