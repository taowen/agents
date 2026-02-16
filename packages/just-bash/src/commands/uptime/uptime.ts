/**
 * uptime - tell how long the system has been running
 *
 * Usage: uptime
 *
 * In sandboxed environment, returns static uptime info with current time.
 */

import type { Command, CommandContext, ExecResult } from "../../types.js";

async function uptimeExecute(
  _args: string[],
  _ctx: CommandContext
): Promise<ExecResult> {
  const now = new Date();
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");
  return {
    stdout: ` ${hh}:${mm}:${ss} up 0 min,  1 user,  load average: 0.00, 0.00, 0.00\n`,
    stderr: "",
    exitCode: 0
  };
}

export const uptimeCommand: Command = {
  name: "uptime",
  execute: uptimeExecute
};

import type { CommandFuzzInfo } from "../fuzz-flags-types.js";

export const flagsForFuzzing: CommandFuzzInfo = {
  name: "uptime",
  flags: []
};
