/**
 * id - print real and effective user and group IDs
 *
 * Usage: id
 *
 * In sandboxed environment, returns static uid/gid info.
 */

import type { Command, CommandContext, ExecResult } from "../../types.js";

async function idExecute(
  _args: string[],
  _ctx: CommandContext
): Promise<ExecResult> {
  return {
    stdout: "uid=1000(user) gid=1000(user) groups=1000(user)\n",
    stderr: "",
    exitCode: 0
  };
}

export const idCommand: Command = {
  name: "id",
  execute: idExecute
};

import type { CommandFuzzInfo } from "../fuzz-flags-types.js";

export const flagsForFuzzing: CommandFuzzInfo = {
  name: "id",
  flags: []
};
