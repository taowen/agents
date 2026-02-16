/**
 * uname - print system information
 *
 * Usage: uname [OPTION]...
 *
 * In sandboxed environment, returns virtual system info.
 */

import type { Command, CommandContext, ExecResult } from "../../types.js";
import { hasHelpFlag, showHelp } from "../help.js";

const unameHelp = {
  name: "uname",
  summary: "print system information",
  usage: "uname [OPTION]...",
  options: [
    "-a          print all information",
    "-s          print the kernel name",
    "-n          print the network node hostname",
    "-r          print the kernel release",
    "-m          print the machine hardware name",
    "-o          print the operating system",
    "    --help  display this help and exit"
  ]
};

const KERNEL = "Linux";
const NODENAME = "localhost";
const RELEASE = "6.0.0-virtual";
const MACHINE = "x86_64";
const OS = "GNU/Linux";

export const unameCommand: Command = {
  name: "uname",

  async execute(args: string[], _ctx: CommandContext): Promise<ExecResult> {
    if (hasHelpFlag(args)) {
      return showHelp(unameHelp);
    }

    // Check for unknown options
    for (const arg of args) {
      if (arg.startsWith("-") && arg !== "--") {
        for (let i = 1; i < arg.length; i++) {
          const ch = arg[i];
          if (!"asnrmo".includes(ch)) {
            return {
              stdout: "",
              stderr: `uname: invalid option -- '${ch}'\n`,
              exitCode: 1
            };
          }
        }
      }
    }

    // Collect flags
    let flagS = false;
    let flagN = false;
    let flagR = false;
    let flagM = false;
    let flagO = false;

    for (const arg of args) {
      if (!arg.startsWith("-") || arg === "--") continue;
      for (let i = 1; i < arg.length; i++) {
        switch (arg[i]) {
          case "a":
            flagS = flagN = flagR = flagM = flagO = true;
            break;
          case "s":
            flagS = true;
            break;
          case "n":
            flagN = true;
            break;
          case "r":
            flagR = true;
            break;
          case "m":
            flagM = true;
            break;
          case "o":
            flagO = true;
            break;
        }
      }
    }

    // Default: print kernel name (like -s)
    if (!flagS && !flagN && !flagR && !flagM && !flagO) {
      flagS = true;
    }

    const parts: string[] = [];
    if (flagS) parts.push(KERNEL);
    if (flagN) parts.push(NODENAME);
    if (flagR) parts.push(RELEASE);
    if (flagM) parts.push(MACHINE);
    if (flagO) parts.push(OS);

    return { stdout: parts.join(" ") + "\n", stderr: "", exitCode: 0 };
  }
};

import type { CommandFuzzInfo } from "../fuzz-flags-types.js";

export const flagsForFuzzing: CommandFuzzInfo = {
  name: "uname",
  flags: [
    { flag: "-a", type: "boolean" },
    { flag: "-s", type: "boolean" },
    { flag: "-n", type: "boolean" },
    { flag: "-r", type: "boolean" },
    { flag: "-m", type: "boolean" },
    { flag: "-o", type: "boolean" }
  ]
};
