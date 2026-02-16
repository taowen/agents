/**
 * df - report file system disk space usage
 *
 * Usage: df [-h] [FILE]...
 *
 * In sandboxed environment, reports virtual filesystem mount points.
 * Sizes are 0 since this is an in-memory filesystem.
 */

import type { Command, CommandContext, ExecResult } from "../../types.js";
import { hasHelpFlag, showHelp } from "../help.js";

const dfHelp = {
  name: "df",
  summary: "report file system disk space usage",
  usage: "df [OPTION]... [FILE]...",
  options: [
    "-h          print sizes in human readable format",
    "    --help  display this help and exit"
  ]
};

interface MountInfo {
  mountPoint: string;
  fsType: string;
}

function getMountInfo(ctx: CommandContext): MountInfo[] {
  const mounts: MountInfo[] = [{ mountPoint: "/", fsType: "memfs" }];

  // Check if fs has getMounts (MountableFs)
  const fs = ctx.fs as {
    getMounts?: () => ReadonlyArray<{ mountPoint: string }>;
  };
  if (typeof fs.getMounts === "function") {
    for (const m of fs.getMounts()) {
      mounts.push({ mountPoint: m.mountPoint, fsType: "agentfs" });
    }
  }

  return mounts;
}

function padRight(s: string, len: number): string {
  return s.length >= len ? s : s + " ".repeat(len - s.length);
}

function padLeft(s: string, len: number): string {
  return s.length >= len ? s : " ".repeat(len - s.length) + s;
}

export const dfCommand: Command = {
  name: "df",

  async execute(args: string[], ctx: CommandContext): Promise<ExecResult> {
    if (hasHelpFlag(args)) {
      return showHelp(dfHelp);
    }

    // Parse flags
    let humanReadable = false;
    for (const arg of args) {
      if (arg === "-h") {
        humanReadable = true;
      } else if (arg.startsWith("-") && arg !== "--") {
        return {
          stdout: "",
          stderr: `df: invalid option -- '${arg.slice(1)}'\n`,
          exitCode: 1
        };
      }
    }

    const mounts = getMountInfo(ctx);

    const sizeCol = humanReadable ? "Size" : "1K-blocks";
    let stdout =
      padRight("Filesystem", 15) +
      padLeft(sizeCol, 10) +
      padLeft("Used", 10) +
      padLeft("Available", 10) +
      " Use%" +
      " Mounted on\n";

    for (const m of mounts) {
      stdout +=
        padRight(m.fsType, 15) +
        padLeft("0", 10) +
        padLeft("0", 10) +
        padLeft("0", 10) +
        "   0%" +
        " " +
        m.mountPoint +
        "\n";
    }

    return { stdout, stderr: "", exitCode: 0 };
  }
};

import type { CommandFuzzInfo } from "../fuzz-flags-types.js";

export const flagsForFuzzing: CommandFuzzInfo = {
  name: "df",
  flags: [{ flag: "-h", type: "boolean" }]
};
