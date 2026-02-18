import fs from "node:fs";
import { execFileSync } from "node:child_process";

export interface DriveInfo {
  mountPoint: string; // e.g. "/mnt/c"
  root: string; // e.g. "C:\\"
}

/**
 * Detect available Windows drives (A:-Z:) and WSL distros.
 * Synchronous — intended to be called once at startup.
 */
export function detectDrives(): DriveInfo[] {
  const drives: DriveInfo[] = [];

  // Scan A:-Z: for existing drives
  for (let code = 65; code <= 90; code++) {
    const letter = String.fromCharCode(code);
    const root = `${letter}:\\`;
    try {
      fs.statSync(root);
      drives.push({
        mountPoint: `/mnt/${letter.toLowerCase()}`,
        root
      });
    } catch {
      // Drive doesn't exist
    }
  }

  // Detect WSL distros
  try {
    const output = execFileSync("wsl", ["--list", "--quiet"], {
      encoding: "utf16le",
      windowsHide: true,
      timeout: 5000
    });
    const distros = output
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter(Boolean);
    if (distros.length > 0) {
      drives.push({
        mountPoint: "/mnt/wsl",
        root: `\\\\wsl.localhost\\${distros[0]}\\`
      });
    }
  } catch {
    // WSL not available
  }

  return drives;
}
