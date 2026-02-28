import { Button } from "@cloudflare/kumo";
import { SunIcon, MoonIcon } from "@phosphor-icons/react";
import { useTheme } from "./hooks";

/* ── ConnectionStatus ──
   Displays a colored dot and label for WebSocket connection state. */

export type ConnectionStatus = "connecting" | "connected" | "disconnected";

interface ConnectionStatusProps {
  status: ConnectionStatus;
}

const statusConfig: Record<
  ConnectionStatus,
  { label: string; dotClass: string; textClass: string }
> = {
  connecting: {
    label: "Connecting...",
    dotClass: "bg-yellow-500",
    textClass: "text-kumo-warning"
  },
  connected: {
    label: "Connected",
    dotClass: "bg-green-500",
    textClass: "text-kumo-success"
  },
  disconnected: {
    label: "Disconnected",
    dotClass: "bg-red-500",
    textClass: "text-kumo-danger"
  }
};

export function ConnectionIndicator({ status }: ConnectionStatusProps) {
  const { label, dotClass, textClass } = statusConfig[status];
  return (
    <div className="flex items-center gap-2" role="status" aria-live="polite">
      <span className={`size-2 rounded-full ${dotClass}`} aria-hidden="true" />
      <span className={textClass}>{label}</span>
    </div>
  );
}

/* ── ModeToggle ──
   Toggles between light and dark theme modes.
   Requires <ThemeProvider> from "./hooks". */

export function ModeToggle() {
  const { mode, setMode } = useTheme();

  const toggle = () => {
    setMode(mode === "light" ? "dark" : "light");
  };

  const icon =
    mode === "light" ? <SunIcon size={16} /> : <MoonIcon size={16} />;

  return (
    <Button
      variant="secondary"
      icon={icon}
      onClick={toggle}
      title={mode === "light" ? "Light" : "Dark"}
    />
  );
}
