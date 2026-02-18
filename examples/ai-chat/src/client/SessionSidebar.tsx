import { useState } from "react";
import {
  PlusIcon,
  GearIcon,
  BrainIcon,
  TrashIcon,
  ChatCircleIcon,
  SignOutIcon,
  PencilSimpleIcon,
  MonitorIcon,
  ArrowsClockwiseIcon,
  TerminalWindowIcon
} from "@phosphor-icons/react";
import type { BridgeDevice } from "./use-bridge-viewer";
import { Button, Text } from "@cloudflare/kumo";

export interface SessionInfo {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

interface UserInfo {
  id: string;
  email: string;
  name: string | null;
  picture: string | null;
}

const DEVICE_STATUS_COLORS: Record<string, string> = {
  connected: "#22c55e",
  running: "#3b82f6"
};

interface SessionSidebarProps {
  sessions: SessionInfo[];
  activeSessionId: string | null;
  user: UserInfo | null;
  onNewSession: () => void;
  onSelectSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, title: string) => void;
  onOpenSettings: () => void;
  onOpenMemory: () => void;
  devices?: BridgeDevice[];
  showActivityPanel?: boolean;
  onToggleActivityPanel?: () => void;
  onResetAgent?: () => void;
}

export function SessionSidebar({
  sessions,
  activeSessionId,
  user,
  onNewSession,
  onSelectSession,
  onDeleteSession,
  onRenameSession,
  onOpenSettings,
  onOpenMemory,
  devices,
  showActivityPanel,
  onToggleActivityPanel,
  onResetAgent
}: SessionSidebarProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");

  const startEditing = (session: SessionInfo) => {
    setEditingId(session.id);
    setEditTitle(session.title);
  };

  const commitEdit = () => {
    if (editingId && editTitle.trim()) {
      onRenameSession(editingId, editTitle.trim());
    }
    setEditingId(null);
  };

  const cancelEdit = () => {
    setEditingId(null);
  };

  return (
    <div className="w-64 h-screen flex flex-col bg-kumo-base border-r border-kumo-line">
      {/* New Chat button */}
      <div className="p-3">
        <Button
          variant="secondary"
          size="sm"
          icon={<PlusIcon size={14} />}
          onClick={onNewSession}
          className="w-full justify-start"
        >
          New Chat
        </Button>
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto px-2 space-y-0.5">
        {sessions.map((session) => {
          const isActive = session.id === activeSessionId;
          const isEditing = editingId === session.id;
          return (
            <div
              key={session.id}
              className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer text-sm transition-colors ${
                isActive
                  ? "bg-kumo-elevated text-kumo-default"
                  : "text-kumo-secondary hover:bg-kumo-elevated/50 hover:text-kumo-default"
              }`}
              onClick={() => !isEditing && onSelectSession(session.id)}
            >
              <ChatCircleIcon
                size={14}
                className="shrink-0 text-kumo-inactive"
              />
              {isEditing ? (
                <input
                  className="flex-1 min-w-0 bg-kumo-base border border-kumo-line rounded px-1 py-0.5 text-sm text-kumo-default outline-none focus:border-kumo-brand"
                  value={editTitle}
                  onChange={(e) => setEditTitle(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") commitEdit();
                    if (e.key === "Escape") cancelEdit();
                  }}
                  onBlur={commitEdit}
                  onClick={(e) => e.stopPropagation()}
                  autoFocus
                />
              ) : (
                <span className="flex-1 min-w-0 truncate">{session.title}</span>
              )}
              {!isEditing && (
                <>
                  <button
                    className="shrink-0 opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-kumo-line text-kumo-inactive hover:text-kumo-default transition-opacity"
                    onClick={(e) => {
                      e.stopPropagation();
                      startEditing(session);
                    }}
                  >
                    <PencilSimpleIcon size={12} />
                  </button>
                  <button
                    className="shrink-0 opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-kumo-line text-kumo-inactive hover:text-kumo-default transition-opacity"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteSession(session.id);
                    }}
                  >
                    <TrashIcon size={12} />
                  </button>
                </>
              )}
            </div>
          );
        })}
        {sessions.length === 0 && (
          <div className="px-3 py-4 text-center">
            <Text size="xs" variant="secondary">
              No sessions yet
            </Text>
          </div>
        )}
      </div>

      {/* Bottom section */}
      <div className="border-t border-kumo-line p-3 space-y-2">
        {/* Device list + activity toggle */}
        {devices && devices.length > 0 && (
          <div className="space-y-1">
            <div className="flex items-center gap-2 px-3 py-1">
              <MonitorIcon size={14} className="shrink-0 text-kumo-inactive" />
              <span className="text-xs font-medium text-kumo-secondary">
                Devices ({devices.length})
              </span>
            </div>
            {devices.map((device) => (
              <div
                key={device.deviceName}
                className="flex items-center gap-2 px-3 py-1"
              >
                <span
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    backgroundColor:
                      DEVICE_STATUS_COLORS[device.status] ?? "#22c55e",
                    display: "inline-block",
                    flexShrink: 0
                  }}
                />
                <span className="flex-1 text-xs text-kumo-secondary truncate">
                  {device.deviceName}
                </span>
              </div>
            ))}
            <div className="flex gap-1 px-3">
              <button
                onClick={onToggleActivityPanel}
                className={`flex items-center gap-1.5 flex-1 px-2 py-1.5 rounded text-xs transition-colors ${
                  showActivityPanel
                    ? "bg-kumo-brand/10 text-kumo-brand"
                    : "text-kumo-secondary hover:bg-kumo-elevated hover:text-kumo-default"
                }`}
              >
                <TerminalWindowIcon size={12} />
                <span>Activity</span>
              </button>
              {onResetAgent && (
                <button
                  onClick={onResetAgent}
                  className="flex items-center gap-1.5 px-2 py-1.5 rounded text-xs text-kumo-secondary hover:bg-kumo-elevated hover:text-kumo-default transition-colors"
                  title="Reset local agent session"
                >
                  <ArrowsClockwiseIcon size={12} />
                  <span>Reset</span>
                </button>
              )}
            </div>
          </div>
        )}

        {/* Memory link */}
        <button
          onClick={onOpenMemory}
          className="flex items-center gap-2 w-full px-3 py-2 rounded-lg text-sm text-kumo-secondary hover:bg-kumo-elevated hover:text-kumo-default transition-colors"
        >
          <BrainIcon size={14} />
          <span>Memory</span>
        </button>

        {/* Settings link */}
        <button
          onClick={onOpenSettings}
          className="flex items-center gap-2 w-full px-3 py-2 rounded-lg text-sm text-kumo-secondary hover:bg-kumo-elevated hover:text-kumo-default transition-colors"
        >
          <GearIcon size={14} />
          <span>Settings</span>
        </button>

        {/* User info + logout */}
        {user && (
          <div className="flex items-center gap-2 px-3 py-2">
            {user.picture ? (
              <img
                src={user.picture}
                alt=""
                className="w-6 h-6 rounded-full shrink-0"
                referrerPolicy="no-referrer"
              />
            ) : (
              <div className="w-6 h-6 rounded-full bg-kumo-elevated shrink-0" />
            )}
            <span className="flex-1 text-xs text-kumo-secondary truncate">
              {user.name || user.email}
            </span>
            <a
              href="/auth/logout"
              className="p-1 rounded hover:bg-kumo-elevated text-kumo-inactive hover:text-kumo-default transition-colors"
              title="Sign out"
            >
              <SignOutIcon size={14} />
            </a>
          </div>
        )}
      </div>
    </div>
  );
}
