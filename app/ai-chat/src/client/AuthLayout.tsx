import { useState, useEffect, useCallback } from "react";
import { Outlet, useNavigate, useParams, useOutletContext } from "react-router";
import { SessionSidebar } from "./SessionSidebar";
import { BugReportFloat } from "./BugReportFloat";
import { useSessions, useDevices } from "./api";
import type { UserInfo } from "./api";

interface RootContext {
  user: UserInfo;
}

export interface AuthLayoutContext {
  user: UserInfo;
  onOpenSidebar: () => void;
  onOpenBugReport: () => void;
  onFirstMessage: (text: string) => void;
}

export function AuthLayout() {
  const { user } = useOutletContext<RootContext>();
  const { sessions, isLoading, createSession, deleteSession, renameSession } =
    useSessions();
  const { devices } = useDevices();
  const navigate = useNavigate();
  const { sessionId: activeSessionId } = useParams<{ sessionId: string }>();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [bugReportOpen, setBugReportOpen] = useState(false);

  // Global keyboard shortcut: Ctrl+Shift+B / Cmd+Shift+B
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (
        e.shiftKey &&
        (e.ctrlKey || e.metaKey) &&
        e.key.toLowerCase() === "b"
      ) {
        e.preventDefault();
        setBugReportOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const handleOpenBugReport = useCallback(() => setBugReportOpen(true), []);
  const handleOpenSidebar = useCallback(() => setSidebarOpen(true), []);

  const handleNewSession = async () => {
    try {
      const session = await createSession();
      navigate("/s/" + session.id);
    } catch (e) {
      console.error("Failed to create session:", e);
    }
  };

  const handleDeleteSession = async (id: string) => {
    try {
      // Check for pending scheduled tasks before deleting
      const schedRes = await fetch(`/api/sessions/${id}/schedules`);
      if (schedRes.ok) {
        const schedules = (await schedRes.json()) as {
          id: string;
          type: string;
          payload?: string | { description?: string };
          time: number;
          cron?: string;
        }[];
        if (schedules.length > 0) {
          const lines = schedules.map((s) => {
            let desc = "";
            try {
              const p =
                typeof s.payload === "string"
                  ? JSON.parse(s.payload)
                  : s.payload;
              desc = p?.description || "unnamed task";
            } catch {
              desc = "unnamed task";
            }
            const when = new Date(s.time * 1000).toLocaleString();
            return `- ${desc} (${s.type === "cron" ? `cron: ${s.cron}` : when})`;
          });
          const confirmed = window.confirm(
            `This session has scheduled tasks that will be cancelled:\n\n${lines.join("\n")}\n\nContinue?`
          );
          if (!confirmed) return;
        }
      }
      // Navigate away before deleting if the deleted session is active
      if (activeSessionId === id) {
        const remaining = (sessions ?? []).filter((s) => s.id !== id);
        if (remaining.length > 0) {
          navigate("/s/" + remaining[0].id);
        } else {
          navigate("/");
        }
      }
      await deleteSession(id);
    } catch (e) {
      console.error("Failed to delete session:", e);
    }
  };

  const handleRenameSession = async (id: string, title: string) => {
    try {
      await renameSession(id, title);
    } catch (e) {
      console.error("Failed to rename session:", e);
    }
  };

  const handleFirstMessage = useCallback(
    async (text: string) => {
      if (!activeSessionId) return;
      if (activeSessionId.startsWith("device-")) return;
      const title = text.length > 50 ? text.slice(0, 50) + "..." : text;
      try {
        await renameSession(activeSessionId, title);
      } catch (e) {
        console.error("Failed to update session title:", e);
      }
    },
    [activeSessionId, renameSession]
  );

  const sidebarProps = {
    sessions: sessions ?? [],
    devices,
    activeSessionId: activeSessionId ?? null,
    user,
    isLoading,
    onNewSession: () => {
      handleNewSession();
      setSidebarOpen(false);
    },
    onSelectSession: (id: string) => {
      navigate("/s/" + id);
      setSidebarOpen(false);
    },
    onDeleteSession: handleDeleteSession,
    onRenameSession: handleRenameSession
  };

  const outletContext: AuthLayoutContext = {
    user,
    onOpenSidebar: handleOpenSidebar,
    onOpenBugReport: handleOpenBugReport,
    onFirstMessage: handleFirstMessage
  };

  return (
    <div className="flex h-screen">
      {/* Desktop sidebar */}
      <div className="hidden md:block">
        <SessionSidebar {...sidebarProps} />
      </div>

      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-40 md:hidden">
          <div
            className="absolute inset-0 bg-black/40"
            onClick={() => setSidebarOpen(false)}
          />
          <div className="relative z-10 h-full w-64">
            <SessionSidebar {...sidebarProps} />
          </div>
        </div>
      )}

      <div className="flex-1 min-w-0 flex flex-col">
        <div className="flex-1 min-h-0">
          <Outlet context={outletContext} />
        </div>
      </div>

      <BugReportFloat
        open={bugReportOpen}
        onClose={() => setBugReportOpen(false)}
        sessionId={activeSessionId ?? null}
      />
    </div>
  );
}
