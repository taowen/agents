import { Suspense, useState, useEffect, useRef, useCallback } from "react";
import { LoginPage } from "./LoginPage";
import { DevicePage } from "./DevicePage";
import { SessionSidebar } from "./SessionSidebar";
import { SettingsPage } from "./SettingsPage";
import { MemoryPage } from "./MemoryPage";
import { UsagePage } from "./UsagePage";
import { Chat } from "./Chat";
import { BugReportFloat } from "./BugReportFloat";
import { useAuth, useSessions } from "./api";
import { AppShellSkeleton } from "./Skeleton";

export interface UserInfo {
  id: string;
  email: string;
  name: string | null;
  picture: string | null;
}

function AuthenticatedApp({ user }: { user: UserInfo }) {
  const { sessions, isLoading, createSession, deleteSession, renameSession } =
    useSessions();
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [view, setView] = useState<"chat" | "settings" | "memory" | "usage">(
    "chat"
  );
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [bugReportOpen, setBugReportOpen] = useState(false);
  const autoCreatingRef = useRef(false);

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

  // Select first session when sessions load and nothing is active
  useEffect(() => {
    if (sessions && sessions.length > 0 && activeSessionId === null) {
      setActiveSessionId(sessions[0].id);
    }
  }, [sessions, activeSessionId]);

  // Auto-create first session if none exist (only after loading completes)
  useEffect(() => {
    if (
      !isLoading &&
      sessions &&
      sessions.length === 0 &&
      activeSessionId === null &&
      !autoCreatingRef.current
    ) {
      autoCreatingRef.current = true;
      handleNewSession();
    }
  }, [isLoading, sessions, activeSessionId]);

  const handleNewSession = async () => {
    try {
      const session = await createSession();
      setActiveSessionId(session.id);
      setView("chat");
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
      // Switch active session BEFORE deleting so the Chat component
      // transitions in the same render as the sidebar update, avoiding a flash.
      if (activeSessionId === id) {
        const remaining = (sessions ?? []).filter((s) => s.id !== id);
        setActiveSessionId(remaining.length > 0 ? remaining[0].id : null);
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

  const handleFirstMessage = async (text: string) => {
    if (!activeSessionId) return;
    const title = text.length > 50 ? text.slice(0, 50) + "..." : text;
    try {
      await renameSession(activeSessionId, title);
    } catch (e) {
      console.error("Failed to update session title:", e);
    }
  };

  const sidebarProps = {
    sessions: sessions ?? [],
    activeSessionId,
    user,
    isLoading,
    onNewSession: () => {
      handleNewSession();
      setSidebarOpen(false);
    },
    onSelectSession: (id: string) => {
      setActiveSessionId(id);
      setView("chat");
      setSidebarOpen(false);
    },
    onDeleteSession: handleDeleteSession,
    onRenameSession: handleRenameSession,
    onOpenSettings: () => {
      setView("settings");
      setSidebarOpen(false);
    },
    onOpenMemory: () => {
      setView("memory");
      setSidebarOpen(false);
    },
    onOpenUsage: () => {
      setView("usage");
      setSidebarOpen(false);
    }
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
          {view === "settings" ? (
            <SettingsPage
              onBack={() => setView("chat")}
              onOpenSidebar={() => setSidebarOpen(true)}
            />
          ) : view === "memory" ? (
            <MemoryPage
              onBack={() => setView("chat")}
              onOpenSidebar={() => setSidebarOpen(true)}
            />
          ) : view === "usage" ? (
            <UsagePage
              onBack={() => setView("chat")}
              onOpenSidebar={() => setSidebarOpen(true)}
            />
          ) : activeSessionId ? (
            <Chat
              key={activeSessionId}
              sessionId={activeSessionId}
              onFirstMessage={handleFirstMessage}
              onOpenSidebar={() => setSidebarOpen(true)}
              onOpenBugReport={handleOpenBugReport}
            />
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center gap-3 p-8">
              <div className="animate-pulse rounded bg-kumo-elevated h-4 w-48" />
              <div className="animate-pulse rounded bg-kumo-elevated h-4 w-32" />
            </div>
          )}
        </div>
      </div>

      <BugReportFloat
        open={bugReportOpen}
        onClose={() => setBugReportOpen(false)}
        sessionId={activeSessionId}
      />
    </div>
  );
}

export default function App() {
  // /device path renders the device approval page
  if (window.location.pathname === "/device") {
    return <DevicePage />;
  }

  const { user, authenticated, isLoading } = useAuth();

  if (isLoading) {
    return <AppShellSkeleton />;
  }

  if (!authenticated) {
    return <LoginPage />;
  }

  return (
    <Suspense fallback={<AppShellSkeleton />}>
      <AuthenticatedApp user={user!} />
    </Suspense>
  );
}
