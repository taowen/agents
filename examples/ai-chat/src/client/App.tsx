import { Suspense, useState, useEffect } from "react";
import { LoginPage } from "./LoginPage";
import { SessionSidebar, type SessionInfo } from "./SessionSidebar";
import { SettingsPage } from "./SettingsPage";
import { Chat } from "./Chat";

export interface UserInfo {
  id: string;
  email: string;
  name: string | null;
  picture: string | null;
}

function AuthenticatedApp({ user }: { user: UserInfo }) {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [view, setView] = useState<"chat" | "settings">("chat");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Load sessions on mount
  useEffect(() => {
    fetch("/api/sessions")
      .then((res) => res.json())
      .then((data: SessionInfo[]) => {
        setSessions(data);
        if (data.length > 0) {
          setActiveSessionId(data[0].id);
        }
      })
      .catch(console.error);
  }, []);

  const handleNewSession = async () => {
    try {
      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
      });
      const session = (await res.json()) as SessionInfo;
      setSessions((prev) => [session, ...prev]);
      setActiveSessionId(session.id);
      setView("chat");
    } catch (e) {
      console.error("Failed to create session:", e);
    }
  };

  const handleDeleteSession = async (id: string) => {
    try {
      await fetch(`/api/sessions/${id}`, { method: "DELETE" });
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (activeSessionId === id) {
        const remaining = sessions.filter((s) => s.id !== id);
        setActiveSessionId(remaining.length > 0 ? remaining[0].id : null);
      }
    } catch (e) {
      console.error("Failed to delete session:", e);
    }
  };

  const handleFirstMessage = async (text: string) => {
    if (!activeSessionId) return;
    const title = text.length > 50 ? text.slice(0, 50) + "..." : text;
    try {
      await fetch(`/api/sessions/${activeSessionId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title })
      });
      setSessions((prev) =>
        prev.map((s) => (s.id === activeSessionId ? { ...s, title } : s))
      );
    } catch (e) {
      console.error("Failed to update session title:", e);
    }
  };

  // Auto-create first session if none exist
  useEffect(() => {
    if (sessions.length === 0 && activeSessionId === null) {
      handleNewSession();
    }
  }, [sessions.length, activeSessionId]);

  const sidebarProps = {
    sessions,
    activeSessionId,
    user,
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
    onOpenSettings: () => {
      setView("settings");
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

      <div className="flex-1 min-w-0">
        {view === "settings" ? (
          <SettingsPage
            onBack={() => setView("chat")}
            onOpenSidebar={() => setSidebarOpen(true)}
          />
        ) : activeSessionId ? (
          <Chat
            key={activeSessionId}
            sessionId={activeSessionId}
            onFirstMessage={handleFirstMessage}
            onOpenSidebar={() => setSidebarOpen(true)}
          />
        ) : (
          <div className="flex items-center justify-center h-full text-kumo-inactive">
            Loading...
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [authState, setAuthState] = useState<
    "loading" | "unauthenticated" | "authenticated"
  >("loading");
  const [user, setUser] = useState<UserInfo | null>(null);

  useEffect(() => {
    fetch("/auth/status")
      .then((res) => res.json())
      .then((data: { authenticated: boolean; user?: UserInfo }) => {
        if (data.authenticated && data.user) {
          setUser(data.user);
          setAuthState("authenticated");
        } else {
          setAuthState("unauthenticated");
        }
      })
      .catch(() => setAuthState("unauthenticated"));
  }, []);

  if (authState === "loading") {
    return (
      <div className="flex items-center justify-center h-screen text-kumo-inactive">
        Loading...
      </div>
    );
  }

  if (authState === "unauthenticated") {
    return <LoginPage />;
  }

  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center h-screen text-kumo-inactive">
          Loading...
        </div>
      }
    >
      <AuthenticatedApp user={user!} />
    </Suspense>
  );
}
