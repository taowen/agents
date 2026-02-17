import useSWR from "swr";
import type { SessionInfo } from "./SessionSidebar";
import type { UserInfo } from "./App";

const fetcher = (url: string) => fetch(url).then((r) => r.json());

// --- Auth ---

export function useAuth() {
  const { data, isLoading } = useSWR<{
    authenticated: boolean;
    user?: UserInfo;
  }>("/auth/status", fetcher);
  return {
    user: data?.user ?? null,
    authenticated: data?.authenticated ?? false,
    isLoading
  };
}

// --- Sessions ---

export function useSessions() {
  const {
    data: sessions,
    isLoading,
    mutate: mutateSessions
  } = useSWR<SessionInfo[]>("/api/sessions", fetcher);

  const createSession = async (): Promise<SessionInfo> => {
    const res = await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });
    const session = (await res.json()) as SessionInfo;
    await mutateSessions((prev) => (prev ? [session, ...prev] : [session]), {
      revalidate: false
    });
    return session;
  };

  const deleteSession = async (id: string) => {
    await fetch(`/api/sessions/${id}`, { method: "DELETE" });
    await mutateSessions((prev) => prev?.filter((s) => s.id !== id), {
      revalidate: false
    });
  };

  const renameSession = async (id: string, title: string) => {
    await fetch(`/api/sessions/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title })
    });
    await mutateSessions(
      (prev) => prev?.map((s) => (s.id === id ? { ...s, title } : s)),
      { revalidate: false }
    );
  };

  return {
    sessions,
    isLoading,
    mutateSessions,
    createSession,
    deleteSession,
    renameSession
  };
}

// --- Settings ---

export interface Settings {
  github_client_id?: string;
  github_configured?: boolean;
  llm_api_key_set?: boolean;
  llm_provider?: string;
  llm_base_url?: string;
  llm_model?: string;
}

export function useSettings() {
  const {
    data: settings,
    isLoading,
    mutate: mutateSettings
  } = useSWR<Settings>("/api/settings", fetcher);
  return { settings, isLoading, mutateSettings };
}

// --- Memory ---

export interface MemoryFiles {
  profile: string;
  preferences: string;
  entities: string;
}

export function useMemory() {
  const {
    data: memory,
    isLoading,
    mutate: mutateMemory
  } = useSWR<MemoryFiles>("/api/memory", fetcher);
  return { memory, isLoading, mutateMemory };
}
