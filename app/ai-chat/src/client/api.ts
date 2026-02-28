import useSWR from "swr";
import * as Sentry from "@sentry/react";
import type { UIMessage } from "ai";
import type { SessionInfo } from "./SessionSidebar";

export interface UserInfo {
  id: string;
  email: string;
  name: string | null;
  picture: string | null;
}

const fetcher = async <T = unknown>(url: string): Promise<T> => {
  const res = await fetch(url);
  return res.json() as Promise<T>;
};

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

// --- Devices ---

export interface DeviceInfo {
  deviceName: string;
  sessionId: string;
  title: string;
}

export function useDevices() {
  const { data, isLoading } = useSWR<DeviceInfo[]>("/api/devices", fetcher, {
    refreshInterval: 10_000
  });
  return { devices: data ?? [], isLoading };
}

// --- /etc JSON fetcher (shared by LLM, GitHub, MCP hooks) ---

const etcJsonFetcher = async (url: string) => {
  const res = await fetch(url);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const buf = await res.arrayBuffer();
  return JSON.parse(new TextDecoder().decode(buf));
};

// --- LLM Config ---

export interface LlmFileConfig {
  provider: string;
  api_key: string;
  base_url: string;
  model: string;
}

export function useLlmConfig() {
  const { data, isLoading, mutate } = useSWR<LlmFileConfig | null>(
    "/api/files/content?path=%2Fetc%2Fllm.json",
    etcJsonFetcher
  );
  return { llmConfig: data ?? null, isLoading, mutateLlmConfig: mutate };
}

// --- GitHub Config ---

export interface GithubFileConfig {
  client_id: string;
  client_secret: string;
}

export function useGithubConfig() {
  const { data, isLoading, mutate } = useSWR<GithubFileConfig | null>(
    "/api/files/content?path=%2Fetc%2Fgithub.json",
    etcJsonFetcher
  );
  return { githubConfig: data ?? null, isLoading, mutateGithubConfig: mutate };
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

// --- MCP Servers ---

export interface McpServerEntry {
  name: string;
  url: string;
  headers?: Record<string, string>;
}

export function useMcpServers() {
  const { data, isLoading, mutate } = useSWR<McpServerEntry[]>(
    "/api/files/content?path=%2Fetc%2Fmcp-servers.json",
    (u: string) => etcJsonFetcher(u).then((d) => d ?? [])
  );
  return { mcpServers: data ?? [], isLoading, mutateMcpServers: mutate };
}

// --- Usage Stats ---

export interface UsageRow {
  hour: string;
  api_key_type: string;
  request_count: number;
  input_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  output_tokens: number;
}

export function useUsageStats(start: string, end: string) {
  const { data, error, isLoading, mutate } = useSWR<UsageRow[]>(
    `/api/usage?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`,
    async (url: string) => {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Usage fetch failed: ${res.status}`);
      return res.json() as Promise<UsageRow[]>;
    }
  );
  return {
    usage: Array.isArray(data) ? data : [],
    error,
    isLoading,
    mutateUsage: mutate
  };
}

// --- Quota Status ---

export interface QuotaStatus {
  exceeded: boolean;
  exceededAt: string | null;
  hourly: {
    requests: number;
    tokens: number;
    requestLimit: number;
    tokenLimit: number;
  };
  daily: {
    requests: number;
    tokens: number;
    requestLimit: number;
    tokenLimit: number;
  };
}

export function useQuotaStatus() {
  const { data, isLoading, mutate } = useSWR<QuotaStatus>(
    "/api/quota",
    fetcher,
    { refreshInterval: 60_000 }
  );
  return { quota: data ?? null, isLoading, mutateQuota: mutate };
}

// --- Initial Messages (for Chat) ---

export function useInitialMessages(sessionId: string | undefined) {
  const { data, isLoading } = useSWR(
    sessionId ? `/agents/chat-agent/${sessionId}/get-messages` : null,
    async (url: string) => {
      const res = await fetch(url);
      if (!res.ok) {
        console.error(`[useInitialMessages] ${url} → ${res.status}`);
        return [];
      }
      const text = await res.text();
      if (!text.trim()) return [];
      return JSON.parse(text);
    }
  );
  return { messages: data ?? [], isLoading };
}

// --- Older Messages (Lazy Loading) ---

export async function fetchOlderMessages(
  sessionId: string,
  beforeId: string,
  limit = 50
): Promise<UIMessage[]> {
  const res = await fetch(
    `/agents/chat-agent/${sessionId}/get-messages?before=${encodeURIComponent(beforeId)}&limit=${limit}`
  );
  if (!res.ok) return [];
  const text = await res.text();
  if (!text.trim()) return [];
  return JSON.parse(text);
}

// --- Bug Reports ---

export async function reportBug(
  sessionId: string,
  description: string
): Promise<{ reportId: string }> {
  const res = await fetch(`/api/sessions/${sessionId}/report-bug`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ description })
  });
  if (!res.ok) throw new Error(`Report bug failed: ${res.status}`);
  const { reportId } = (await res.json()) as { reportId: string };
  Sentry.withScope((scope) => {
    scope.setTag("report_id", reportId);
    scope.setTag("session_uuid", sessionId);
    Sentry.captureMessage(
      `[Client Bug Report ${reportId}] ${description}`,
      "warning"
    );
  });
  return { reportId };
}
