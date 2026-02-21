import useSWR from "swr";
import type { SessionInfo } from "./SessionSidebar";
import type { UserInfo } from "./App";

const fetcher = async <T = any>(url: string): Promise<T> => {
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
  return res.json() as Promise<{ reportId: string }>;
}
