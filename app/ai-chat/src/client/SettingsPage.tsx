import { useState } from "react";
import { useNavigate, useOutletContext } from "react-router";
import {
  GithubLogoIcon,
  ArrowSquareOutIcon,
  FloppyDiskIcon,
  ArrowLeftIcon,
  ListIcon,
  PlusIcon,
  XIcon,
  SignOutIcon
} from "@phosphor-icons/react";
import { Button, Text } from "@cloudflare/kumo";
import {
  useLlmConfig,
  useGithubConfig,
  useMcpServers,
  type McpServerEntry
} from "./api";
import type { AuthLayoutContext } from "./AuthLayout";
import { FormFieldSkeleton, Skeleton } from "./Skeleton";

export function SettingsPage() {
  const navigate = useNavigate();
  const { onOpenSidebar } = useOutletContext<AuthLayoutContext>();
  const { llmConfig, isLoading: llmLoading, mutateLlmConfig } = useLlmConfig();
  const {
    githubConfig,
    isLoading: githubLoading,
    mutateGithubConfig
  } = useGithubConfig();
  const {
    mcpServers,
    isLoading: mcpLoading,
    mutateMcpServers
  } = useMcpServers();

  // Dirty tracking: only track fields the user has explicitly changed
  const [dirty, setDirty] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");

  // MCP form state (add-server form only)
  const [mcpError, setMcpError] = useState("");
  const [mcpName, setMcpName] = useState("");
  const [mcpUrl, setMcpUrl] = useState("");
  const [mcpHeaders, setMcpHeaders] = useState("");
  const [mcpSaving, setMcpSaving] = useState(false);

  // Derive field values: dirty value > /etc file value > default
  const setField = (key: string, value: string) =>
    setDirty((prev) => ({ ...prev, [key]: value }));

  const llmProvider = dirty.provider ?? llmConfig?.provider ?? "builtin";
  const llmBaseUrl = dirty.base_url ?? llmConfig?.base_url ?? "";
  const llmModel = dirty.model ?? llmConfig?.model ?? "";
  const githubClientId = dirty.client_id ?? githubConfig?.client_id ?? "";
  // Password fields: always start empty, only tracked in dirty
  const llmApiKey = dirty.api_key ?? "";
  const githubClientSecret = dirty.client_secret ?? "";
  // Derived booleans
  const llmApiKeySet = !!llmConfig?.api_key;
  const githubConfigured = !!(
    githubConfig?.client_id && githubConfig?.client_secret
  );

  const saveMcpConfig = async (entries: McpServerEntry[]) => {
    const text = JSON.stringify(entries, null, 2);
    const res = await fetch(
      `/api/files/content?path=${encodeURIComponent("/etc/mcp-servers.json")}`,
      { method: "PUT", body: text }
    );
    if (!res.ok) throw new Error("Failed to save MCP config");
    await mutateMcpServers(entries, { revalidate: false });
  };

  const putEtcFile = async (path: string, data: unknown) => {
    const res = await fetch(
      `/api/files/content?path=${encodeURIComponent(path)}`,
      { method: "PUT", body: JSON.stringify(data) }
    );
    if (!res.ok) throw new Error(`Failed to save ${path}`);
  };

  const deleteEtcFile = async (path: string) => {
    await fetch(`/api/files?path=${encodeURIComponent(path)}`, {
      method: "DELETE"
    });
  };

  const handleSave = async () => {
    setSaving(true);
    setError("");
    setSaved(false);
    try {
      // LLM config
      if (llmProvider === "builtin") {
        await deleteEtcFile("/etc/llm.json");
        await mutateLlmConfig(null, { revalidate: false });
      } else {
        const llmData = {
          provider: llmProvider,
          api_key: llmApiKey || llmConfig?.api_key || "",
          base_url: llmBaseUrl,
          model: llmModel
        };
        await putEtcFile("/etc/llm.json", llmData);
        await mutateLlmConfig(llmData, { revalidate: false });
      }

      // GitHub config
      if (dirty.client_id !== undefined || dirty.client_secret !== undefined) {
        const ghData = {
          client_id: githubClientId,
          client_secret: githubClientSecret || githubConfig?.client_secret || ""
        };
        await putEtcFile("/etc/github.json", ghData);
        await mutateGithubConfig(ghData, { revalidate: false });
      }

      setSaved(true);
      setDirty({});
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const callbackUrl =
    typeof window !== "undefined"
      ? `${window.location.origin}/oauth/github/callback`
      : "";

  const inputClass =
    "w-full px-3 py-2 rounded-lg border border-kumo-line bg-kumo-elevated text-kumo-default text-sm focus:outline-none focus:ring-2 focus:ring-kumo-ring font-mono";

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-2xl mx-auto px-5 py-6">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <button
            onClick={onOpenSidebar}
            className="md:hidden p-1.5 rounded-lg hover:bg-kumo-elevated text-kumo-secondary hover:text-kumo-default transition-colors"
          >
            <ListIcon size={20} />
          </button>
          <button
            onClick={() => navigate("/")}
            className="p-1.5 rounded-lg hover:bg-kumo-elevated text-kumo-secondary hover:text-kumo-default transition-colors"
          >
            <ArrowLeftIcon size={18} />
          </button>
          <h2 className="text-lg font-semibold text-kumo-default">Settings</h2>
        </div>

        {/* LLM Configuration */}
        <div className="rounded-xl ring ring-kumo-line bg-kumo-base p-5 mb-5">
          <h3 className="text-sm font-semibold text-kumo-default mb-4">
            LLM Configuration
          </h3>

          {llmLoading ? (
            <div className="space-y-4">
              <FormFieldSkeleton />
              <FormFieldSkeleton />
              <FormFieldSkeleton />
              <FormFieldSkeleton />
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-kumo-secondary mb-1">
                  Provider
                </label>
                <select
                  value={llmProvider}
                  onChange={(e) => setField("provider", e.target.value)}
                  className={inputClass}
                >
                  <option value="builtin">Built-in</option>
                  <option value="google">Google Gemini</option>
                  <option value="openai-compatible">OpenAI Compatible</option>
                </select>
              </div>

              {llmProvider === "builtin" ? (
                <div className="px-3 py-2 rounded-lg bg-kumo-elevated text-sm text-kumo-secondary">
                  Using built-in model. No API key needed.
                </div>
              ) : (
                <>
                  <div>
                    <label className="block text-xs font-medium text-kumo-secondary mb-1">
                      API Key
                      {llmApiKeySet && (
                        <span className="ml-1 text-kumo-inactive font-normal">
                          (configured — leave blank to keep current)
                        </span>
                      )}
                    </label>
                    <input
                      type="password"
                      value={llmApiKey}
                      onChange={(e) => setField("api_key", e.target.value)}
                      placeholder={llmApiKeySet ? "********" : "Enter API key"}
                      className={inputClass}
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-kumo-secondary mb-1">
                      Base URL
                    </label>
                    <input
                      type="text"
                      value={llmBaseUrl}
                      onChange={(e) => setField("base_url", e.target.value)}
                      className={inputClass}
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-kumo-secondary mb-1">
                      Model
                    </label>
                    <input
                      type="text"
                      value={llmModel}
                      onChange={(e) => setField("model", e.target.value)}
                      className={inputClass}
                    />
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {/* GitHub Configuration */}
        <div className="rounded-xl ring ring-kumo-line bg-kumo-base p-5 mb-5">
          <div className="flex items-center gap-2 mb-4">
            <GithubLogoIcon size={18} className="text-kumo-default" />
            <h3 className="text-sm font-semibold text-kumo-default">
              GitHub Integration
            </h3>
            {githubConfigured && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
                Connected
              </span>
            )}
          </div>

          <div className="space-y-4">
            <div className="space-y-2 text-sm text-kumo-secondary">
              <span className="text-xs font-semibold text-kumo-default">
                Setup Instructions
              </span>
              <ol className="list-decimal list-inside space-y-1.5">
                <li>
                  <a
                    href="https://github.com/settings/applications/new"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-kumo-brand hover:underline inline-flex items-center gap-1"
                  >
                    Create a GitHub OAuth App
                    <ArrowSquareOutIcon size={12} />
                  </a>
                </li>
                <li>
                  Set <strong>Homepage URL</strong> to:{" "}
                  <code className="px-1 py-0.5 rounded bg-kumo-elevated text-xs">
                    {typeof window !== "undefined"
                      ? window.location.origin
                      : ""}
                  </code>
                </li>
                <li>
                  Set <strong>Authorization callback URL</strong> to:
                  <div className="mt-1">
                    <code className="block px-2 py-1 rounded bg-kumo-elevated text-xs break-all select-all">
                      {callbackUrl}
                    </code>
                  </div>
                </li>
              </ol>
            </div>

            <div>
              <label className="block text-xs font-medium text-kumo-secondary mb-1">
                Client ID
              </label>
              <input
                type="text"
                value={githubClientId}
                onChange={(e) => setField("client_id", e.target.value)}
                placeholder="Ov23li..."
                className={inputClass}
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-kumo-secondary mb-1">
                Client Secret
                {githubConfigured && (
                  <span className="ml-1 text-kumo-inactive font-normal">
                    (leave blank to keep current)
                  </span>
                )}
              </label>
              <input
                type="password"
                value={githubClientSecret}
                onChange={(e) => setField("client_secret", e.target.value)}
                placeholder={
                  githubConfigured ? "********" : "Enter client secret"
                }
                className={inputClass}
              />
            </div>

            {githubConfigured && (
              <Button
                variant="secondary"
                size="sm"
                icon={<GithubLogoIcon size={14} />}
                onClick={() => {
                  window.location.href = "/oauth/github";
                }}
              >
                Reconnect GitHub
              </Button>
            )}
          </div>
        </div>

        {/* MCP Servers */}
        <div className="rounded-xl ring ring-kumo-line bg-kumo-base p-5 mb-5">
          <h3 className="text-sm font-semibold text-kumo-default mb-4">
            MCP Servers
          </h3>

          {mcpLoading ? (
            <div className="space-y-2">
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
            </div>
          ) : (
            <div className="space-y-3">
              {mcpServers.length > 0 && (
                <div className="space-y-2">
                  {mcpServers.map((server, i) => (
                    <div
                      key={server.name}
                      className="flex items-center justify-between px-3 py-2 rounded-lg bg-kumo-elevated"
                    >
                      <div className="text-sm text-kumo-default min-w-0">
                        <span className="font-medium">{server.name}</span>
                        <span className="text-kumo-secondary mx-2">
                          &mdash;
                        </span>
                        <span className="text-kumo-secondary break-all">
                          {server.url}
                        </span>
                      </div>
                      <button
                        onClick={async () => {
                          const updated = mcpServers.filter(
                            (_, idx) => idx !== i
                          );
                          try {
                            await saveMcpConfig(updated);
                          } catch {
                            setMcpError("Failed to remove server");
                          }
                        }}
                        className="ml-2 p-1 rounded hover:bg-kumo-base text-kumo-secondary hover:text-kumo-default transition-colors shrink-0"
                      >
                        <XIcon size={14} />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* Add Server form */}
              <div className="space-y-2 pt-2">
                <div className="flex gap-2">
                  <div className="flex-1">
                    <label className="block text-xs font-medium text-kumo-secondary mb-1">
                      Name
                    </label>
                    <input
                      type="text"
                      value={mcpName}
                      onChange={(e) => setMcpName(e.target.value)}
                      placeholder="weather"
                      className={inputClass}
                    />
                  </div>
                  <div className="flex-[2]">
                    <label className="block text-xs font-medium text-kumo-secondary mb-1">
                      URL
                    </label>
                    <input
                      type="text"
                      value={mcpUrl}
                      onChange={(e) => setMcpUrl(e.target.value)}
                      placeholder="https://example.com/sse"
                      className={inputClass}
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-xs font-medium text-kumo-secondary mb-1">
                    Headers (JSON, optional)
                  </label>
                  <input
                    type="text"
                    value={mcpHeaders}
                    onChange={(e) => setMcpHeaders(e.target.value)}
                    placeholder='{"Authorization": "Bearer sk-xxx"}'
                    className={inputClass}
                  />
                </div>
                <div className="flex items-center gap-3">
                  <Button
                    variant="secondary"
                    size="sm"
                    icon={<PlusIcon size={14} />}
                    loading={mcpSaving}
                    onClick={async () => {
                      if (!mcpName.trim() || !mcpUrl.trim()) {
                        setMcpError("Name and URL are required");
                        return;
                      }
                      if (mcpServers.some((s) => s.name === mcpName.trim())) {
                        setMcpError("A server with this name already exists");
                        return;
                      }
                      setMcpSaving(true);
                      setMcpError("");
                      try {
                        const entry: McpServerEntry = {
                          name: mcpName.trim(),
                          url: mcpUrl.trim()
                        };
                        if (mcpHeaders.trim()) {
                          entry.headers = JSON.parse(mcpHeaders.trim());
                        }
                        await saveMcpConfig([...mcpServers, entry]);
                        setMcpName("");
                        setMcpUrl("");
                        setMcpHeaders("");
                      } catch (e) {
                        setMcpError(
                          e instanceof SyntaxError
                            ? "Invalid JSON in headers"
                            : "Failed to add server"
                        );
                      } finally {
                        setMcpSaving(false);
                      }
                    }}
                  >
                    Add
                  </Button>
                  {mcpError && (
                    <Text size="xs" variant="error">
                      {mcpError}
                    </Text>
                  )}
                </div>
              </div>

              <p className="text-xs text-kumo-inactive pt-1">
                Changes take effect in new sessions.
              </p>
            </div>
          )}
        </div>

        {/* Save */}
        <div className="flex items-center gap-3">
          <Button
            variant="primary"
            size="sm"
            icon={<FloppyDiskIcon size={14} />}
            onClick={handleSave}
            loading={saving}
          >
            Save Settings
          </Button>
          {saved && (
            <Text size="xs" variant="secondary">
              Settings saved successfully.
            </Text>
          )}
          {error && (
            <Text size="xs" variant="error">
              {error}
            </Text>
          )}
        </div>

        {/* Logout */}
        <div className="mt-8 pt-5 border-t border-kumo-line">
          <button
            onClick={() => {
              window.location.href = "/auth/logout";
            }}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
          >
            <SignOutIcon size={16} />
            退出登录
          </button>
        </div>
      </div>
    </div>
  );
}
