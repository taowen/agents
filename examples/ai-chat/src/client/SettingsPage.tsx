import { useState, useEffect } from "react";
import {
  GithubLogoIcon,
  ArrowSquareOutIcon,
  FloppyDiskIcon,
  ArrowLeftIcon,
  ListIcon
} from "@phosphor-icons/react";
import { Button, Text } from "@cloudflare/kumo";

interface SettingsPageProps {
  onBack: () => void;
  onOpenSidebar?: () => void;
}

interface Settings {
  github_client_id?: string;
  github_configured?: boolean;
  llm_api_key_set?: boolean;
  llm_provider?: string;
  llm_base_url?: string;
  llm_model?: string;
}

export function SettingsPage({ onBack, onOpenSidebar }: SettingsPageProps) {
  const [settings, setSettings] = useState<Settings>({});
  const [llmApiKey, setLlmApiKey] = useState("");
  const [llmProvider, setLlmProvider] = useState("builtin");
  const [llmBaseUrl, setLlmBaseUrl] = useState("");
  const [llmModel, setLlmModel] = useState("");
  const [githubClientId, setGithubClientId] = useState("");
  const [githubClientSecret, setGithubClientSecret] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/settings")
      .then((res) => res.json())
      .then((data: Settings) => {
        setSettings(data);
        if (data.llm_provider) setLlmProvider(data.llm_provider);
        if (data.llm_base_url) setLlmBaseUrl(data.llm_base_url);
        if (data.llm_model) setLlmModel(data.llm_model);
        if (data.github_client_id) setGithubClientId(data.github_client_id);
      })
      .catch(() => {});
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setError("");
    setSaved(false);
    try {
      const body: Record<string, string | null> = {
        llm_provider: llmProvider,
        llm_base_url: llmBaseUrl,
        llm_model: llmModel
      };
      if (llmApiKey) body.llm_api_key = llmApiKey;
      if (githubClientId) body.github_client_id = githubClientId;
      if (githubClientSecret) body.github_client_secret = githubClientSecret;

      const res = await fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      if (!res.ok) throw new Error(await res.text());
      setSaved(true);
      // Refresh settings display
      const updated = (await fetch("/api/settings").then((r) =>
        r.json()
      )) as Settings;
      setSettings(updated);
      setLlmApiKey("");
      setGithubClientSecret("");
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
            onClick={onBack}
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

          <div className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-kumo-secondary mb-1">
                Provider
              </label>
              <select
                value={llmProvider}
                onChange={(e) => setLlmProvider(e.target.value)}
                className={inputClass}
              >
                <option value="builtin">Built-in (Doubao)</option>
                <option value="google">Google Gemini</option>
                <option value="openai-compatible">OpenAI Compatible</option>
              </select>
            </div>

            {llmProvider === "builtin" ? (
              <div className="px-3 py-2 rounded-lg bg-kumo-elevated text-sm text-kumo-secondary">
                Using built-in Doubao model. No API key needed.
              </div>
            ) : (
              <>
                <div>
                  <label className="block text-xs font-medium text-kumo-secondary mb-1">
                    API Key
                    {settings.llm_api_key_set && (
                      <span className="ml-1 text-kumo-inactive font-normal">
                        (configured — leave blank to keep current)
                      </span>
                    )}
                  </label>
                  <input
                    type="password"
                    value={llmApiKey}
                    onChange={(e) => setLlmApiKey(e.target.value)}
                    placeholder={
                      settings.llm_api_key_set ? "********" : "Enter API key"
                    }
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
                    onChange={(e) => setLlmBaseUrl(e.target.value)}
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
                    onChange={(e) => setLlmModel(e.target.value)}
                    className={inputClass}
                  />
                </div>
              </>
            )}
          </div>
        </div>

        {/* GitHub Configuration */}
        <div className="rounded-xl ring ring-kumo-line bg-kumo-base p-5 mb-5">
          <div className="flex items-center gap-2 mb-4">
            <GithubLogoIcon size={18} className="text-kumo-default" />
            <h3 className="text-sm font-semibold text-kumo-default">
              GitHub Integration
            </h3>
            {settings.github_configured && (
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
                onChange={(e) => setGithubClientId(e.target.value)}
                placeholder="Ov23li..."
                className={inputClass}
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-kumo-secondary mb-1">
                Client Secret
                {settings.github_configured && (
                  <span className="ml-1 text-kumo-inactive font-normal">
                    (leave blank to keep current)
                  </span>
                )}
              </label>
              <input
                type="password"
                value={githubClientSecret}
                onChange={(e) => setGithubClientSecret(e.target.value)}
                placeholder={
                  settings.github_configured
                    ? "********"
                    : "Enter client secret"
                }
                className={inputClass}
              />
            </div>

            {settings.github_configured && (
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
      </div>
    </div>
  );
}
