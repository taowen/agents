import { useState } from "react";
import {
  FloppyDiskIcon,
  ArrowLeftIcon,
  ListIcon,
  BrainIcon
} from "@phosphor-icons/react";
import { Button, Text } from "@cloudflare/kumo";
import { useMemory } from "./api";
import { Skeleton } from "./Skeleton";

interface MemoryPageProps {
  onBack: () => void;
  onOpenSidebar?: () => void;
}

export function MemoryPage({ onBack, onOpenSidebar }: MemoryPageProps) {
  const { memory, isLoading, mutateMemory } = useMemory();

  // Dirty tracking: only track fields the user has explicitly changed
  const [dirty, setDirty] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");

  const field = (key: string) =>
    dirty[key] ?? memory?.[key as keyof typeof memory] ?? "";
  const setField = (key: string, value: string) =>
    setDirty((prev) => ({ ...prev, [key]: value }));

  const handleSave = async () => {
    setSaving(true);
    setError("");
    setSaved(false);
    try {
      const res = await fetch("/api/memory", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          profile: field("profile"),
          preferences: field("preferences"),
          entities: field("entities")
        })
      });
      if (!res.ok) throw new Error(await res.text());
      setSaved(true);
      await mutateMemory();
      setDirty({});
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

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
          <BrainIcon size={20} className="text-kumo-default" />
          <h2 className="text-lg font-semibold text-kumo-default">
            Agent Memory
          </h2>
        </div>

        <p className="text-sm text-kumo-secondary mb-5">
          Information the agent remembers across sessions. The agent can also
          update these files itself via bash commands.
        </p>

        <div className="space-y-5">
          <div className="rounded-xl ring ring-kumo-line bg-kumo-base p-5">
            <label className="block text-xs font-medium text-kumo-secondary mb-1">
              User Profile
              <span className="ml-1 text-kumo-inactive font-normal">
                — name, role, background
              </span>
            </label>
            {isLoading ? (
              <Skeleton className="h-[100px] w-full" />
            ) : (
              <textarea
                value={field("profile")}
                onChange={(e) => setField("profile", e.target.value)}
                placeholder="e.g. Name: Alice, Role: Frontend developer"
                className={`${inputClass} min-h-[100px] resize-y`}
              />
            )}
          </div>

          <div className="rounded-xl ring ring-kumo-line bg-kumo-base p-5">
            <label className="block text-xs font-medium text-kumo-secondary mb-1">
              Preferences
              <span className="ml-1 text-kumo-inactive font-normal">
                — coding style, communication habits
              </span>
            </label>
            {isLoading ? (
              <Skeleton className="h-[100px] w-full" />
            ) : (
              <textarea
                value={field("preferences")}
                onChange={(e) => setField("preferences", e.target.value)}
                placeholder={
                  "e.g. - Prefers TypeScript\n- Likes concise answers"
                }
                className={`${inputClass} min-h-[100px] resize-y`}
              />
            )}
          </div>

          <div className="rounded-xl ring ring-kumo-line bg-kumo-base p-5">
            <label className="block text-xs font-medium text-kumo-secondary mb-1">
              Known Entities
              <span className="ml-1 text-kumo-inactive font-normal">
                — projects, people, companies
              </span>
            </label>
            {isLoading ? (
              <Skeleton className="h-[100px] w-full" />
            ) : (
              <textarea
                value={field("entities")}
                onChange={(e) => setField("entities", e.target.value)}
                placeholder={
                  "e.g. - Project: ai-chat (Cloudflare Workers app)\n- Company: Acme Corp"
                }
                className={`${inputClass} min-h-[100px] resize-y`}
              />
            )}
          </div>

          <div className="flex items-center gap-3">
            <Button
              variant="primary"
              size="sm"
              icon={<FloppyDiskIcon size={14} />}
              onClick={handleSave}
              loading={saving}
            >
              Save Memory
            </Button>
            {saved && (
              <Text size="xs" variant="secondary">
                Memory saved successfully.
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
    </div>
  );
}
