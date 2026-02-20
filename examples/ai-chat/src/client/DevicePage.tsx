import { useState } from "react";
import { useAuth } from "./api";
import { LoginPage } from "./LoginPage";
import { CloudSunIcon, SpinnerIcon } from "@phosphor-icons/react";

type Step = "idle" | "approving" | "done" | "error";

export function DevicePage() {
  const { user, authenticated, isLoading } = useAuth();
  const [code, setCode] = useState("");
  const [step, setStep] = useState<Step>("idle");
  const [error, setError] = useState("");

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-kumo-elevated">
        <SpinnerIcon size={32} className="animate-spin text-kumo-secondary" />
      </div>
    );
  }

  if (!authenticated) {
    return <LoginPage />;
  }

  async function handleApprove() {
    const trimmed = code.trim().toUpperCase();
    if (trimmed.length !== 6) {
      setError("Please enter a 6-character code");
      return;
    }
    setStep("approving");
    setError("");
    try {
      const res = await fetch("/api/device/approve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: trimmed })
      });
      if (!res.ok) {
        const data = (await res.json()) as { error?: string };
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      setStep("done");
    } catch (e: any) {
      setError(e.message || "Approval failed");
      setStep("error");
    }
  }

  return (
    <div className="flex items-center justify-center h-screen bg-kumo-elevated">
      <div className="w-full max-w-sm mx-4 rounded-xl ring ring-kumo-line overflow-hidden bg-kumo-base p-8 text-center">
        <CloudSunIcon size={48} className="text-kumo-brand mx-auto mb-4" />
        <h1 className="text-xl font-semibold text-kumo-default mb-2">
          Approve Device
        </h1>
        <p className="text-sm text-kumo-secondary mb-1">
          Signed in as <strong>{user?.name || user?.email}</strong>
        </p>
        <p className="text-sm text-kumo-secondary mb-6">
          Enter the 6-character code shown on your device.
        </p>

        {step === "done" ? (
          <div className="space-y-3">
            <p className="text-sm text-green-600 font-medium">
              Device approved! You can close this page.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <input
              type="text"
              maxLength={6}
              value={code}
              onChange={(e) => setCode(e.target.value.toUpperCase())}
              placeholder="A3F7K2"
              className="w-full text-center text-2xl tracking-[0.3em] font-mono py-3 px-4 rounded-lg bg-kumo-elevated ring ring-kumo-line text-kumo-default placeholder:text-kumo-secondary/40 focus:outline-none focus:ring-2 focus:ring-kumo-brand"
              disabled={step === "approving"}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleApprove();
              }}
            />
            <button
              onClick={handleApprove}
              disabled={step === "approving" || code.trim().length !== 6}
              className="w-full px-5 py-2.5 rounded-lg bg-kumo-contrast text-kumo-inverse text-sm font-medium hover:opacity-90 transition-opacity disabled:opacity-40"
            >
              {step === "approving" ? (
                <span className="inline-flex items-center gap-2">
                  <SpinnerIcon size={16} className="animate-spin" />
                  Approving...
                </span>
              ) : (
                "Approve"
              )}
            </button>
          </div>
        )}

        {error && <p className="mt-3 text-sm text-red-500">{error}</p>}
      </div>
    </div>
  );
}
