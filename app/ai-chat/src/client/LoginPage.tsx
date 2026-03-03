import {
  CloudSunIcon,
  CopyIcon,
  CheckIcon,
  EnvelopeIcon,
  SpinnerIcon
} from "@phosphor-icons/react";
import { useState, useEffect, useRef, useCallback } from "react";

type EmailStep = "idle" | "waiting" | "confirm" | "rejected";

export function LoginPage() {
  const [emailStep, setEmailStep] = useState<EmailStep>("idle");
  const [token, setToken] = useState("");
  const [address, setAddress] = useState("");
  const [senderEmail, setSenderEmail] = useState("");
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState("");
  const [rejectReason, setRejectReason] = useState("");
  const pollRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = undefined;
    }
  }, []);

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  async function startEmailLogin() {
    setError("");
    try {
      const res = await fetch("/auth/email/start", { method: "POST" });
      const data = (await res.json()) as { token: string; address: string };
      setToken(data.token);
      setAddress(data.address);
      setEmailStep("waiting");

      // Start polling
      pollRef.current = setInterval(async () => {
        try {
          const checkRes = await fetch(`/auth/email/check?token=${data.token}`);
          const checkData = (await checkRes.json()) as {
            status: string;
            email?: string;
            reason?: string;
            debug?: unknown;
          };
          if (checkData.status === "received" && checkData.email) {
            stopPolling();
            setSenderEmail(checkData.email);
            setEmailStep("confirm");
          } else if (checkData.status === "rejected") {
            stopPolling();
            setRejectReason(checkData.reason || "");
            setEmailStep("rejected");
          } else if (checkData.status === "expired") {
            stopPolling();
            setError("Token expired. Please try again.");
            setEmailStep("idle");
          }
        } catch {
          // ignore polling errors
        }
      }, 2000);
    } catch {
      setError("Failed to start email login");
    }
  }

  async function confirmLogin() {
    setError("");
    try {
      const res = await fetch(`/auth/email/confirm?token=${token}`, {
        method: "POST"
      });
      if (res.ok) {
        window.location.reload();
      } else {
        const text = await res.text();
        setError(text || "Confirm failed");
      }
    } catch {
      setError("Confirm request failed");
    }
  }

  function cancelEmailLogin() {
    stopPolling();
    setEmailStep("idle");
    setToken("");
    setAddress("");
    setSenderEmail("");
    setError("");
  }

  async function copyAddress() {
    try {
      await navigator.clipboard.writeText(address);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // fallback: select text
    }
  }

  return (
    <div className="min-h-screen bg-kumo-elevated overflow-y-auto">
      {/* Hero / Landing Section */}
      <div className="w-full max-w-2xl mx-auto px-6 pt-16 pb-12 text-center">
        <CloudSunIcon size={56} className="text-kumo-brand mx-auto mb-4" />
        <h1 className="text-3xl font-bold text-kumo-default mb-3">AI Chat</h1>
        <p className="text-lg text-kumo-secondary mb-4">
          An AI assistant that works with your files
        </p>
        <p className="text-sm text-kumo-secondary mb-10 max-w-lg mx-auto leading-relaxed">
          AI Chat connects your Google Drive as a virtual filesystem so your AI
          assistant can browse, read, and edit documents on your behalf. It also
          provides a sandboxed shell environment with persistent storage for
          running commands, mounting Git repos, and scheduling recurring tasks.
        </p>

        {/* Features */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-left mb-10">
          <div className="rounded-xl bg-kumo-base ring ring-kumo-line p-5">
            <h3 className="text-sm font-semibold text-kumo-default mb-1">
              Google Drive Integration
            </h3>
            <p className="text-sm text-kumo-secondary leading-relaxed">
              Mount your Google Drive as a local path. The AI assistant can
              list, read, create, and update files — including exporting Google
              Docs and Sheets to text.
            </p>
          </div>
          <div className="rounded-xl bg-kumo-base ring ring-kumo-line p-5">
            <h3 className="text-sm font-semibold text-kumo-default mb-1">
              Sandboxed Shell Environment
            </h3>
            <p className="text-sm text-kumo-secondary leading-relaxed">
              Run shell commands in a secure sandbox with persistent storage
              backed by Cloudflare D1 and R2.
            </p>
          </div>
          <div className="rounded-xl bg-kumo-base ring ring-kumo-line p-5">
            <h3 className="text-sm font-semibold text-kumo-default mb-1">
              Git Repository Mounting
            </h3>
            <p className="text-sm text-kumo-secondary leading-relaxed">
              Mount Git repositories as local directories. Auto-commit and push
              changes as you work.
            </p>
          </div>
          <div className="rounded-xl bg-kumo-base ring ring-kumo-line p-5">
            <h3 className="text-sm font-semibold text-kumo-default mb-1">
              Scheduled Tasks & Memory
            </h3>
            <p className="text-sm text-kumo-secondary leading-relaxed">
              Set up recurring tasks and let the assistant remember context
              across conversations.
            </p>
          </div>
        </div>

        {/* Google data usage disclosure */}
        <div className="rounded-xl bg-kumo-base ring ring-kumo-line p-5 text-left mb-10">
          <h3 className="text-sm font-semibold text-kumo-default mb-2">
            How we use your Google data
          </h3>
          <ul className="text-sm text-kumo-secondary leading-relaxed space-y-2">
            <li>
              <strong className="text-kumo-default">Authentication</strong> — We
              use Google Sign-In to verify your identity and create your account
              (name, email, profile picture).
            </li>
            <li>
              <strong className="text-kumo-default">Google Drive access</strong>{" "}
              — When you mount your Google Drive, the AI assistant can browse,
              read, create, and edit files on your behalf. Drive is only
              accessed after you explicitly run the mount command. We do not
              scan or index your files in the background.
            </li>
            <li>
              <strong className="text-kumo-default">No other services</strong> —
              We do not access Gmail, Google Calendar, Contacts, or any other
              Google services.
            </li>
          </ul>
          <p className="text-sm text-kumo-secondary leading-relaxed mt-3">
            See our{" "}
            <a
              href="/privacy.html"
              className="underline hover:text-kumo-default transition-colors"
            >
              Privacy Policy
            </a>{" "}
            for full details on data handling and retention.
          </p>
        </div>
      </div>

      {/* Sign In Card */}
      <div className="w-full max-w-sm mx-auto px-4 pb-16">
        <div className="rounded-xl ring ring-kumo-line overflow-hidden bg-kumo-base p-8 text-center">
          <h2 className="text-xl font-semibold text-kumo-default mb-2">
            Sign In
          </h2>
          <p className="text-sm text-kumo-secondary mb-6">
            Sign in to start chatting with your personal AI assistant.
          </p>
          <a
            href="/auth/google"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-kumo-contrast text-kumo-inverse text-sm font-medium hover:opacity-90 transition-opacity"
          >
            <svg
              viewBox="0 0 24 24"
              width="18"
              height="18"
              className="shrink-0"
            >
              <path
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"
                fill="#4285F4"
              />
              <path
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                fill="#34A853"
              />
              <path
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                fill="#FBBC05"
              />
              <path
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                fill="#EA4335"
              />
            </svg>
            Sign in with Google
          </a>

          {/* Divider */}
          <div className="flex items-center gap-3 my-5">
            <div className="flex-1 h-px bg-kumo-line" />
            <span className="text-xs text-kumo-secondary">or</span>
            <div className="flex-1 h-px bg-kumo-line" />
          </div>

          {/* Email login */}
          {emailStep === "idle" && (
            <button
              onClick={startEmailLogin}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg ring ring-kumo-line text-kumo-default text-sm font-medium hover:bg-kumo-elevated transition-colors w-full justify-center"
            >
              <EnvelopeIcon size={18} className="shrink-0" />
              Sign in with Email
            </button>
          )}

          {emailStep === "waiting" && (
            <div className="space-y-3">
              <p className="text-sm text-kumo-secondary">
                Send an email from your mailbox to the address below. Subject
                and body can be anything.
              </p>
              <div className="flex items-center gap-2 bg-kumo-elevated rounded-lg px-3 py-2 ring ring-kumo-line">
                <code className="text-xs text-kumo-default flex-1 text-left break-all">
                  {address}
                </code>
                <button
                  onClick={copyAddress}
                  className="shrink-0 p-1 rounded hover:bg-kumo-base transition-colors"
                  title="Copy address"
                >
                  {copied ? (
                    <CheckIcon size={16} className="text-green-500" />
                  ) : (
                    <CopyIcon size={16} className="text-kumo-secondary" />
                  )}
                </button>
              </div>
              <div className="flex items-center justify-center gap-2 text-sm text-kumo-secondary">
                <SpinnerIcon size={16} className="animate-spin" />
                Waiting for your email...
              </div>
              <button
                onClick={cancelEmailLogin}
                className="text-xs text-kumo-secondary hover:text-kumo-default transition-colors"
              >
                Cancel
              </button>
            </div>
          )}

          {emailStep === "confirm" && (
            <div className="space-y-3">
              <p className="text-sm text-kumo-default">
                Received email from <strong>{senderEmail}</strong>
              </p>
              <p className="text-sm text-kumo-secondary">
                Confirm to sign in with this email?
              </p>
              <div className="flex gap-2 justify-center">
                <button
                  onClick={confirmLogin}
                  className="px-5 py-2 rounded-lg bg-kumo-contrast text-kumo-inverse text-sm font-medium hover:opacity-90 transition-opacity"
                >
                  Confirm
                </button>
                <button
                  onClick={cancelEmailLogin}
                  className="px-5 py-2 rounded-lg ring ring-kumo-line text-kumo-default text-sm font-medium hover:bg-kumo-elevated transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {emailStep === "rejected" && (
            <div className="space-y-3">
              <p className="text-sm text-kumo-default">
                We received your email, but it failed authentication.
              </p>
              <p className="text-sm text-kumo-secondary">
                {rejectReason || "The email could not be verified."} Please use
                Google Sign-In or try with a different email provider (e.g.
                Gmail, Outlook).
              </p>
              <button
                onClick={cancelEmailLogin}
                className="px-5 py-2 rounded-lg ring ring-kumo-line text-kumo-default text-sm font-medium hover:bg-kumo-elevated transition-colors"
              >
                Back
              </button>
            </div>
          )}

          {error && <p className="mt-3 text-sm text-red-500">{error}</p>}

          <div className="mt-6 pt-4 border-t border-kumo-line flex justify-center gap-4 text-xs text-kumo-secondary">
            <a
              href="/privacy.html"
              className="hover:text-kumo-default transition-colors"
            >
              Privacy Policy
            </a>
            <a
              href="/terms.html"
              className="hover:text-kumo-default transition-colors"
            >
              Terms of Service
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
