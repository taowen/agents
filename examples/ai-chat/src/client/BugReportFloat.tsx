import { useState, useCallback } from "react";
import { Button } from "@cloudflare/kumo";
import { CopyIcon, XIcon } from "@phosphor-icons/react";
import { reportBug } from "./api";

interface BugReportFloatProps {
  open: boolean;
  onClose: () => void;
  sessionId: string | null;
}

const isMac =
  typeof navigator !== "undefined" &&
  /Mac|iPhone|iPad/.test(navigator.platform);

export function BugReportFloat({
  open,
  onClose,
  sessionId
}: BugReportFloatProps) {
  const [description, setDescription] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [reportId, setReportId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleClose = useCallback(() => {
    onClose();
    setDescription("");
    setReportId(null);
    setError(null);
  }, [onClose]);

  const handleSubmit = useCallback(async () => {
    if (!description.trim() || !sessionId) return;
    setSubmitting(true);
    setError(null);
    try {
      const { reportId: id } = await reportBug(sessionId, description.trim());
      setReportId(id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to submit report");
    } finally {
      setSubmitting(false);
    }
  }, [description, sessionId]);

  if (!open) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 w-[380px] max-w-[calc(100vw-2rem)] bg-kumo-base rounded-xl border border-kumo-line shadow-lg">
      <div className="flex items-center justify-between px-4 pt-4 pb-2">
        <h2 className="text-lg font-semibold text-kumo-default">Report Bug</h2>
        <button
          onClick={handleClose}
          className="p-1 rounded-lg hover:bg-kumo-elevated text-kumo-secondary hover:text-kumo-default transition-colors"
        >
          <XIcon size={18} />
        </button>
      </div>

      <div className="px-4 pb-2">
        {!reportId ? (
          <>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe what went wrong..."
              rows={4}
              className="w-full rounded-lg border border-kumo-line bg-kumo-elevated text-kumo-default placeholder:text-kumo-tertiary p-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-kumo-ring"
              disabled={submitting}
            />
            {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
            <div className="flex justify-end gap-2 mt-3">
              <Button variant="secondary" onClick={handleClose}>
                Cancel
              </Button>
              <Button
                variant="primary"
                onClick={handleSubmit}
                disabled={!description.trim() || submitting || !sessionId}
              >
                {submitting ? "Submitting..." : "Submit"}
              </Button>
            </div>
          </>
        ) : (
          <>
            <p className="text-sm text-kumo-secondary mb-3">
              Bug report submitted. Share this ID with the developer:
            </p>
            <div className="flex items-center gap-2 bg-kumo-elevated rounded-lg border border-kumo-line p-3">
              <code className="flex-1 text-sm font-mono text-kumo-default break-all">
                {reportId}
              </code>
              <button
                onClick={() => navigator.clipboard.writeText(reportId)}
                className="p-1.5 rounded-md hover:bg-kumo-base text-kumo-secondary hover:text-kumo-default transition-colors"
                title="Copy to clipboard"
              >
                <CopyIcon size={16} />
              </button>
            </div>
            <div className="flex justify-end mt-3">
              <Button variant="secondary" onClick={handleClose}>
                Close
              </Button>
            </div>
          </>
        )}
      </div>

      <div className="px-4 pb-3 pt-1 border-t border-kumo-line">
        <p className="text-xs text-kumo-tertiary text-center">
          Press{" "}
          <kbd className="px-1.5 py-0.5 rounded bg-kumo-elevated border border-kumo-line text-[11px] font-mono">
            {isMac ? "⌘⇧B" : "Ctrl+Shift+B"}
          </kbd>{" "}
          to toggle
        </p>
      </div>
    </div>
  );
}
