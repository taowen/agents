import { getToolName } from "ai";
import type { UIMessage } from "ai";
import { Button, Badge, Surface, Text } from "@cloudflare/kumo";
import { GearIcon, CheckCircleIcon, XCircleIcon } from "@phosphor-icons/react";

type ToolUIPart = Extract<
  UIMessage["parts"][number],
  { type: `tool-${string}` } | { type: "dynamic-tool" }
>;

function BashToolOutput({ part }: { part: ToolUIPart }) {
  const bashInput = part.input as { command?: string } | undefined;
  const bashOutput = part.output as
    | {
        stdout?: string;
        stderr?: string;
        exitCode?: number;
      }
    | undefined;

  const cmd = bashInput?.command || "";
  const cmdShort = cmd.length > 80 ? cmd.slice(0, 80) + "\u2026" : cmd;
  const output = bashOutput?.stdout || "";
  const outputShort = output.split("\n")[0]?.slice(0, 80) || "";
  const exitOk = bashOutput?.exitCode === 0;

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] space-y-1">
        {/* Request (command) — collapsed by default */}
        <details className="group">
          <summary className="cursor-pointer list-none flex items-center gap-2 px-3 py-1.5 rounded-lg bg-kumo-base ring ring-kumo-line hover:bg-kumo-elevated transition-colors">
            <GearIcon size={12} className="text-kumo-inactive shrink-0" />
            <span className="font-mono text-xs text-kumo-secondary truncate">
              $ {cmdShort}
            </span>
          </summary>
          <div className="mt-1 px-3 py-2 rounded-lg bg-kumo-base ring ring-kumo-line font-mono text-xs whitespace-pre-wrap overflow-x-auto max-h-[300px] overflow-y-auto">
            <Text size="xs" variant="secondary">
              {cmd}
            </Text>
          </div>
        </details>
        {/* Response (output) — collapsed by default */}
        <details className="group">
          <summary className="cursor-pointer list-none flex items-center gap-2 px-3 py-1.5 rounded-lg bg-kumo-base ring ring-kumo-line hover:bg-kumo-elevated transition-colors">
            {exitOk ? (
              <CheckCircleIcon
                size={12}
                className="text-kumo-inactive shrink-0"
              />
            ) : (
              <XCircleIcon size={12} className="text-kumo-inactive shrink-0" />
            )}
            {exitOk ? (
              <span className="text-xs text-kumo-secondary truncate">
                {outputShort || "OK"}
              </span>
            ) : (
              <span className="text-xs text-kumo-secondary truncate">
                Exit {bashOutput?.exitCode}
                {bashOutput?.stderr
                  ? `: ${bashOutput.stderr.split("\n")[0]?.slice(0, 60)}`
                  : ""}
              </span>
            )}
          </summary>
          <div className="mt-1 px-3 py-2 rounded-lg bg-kumo-base ring ring-kumo-line font-mono text-xs whitespace-pre-wrap overflow-x-auto max-h-[400px] overflow-y-auto">
            {bashOutput?.stdout && (
              <Text size="xs" variant="secondary">
                {bashOutput.stdout}
              </Text>
            )}
            {bashOutput?.stderr && (
              <Text size="xs" variant="error">
                {bashOutput.stderr}
              </Text>
            )}
          </div>
        </details>
      </div>
    </div>
  );
}

function BrowserToolOutput({ part }: { part: ToolUIPart }) {
  const browserInput = part.input as
    | {
        action?: string;
        url?: string;
        selector?: string;
        text?: string;
        direction?: string;
      }
    | undefined;
  const browserOutput = part.output as
    | {
        action?: string;
        success?: boolean;
        url?: string;
        title?: string;
        text?: string;
        screenshot?: string;
        error?: string;
      }
    | undefined;

  if (!browserOutput) return null;

  const action = browserOutput.action || browserInput?.action || "";
  const url = browserOutput.url || browserInput?.url || "";
  const summaryText = `${action}${url ? " " + url : ""}`;
  const summaryShort =
    summaryText.length > 70 ? summaryText.slice(0, 70) + "\u2026" : summaryText;

  return (
    <div className="flex justify-start">
      <details className="max-w-[85%]">
        <summary className="cursor-pointer list-none flex items-center gap-2 px-3 py-1.5 rounded-lg bg-kumo-base ring ring-kumo-line hover:bg-kumo-elevated transition-colors">
          <GearIcon size={12} className="text-kumo-inactive shrink-0" />
          <span className="font-mono text-xs text-kumo-secondary truncate">
            {summaryShort}
          </span>
          {browserOutput.success ? (
            <Badge variant="secondary">OK</Badge>
          ) : (
            <Badge variant="destructive">Failed</Badge>
          )}
        </summary>
        <div className="mt-1 px-3 py-2 rounded-lg bg-kumo-base ring ring-kumo-line space-y-2">
          {browserOutput.error && (
            <Text size="xs" variant="error">
              {browserOutput.error}
            </Text>
          )}
          {browserOutput.url && browserOutput.title && (
            <Text size="xs" variant="secondary">
              {browserOutput.title} — {browserOutput.url}
            </Text>
          )}
          {browserOutput.screenshot && (
            <img
              src={`data:image/png;base64,${browserOutput.screenshot}`}
              alt="Browser screenshot"
              className="rounded border border-kumo-line max-w-full"
            />
          )}
          {browserOutput.text && (
            <pre className="p-2 bg-kumo-elevated rounded overflow-x-auto text-xs text-kumo-secondary max-h-[300px] overflow-y-auto">
              {browserOutput.text}
            </pre>
          )}
        </div>
      </details>
    </div>
  );
}

function GenericToolOutput({ part }: { part: ToolUIPart }) {
  const toolName = getToolName(part);

  return (
    <div className="flex justify-start">
      <details className="max-w-[85%]">
        <summary className="cursor-pointer list-none flex items-center gap-2 px-3 py-1.5 rounded-lg bg-kumo-base ring ring-kumo-line hover:bg-kumo-elevated transition-colors">
          <GearIcon size={12} className="text-kumo-inactive shrink-0" />
          <Text size="xs" variant="secondary" bold>
            {toolName}
          </Text>
          <Badge variant="secondary">Done</Badge>
        </summary>
        <div className="mt-1 px-3 py-2 rounded-lg bg-kumo-base ring ring-kumo-line font-mono text-xs whitespace-pre-wrap">
          <Text size="xs" variant="secondary">
            {JSON.stringify(part.output, null, 2)}
          </Text>
        </div>
      </details>
    </div>
  );
}

function ToolApproval({
  part,
  addToolApprovalResponse
}: {
  part: ToolUIPart;
  addToolApprovalResponse: (response: {
    id: string;
    approved: boolean;
  }) => void;
}) {
  const toolName = getToolName(part);
  const approvalId = ((part as any).approval as { id?: string })?.id;

  return (
    <div className="flex justify-start">
      <Surface className="max-w-[85%] px-4 py-3 rounded-xl ring-2 ring-kumo-warning">
        <div className="flex items-center gap-2 mb-2">
          <GearIcon size={14} className="text-kumo-warning" />
          <Text size="sm" bold>
            Approval needed: {toolName}
          </Text>
        </div>
        <div className="font-mono mb-3">
          <Text size="xs" variant="secondary">
            {JSON.stringify(part.input, null, 2)}
          </Text>
        </div>
        <div className="flex gap-2">
          <Button
            variant="primary"
            size="sm"
            icon={<CheckCircleIcon size={14} />}
            onClick={() => {
              if (approvalId) {
                addToolApprovalResponse({
                  id: approvalId,
                  approved: true
                });
              }
            }}
          >
            Approve
          </Button>
          <Button
            variant="secondary"
            size="sm"
            icon={<XCircleIcon size={14} />}
            onClick={() => {
              if (approvalId) {
                addToolApprovalResponse({
                  id: approvalId,
                  approved: false
                });
              }
            }}
          >
            Reject
          </Button>
        </div>
      </Surface>
    </div>
  );
}

function ToolRunning({ part }: { part: ToolUIPart }) {
  const toolName = getToolName(part);

  return (
    <div className="flex justify-start">
      <Surface className="max-w-[85%] px-4 py-2.5 rounded-xl ring ring-kumo-line">
        <div className="flex items-center gap-2">
          <GearIcon size={14} className="text-kumo-inactive animate-spin" />
          <Text size="xs" variant="secondary">
            Running {toolName}...
          </Text>
        </div>
      </Surface>
    </div>
  );
}

export function ToolOutput({
  part,
  addToolApprovalResponse
}: {
  part: ToolUIPart;
  addToolApprovalResponse: (response: {
    id: string;
    approved: boolean;
  }) => void;
}) {
  if (part.state === "output-available") {
    const toolName = getToolName(part);
    if (toolName === "bash") return <BashToolOutput part={part} />;
    if (toolName === "browser") return <BrowserToolOutput part={part} />;
    return <GenericToolOutput part={part} />;
  }

  if ("approval" in part && part.state === "approval-requested") {
    return (
      <ToolApproval
        part={part}
        addToolApprovalResponse={addToolApprovalResponse}
      />
    );
  }

  if (part.state === "input-available" || part.state === "input-streaming") {
    return <ToolRunning part={part} />;
  }

  return null;
}
