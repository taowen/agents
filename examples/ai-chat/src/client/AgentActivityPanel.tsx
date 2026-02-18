import { useEffect, useRef, useCallback, useState } from "react";
import { XIcon } from "@phosphor-icons/react";
import type { BridgeLog } from "./use-bridge";

interface AgentActivityPanelProps {
  logs: BridgeLog[];
  onClose: () => void;
  onClear: () => void;
}

const MIN_HEIGHT = 100;
const DEFAULT_HEIGHT = 200;
const MAX_HEIGHT_RATIO = 0.5;

export function AgentActivityPanel({
  logs,
  onClose,
  onClear
}: AgentActivityPanelProps) {
  const [height, setHeight] = useState(DEFAULT_HEIGHT);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const startY = useRef(0);
  const startHeight = useRef(0);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      dragging.current = true;
      startY.current = e.clientY;
      startHeight.current = height;

      const onMouseMove = (ev: MouseEvent) => {
        if (!dragging.current) return;
        const delta = startY.current - ev.clientY;
        const maxH = window.innerHeight * MAX_HEIGHT_RATIO;
        const newH = Math.min(
          maxH,
          Math.max(MIN_HEIGHT, startHeight.current + delta)
        );
        setHeight(newH);
      };

      const onMouseUp = () => {
        dragging.current = false;
        document.removeEventListener("mousemove", onMouseMove);
        document.removeEventListener("mouseup", onMouseUp);
      };

      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", onMouseUp);
    },
    [height]
  );

  return (
    <div className="activity-panel" style={{ height }}>
      {/* Drag handle */}
      <div className="activity-panel-handle" onMouseDown={onMouseDown} />

      {/* Header */}
      <div className="activity-panel-header">
        <span className="activity-panel-title">Agent Activity</span>
        <div className="activity-panel-actions">
          <button className="activity-panel-btn" onClick={onClear}>
            Clear
          </button>
          <button className="activity-panel-btn" onClick={onClose}>
            <XIcon size={14} />
          </button>
        </div>
      </div>

      {/* Logs */}
      <div className="activity-panel-logs">
        {logs.length === 0 ? (
          <div className="activity-log-empty">No bridge activity yet</div>
        ) : (
          logs.map((log, i) => (
            <div key={i} className="activity-log-entry">
              <span className="activity-log-time">{log.time}</span>
              <span className="activity-log-msg">{log.message}</span>
            </div>
          ))
        )}
        <div ref={logsEndRef} />
      </div>
    </div>
  );
}
