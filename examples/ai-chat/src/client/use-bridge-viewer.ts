import { useCallback, useEffect, useRef, useState } from "react";

export interface BridgeDevice {
  deviceName: string;
  status: "connected" | "running";
}

export interface BridgeLog {
  time: string;
  message: string;
  deviceName?: string;
}

export function useBridgeViewer() {
  const [devices, setDevices] = useState<BridgeDevice[]>([]);
  const [logs, setLogs] = useState<BridgeLog[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(
      `${proto}//${location.host}/agents/bridge-manager/default`
    );
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: "cf_agent_bridge_subscribe" }));
    };

    ws.onmessage = (event) => {
      let data: any;
      try {
        data = JSON.parse(event.data);
      } catch {
        return;
      }

      if (data.type === "cf_agent_bridge_devices") {
        setDevices(data.devices ?? []);
      }

      if (data.type === "cf_agent_bridge_device_log") {
        const time = data.time
          ? new Date(data.time).toLocaleTimeString()
          : new Date().toLocaleTimeString();
        setLogs((prev) => [
          ...prev.slice(-199),
          { time, message: data.message, deviceName: data.deviceName }
        ]);
      }
    };

    ws.onclose = () => {
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      // Will trigger onclose → reconnect
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const clearLogs = useCallback(() => {
    setLogs([]);
  }, []);

  return { devices, logs, clearLogs };
}
