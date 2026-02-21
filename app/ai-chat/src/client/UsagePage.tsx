import { useState, useMemo } from "react";
import { useNavigate, useOutletContext } from "react-router";
import { ArrowLeftIcon, ListIcon, ChartBarIcon } from "@phosphor-icons/react";
import { Text } from "@cloudflare/kumo";
import { useUsageStats, useQuotaStatus } from "./api";
import type { UsageRow } from "./api";
import type { AuthLayoutContext } from "./AuthLayout";
import { Skeleton } from "./Skeleton";

function todayStr() {
  return new Date().toISOString().slice(0, 10);
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return String(n);
}

type KeyFilter = "all" | "builtin" | "custom";

/** Re-aggregate rows by hour after filtering by api_key_type. */
function aggregateByHour(
  rows: UsageRow[],
  filter: KeyFilter
): {
  hour: string;
  request_count: number;
  input_tokens: number;
  cache_read_tokens: number;
  output_tokens: number;
}[] {
  const filtered =
    filter === "all" ? rows : rows.filter((r) => r.api_key_type === filter);

  const map = new Map<
    string,
    {
      hour: string;
      request_count: number;
      input_tokens: number;
      cache_read_tokens: number;
      output_tokens: number;
    }
  >();

  for (const r of filtered) {
    const existing = map.get(r.hour);
    if (existing) {
      existing.request_count += r.request_count || 0;
      existing.input_tokens += r.input_tokens || 0;
      existing.cache_read_tokens += r.cache_read_tokens || 0;
      existing.output_tokens += r.output_tokens || 0;
    } else {
      map.set(r.hour, {
        hour: r.hour,
        request_count: r.request_count || 0,
        input_tokens: r.input_tokens || 0,
        cache_read_tokens: r.cache_read_tokens || 0,
        output_tokens: r.output_tokens || 0
      });
    }
  }

  return [...map.values()].sort((a, b) => a.hour.localeCompare(b.hour));
}

export function UsagePage() {
  const navigate = useNavigate();
  const { onOpenSidebar } = useOutletContext<AuthLayoutContext>();
  const [selectedDate, setSelectedDate] = useState(todayStr);
  const [keyFilter, setKeyFilter] = useState<KeyFilter>("all");

  // Build hourly range for the selected day
  const start = selectedDate + "T00";
  const end = selectedDate + "T23";

  const { usage, error, isLoading } = useUsageStats(start, end);
  const { quota } = useQuotaStatus();

  const filteredUsage = useMemo(
    () => aggregateByHour(usage, keyFilter),
    [usage, keyFilter]
  );

  // Summary totals
  const totals = filteredUsage.reduce(
    (acc, r) => ({
      request_count: acc.request_count + r.request_count,
      input_tokens: acc.input_tokens + r.input_tokens,
      cache_read_tokens: acc.cache_read_tokens + r.cache_read_tokens,
      output_tokens: acc.output_tokens + r.output_tokens
    }),
    {
      request_count: 0,
      input_tokens: 0,
      cache_read_tokens: 0,
      output_tokens: 0
    }
  );

  const cellClass = "px-3 py-2 text-right text-sm tabular-nums";
  const headerCellClass =
    "px-3 py-2 text-right text-xs font-medium text-kumo-secondary";

  const filterBtnClass = (active: boolean) =>
    `px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
      active
        ? "bg-kumo-contrast text-kumo-inverse"
        : "bg-kumo-elevated text-kumo-secondary hover:text-kumo-default"
    }`;

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-3xl mx-auto px-5 py-6">
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
          <ChartBarIcon size={20} className="text-kumo-default" />
          <h2 className="text-lg font-semibold text-kumo-default">
            Token Usage
          </h2>
        </div>

        {/* Quota exceeded banner */}
        {quota?.exceeded && (
          <div className="rounded-xl bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 p-3 mb-5">
            <Text size="sm" variant="secondary">
              <span className="text-amber-700 dark:text-amber-400">
                Builtin API key quota exceeded
                {quota.exceededAt
                  ? ` (since ${new Date(quota.exceededAt + "Z").toLocaleString()})`
                  : ""}
                . Please configure your own API key in Settings to continue.
              </span>
            </Text>
          </div>
        )}

        {/* Date picker + filter toggle */}
        <div className="flex items-center gap-3 mb-5 flex-wrap">
          <label className="text-sm text-kumo-secondary">Date:</label>
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="px-3 py-1.5 rounded-lg border border-kumo-line bg-kumo-elevated text-kumo-default text-sm focus:outline-none focus:ring-2 focus:ring-kumo-ring"
          />
          <div className="flex gap-1 ml-auto">
            <button
              className={filterBtnClass(keyFilter === "all")}
              onClick={() => setKeyFilter("all")}
            >
              All
            </button>
            <button
              className={filterBtnClass(keyFilter === "builtin")}
              onClick={() => setKeyFilter("builtin")}
            >
              Builtin
            </button>
            <button
              className={filterBtnClass(keyFilter === "custom")}
              onClick={() => setKeyFilter("custom")}
            >
              Custom
            </button>
          </div>
        </div>

        {/* Error banner */}
        {error && (
          <div className="rounded-xl bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 p-3 mb-5">
            <Text size="sm" variant="secondary">
              <span className="text-red-700 dark:text-red-400">
                Failed to load usage data. Please try again later.
              </span>
            </Text>
          </div>
        )}

        {/* Summary row */}
        <div className="rounded-xl ring ring-kumo-line bg-kumo-base p-4 mb-5">
          {isLoading ? (
            <Skeleton className="h-12 w-full" />
          ) : (
            <div className="grid grid-cols-4 gap-4 text-center">
              <div>
                <div className="text-lg font-semibold text-kumo-default tabular-nums">
                  {formatNumber(totals.request_count)}
                </div>
                <Text size="xs" variant="secondary">
                  Requests
                </Text>
              </div>
              <div>
                <div className="text-lg font-semibold text-kumo-default tabular-nums">
                  {formatNumber(totals.input_tokens)}
                </div>
                <Text size="xs" variant="secondary">
                  Input
                </Text>
              </div>
              <div>
                <div className="text-lg font-semibold text-kumo-default tabular-nums">
                  {formatNumber(totals.cache_read_tokens)}
                </div>
                <Text size="xs" variant="secondary">
                  Cache Read
                </Text>
              </div>
              <div>
                <div className="text-lg font-semibold text-kumo-default tabular-nums">
                  {formatNumber(totals.output_tokens)}
                </div>
                <Text size="xs" variant="secondary">
                  Output
                </Text>
              </div>
            </div>
          )}
        </div>

        {/* Hourly table */}
        <div className="rounded-xl ring ring-kumo-line bg-kumo-base overflow-hidden">
          {isLoading ? (
            <div className="p-4">
              <Skeleton className="h-48 w-full" />
            </div>
          ) : filteredUsage.length === 0 ? (
            <div className="px-4 py-8 text-center">
              <Text size="sm" variant="secondary">
                No usage data for this date.
              </Text>
            </div>
          ) : (
            <table className="w-full">
              <thead>
                <tr className="border-b border-kumo-line">
                  <th className="px-3 py-2 text-left text-xs font-medium text-kumo-secondary">
                    Hour
                  </th>
                  <th className={headerCellClass}>Requests</th>
                  <th className={headerCellClass}>Input</th>
                  <th className={headerCellClass}>Cache Read</th>
                  <th className={headerCellClass}>Output</th>
                </tr>
              </thead>
              <tbody>
                {filteredUsage.map((row) => (
                  <tr
                    key={row.hour}
                    className="border-b border-kumo-line last:border-0 hover:bg-kumo-elevated/50"
                  >
                    <td className="px-3 py-2 text-sm text-kumo-default">
                      {row.hour.slice(-2)}:00
                    </td>
                    <td className={cellClass}>{row.request_count}</td>
                    <td className={cellClass}>
                      {formatNumber(row.input_tokens)}
                    </td>
                    <td className={cellClass}>
                      {formatNumber(row.cache_read_tokens)}
                    </td>
                    <td className={cellClass}>
                      {formatNumber(row.output_tokens)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
