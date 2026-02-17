/**
 * Shared timezone utilities for date, uptime, printf/strftime, etc.
 *
 * All functions use Intl.DateTimeFormat to resolve timezone-aware date parts,
 * offsets, and abbreviations. They fall back gracefully on invalid TZ values.
 */

/**
 * Date components broken out by field, with month 1-based.
 */
export interface DateParts {
  year: number;
  /** 1-based month (1 = January) */
  month: number;
  day: number;
  hour: number;
  minute: number;
  second: number;
  /** 0 = Sunday … 6 = Saturday */
  weekday: number;
}

const WEEKDAY_MAP = new Map<string, number>([
  ["Sun", 0],
  ["Mon", 1],
  ["Tue", 2],
  ["Wed", 3],
  ["Thu", 4],
  ["Fri", 5],
  ["Sat", 6]
]);

/**
 * Get date/time parts in a specific timezone using Intl.DateTimeFormat.
 * Falls back to local time if the timezone string is invalid.
 */
export function getDatePartsInTz(date: Date, tz?: string): DateParts {
  const options: Intl.DateTimeFormatOptions = {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    weekday: "short",
    hour12: false,
    timeZone: tz
  };

  try {
    const formatter = new Intl.DateTimeFormat("en-US", options);
    const parts = formatter.formatToParts(date);
    const getValue = (type: string): string =>
      parts.find((p) => p.type === type)?.value ?? "";

    const weekdayStr = getValue("weekday");

    return {
      year: Number.parseInt(getValue("year"), 10) || date.getFullYear(),
      month: Number.parseInt(getValue("month"), 10) || date.getMonth() + 1,
      day: Number.parseInt(getValue("day"), 10) || date.getDate(),
      hour: Number.parseInt(getValue("hour"), 10) || date.getHours(),
      minute: Number.parseInt(getValue("minute"), 10) || date.getMinutes(),
      second: Number.parseInt(getValue("second"), 10) || date.getSeconds(),
      weekday: WEEKDAY_MAP.get(weekdayStr) ?? date.getDay()
    };
  } catch {
    return {
      year: date.getFullYear(),
      month: date.getMonth() + 1,
      day: date.getDate(),
      hour: date.getHours(),
      minute: date.getMinutes(),
      second: date.getSeconds(),
      weekday: date.getDay()
    };
  }
}

/**
 * Get timezone offset string like "+0800" or "-0500".
 * If no tz is provided, uses the local timezone offset.
 */
export function getTzOffset(date: Date, tz?: string): string {
  if (tz) {
    try {
      const formatter = new Intl.DateTimeFormat("en-US", {
        timeZone: tz,
        timeZoneName: "longOffset"
      });
      const parts = formatter.formatToParts(date);
      const tzPart = parts.find((p) => p.type === "timeZoneName");
      if (tzPart) {
        const match = tzPart.value.match(/GMT([+-])(\d{2}):(\d{2})/);
        if (match) return `${match[1]}${match[2]}${match[3]}`;
        if (tzPart.value === "GMT" || tzPart.value === "UTC") return "+0000";
      }
    } catch {
      // fall through to local offset
    }
  }

  const offset = -date.getTimezoneOffset();
  const sign = offset >= 0 ? "+" : "-";
  const hours = Math.floor(Math.abs(offset) / 60);
  const mins = Math.abs(offset) % 60;
  return `${sign}${String(hours).padStart(2, "0")}${String(mins).padStart(2, "0")}`;
}

/**
 * Get timezone abbreviation like "CST", "EDT", "UTC".
 * Falls back to "UTC" on error.
 */
export function getTzName(date: Date, tz?: string): string {
  try {
    const formatter = new Intl.DateTimeFormat("en-US", {
      timeZone: tz,
      timeZoneName: "short"
    });
    const parts = formatter.formatToParts(date);
    const tzPart = parts.find((p) => p.type === "timeZoneName");
    return tzPart?.value ?? "UTC";
  } catch {
    return "UTC";
  }
}
