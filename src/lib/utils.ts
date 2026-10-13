import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatMoney(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value);
}

export function formatPct(value: number) {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

export function pctTone(value: number) {
  if (value > 0) return "text-mint";
  if (value < 0) return "text-crimson";
  return "text-slateui";
}

/** Absolute local datetime for update stamps. */
export function formatWhen(value: string | null | undefined) {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    timeZoneName: "short",
  });
}

function formatDuration(ms: number) {
  const sec = Math.floor(ms / 1000);
  if (sec < 60) return `${sec}s`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m`;
  const hr = Math.floor(min / 60);
  if (hr < 48) return `${hr}h`;
  const day = Math.floor(hr / 24);
  return `${day}d`;
}

/** Relative age, e.g. "12m ago". */
export function formatAge(value: string | null | undefined) {
  if (!value) return "never";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "unknown";
  const ms = Date.now() - d.getTime();
  if (ms < 0) return "just now";
  return `${formatDuration(ms)} ago`;
}

/** Time until a future timestamp, e.g. "in 42m". */
export function formatUntil(value: string | null | undefined) {
  if (!value) return "pending";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "unknown";
  const ms = d.getTime() - Date.now();
  if (ms <= 0) return "due now";
  return `in ${formatDuration(ms)}`;
}

export function directionTone(direction: string) {
  if (direction === "up" || direction === "bullish") return "text-mint";
  if (direction === "down" || direction === "bearish") return "text-crimson";
  return "text-slateui";
}
