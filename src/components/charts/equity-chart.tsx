"use client";

import {
  ColorType,
  createChart,
  LineSeries,
  type IChartApi,
} from "lightweight-charts";
import { useEffect, useRef } from "react";

const COLORS = ["#3ecf8e", "#94a3b8", "#e11d48", "#cbd5e1"];

type Props = {
  curves: Record<string, Array<{ ts: string; equity: number }>>;
  labels: Record<string, string>;
  height?: number;
};

export function EquityChart({ curves, labels, height = 300 }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart: IChartApi = createChart(containerRef.current, {
      height,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#94a3b8",
      },
      grid: {
        vertLines: { color: "rgba(148, 163, 184, 0.08)" },
        horzLines: { color: "rgba(148, 163, 184, 0.08)" },
      },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false },
    });

    Object.entries(curves).forEach(([agentId, points], idx) => {
      const series = chart.addSeries(LineSeries, {
        color: COLORS[idx % COLORS.length],
        lineWidth: 2,
        title: labels[agentId] ?? agentId,
      });
      const sampled =
        points.length > 180
          ? points.filter((_, i) => i % Math.ceil(points.length / 180) === 0)
          : points;
      series.setData(
        sampled.map((p) => ({
          time: Math.floor(new Date(p.ts).getTime() / 1000) as unknown as string,
          value: p.equity,
        })),
      );
    });
    chart.timeScale().fitContent();

    const observer = new ResizeObserver(() => {
      if (!containerRef.current) return;
      chart.applyOptions({ width: containerRef.current.clientWidth });
    });
    observer.observe(containerRef.current);

    return () => {
      observer.disconnect();
      chart.remove();
    };
  }, [curves, labels, height]);

  return <div ref={containerRef} className="w-full" style={{ height }} />;
}
