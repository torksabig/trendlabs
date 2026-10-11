"use client";

import {
  ColorType,
  createChart,
  LineSeries,
  type IChartApi,
} from "lightweight-charts";
import { useEffect, useRef } from "react";

const COLORS = ["#0f766e", "#1d4e89", "#b45309", "#365314"];

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
        textColor: "#2a3a34",
      },
      grid: {
        vertLines: { color: "rgba(15, 40, 32, 0.06)" },
        horzLines: { color: "rgba(15, 40, 32, 0.06)" },
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
