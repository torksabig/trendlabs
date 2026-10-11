"use client";

import {
  CandlestickSeries,
  ColorType,
  createChart,
  type IChartApi,
  type ISeriesApi,
} from "lightweight-charts";
import { useEffect, useRef } from "react";

import type { Candle } from "@/lib/types";

type Props = {
  candles: Candle[];
  height?: number;
};

export function CandleChart({ candles, height = 360 }: Props) {
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

    const series: ISeriesApi<"Candlestick"> = chart.addSeries(CandlestickSeries, {
      upColor: "#0f766e",
      downColor: "#b45309",
      borderVisible: false,
      wickUpColor: "#0f766e",
      wickDownColor: "#b45309",
    });

    series.setData(
      candles.map((c) => ({
        time: c.time as unknown as string,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      })),
    );
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
  }, [candles, height]);

  return <div ref={containerRef} className="w-full" style={{ height }} />;
}
