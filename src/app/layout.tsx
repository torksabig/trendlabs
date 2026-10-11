import type { Metadata } from "next";
import { IBM_Plex_Mono, Manrope, Newsreader } from "next/font/google";

import { SiteHeader } from "@/components/layout/site-header";

import "./globals.css";

const display = Newsreader({
  variable: "--font-display",
  subsets: ["latin"],
  style: ["normal", "italic"],
});

const sans = Manrope({
  variable: "--font-sans",
  subsets: ["latin"],
});

const mono = IBM_Plex_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "TrendLab — Multi-Agent Paper Trading Research",
  description:
    "Automated multi-agent stock market research and paper trading on a $0/month stack.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${display.variable} ${sans.variable} ${mono.variable} font-sans antialiased`}>
        <div className="relative z-10 min-h-screen">
          <SiteHeader />
          <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">{children}</main>
        </div>
      </body>
    </html>
  );
}
