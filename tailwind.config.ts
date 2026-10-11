import type { Config } from "tailwindcss";

export default {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        paper: "var(--paper)",
        ink: "var(--ink)",
        charcoal: {
          DEFAULT: "var(--charcoal)",
          2: "var(--charcoal-2)",
          3: "var(--charcoal-3)",
        },
        slateui: {
          DEFAULT: "var(--slate)",
          dim: "var(--slate-dim)",
        },
        mint: {
          DEFAULT: "var(--mint)",
          dim: "var(--mint-dim)",
        },
        crimson: {
          DEFAULT: "var(--crimson)",
          dim: "var(--crimson-dim)",
        },
        teal: {
          DEFAULT: "var(--mint)",
          700: "var(--mint)",
          800: "var(--mint)",
        },
      },
      fontFamily: {
        display: ["var(--font-display)", "Georgia", "serif"],
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;
