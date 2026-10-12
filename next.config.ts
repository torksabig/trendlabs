import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Keep Next from auto-writing AGENTS.md into the repo.
  agentRules: false,
};

export default nextConfig;
