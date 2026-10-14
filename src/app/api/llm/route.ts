import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

/**
 * Status of configured research LLM providers.
 * Never returns secret values — only booleans + default model ids from env.
 */
export async function GET() {
  const google = Boolean(
    process.env.GOOGLE_AI_API_KEY || process.env.GEMINI_API_KEY,
  );
  const groq = Boolean(process.env.GROQ_API_KEY);
  const openrouter = Boolean(process.env.OPENROUTER_API_KEY);
  const cloudflare = Boolean(
    process.env.CLOUDFLARE_ACCOUNT_ID && process.env.CLOUDFLARE_API_TOKEN,
  );
  const cerebras = Boolean(process.env.CEREBRAS_API_KEY);
  const mistral = Boolean(process.env.MISTRAL_API_KEY);
  const ollama = process.env.VERCEL
    ? false
    : process.env.OLLAMA_AVAILABLE === "1" || Boolean(process.env.OLLAMA_HOST);

  const providers = [
    {
      id: "auto",
      label: "Auto (free cloud → Ollama → fallback)",
      configured: true,
      default_model: "",
    },
    {
      id: "google",
      label: "Google AI Studio (Gemini)",
      configured: google,
      default_model: process.env.GOOGLE_AI_MODEL || "gemini-2.0-flash",
    },
    {
      id: "groq",
      label: "Groq",
      configured: groq,
      default_model: process.env.GROQ_MODEL || "llama-3.3-70b-versatile",
    },
    {
      id: "openrouter",
      label: "OpenRouter",
      configured: openrouter,
      default_model: process.env.OPENROUTER_MODEL || "openai/gpt-6-luna",
    },
    {
      id: "cloudflare",
      label: "Cloudflare Workers AI",
      configured: cloudflare,
      default_model:
        process.env.CLOUDFLARE_AI_MODEL || "@cf/meta/llama-3.1-8b-instruct",
    },
    {
      id: "cerebras",
      label: "Cerebras",
      configured: cerebras,
      default_model: process.env.CEREBRAS_MODEL || "llama3.1-8b",
    },
    {
      id: "mistral",
      label: "Mistral",
      configured: mistral,
      default_model: process.env.MISTRAL_MODEL || "mistral-small-latest",
    },
    {
      id: "ollama",
      label: "Ollama (local)",
      configured: ollama,
      default_model: process.env.OLLAMA_MODEL || "qwen3:14b-q4_K_M",
    },
    {
      id: "fallback",
      label: "Rule-based fallback",
      configured: true,
      default_model: "rule_based_fallback",
    },
  ];

  const anyCloud = google || groq || openrouter || cloudflare || cerebras || mistral;

  return NextResponse.json({
    ok: true,
    default_provider: process.env.TRENDLAB_LLM_PROVIDER || "auto",
    any_cloud_configured: anyCloud,
    job_mode_hint: process.env.TRENDLAB_JOB_MODE || (process.env.VERCEL ? "github_actions" : "local"),
    providers,
  });
}
