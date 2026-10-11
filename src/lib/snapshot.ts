import { promises as fs } from "fs";
import path from "path";

import type { Snapshot } from "@/lib/types";

export async function loadSnapshot(): Promise<Snapshot | null> {
  const file = path.join(process.cwd(), "public", "data", "snapshot.json");
  try {
    const raw = await fs.readFile(/* turbopackIgnore: true */ file, "utf8");
    return JSON.parse(raw) as Snapshot;
  } catch {
    return null;
  }
}
