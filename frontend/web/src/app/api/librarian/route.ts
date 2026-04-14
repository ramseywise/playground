import { NextRequest, NextResponse } from "next/server";
import { queryLibrarian } from "@/lib/librarian";
import type { RagRequest } from "@/lib/types";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  const start = Date.now();

  let body: RagRequest;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  if (!body.query?.trim()) {
    return NextResponse.json({ error: "query is required" }, { status: 400 });
  }

  try {
    const result = await queryLibrarian(body.query, body.session_id);
    return NextResponse.json({
      ...result,
      latency_ms: Date.now() - start,
      backend: "librarian" as const,
    });
  } catch (err) {
    const message =
      err instanceof Error ? err.message : "Librarian call failed";
    const status = message.includes("fetch") ? 503 : 502;
    return NextResponse.json({ error: message }, { status });
  }
}
