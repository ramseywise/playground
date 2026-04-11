import { NextRequest, NextResponse } from "next/server";
import { retrieveAndGenerate } from "@/lib/bedrock";
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

  if (!process.env.BEDROCK_KNOWLEDGE_BASE_ID) {
    return NextResponse.json(
      { error: "BEDROCK_KNOWLEDGE_BASE_ID is not configured" },
      { status: 503 },
    );
  }

  try {
    const result = await retrieveAndGenerate(body.query, body.session_id);
    return NextResponse.json({
      ...result,
      latency_ms: Date.now() - start,
      backend: "bedrock" as const,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Bedrock call failed";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
