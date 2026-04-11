import type { RagResponse } from "./types";

const LIBRARIAN_URL =
  process.env.LIBRARIAN_API_URL ?? "http://localhost:8000";

export async function queryLibrarian(
  query: string,
  sessionId?: string,
): Promise<Omit<RagResponse, "latency_ms" | "backend">> {
  const resp = await fetch(`${LIBRARIAN_URL}/api/v1/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Request-ID": crypto.randomUUID(),
    },
    body: JSON.stringify({ query, session_id: sessionId }),
    signal: AbortSignal.timeout(60_000),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Librarian API ${resp.status}: ${text}`);
  }

  const data = await resp.json();

  return {
    response: data.response ?? "",
    citations: (data.citations ?? []).map(
      (c: { url?: string; title?: string }) => ({
        url: c.url ?? "",
        title: c.title ?? "Source",
      }),
    ),
    confidence_score: data.confidence_score ?? null,
    intent: data.intent ?? null,
    trace_id: data.trace_id ?? "",
    session_id: undefined,
  };
}
