export interface Citation {
  url: string;
  title: string;
  excerpt?: string;
}

export interface RagResponse {
  response: string;
  citations: Citation[];
  confidence_score: number | null;
  intent: string | null;
  trace_id: string;
  latency_ms: number;
  backend: Backend;
  session_id?: string;
}

export interface RagRequest {
  query: string;
  session_id?: string;
}

export type Backend = "librarian" | "bedrock" | "triage" | "both";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  /** Present on assistant messages — one per active backend. */
  responses?: RagResponse[];
  errors?: string[];
}

export function isRagResponse(data: unknown): data is RagResponse {
  return (
    typeof data === "object" &&
    data !== null &&
    "response" in data &&
    typeof (data as RagResponse).response === "string" &&
    "citations" in data &&
    Array.isArray((data as RagResponse).citations)
  );
}
