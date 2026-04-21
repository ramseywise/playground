"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Sidebar } from "./Sidebar";
import { LatencyBadge } from "./LatencyBadge";
import { CitationList } from "./CitationList";
import type { Backend, ChatMessage, RagResponse } from "@/lib/types";
import { isRagResponse } from "@/lib/types";

async function fetchBackend(
  endpoint: string,
  query: string,
  sessionId?: string,
): Promise<RagResponse> {
  const resp = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, session_id: sessionId }),
  });
  const data = await resp.json();
  if (!isRagResponse(data)) {
    throw new Error(data.error ?? `Backend returned ${resp.status}`);
  }
  return data;
}

export function ComparisonView() {
  const [backend, setBackend] = useState<Backend>("both");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [librarianStatus, setLibrarianStatus] = useState<
    "connected" | "error" | "unknown"
  >("unknown");
  const [bedrockSessionId, setBedrockSessionId] = useState<
    string | undefined
  >();
  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Check librarian health on mount
  useEffect(() => {
    fetch("/api/librarian", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: "__health_check__" }),
    })
      .then((r) => {
        // Any non-network response means the proxy reached something
        setLibrarianStatus(r.status === 502 || r.status === 503 ? "error" : "connected");
      })
      .catch(() => setLibrarianStatus("error"));
  }, []);

  // Auto-scroll on new messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const lastResponses: RagResponse[] = (() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "assistant" && messages[i].responses?.length) {
        return messages[i].responses!;
      }
    }
    return [];
  })();

  const handleClearChat = useCallback(() => {
    setMessages([]);
    setBedrockSessionId(undefined);
  }, []);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const q = query.trim();
      if (!q || loading) return;

      setQuery("");
      setLoading(true);

      // Add user message
      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: q,
      };
      setMessages((prev) => [...prev, userMsg]);

      // Determine which backends to call
      const calls: Array<{
        endpoint: string;
        label: string;
        sessionId?: string;
      }> = [];
      if (backend === "librarian" || backend === "both") {
        calls.push({ endpoint: "/api/librarian", label: "librarian" });
      }
      if (backend === "bedrock" || backend === "both") {
        calls.push({
          endpoint: "/api/bedrock",
          label: "bedrock",
          sessionId: bedrockSessionId,
        });
      }

      const results = await Promise.allSettled(
        calls.map((c) => fetchBackend(c.endpoint, q, c.sessionId)),
      );

      const responses: RagResponse[] = [];
      const errors: string[] = [];

      results.forEach((result, i) => {
        if (result.status === "fulfilled") {
          responses.push(result.value);
          if (result.value.backend === "bedrock" && result.value.session_id) {
            setBedrockSessionId(result.value.session_id);
          }
        } else {
          errors.push(`${calls[i].label}: ${result.reason}`);
        }
      });

      // Build display content — in "both" mode, show side by side
      let content: string;
      if (responses.length === 1) {
        content = responses[0].response;
      } else if (responses.length > 1) {
        content = responses.map((r) => r.response).join("\n");
      } else {
        content = errors.join("\n") || "No response received.";
      }

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content,
        responses,
        errors: errors.length > 0 ? errors : undefined,
      };
      setMessages((prev) => [...prev, assistantMsg]);
      setLoading(false);
      inputRef.current?.focus();
    },
    [query, loading, backend, bedrockSessionId],
  );

  return (
    <div className="flex h-full">
      <Sidebar
        backend={backend}
        onBackendChange={setBackend}
        onClearChat={handleClearChat}
        librarianStatus={librarianStatus}
        lastResponses={lastResponses}
      />

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 && (
            <p className="text-sm text-gray-400 text-center mt-8">
              Ask the librarian...
            </p>
          )}

          {messages.map((msg) => (
            <div key={msg.id}>
              {msg.role === "user" ? (
                /* User bubble */
                <div className="flex justify-end">
                  <div className="bg-blue-600 text-white rounded-lg px-4 py-2 max-w-[70%]">
                    <p className="text-sm">{msg.content}</p>
                  </div>
                </div>
              ) : (
                /* Assistant bubble(s) */
                <div className="flex justify-start">
                  <div className="max-w-[85%] space-y-2">
                    {/* Errors */}
                    {msg.errors?.map((err, i) => (
                      <div
                        key={i}
                        className="bg-red-50 border border-red-200 rounded-lg px-4 py-2 text-sm text-red-700"
                      >
                        {err}
                      </div>
                    ))}

                    {/* Single backend — one bubble */}
                    {msg.responses?.length === 1 && (
                      <div className="bg-gray-100 rounded-lg px-4 py-3">
                        <p className="text-sm whitespace-pre-wrap">
                          {msg.responses[0].response}
                        </p>
                      </div>
                    )}

                    {/* Both backends — side by side */}
                    {msg.responses && msg.responses.length > 1 && (
                      <div className="grid grid-cols-2 gap-3">
                        {msg.responses.map((resp) => (
                          <div
                            key={resp.backend}
                            className={`rounded-lg px-4 py-3 ${
                              resp.backend === "librarian"
                                ? "bg-purple-50 border border-purple-200"
                                : "bg-orange-50 border border-orange-200"
                            }`}
                          >
                            <div className="flex items-center gap-2 mb-2">
                              <span className="text-xs font-semibold uppercase text-gray-500">
                                {resp.backend === "librarian"
                                  ? "Python RAG"
                                  : "AWS Bedrock"}
                              </span>
                              <LatencyBadge ms={resp.latency_ms} />
                            </div>
                            <p className="text-sm whitespace-pre-wrap">
                              {resp.response}
                            </p>
                            {resp.citations.length > 0 && (
                              <div className="mt-2 pt-2 border-t border-gray-200">
                                <CitationList citations={resp.citations} />
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-gray-100 rounded-lg px-4 py-3">
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <svg
                    className="animate-spin h-4 w-4"
                    viewBox="0 0 24 24"
                    fill="none"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                    />
                  </svg>
                  Thinking...
                </div>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* Chat input */}
        <div className="border-t bg-white p-4">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask the librarian..."
              disabled={loading}
              className="flex-1 border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
              autoFocus
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Querying..." : "Send"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
