import type { RagResponse } from "@/lib/types";
import { CitationList } from "./CitationList";
import { LatencyBadge } from "./LatencyBadge";

interface ChatPanelProps {
  title: string;
  response: RagResponse | null;
  loading: boolean;
  error: string | null;
  accentClass?: string;
}

export function ChatPanel({
  title,
  response,
  loading,
  error,
  accentClass,
}: ChatPanelProps) {
  return (
    <div className="flex flex-col h-full border rounded-lg overflow-hidden bg-white">
      {/* Header */}
      <div
        className={`px-4 py-2 font-semibold text-sm flex items-center gap-2 border-b ${accentClass ?? "bg-gray-50"}`}
      >
        <span>{title}</span>
        {response && <LatencyBadge ms={response.latency_ms} />}
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {loading && (
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
        )}

        {error && (
          <div className="rounded bg-red-50 border border-red-200 p-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {response && (
          <>
            <div className="prose prose-sm max-w-none">
              <p className="whitespace-pre-wrap">{response.response}</p>
            </div>

            {/* Metadata */}
            <div className="flex flex-wrap gap-3 text-xs text-gray-500 border-t pt-2">
              {response.confidence_score !== null && (
                <span>
                  Confidence:{" "}
                  {(response.confidence_score * 100).toFixed(0)}%
                </span>
              )}
              {response.confidence_score === null && (
                <span className="text-gray-400">Confidence: N/A</span>
              )}
              {response.intent && <span>Intent: {response.intent}</span>}
              {response.trace_id && (
                <span className="font-mono">
                  trace: {response.trace_id.slice(0, 8)}
                </span>
              )}
            </div>

            {/* Citations */}
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-1">
                Sources ({response.citations.length})
              </p>
              <CitationList citations={response.citations} />
            </div>
          </>
        )}

        {!loading && !error && !response && (
          <p className="text-sm text-gray-400">
            Responses will appear here.
          </p>
        )}
      </div>
    </div>
  );
}
