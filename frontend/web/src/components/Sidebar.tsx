"use client";

import type { Backend, RagResponse } from "@/lib/types";
import { CitationList } from "./CitationList";
import { LatencyBadge } from "./LatencyBadge";

interface SidebarProps {
  backend: Backend;
  onBackendChange: (b: Backend) => void;
  onClearChat: () => void;
  librarianStatus: "connected" | "error" | "unknown";
  lastResponses: RagResponse[];
}

export function Sidebar({
  backend,
  onBackendChange,
  onClearChat,
  librarianStatus,
  lastResponses,
}: SidebarProps) {
  return (
    <aside className="w-72 bg-white border-r flex flex-col h-full overflow-y-auto">
      <div className="p-4 space-y-4">
        {/* Title */}
        <h1 className="text-lg font-bold">Librarian RAG Playground</h1>
        <hr />

        {/* Connection status */}
        <div>
          {librarianStatus === "connected" && (
            <div className="text-sm text-green-700 bg-green-50 border border-green-200 rounded px-3 py-1.5">
              Librarian API connected
            </div>
          )}
          {librarianStatus === "error" && (
            <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded px-3 py-1.5">
              Cannot reach Librarian API
            </div>
          )}
          {librarianStatus === "unknown" && (
            <div className="text-sm text-gray-500 bg-gray-50 border border-gray-200 rounded px-3 py-1.5">
              Librarian: checking...
            </div>
          )}
        </div>

        {/* Backend selector */}
        <div>
          <label className="text-xs font-semibold uppercase tracking-wide text-gray-500 block mb-1.5">
            Backend
          </label>
          <select
            value={backend}
            onChange={(e) => onBackendChange(e.target.value as Backend)}
            className="w-full border rounded px-2 py-1.5 text-sm bg-white"
          >
            <option value="librarian">Python RAG (Librarian)</option>
            <option value="bedrock">AWS Bedrock KB</option>
            <option value="both">Both (side by side)</option>
          </select>
        </div>

        {/* Clear chat */}
        <button
          onClick={onClearChat}
          className="w-full text-sm border rounded px-3 py-1.5 hover:bg-gray-50"
        >
          Clear chat
        </button>

        <hr />

        {/* Last response metadata */}
        <div>
          <h2 className="text-sm font-semibold mb-2">Last response metadata</h2>
          {lastResponses.length === 0 && (
            <p className="text-xs text-gray-400">No responses yet.</p>
          )}
          {lastResponses.map((resp) => (
            <div
              key={resp.backend}
              className="mb-3 last:mb-0"
            >
              {lastResponses.length > 1 && (
                <p className="text-xs font-semibold text-gray-500 mb-1">
                  {resp.backend === "librarian"
                    ? "Python RAG"
                    : "AWS Bedrock"}
                </p>
              )}

              {/* Confidence metric */}
              <div className="mb-1">
                <span className="text-xs text-gray-500">Confidence</span>
                <p className="text-xl font-bold">
                  {resp.confidence_score !== null
                    ? resp.confidence_score.toFixed(2)
                    : "N/A"}
                </p>
              </div>

              {/* Intent */}
              <p className="text-xs text-gray-600">
                Intent: {resp.intent ?? "\u2014"}
              </p>

              {/* Latency */}
              <div className="flex items-center gap-1 mt-1">
                <span className="text-xs text-gray-500">Latency:</span>
                <LatencyBadge ms={resp.latency_ms} />
              </div>

              {/* Citations */}
              {resp.citations.length > 0 && (
                <div className="mt-2">
                  <h3 className="text-xs font-semibold text-gray-500 mb-1">
                    Citations
                  </h3>
                  <CitationList citations={resp.citations} />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
