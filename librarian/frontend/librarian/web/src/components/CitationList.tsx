import type { Citation } from "@/lib/types";

interface Props {
  citations: Citation[];
}

export function CitationList({ citations }: Props) {
  if (!citations.length) {
    return <p className="text-xs text-gray-400">No citations</p>;
  }

  return (
    <ul className="space-y-1">
      {citations.map((c, i) => (
        <li key={i} className="text-xs">
          <a
            href={c.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:underline"
          >
            {c.title}
          </a>
          {c.excerpt && (
            <p className="text-gray-500 mt-0.5 line-clamp-2">{c.excerpt}</p>
          )}
        </li>
      ))}
    </ul>
  );
}
