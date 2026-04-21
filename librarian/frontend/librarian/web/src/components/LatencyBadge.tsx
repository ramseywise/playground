interface Props {
  ms: number | null;
}

export function LatencyBadge({ ms }: Props) {
  if (ms === null) return null;

  const colorClass =
    ms < 2000
      ? "bg-green-100 text-green-800"
      : ms < 5000
        ? "bg-yellow-100 text-yellow-800"
        : "bg-red-100 text-red-800";

  return (
    <span className={`text-xs font-mono px-1.5 py-0.5 rounded ${colorClass}`}>
      {ms}ms
    </span>
  );
}
