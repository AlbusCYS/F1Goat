"use client";

import React, { useMemo, useState } from "react";
import * as Slider from "@radix-ui/react-slider";

type Weights = {
  career: number;
  peak: number;
  context: number;
  longevity: number;
  quali: number;
};

type RankRow = {
  rank: number;
  full_name: string;
  goat_score: number;

  career_score?: number;
  peak_score?: number;
  context_score?: number;
  longevity_score?: number;
  quali_score?: number;

  starts?: number;
  wins?: number;
  podiums?: number;

  // NEW (from updated backend_goat.py)
  championships?: number;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

function clamp(n: number, a: number, b: number) {
  return Math.max(a, Math.min(b, n));
}

function normalizeWeights(w: Weights): Weights {
  const sum = w.career + w.peak + w.context + w.longevity + w.quali;
  if (sum <= 0) return w;
  return {
    career: w.career / sum,
    peak: w.peak / sum,
    context: w.context / sum,
    longevity: w.longevity / sum,
    quali: w.quali / sum,
  };
}

function WeightSlider({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  const v100 = Math.round(value * 100);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="text-sm font-medium">{label}</div>
        <div className="text-sm tabular-nums">{v100}</div>
      </div>

      <Slider.Root
        className="relative flex h-5 w-full touch-none select-none items-center"
        value={[v100]}
        min={0}
        max={100}
        step={1}
        onValueChange={(arr) => onChange((arr[0] ?? 0) / 100)}
      >
        <Slider.Track className="relative h-2 w-full grow rounded-full bg-gray-200">
          <Slider.Range className="absolute h-full rounded-full bg-gray-900" />
        </Slider.Track>
        <Slider.Thumb className="block h-5 w-5 rounded-full bg-white shadow ring-1 ring-gray-300 focus:outline-none" />
      </Slider.Root>
    </div>
  );
}

export default function Page() {
  const [weights, setWeights] = useState<Weights>({
    career: 0.30,
    peak: 0.25,
    context: 0.20,
    longevity: 0.15,
    quali: 0.10,
  });

  const normalized = useMemo(() => normalizeWeights(weights), [weights]);

  const [eraNormalize, setEraNormalize] = useState(true);
  const [minStarts, setMinStarts] = useState(30);
  const [topN, setTopN] = useState(50);

  const [rows, setRows] = useState<RankRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function fetchRanking() {
    setLoading(true);
    setError(null);

    try {
      const payload = {
        weights: normalized,
        era_normalize: eraNormalize,
        min_starts: minStarts,
        top_n: topN,
      };

      const res = await fetch(`${API_BASE}/rank`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`API error ${res.status}: ${txt}`);
      }

      const data = (await res.json()) as { rows: RankRow[] } | RankRow[];
      const outRows = Array.isArray(data) ? data : data.rows;

      setRows(outRows);
    } catch (e: any) {
      setError(e?.message ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  const weightPct = (v: number) => `${Math.round(v * 100)}%`;
  const score = (v: number | undefined) => (typeof v === "number" ? v.toFixed(2) : "—");

  return (
    <main className="min-h-screen bg-gray-50 text-gray-900">
      <div className="mx-auto max-w-6xl p-6 space-y-6">
        <header className="space-y-1">
          <h1 className="text-2xl font-bold">F1 GOAT Index</h1>
          <p className="text-sm text-gray-600">
            Tune the definition of “greatness” with weights. The backend computes the ranking.
          </p>
        </header>

        <section className="grid gap-6 md:grid-cols-2">
          <div className="rounded-xl bg-white p-5 shadow-sm ring-1 ring-gray-200 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Weights</h2>
              <div className="text-xs text-gray-600">
                Auto-normalized:{" "}
                {weightPct(normalized.career)} / {weightPct(normalized.peak)} /{" "}
                {weightPct(normalized.context)} / {weightPct(normalized.longevity)} /{" "}
                {weightPct(normalized.quali)}
              </div>
            </div>

            <div className="space-y-4">
              <WeightSlider
                label="Career (championships, wins, podiums)"
                value={weights.career}
                onChange={(v) => setWeights((w) => ({ ...w, career: v }))}
              />
              <WeightSlider
                label="Peak (best seasons)"
                value={weights.peak}
                onChange={(v) => setWeights((w) => ({ ...w, peak: v }))}
              />
              <WeightSlider
                label="Context (car strength proxy)"
                value={weights.context}
                onChange={(v) => setWeights((w) => ({ ...w, context: v }))}
              />
              <WeightSlider
                label="Longevity & consistency"
                value={weights.longevity}
                onChange={(v) => setWeights((w) => ({ ...w, longevity: v }))}
              />
              <WeightSlider
                label="Qualifying / speed"
                value={weights.quali}
                onChange={(v) => setWeights((w) => ({ ...w, quali: v }))}
              />
            </div>
          </div>

          <div className="rounded-xl bg-white p-5 shadow-sm ring-1 ring-gray-200 space-y-4">
            <h2 className="text-lg font-semibold">Options</h2>

            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={eraNormalize}
                onChange={(e) => setEraNormalize(e.target.checked)}
                className="h-4 w-4"
              />
              <span className="text-sm">Enable era normalization (multiplier)</span>
            </label>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <label className="text-sm font-medium">Min starts</label>
                <input
                  type="number"
                  value={minStarts}
                  min={0}
                  max={9999}
                  onChange={(e) => setMinStarts(clamp(Number(e.target.value), 0, 9999))}
                  className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
                />
              </div>

              <div className="space-y-1">
                <label className="text-sm font-medium">Top N</label>
                <input
                  type="number"
                  value={topN}
                  min={1}
                  max={500}
                  onChange={(e) => setTopN(clamp(Number(e.target.value), 1, 500))}
                  className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
                />
              </div>
            </div>

            <button
              onClick={fetchRanking}
              disabled={loading}
              className="w-full rounded-xl bg-gray-900 px-4 py-3 text-sm font-semibold text-white disabled:opacity-60"
            >
              {loading ? "Computing..." : "Recompute ranking"}
            </button>

            {error && (
              <div className="rounded-lg bg-red-50 p-3 text-sm text-red-700 ring-1 ring-red-200">
                {error}
              </div>
            )}

            <div className="text-xs text-gray-600">
              Backend: <span className="font-mono">{API_BASE}</span>
            </div>
          </div>
        </section>

        <section className="rounded-xl bg-white p-5 shadow-sm ring-1 ring-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Leaderboard</h2>
            <div className="text-xs text-gray-600">Showing {rows.length} drivers</div>
          </div>

          <div className="mt-4 overflow-auto">
            <table className="w-full min-w-[1000px] text-sm">
              <thead className="sticky top-0 bg-white">
                <tr className="border-b border-gray-200 text-left text-gray-600">
                  <th className="py-2 pr-4">Rank</th>
                  <th className="py-2 pr-4">Driver</th>
                  <th className="py-2 pr-4">GOAT</th>

                  {/* NEW */}
                  <th className="py-2 pr-4">Titles</th>

                  <th className="py-2 pr-4">Career</th>
                  <th className="py-2 pr-4">Peak</th>
                  <th className="py-2 pr-4">Context</th>
                  <th className="py-2 pr-4">Longevity</th>
                  <th className="py-2 pr-4">Quali</th>

                  <th className="py-2 pr-4">Starts</th>
                  <th className="py-2 pr-4">Wins</th>
                  <th className="py-2 pr-4">Podiums</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={`${r.rank}-${r.full_name}`} className="border-b border-gray-100">
                    <td className="py-2 pr-4 tabular-nums">{r.rank}</td>
                    <td className="py-2 pr-4 font-medium">{r.full_name}</td>
                    <td className="py-2 pr-4 tabular-nums">{score(r.goat_score)}</td>

                    {/* NEW */}
                    <td className="py-2 pr-4 tabular-nums">{r.championships ?? "—"}</td>

                    <td className="py-2 pr-4 tabular-nums">{score(r.career_score)}</td>
                    <td className="py-2 pr-4 tabular-nums">{score(r.peak_score)}</td>
                    <td className="py-2 pr-4 tabular-nums">{score(r.context_score)}</td>
                    <td className="py-2 pr-4 tabular-nums">{score(r.longevity_score)}</td>
                    <td className="py-2 pr-4 tabular-nums">{score(r.quali_score)}</td>

                    <td className="py-2 pr-4 tabular-nums">{r.starts ?? "—"}</td>
                    <td className="py-2 pr-4 tabular-nums">{r.wins ?? "—"}</td>
                    <td className="py-2 pr-4 tabular-nums">{r.podiums ?? "—"}</td>
                  </tr>
                ))}

                {rows.length === 0 && (
                  <tr>
                    <td colSpan={12} className="py-8 text-center text-gray-500">
                      Click “Recompute ranking” to load results.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        <footer className="text-xs text-gray-500">
          Tip: If the table is empty or errors, make sure FastAPI is running and CORS is enabled.
        </footer>
      </div>
    </main>
  );
}
