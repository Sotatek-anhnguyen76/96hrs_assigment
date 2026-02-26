"use client";

import { Character } from "@/lib/api";

interface Props {
  characters: Character[];
  selected: Character | null;
  onSelect: (c: Character) => void;
}

export default function CharacterSelect({
  characters,
  selected,
  onSelect,
}: Props) {
  return (
    <div className="flex flex-col gap-3 p-4">
      <h2 className="text-lg font-semibold text-[var(--foreground)] mb-1">
        Choose a character
      </h2>
      {characters.map((c) => (
        <button
          key={c.id}
          onClick={() => onSelect(c)}
          className={`text-left p-4 rounded-xl border transition-all duration-150 ${
            selected?.id === c.id
              ? "border-[var(--accent)] bg-[var(--accent)]/10"
              : "border-[var(--border)] bg-[var(--card)] hover:bg-[var(--card-hover)] hover:border-[var(--muted)]"
          }`}
        >
          <div className="font-semibold text-[var(--foreground)]">{c.name}</div>
          <div className="text-sm text-[var(--muted)] mt-1">
            {c.description}
          </div>
        </button>
      ))}
    </div>
  );
}
