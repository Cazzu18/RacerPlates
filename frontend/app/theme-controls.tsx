"use client";

import { useThemePreferences, type FontPreference, type ThemePreference } from "./theme-provider";

const THEME_OPTIONS: { value: ThemePreference; label: string }[] = [
  { value: "system", label: "Match system" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
];

const FONT_OPTIONS: { value: FontPreference; label: string }[] = [
  { value: "sans", label: "Sans Serif" },
  { value: "serif", label: "Serif" },
  { value: "mono", label: "Monospace" },
];

export default function ThemeControls() {
  const { theme, font, setTheme, setFont } = useThemePreferences();

  return (
    <div
      className="fixed bottom-4 right-4 z-40 w-64 max-w-[90vw] space-y-3 rounded-2xl border px-4 py-3 text-xs shadow-lg"
      style={{
        backgroundColor: "var(--background)",
        color: "var(--foreground)",
        borderColor: "rgba(148, 163, 184, 0.4)",
      }}
    >
      <div className="flex flex-col gap-0.5">
        <span
          className="text-[11px] font-semibold uppercase tracking-wide"
          style={{ color: "var(--foreground)" }}
        >
          Display preferences
        </span>
        <p
          className="text-[11px] opacity-70"
          style={{ color: "var(--foreground)" }}
        >
          Choose how RacerPlates should follow your system appearance.
        </p>
      </div>

      <div className="space-y-1">
        <label className="text-[11px] font-semibold uppercase tracking-wide">
          Color mode
        </label>
        <select
          className="w-full rounded-md border px-2 py-1 text-xs"
          value={theme}
          onChange={(event) => setTheme(event.target.value as ThemePreference)}
          style={{
            backgroundColor: "var(--background)",
            color: "var(--foreground)",
            borderColor: "rgba(148, 163, 184, 0.6)",
          }}
        >
          {THEME_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <div className="space-y-1">
        <label className="text-[11px] font-semibold uppercase tracking-wide">
          Font style
        </label>
        <select
          className="w-full rounded-md border px-2 py-1 text-xs"
          value={font}
          onChange={(event) => setFont(event.target.value as FontPreference)}
          style={{
            backgroundColor: "var(--background)",
            color: "var(--foreground)",
            borderColor: "rgba(148, 163, 184, 0.6)",
          }}
        >
          {FONT_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
