"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

export type ThemePreference = "system" | "light" | "dark";
export type FontPreference = "sans" | "serif" | "mono";

type ThemeContextValue = {
  theme: ThemePreference;
  font: FontPreference;
  setTheme: (preference: ThemePreference) => void;
  setFont: (preference: FontPreference) => void;
};

const THEME_KEY = "racerplates-theme";
const FONT_KEY = "racerplates-font";

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<ThemePreference>("system");
  const [font, setFont] = useState<FontPreference>("sans");

  useEffect(() => {
    if (typeof window === "undefined") return;
    const storedTheme = localStorage.getItem(THEME_KEY) as ThemePreference | null;
    const storedFont = localStorage.getItem(FONT_KEY) as FontPreference | null;
    if (storedTheme) {
      setTheme(storedTheme);
    }
    if (storedFont) {
      setFont(storedFont);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");

    const applyPreferences = () => {
      const resolvedTheme =
        theme === "system" ? (media.matches ? "dark" : "light") : theme;
      const root = document.documentElement;
      root.dataset.theme = resolvedTheme;
      root.dataset.font = font;
    };

    applyPreferences();

    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", applyPreferences);
      return () => media.removeEventListener("change", applyPreferences);
    }

    media.addListener(applyPreferences);
    return () => media.removeListener(applyPreferences);
  }, [theme, font]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    localStorage.setItem(FONT_KEY, font);
  }, [font]);

  const handleThemeChange = useCallback(
    (preference: ThemePreference) => {
      setTheme(preference);
    },
    []
  );

  const handleFontChange = useCallback(
    (preference: FontPreference) => {
      setFont(preference);
    },
    []
  );

  const value = useMemo(
    () => ({
      theme,
      font,
      setTheme: handleThemeChange,
      setFont: handleFontChange,
    }),
    [theme, font, handleThemeChange, handleFontChange]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useThemePreferences() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useThemePreferences must be used within ThemeProvider");
  }
  return context;
}

export default ThemeProvider;
