/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#090d16",
        surface: "#0f172a",
        surfaceHighlight: "#1e293b",
        borderDark: "#1e293b",
        borderLight: "#334155",
        accentBlue: "#3b82f6",
        accentCyan: "#06b6d4",
      }
    },
  },
  plugins: [],
}
