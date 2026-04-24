/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f6f7fb",
          100: "#eceef6",
          200: "#ced4e8",
          300: "#a9b2d6",
          500: "#4f5bd5",
          600: "#3e4ab1",
          700: "#2f3888",
          900: "#1a1f4b",
        },
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "Inter",
          "system-ui",
          "sans-serif",
        ],
      },
    },
  },
  plugins: [],
};
