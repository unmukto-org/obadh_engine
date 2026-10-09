/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./*.html", "./js/**/*.js"],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        gray: {
          50: '#f3f8f7', 100: '#eaf5f4', 200: '#d8e9e9', 300: '#c2d6da',
          400: '#a8c0c5', 500: '#93afb7', 600: '#89a7b0', 700: '#345d6c',
          800: '#24434d', 900: '#0a2530', 950: '#061a22',
        },
        primary: {
          50: '#effaf9',
          100: '#d8f1ee',
          200: '#b3e4de',
          300: '#80d5cc',
          400: '#3cbfbc',
          500: '#28b3ae',
          600: '#188783',
          700: '#146c6d',
          800: '#16506f',
          900: '#103f50',
          950: '#0a2530',
        },
      },
      fontFamily: {
        sans: ['"Schibsted Grotesk"', 'system-ui', 'sans-serif'],
        bengali: ['"Anek Bangla"', 'sans-serif'],
      }
    }
  },
  plugins: [
    require('@tailwindcss/forms')
  ],
} 