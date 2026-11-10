/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#edf5ff',
          100: '#dfeeff',
          200: '#bfd8ff',
          300: '#8ab8ff',
          400: '#5f9efa',
          500: '#2e7cf6',
          600: '#1f63d7',
          700: '#1f4fb0',
          800: '#1b3f85',
          900: '#183769',
        },
        slate: {
          950: '#081521',
        },
      },
      boxShadow: {
        soft: '0 8px 22px rgba(15, 23, 42, 0.08)',
      },
      borderRadius: {
        xl: '1rem',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
