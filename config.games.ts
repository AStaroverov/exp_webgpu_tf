import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
    plugins: [
        wasm(),
        topLevelAwait(),
        react(),
        tailwindcss(),
    ],
    optimizeDeps: {
        include: ['lodash/fp'],
        exclude: ['@dimforge/rapier2d-simd'],
    },
});