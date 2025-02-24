import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
    plugins: [
        wasm(),
        topLevelAwait(),
    ],
    server: {
        hmr: false, // Disable hot module reload
    },
    optimizeDeps: {
        include: ['lodash/fp'],
        exclude: ['@dimforge/rapier2d'],
    },
});