import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
    plugins: [
        wasm(),
        topLevelAwait(),
    ],
    optimizeDeps: {
        include: ['lodash/fp'],
        exclude: ['@dimforge/rapier2d-simd'],
    },
});