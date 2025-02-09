import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
    plugins: [
        wasm(),
        topLevelAwait(),
    ],
    build: {
        rollupOptions: {
            input: {
                app: './games/pingpong/index.html', // default
            },
        },
    },
    optimizeDeps: {
        exclude: [
            '@dimforge/rapier2d',
        ],
    },
});