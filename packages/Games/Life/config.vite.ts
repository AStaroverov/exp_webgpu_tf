import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import topLevelAwait from 'vite-plugin-top-level-await';
import wasm from 'vite-plugin-wasm';
import path from 'path';

export default defineConfig({
    server: {
        hmr: false
    },
    plugins: [
        wasm(),
        topLevelAwait(),
        react(),
        tailwindcss(),
    ],
    optimizeDeps: {
        include: ['lodash/fp'],
    },
    resolve: {
        alias: {
            'renderer': path.resolve(__dirname, '../../renderer'),
        }
    }
});
