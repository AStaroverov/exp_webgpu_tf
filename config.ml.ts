import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
    plugins: [
        wasm(),
        topLevelAwait(),
        {
            name: 'configure-response-headers',
            configureServer: (server) => {
                server.middlewares.use((_req, res, next) => {
                    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
                    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
                    next();
                });
            },
        },
    ],
    server: {
        hmr: false, // Disable hot module reload
    },
    optimizeDeps: {
        include: ['lodash/fp'],
        exclude: ['@dimforge/rapier2d-simd'],
    },
});