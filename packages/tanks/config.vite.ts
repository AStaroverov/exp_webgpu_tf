import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

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

// {
//     name: 'configure-response-headers',
//         configureServer: (server) => {
//     server.middlewares.use((_req, res, next) => {
//         res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
//         res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
//         next();
//     });
// },
// },
