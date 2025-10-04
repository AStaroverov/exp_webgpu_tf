import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import topLevelAwait from 'vite-plugin-top-level-await';
import wasm from 'vite-plugin-wasm';

export default defineConfig({
    envPrefix: ['VITE_', 'ABLY_', 'SUPABASE_'],
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
