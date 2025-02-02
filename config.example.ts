import { defineConfig } from 'vite';

export default defineConfig({
    build: {
        rollupOptions: {
            input: {
                app: './src/example/index.html', // default
            },
        },
    },
});