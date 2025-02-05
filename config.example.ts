import { defineConfig } from 'vite';

export default defineConfig({
    build: {
        rollupOptions: {
            input: {
                app: './example/index.html', // default
            },
        },
    },
});