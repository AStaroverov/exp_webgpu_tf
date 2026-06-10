import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import topLevelAwait from "vite-plugin-top-level-await";
import wasm from "vite-plugin-wasm";

export default defineConfig({
  server: {
    hmr: false,
  },
  plugins: [wasm(), topLevelAwait(), react(), tailwindcss()],
  worker: {
    format: "es",
    plugins: () => [wasm(), topLevelAwait()],
  },
  optimizeDeps: {
    include: ["lodash/fp"],
    exclude: ["@dimforge/rapier2d-simd"],
  },
});
