import { defineConfig } from "vite";
import topLevelAwait from "vite-plugin-top-level-await";
import wasm from "vite-plugin-wasm";

export default defineConfig({
  // Relative base so built asset URLs resolve under the GitHub Pages
  // project path (astaroverov.github.io/exp_webgpu_tf/) as well as locally.
  base: "./",
  server: {
    hmr: false,
  },
  plugins: [wasm(), topLevelAwait()],
  optimizeDeps: {
    include: ["lodash/fp"],
    exclude: ["@dimforge/rapier2d-simd"],
  },
});
