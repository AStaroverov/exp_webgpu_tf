import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import topLevelAwait from "vite-plugin-top-level-await";
import wasm from "vite-plugin-wasm";

// Mirrors ppo_unknown/config.vite.ts: the single-thread Burn loop imports the real
// game (`unknown`) ECS + Rapier WASM + ppo_unknown env/state/reward modules and the
// MetricsBrowser charts panel, so it needs the same plugin set. We additionally need
// the burn_rl wasm (covered by vite-plugin-wasm + top-level-await).
export default defineConfig({
  server: {
    hmr: false,
  },
  plugins: [wasm(), topLevelAwait(), react(), tailwindcss()],
  optimizeDeps: {
    include: ["lodash/fp"],
    exclude: ["@dimforge/rapier2d-simd"],
  },
});
