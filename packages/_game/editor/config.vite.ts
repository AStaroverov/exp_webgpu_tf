import { defineConfig } from "vite";
import topLevelAwait from "vite-plugin-top-level-await";
import wasm from "vite-plugin-wasm";

// SharedArrayBuffer (the physics-worker bridge, plan §3/§6.4) requires the page to
// be cross-origin isolated, which needs COOP/COEP response headers on BOTH the dev
// server and the preview server. The production host MUST send the same headers or
// SAB fails only in production (there is no single-thread fallback).
const crossOriginIsolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  plugins: [wasm(), topLevelAwait()],
  optimizeDeps: {
    exclude: ["@dimforge/rapier3d-simd"],
  },
  server: {
    headers: crossOriginIsolationHeaders,
  },
  preview: {
    headers: crossOriginIsolationHeaders,
  },
  // The worker bundle must be ES so vite-plugin-wasm + top-level-await apply inside
  // it (Rapier WASM is imported in the worker at Step 3). Classic workers can't TLA.
  worker: {
    format: "es",
    plugins: () => [wasm(), topLevelAwait()],
  },
});
