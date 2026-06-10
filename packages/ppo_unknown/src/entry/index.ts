import "@tensorflow/tfjs-backend-wasm";
import { macroTasks } from "../../../../lib/TasksScheduler/macroTasks.ts";
import { setConsolePrefix } from "../../../ppo/src/infra/console.ts";
import { initTensorFlow } from "../../../ppo/src/infra/initTensorFlow.ts";
// Side-effect import: register custom layers + AdamW so the main tab can LOAD the
// policy model for the debug visualizer (see ActorWorker.ts).
import "../models/createUnknownNetworks.ts";
import { CONFIG } from "../config.ts";
import { UnknownVisTestEpisodeManager } from "../agents/UnknownVisTestEpisodeManager.ts";
import { createDebugVisualization } from "../ui/debug.ts";

setConsolePrefix(`[TAB]`);

await initTensorFlow("wasm");

// Actors run the simulation + policy inference (WASM backend).
Array.from(
  { length: CONFIG.workerCount },
  () => new Worker(new URL("./ActorWorker.ts", import.meta.url), { type: "module" }),
);

// Learners train policy + value (WebGPU backend); start them slightly later so
// the actors and TF backends are up first.
macroTasks.addTimeout(() => {
  new Worker(new URL("./LearnerPolicyWorker.ts", import.meta.url), { type: "module" });
  new Worker(new URL("./LearnerValueWorker.ts", import.meta.url), { type: "module" });
}, 1000);

// Debug visualizer: render a live episode driven by the current policy to the
// page canvas (no training data emitted). Starts once the learner has saved a
// first model; retries until then. The dashboard adds the realtime metrics
// charts panel (press M) reading the learners' metricsChannels.
const visManager = new UnknownVisTestEpisodeManager();
createDebugVisualization(document.body, visManager);
visManager.start();
