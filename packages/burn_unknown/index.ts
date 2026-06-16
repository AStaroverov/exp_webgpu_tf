/**
 * burn_unknown — REAL single-thread PPO training on the ACTUAL ppo_unknown game,
 * driven by Burn's `V4Trainer`.
 *
 * This replaces the old synthetic "reach the goal" demo. It runs the real headless
 * `unknown` ECS + Rapier game, builds observations / masks / rewards with ppo_unknown's
 * own modules, and learns with one in-process `V4Trainer` — the actor + learner worker
 * split collapsed into a single sample→learn→sample loop (see src/trainingLoop.ts).
 *
 * UI (reused from ppo_unknown's debug interface):
 *   - live canvas: press P to render one episode of the current policy (V4Trainer.act),
 *     G to watch it greedily;
 *   - metrics dashboard: press M for the charts, fed off the SAME `metricsChannels`
 *     payloads the real dashboard subscribes to.
 *
 * Training and rendering share the global game state (GameDI/PluginDI/scoreTracker), so
 * they are MUTUALLY EXCLUSIVE: while Render is on we pause training, run one rendered
 * episode, then resume — exactly the trade-off the real vis tab makes (it idles training
 * in that tab while the workers keep going; here there are no workers, so we alternate).
 */

import { ensureTrainer, getTrainerVersion } from "./src/trainer.ts";
import { runTrainingIteration } from "./src/trainingLoop.ts";
import { subscribeOnMetrics } from "./src/metricsPanel.ts";
import { BurnVisManager } from "./src/BurnVisManager.ts";
import { createDashboard } from "./src/dashboard.ts";
import type { CurriculumState } from "../ppo_unknown/src/curriculum/types.ts";

const statusEl = document.getElementById("status")!;
const logEl = document.getElementById("log")!;
const canvas = document.getElementById("game") as HTMLCanvasElement;

function log(line: string) {
  logEl.textContent = line + "\n" + logEl.textContent;
}

async function main() {
  if (!("gpu" in navigator)) {
    statusEl.textContent = "WebGPU not available in this browser.";
    return;
  }

  await ensureTrainer();
  subscribeOnMetrics();

  const visManager = new BurnVisManager();
  const controls = createDashboard(document.body, visManager, () => ({
    version: getTrainerVersion(),
  }));

  // Curriculum iteration tracks completed updates (drives shaping anneal + rung unlock).
  const state: CurriculumState = { iteration: 0, mapScenarioIndexToSuccessRatio: {} };

  statusEl.textContent =
    "ready — training the real ppo_unknown game with V4Trainer. Press P to render a live episode, M for charts, G for greedy.";

  // Single sample→learn→sample loop. When Render is on, run a rendered episode instead
  // of a training iteration (mutually exclusive: shared global game state).
  while (true) {
    if (controls.render) {
      try {
        await visManager.runOnce(state, canvas);
      } catch (e) {
        console.error("Vis episode error:", e);
      }
      continue;
    }

    try {
      // Abort the in-progress episode as soon as Render is toggled, so P engages
      // promptly instead of waiting out the whole match.
      const report = await runTrainingIteration(state, () => controls.render);
      if (report == null) continue; // aborted for Render — switch on the next pass
      state.iteration = report.iteration;
      const s = report.stats;
      log(
        `iter ${report.iteration}: eps=${report.episodes} steps=${report.steps} ` +
          `ep_ret=${report.meanEpisodeReturn.toFixed(3)} succ=${report.meanSuccess.toFixed(2)} | ` +
          `ret=${s.avg_return.toFixed(3)} pl=${s.policy_loss.toFixed(3)} vl=${s.value_loss.toFixed(3)} ` +
          `ent=${s.entropy.toFixed(3)} kl=${s.kl.toFixed(4)} lr=${s.lr.toExponential(2)} gn=${s.grad_norm.toFixed(2)}`,
      );
    } catch (e) {
      console.error("Training iteration error:", e);
      log("training error: " + e);
    }
    // Yield so the dashboard / charts stay responsive between iterations.
    await new Promise((res) => setTimeout(res, 0));
  }
}

main().catch((e) => {
  statusEl.textContent = "error: " + e;
  console.error(e);
});
