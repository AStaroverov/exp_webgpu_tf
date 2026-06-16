/**
 * BurnVisManager — the live-canvas debug visualizer, the Burn analogue of
 * ppo_unknown's `VisTestEpisodeManager`. It runs ONE real episode of the CURRENT
 * policy and renders it to the page canvas at frame cadence so you can watch it.
 *
 * The original VisTestEpisodeManager is tfjs-coupled through `UnknownAgent` (loads a
 * tf.LayersModel from IndexedDB, runs `batchActAsync`). The minimal seam swapped here:
 * the scenario is built with `createBurnScenario`, whose `BurnUnknownAgent` drives
 * rendering via `V4Trainer.act` instead. Everything else — the renderer
 * (`setRenderTarget` → WebGPU init in the `unknown` game), the policy driver, the
 * frame-cadence loop, the fire-and-forget (no drain) decision scheduling — is the real
 * game's machinery, unchanged.
 *
 * Vis episodes use `train: false`: agents act (and render) but record no training data,
 * exactly like the real vis tab's no-op `afterEpisode`.
 */

import { frameTasks } from "../../../lib/TasksScheduler/frameTasks.ts";
import { TICK_TIME_SIMULATION } from "../../ppo_unknown/src/consts.ts";
import { CONFIG } from "../../ppo_unknown/src/config.ts";
import { scenarioCompositions, type CurriculumState } from "../../ppo_unknown/src/curriculum/types.ts";
import { createBurnScenario, type BurnScenario } from "./createBurnScenario.ts";
import { ensureTrainer, getTrainerVersion } from "./trainer.ts";

const MAX_FRAMES = CONFIG.episodeFrames;

export class BurnVisManager {
  private current?: BurnScenario;

  getVersion(): number {
    return getTrainerVersion();
  }

  getSuccessRatio(): number {
    return this.current?.getSuccessRatio() ?? 0;
  }

  getAgents() {
    return this.current?.agents ?? [];
  }

  /** Run one rendered episode of the current policy (resolves on termination). */
  async runOnce(state: CurriculumState, canvas: HTMLCanvasElement): Promise<void> {
    await ensureTrainer();

    // Watch the bread-and-butter rungs (standing/moving enemies) so the canvas shows
    // the policy fighting scripted opponents.
    const index = Math.floor(Math.random() * Math.min(2, scenarioCompositions.length));
    const scenario = createBurnScenario({
      index,
      train: false, // vis: act + render, record nothing
      config: scenarioCompositions[index],
      iteration: state.iteration,
    });
    this.current = scenario;

    scenario.setRenderTarget(canvas); // async; renders once WebGPU is ready
    canvas.style.width = scenario.width + "px";
    canvas.style.height = scenario.height + "px";

    try {
      await this.runGameLoop(scenario);
    } finally {
      scenario.agents.forEach((a) => a.dispose());
      scenario.destroy();
      this.current = undefined;
    }
  }

  /** Tick at rAF cadence so the episode is watchable (fire-and-forget, no drain). */
  private runGameLoop(scenario: BurnScenario): Promise<void> {
    return new Promise((resolve) => {
      let frame = 0;
      const stop = frameTasks.addInterval(() => {
        const aliveTanks = scenario.getVehicleEids();
        const gameOver =
          scenario.getTeamsCount() <= 1 || aliveTanks.length <= 1 || frame > MAX_FRAMES;
        scenario.gameTick(TICK_TIME_SIMULATION);
        frame++;
        if (gameOver) {
          stop();
          resolve();
        }
      }, 1);
    });
  }
}
