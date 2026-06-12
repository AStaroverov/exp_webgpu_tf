/**
 * UnknownVisTestEpisodeManager — debug visualizer, the ppo_unknown analogue of
 * tanks' `VisTestEpisodeManager`. Runs a real episode driven by the CURRENT
 * policy (pulled from storage like an actor) but renders it to the page canvas
 * and ticks at frame cadence so you can watch it. It does NOT emit training data
 * (afterEpisode is a no-op) — purely for inspection in the main tab.
 */

import { filter, firstValueFrom, interval, take } from "rxjs";
import { max } from "../../../../lib/math.ts";
import { frameTasks } from "../../../../lib/TasksScheduler/frameTasks.ts";
import { macroTasks } from "../../../../lib/TasksScheduler/macroTasks.ts";
import { TICK_TIME_SIMULATION } from "../consts.ts";
import { Scenario } from "../env/createUnknownScenario.ts";
import { UnknownAgent } from "../env/UnknownAgent.ts";
import { createScenarioByCurriculumState } from "../curriculum/createScenarioByCurriculumState.ts";
import { getDrawState, settingsReady } from "../ui/uiUtils.ts";
import { EpisodeManager } from "./EpisodeManager.ts";

export class VisTestEpisodeManager extends EpisodeManager {
  private current?: Scenario;

  public getVersion(): number {
    return max(...(this.current?.agents ?? []).map((a) => a.getVersion()), 0);
  }

  public getSuccessRatio(): number {
    return this.current?.getSuccessRatio() ?? 0;
  }

  public getAgents(): UnknownAgent[] {
    return this.current?.agents ?? [];
  }

  public async start(): Promise<void> {
    await settingsReady;
    while (true) {
      try {
        // Only run a visual episode while rendering is enabled (toggle in the
        // dashboard / 'P'). When off, the tab idles — actors + learners keep
        // training in their workers and the charts still update.
        await this.waitEnabling();
        await this.runEpisode();
      } catch (error) {
        console.error("Vis episode error (retrying):", error);
        await firstValueFrom(interval(1000).pipe(take(1)));
      }
    }
  }

  /** Resolve once rendering is enabled (polls the persisted draw state). */
  private waitEnabling(): Promise<unknown> {
    if (getDrawState()) return Promise.resolve(null);
    return firstValueFrom(interval(100).pipe(filter(getDrawState), take(1)));
  }

  protected beforeEpisode(): Scenario {
    const scenario = createScenarioByCurriculumState(this.curriculumState, { train: true });

    const canvas = document.querySelector("canvas");
    if (canvas) {
      scenario.setRenderTarget(canvas); // async; renders once WebGPU is ready
      canvas.style.width = scenario.width + "px";
      canvas.style.height = scenario.height + "px";
    }

    this.current = scenario;
    return scenario;
  }

  /** Don't publish visual-run data into the training stream. */
  protected afterEpisode(): void {}

  protected cleanupEpisode(scenario: Scenario): void {
    this.current = undefined;
    super.cleanupEpisode(scenario);
  }

  /** Tick at rAF cadence so the episode is watchable (vs the burst loop). */
  protected runGameLoop(episode: Scenario): Promise<unknown> {
    return new Promise((resolve) => {
      let frame = 0;
      const stop = frameTasks.addInterval(() => {
        const gameOver = this.runGameTick(frame++, TICK_TIME_SIMULATION, episode);
        // Stop on termination OR when rendering is toggled off mid-episode.
        if (gameOver || !getDrawState()) {
          stop();
          macroTasks.addTimeout(() => resolve(null), 0);
        }
      }, 1);
    });
  }
}
