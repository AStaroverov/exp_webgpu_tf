import { filter, first, firstValueFrom, race, shareReplay, startWith, timer } from "rxjs";
import { macroTasks } from "../../../../lib/TasksScheduler/macroTasks.ts";
import { queueSizeChannel } from "./channels.ts";

const queueSize$ = queueSizeChannel.obs.pipe(startWith(0), shareReplay(1));

export abstract class AbstractEpisodeManager<Scen> {
  private backpressure$;

  constructor(
    protected readonly config: { backpressureQueueSize: number; simulationTickTime: number },
  ) {
    this.backpressure$ = race([
      timer(60 * 1000),
      queueSize$.pipe(filter((queueSize) => queueSize <= config.backpressureQueueSize)),
    ]).pipe(first());
  }

  public async start() {
    while (true) {
      try {
        await firstValueFrom(this.backpressure$);
        await this.runEpisode();
      } catch (error) {
        console.error("Error during episode:", error);
      }
    }
  }

  protected async runEpisode() {
    const episode = this.beforeEpisode();

    try {
      await this.awaitAgentsSync();
      await this.runGameLoop(episode);
      this.afterEpisode(episode);
    } finally {
      this.cleanupEpisode(episode);
    }
  }

  protected async runGameLoop(episode: Scen): Promise<unknown> {
    let frame = 0;
    while (true) {
      for (let i = 0; i < 100; i++) {
        const gameOver = this.runGameTick(frame++, this.config.simulationTickTime, episode);
        await this.drainDecisions(episode);
        if (gameOver) return null;
      }
      // Yield a macrotask every 100 ticks so backpressure / other workers get a turn
      // (ticks with no decision resolve drain on a microtask and would otherwise spin).
      await new Promise<void>((resolve) => macroTasks.addTimeout(() => resolve(), 0));
    }
  }

  protected abstract beforeEpisode(): Scen;
  protected abstract afterEpisode(scenario: Scen): void;
  protected abstract cleanupEpisode(scenario: Scen): void;
  protected abstract runGameTick(frame: number, deltaTime: number, scenario: Scen): boolean;
  /** Await decisions kicked off by the latest tick (no-op if the env has none). */
  protected abstract drainDecisions(scenario: Scen): Promise<void>;
  protected abstract awaitAgentsSync(): Promise<unknown>;
}
