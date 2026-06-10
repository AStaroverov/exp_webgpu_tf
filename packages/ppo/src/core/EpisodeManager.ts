import { filter, first, firstValueFrom, race, shareReplay, startWith, timer } from "rxjs";
import { macroTasks } from "../../../../lib/TasksScheduler/macroTasks.ts";
import { queueSizeChannel } from "./channels.ts";

const queueSize$ = queueSizeChannel.obs.pipe(startWith(0), shareReplay(1));

export abstract class EpisodeManager<Scen> {
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
    } catch (error) {
      throw error;
    } finally {
      this.cleanupEpisode(episode);
    }
  }

  protected runGameLoop(episode: Scen) {
    return new Promise((resolve, reject) => {
      let frame = 0;

      const stop = macroTasks.addInterval(() => {
        try {
          for (let i = 0; i < 100; i++) {
            const gameOver = this.runGameTick(frame++, this.config.simulationTickTime, episode);

            if (gameOver) {
              stop();
              resolve(null);
              break;
            }
          }
        } catch (error) {
          stop();
          reject(error);
        }
      }, 1);
    });
  }

  protected abstract beforeEpisode(): Scen;
  protected abstract afterEpisode(scenario: Scen): void;
  protected abstract cleanupEpisode(scenario: Scen): void;
  protected abstract runGameTick(frame: number, deltaTime: number, scenario: Scen): boolean;
  protected abstract awaitAgentsSync(): Promise<unknown>;
}
