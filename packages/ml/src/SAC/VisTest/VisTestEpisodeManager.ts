import { EntityId } from 'bitecs';
import { first, firstValueFrom, interval } from 'rxjs';
import { log, max, min, round } from '../../../../../lib/math.ts';
import { frameTasks } from '../../../../../lib/TasksScheduler/frameTasks.ts';
import { macroTasks } from '../../../../../lib/TasksScheduler/macroTasks.ts';
import { SAC_CONFIG } from '../../../../ml-common/config.ts';
import { createScenarioByCurriculumState } from '../../../../ml-common/Curriculum/createScenarioByCurriculumState.ts';
import { Scenario } from '../../../../ml-common/Curriculum/types.ts';
import { getDrawState } from '../../../../ml-common/uiUtils.ts';
import { EpisodeManager } from '../Actor/EpisodeManager.ts';

/**
 * SAC Visual Test Episode Manager
 * Manages visualization and testing of SAC agents in the browser tab
 */
export class VisTestEpisodeManager extends EpisodeManager {
    private currentScenario?: Scenario;

    public async start() {
        while (true) {
            try {
                await this.waitEnabling();
                await this.runEpisode();
            } catch (error) {
                console.error('Error during episode:', error);
                await new Promise(resolve => macroTasks.addTimeout(resolve, 1000));
            }
        }
    }

    public getDiscounterReward(tankEid: EntityId) {
        const discounterLen = round(log(0.01) / log(SAC_CONFIG.gamma));
        return this.getReward(tankEid, discounterLen);
    }

    public getRecentReward(tankEid: EntityId) {
        return this.getReward(tankEid, 10);
    }

    public getVersion() {
        return this.currentScenario?.getAlivePilots()
            .reduce((acc, agent) => max(acc, agent.getVersion?.() ?? 0), 0) ?? 0;
    }

    public getSuccessRatio() {
        return this.currentScenario?.getSuccessRatio() ?? 0;
    }

    protected beforeEpisode() {
        return createScenarioByCurriculumState(this.curriculumState, {
            iteration: this.curriculumState.currentVersion,
        })
            .then((scenario) => {
                (this.currentScenario = scenario);
                const canvas = document.querySelector('canvas')!;
                scenario.setRenderTarget(canvas);
                canvas.style.width = scenario.width + 'px';
                canvas.style.height = scenario.height + 'px';
                return scenario;
            });
    }

    protected afterEpisode() {
        // don't publish visual data
    }

    protected cleanupEpisode(episode: Scenario) {
        this.currentScenario = undefined;
        return super.cleanupEpisode(episode);
    }

    protected waitEnabling() {
        return firstValueFrom(interval(100).pipe(
            first(getDrawState),
        ));
    }

    protected runGameLoop(episode: Scenario) {
        return new Promise(resolve => {
            let frame = 0;

            const stop = frameTasks.addInterval(() => {
                const gameOver = this.runGameTick(
                    frame++,
                    16.667,
                    episode,
                );

                if (gameOver || !getDrawState()) {
                    stop();
                    resolve(null);
                }
            }, 1);
        });
    }

    private getReward(tankEid: EntityId, maxLen: number) {
        const memory = this.currentScenario?.getPilot(tankEid)?.getMemory?.();

        if (memory == null || memory.rewards.length === 0) return 0;

        let sum = 0;
        let len = min(memory.rewards.length - 1, maxLen);
        let i = 1;
        while (len > 0) {
            sum += memory.rewards[memory.rewards.length - i];
            len--;
            i++;
        }
        for (let i = 0; i < len; i++) {
            sum += memory.rewards[i];
        }
        return sum;
    }
}
