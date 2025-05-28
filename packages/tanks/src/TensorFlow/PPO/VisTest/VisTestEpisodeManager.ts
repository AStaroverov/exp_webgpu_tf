import { getDrawState } from '../../Common/uiUtils.ts';
import { EntityId } from 'bitecs';
import { first, firstValueFrom, interval } from 'rxjs';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { EpisodeManager } from '../Actor/EpisodeManager.ts';
import { Scenario } from '../../Common/Curriculum/types.ts';
import { log, max, min, round } from '../../../../../../lib/math.ts';
import { TankAgent } from '../../Common/Curriculum/Agents/CurrentActorAgent.ts';
import { SNAPSHOT_EVERY } from '../../Common/consts.ts';
import { CONFIG } from '../config.ts';
import { frameTasks } from '../../../../../../lib/TasksScheduler/frameTasks.ts';
import { createScenarioByCurriculumState } from '../../Common/Curriculum/createScenarioByCurriculumState.ts';

const discounterLen = round(log(0.01) / log(CONFIG.gamma));

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

    public getReward(tankEid: EntityId) {
        const memory = this.currentScenario?.getAgent(tankEid)?.getMemory?.();

        if (memory == null || memory.rewards.length === 0) return 0;

        let sum = 0;
        let len = min(memory.rewards.length - 1, discounterLen);
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

    public getVersion() {
        return this.currentScenario?.getAliveAgents()
            .reduce((acc, agent) => max(acc, agent.getVersion?.() ?? 0), 0) ?? 0;
    }

    public getSuccessRatio() {
        return this.currentScenario?.getSuccessRatio() ?? 0;
    }

    protected beforeEpisode() {
        return createScenarioByCurriculumState(this.curriculumState, {})
            // return createScenarioWithHeuristicAgents({})
            .then((scenario) => {
                (this.currentScenario = scenario);
                const canvas = document.querySelector('canvas')!;
                scenario.setRenderTarget(canvas);
                canvas.style.width = scenario.width + 'px';
                canvas.style.height = scenario.height + 'px';
                return scenario;
            });
    }


    protected afterEpisode(episode: Scenario) {
        if (getDrawState()) {
            super.afterEpisode(episode);
        }
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
            const shouldEvery = SNAPSHOT_EVERY;
            const maxFramesCount = (CONFIG.episodeFrames - (CONFIG.episodeFrames % shouldEvery) + shouldEvery);
            let regardedAgents: TankAgent[] = [];
            let frame = 0;

            const stop = frameTasks.addInterval(() => {
                frame++;
                const nextRegardedAgents = this.runGameTick(
                    16.667,
                    episode,
                    regardedAgents,
                    frame,
                    maxFramesCount,
                    shouldEvery,
                );

                if (nextRegardedAgents == null || !getDrawState()) {
                    stop();
                    resolve(null);
                } else {
                    regardedAgents = nextRegardedAgents;
                }
            }, 1);
        });
    }
}