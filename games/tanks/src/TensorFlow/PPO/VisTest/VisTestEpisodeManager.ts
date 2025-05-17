import { getDrawState } from '../../Common/uiUtils.ts';
import { EntityId } from 'bitecs';
import { first, firstValueFrom, interval } from 'rxjs';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { EpisodeManager } from '../Actor/EpisodeManager.ts';
import { Scenario } from '../../Common/Curriculum/types.ts';
import { max } from '../../../../../../lib/math.ts';
import { TankAgent } from '../../Common/Curriculum/Agents/CurrentActorAgent.ts';
import { SNAPSHOT_EVERY } from '../../Common/consts.ts';
import { CONFIG } from '../config.ts';
import { frameTasks } from '../../../../../../lib/TasksScheduler/frameTasks.ts';
import { createScenarioByCurriculumState } from '../../Common/Curriculum/createScenarioByCurriculumState.ts';

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

        if (memory == null || memory.actionRewards.length === 0) return 0;

        let sum = 0;
        const len = memory.actionRewards.length - 1;
        for (let i = 0; i < len; i++) {
            const lastStateReward = memory.stateRewards[i];
            const lastActionReward = memory.actionRewards[i];

            sum += lastActionReward - lastStateReward;
        }
        return sum;
    }

    public getVersion() {
        return this.currentScenario?.getAgents()
            .reduce((acc, agent) => max(acc, agent.getVersion?.() ?? 0), 0) ?? 0;
    }

    public getSuccessRatio() {
        return this.currentScenario?.getSuccessRatio() ?? 0;
    }

    protected beforeEpisode() {
        return createScenarioByCurriculumState(this.curriculumState, {
            // return createScenarioWithHistoricalAgents({
            withPlayer: false,
        }).then((scenario) => {
            (this.currentScenario = scenario);
            scenario.setRenderTarget(document.querySelector('canvas'));
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