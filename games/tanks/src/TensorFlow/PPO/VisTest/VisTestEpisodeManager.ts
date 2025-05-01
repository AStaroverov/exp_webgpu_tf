import { getDrawState } from '../../Common/uiUtils.ts';
import { EntityId } from 'bitecs';
import { first, firstValueFrom, interval } from 'rxjs';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { EpisodeManager } from '../Actor/EpisodeManager.ts';
import { Scenario } from '../../Common/Curriculum/types.ts';
import { max } from '../../../../../../lib/math.ts';
import { TankAgent } from '../../Common/Curriculum/Agents/ActorAgent.ts';
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

        const index = memory.actionRewards.length - 1;
        const lastStateReward = memory.stateRewards[index];
        const lastActionReward = memory.actionRewards[index];
        return lastActionReward - lastStateReward;
    }

    public getVersion() {
        return this.currentScenario?.getAgents()
            .reduce((acc, agent) => max(acc, agent.getVersion?.() ?? 0), 0) ?? 0;
    }

    protected beforeEpisode() {
        return createScenarioByCurriculumState(this.curriculumState, {
            // return createScenarioWithHeuristicAgents({
            withRender: true,
            withPlayer: false,
        }).then((scenario) => (this.currentScenario = scenario));
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
            let regardedActors: TankAgent[] = [];
            let frame = 0;

            const stop = frameTasks.addInterval(() => {
                frame++;
                const nextRegardedActors = this.runGameTick(
                    episode,
                    regardedActors,
                    frame,
                    maxFramesCount,
                    shouldEvery,
                );

                if (nextRegardedActors == null || !getDrawState()) {
                    stop();
                    resolve(null);
                } else {
                    regardedActors = nextRegardedActors;
                }
            }, 1);
        });
    }
}