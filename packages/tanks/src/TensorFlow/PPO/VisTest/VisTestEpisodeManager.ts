import { getDrawState } from '../../Common/uiUtils.ts';
import { EntityId } from 'bitecs';
import { first, firstValueFrom, interval } from 'rxjs';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { EpisodeManager } from '../Actor/EpisodeManager.ts';
import { Scenario } from '../../Common/Curriculum/types.ts';
import { log, max, min, round } from '../../../../../../lib/math.ts';
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
        const memory = this.currentScenario?.getPilot(tankEid)?.getMemory?.();

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
        return this.currentScenario?.getAlivePilots()
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

    protected runGameLoop(scenario: Scenario) {
        return new Promise(resolve => {
            let frame = 0;

            // warmup tick
            scenario.gameTick(16.667);

            const stop = frameTasks.addInterval(() => {
                frame++;
                const gameOver = this.runGameTick(
                    frame,
                    16.667,
                    scenario,
                );

                if (gameOver || !getDrawState()) {
                    stop();
                    resolve(null);
                }
            }, 1);
        });
    }
}