import { PolicyLearnerAgent } from './PolicyLearner/PolicyLearnerAgent.ts';
import { macroTasks } from '../../../../../lib/TasksScheduler/macroTasks.ts';
import { TICK_TRAIN_TIME } from '../Common/consts.ts';
import { EntityId } from 'bitecs';
import { ValueLearnerAgent } from './ValueLearner/ValueLearnerAgent.ts';

export class LearnerManager {
    private tankRewards = new Map<EntityId, number>();

    constructor(public agent: PolicyLearnerAgent | ValueLearnerAgent) {

    }

    static create(agent: PolicyLearnerAgent | ValueLearnerAgent) {
        return new LearnerManager(agent).init();
    }

    public start() {
        this.trainingLoop();
    }

    public getReward(tankEid: EntityId) {
        return this.tankRewards.get(tankEid) || 0;
    }

    // Save models
    private async save() {
        return this.agent.save();
    }

    private async init() {
        await this.agent.init();
        await this.agent.save();
        return this;
    }

    private trainingLoop() {
        macroTasks.addTimeout(async () => {
            try {
                const trained = await this.agent.tryTrain();

                if (!trained) return;

                const health = await this.agent.healthCheck();

                if (health) {
                    await this.save();
                } else {
                    await this.agent.load();
                }
            } catch (error) {
                console.error('Error during training loop:', error);
            } finally {
                this.trainingLoop();
            }
        }, TICK_TRAIN_TIME);
    }
}