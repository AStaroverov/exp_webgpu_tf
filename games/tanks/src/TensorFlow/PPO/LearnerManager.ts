import { PolicyLearnerAgent } from './PolicyLearner/PolicyLearnerAgent.ts';
import { macroTasks } from '../../../../../lib/TasksScheduler/macroTasks.ts';
import { TICK_TRAIN_TIME } from '../Common/consts.ts';
import { EntityId } from 'bitecs';
import { ValueLearnerAgent } from './ValueLearner/ValueLearnerAgent.ts';

export class LearnerManager {
    private stopTrainingTimeout: VoidFunction | null = null;
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
        this.stopTrainingTimeout = macroTasks.addTimeout(async () => {
            try {
                if (await this.agent.tryTrain()) {
                    void this.save();
                }

                this.trainingLoop();
            } catch (error) {
                this.stopTrainingTimeout?.();
                this.stopTrainingTimeout = null;

                console.error('Error during training:', error);
                macroTasks.addTimeout(() => this.trainingLoop(), TICK_TRAIN_TIME * 100);
            }
        }, TICK_TRAIN_TIME);
    }
}