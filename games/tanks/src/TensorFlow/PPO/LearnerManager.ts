import { PolicyLearnerAgent } from './PolicyLearner/PolicyLearnerAgent.ts';
import { macroTasks } from '../../../../../lib/TasksScheduler/macroTasks.ts';
import { EntityId } from 'bitecs';
import { ValueLearnerAgent } from './ValueLearner/ValueLearnerAgent.ts';
import { forceExitChannel } from '../Common/channels.ts';

export class LearnerManager {
    private tankRewards = new Map<EntityId, number>();

    constructor(public agent: PolicyLearnerAgent | ValueLearnerAgent) {

    }

    static create(agent: PolicyLearnerAgent | ValueLearnerAgent) {
        return new LearnerManager(agent).init();
    }

    public start() {
        this.trainLoop();
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

    private async trainLoop() {
        let errorCount = 0;

        while (true) {
            await new Promise(resolve => macroTasks.addTimeout(resolve, 100));

            await this.agent.collectBatches();
            if (!this.agent.hasEnoughBatches()) continue;

            errorCount = await this.train() ? 0 : errorCount + 1;

            if (errorCount > 5) {
                console.error('Error count exceeded');
                forceExitChannel.postMessage(null);
                break;
            }
        }
    }

    private async train() {
        try {
            await this.agent.train();
            this.agent.finishTrain();

            const health = await this.agent.healthCheck();

            if (health) {
                await this.save();
                return true;
            } else {
                await this.agent.load();
                return false;
            }
        } catch (error) {
            console.error('Error during training loop:', error);
            return false;
        }
    }
}