import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { forceExitChannel } from '../../Common/channels.ts';
import { learnerStateChannel } from '../../DB';
import { LearnerAgent } from './LearnerAgent.ts';

export class LearnerManager {
    constructor(public agent: LearnerAgent) {

    }

    static create(agent: LearnerAgent) {
        return new LearnerManager(agent).init();
    }

    public start() {
        this.trainLoop();
    }

    // Save models
    private async save() {
        return this.agent.save();
    }

    private async init() {
        await this.agent.init();
        await this.agent.save();

        learnerStateChannel.emit({
            version: this.agent.getVersion(),
            training: false,
        });

        return this;
    }

    private async trainLoop() {
        let errorCount = 0;

        while (true) {
            await new Promise(resolve => macroTasks.addTimeout(resolve, 100));

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
            learnerStateChannel.emit({
                version: this.agent.getVersion(),
                training: true,
            });

            await this.agent.train();

            const health = await this.agent.healthCheck();

            if (!health) {
                throw new Error('Health check failed');
            }

            await this.save();
            return true;
        } catch (error) {
            console.error('Error during training loop:', error);
            await this.agent.load();
            return false;
        } finally {
            learnerStateChannel.emit({
                version: this.agent.getVersion(),
                training: false,
            });
        }
    }
}