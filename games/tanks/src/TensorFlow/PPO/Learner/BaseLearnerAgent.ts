import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { networkHealthCheck } from '../train.ts';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { Model } from '../../Models/Transfer.ts';

export class BaseLearnerAgent {
    public network: tf.LayersModel;

    constructor(createNetwork: () => tf.LayersModel, public modelName: Model) {
        this.network = createNetwork();
    }

    public async init() {
        await this.load();
    }

    public async save() {
        while (!(await this.upload())) {
            await new Promise(resolve => macroTasks.addTimeout(resolve, 100));
        }
    }

    public async healthCheck(): Promise<boolean> {
        try {
            return await networkHealthCheck(this.network);
        } catch (error) {
            await new Promise(resolve => macroTasks.addTimeout(resolve, 100));
            return this.healthCheck();
        }
    }

    public async upload(): Promise<boolean> {
        throw new Error('Not implemented');
    }

    public async load(): Promise<boolean> {
        throw new Error('Not implemented');
    }

    public getVersion() {
        return this.network.optimizer.iterations;
    }

    public updateLR(lr: number) {
        setLR(this.network, lr);
    }
}

function setLR(o: tf.LayersModel, lr: number) {
    // @ts-expect-error
    o.optimizer.learningRate = lr;
}
