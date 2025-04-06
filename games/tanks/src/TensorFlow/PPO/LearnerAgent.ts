import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { networkHealthCheck } from './train.ts';
import { CONFIG } from './config.ts';
import { macroTasks } from '../../../../../lib/TasksScheduler/macroTasks.ts';

export class LearnerAgent<B extends { version: number }> {
    protected version = 0;
    protected batches: B[] = [];

    protected network: tf.LayersModel;

    constructor(createNetwork: () => tf.LayersModel) {
        this.network = createNetwork();
    }

    public async init() {
        await this.load();

        return this;
    }

    public async save() {
        while (!(await this.upload())) {
            await new Promise(resolve => macroTasks.addTimeout(resolve, 100));
        }
    }

    public collectBatches(): Promise<unknown> {
        throw new Error('Not implemented');
    }

    public hasEnoughBatches(): boolean {
        return this.batches.length >= CONFIG.workerCount;
    }

    public extractBatches(): B[] {
        const batches = this.batches.filter(b => {
            const delta = this.version - b.version;
            if (delta > 2) {
                console.warn('[Train]: skipping batch with diff', delta);
                return false;
            }
            return true;
        });
        this.batches = [];
        return batches;
    }

    public async train(_batches: B[]) {
        throw new Error('Not implemented');
    }

    public finishTrain() {
        this.version += 1;
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

    protected updateOptimizersLR(lr: number) {
        setLR(this.network, lr);
    }
}

function setLR(o: tf.LayersModel, lr: number) {
    // @ts-expect-error
    o.optimizer.learningRate = lr;
}
