import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { networkHealthCheck } from './train.ts';
import { CONFIG } from './config.ts';
import { macroTasks } from '../../../../../lib/TasksScheduler/macroTasks.ts';
import { Model } from '../Models/Transfer.ts';
import { memoryChannel } from '../DB';
import { Batch } from '../Common/Memory.ts';

export class LearnerAgent<
    B extends { memories: Batch, version: Record<Model, number> }
        = { memories: Batch, version: Record<Model, number> }
> {
    protected batches: B[] = [];

    protected network: tf.LayersModel;

    constructor(createNetwork: () => tf.LayersModel, public modelName: Model) {
        this.network = createNetwork();

        memoryChannel.obs.subscribe((batch) => {
            const delta = this.getVersion() - batch.version[this.modelName];

            if (delta > 3_000) {
                console.warn('[Train]: skipping batch with diff', delta);
            } else {
                this.batches.push(batch as B);
            }
        });
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

    public collectBatches(batches: B[]) {
        if (batches.length === 0) return;

        this.batches.push();
    }

    public hasEnoughBatches(): boolean {
        return this.batches.length >= CONFIG.workerCount;
    }

    public useBatches(): B[] {
        const batches = this.batches;
        this.batches = [];
        return batches;
    }

    public async train(_batches: B[]) {
        throw new Error('Not implemented');
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

    protected updateOptimizersLR(lr: number) {
        setLR(this.network, lr);
    }
}

function setLR(o: tf.LayersModel, lr: number) {
    // @ts-expect-error
    o.optimizer.learningRate = lr;
}
