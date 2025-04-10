import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { networkHealthCheck } from '../train.ts';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { loadNetworkFromDB, Model, saveNetworkToDB } from '../../Models/Transfer.ts';
import { setModelState } from '../../Common/modelsCopy.ts';

export class BaseLearnerAgent {
    public network: tf.LayersModel;

    constructor(createNetwork: () => tf.LayersModel, public modelName: Model) {
        this.network = createNetwork();
    }

    public async init() {
        await this.load();
    }

    public async healthCheck(): Promise<boolean> {
        try {
            return networkHealthCheck(this.network);
        } catch (error) {
            await new Promise(resolve => macroTasks.addTimeout(resolve, 100));
            return this.healthCheck();
        }
    }

    public getVersion() {
        return this.network.optimizer.iterations;
    }

    public updateLR(lr: number) {
        setLR(this.network, lr);
    }

    public async load() {
        try {
            const network = await loadNetworkFromDB(this.modelName);

            if (!network) return false;

            this.network = await setModelState(this.network, network);

            network.dispose();

            console.log('Models loaded successfully');
            return true;
        } catch (error) {
            console.warn('Could not load models, starting with new ones:', error);
            return false;
        }
    }

    public async upload() {
        try {
            await saveNetworkToDB(this.network, this.modelName);
            return true;
        } catch (error) {
            console.error('Error saving models:', error);
            return false;
        }
    }
}

function setLR(o: tf.LayersModel, lr: number) {
    // @ts-expect-error
    o.optimizer.learningRate = lr;
}
