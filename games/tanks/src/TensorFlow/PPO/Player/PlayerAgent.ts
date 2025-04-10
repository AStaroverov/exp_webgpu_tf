import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { createPolicyNetwork } from '../../Models/Create.ts';
import { predict } from '../train.ts';
import { InputArrays } from '../../Common/InputArrays.ts';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { setModelState } from '../../Common/modelsCopy.ts';
import { loadNetworkFromDB, Model } from '../../Models/Transfer.ts';

export class PlayerAgent {
    private version = 0;
    private policyNetwork: tf.LayersModel = createPolicyNetwork();

    constructor() {
    }

    public static create() {
        return new PlayerAgent();
    }

    getVersion() {
        return this.version;
    }

    predict(state: InputArrays): { actions: Float32Array } {
        return predict(
            this.policyNetwork,
            state,
        );
    }

    async sync() {
        while (!(await this.load())) {
            await new Promise((resolve) => macroTasks.addTimeout(resolve, 1000));
        }

        return this;
    }

    private async load() {
        try {
            const policyNetwork = await loadNetworkFromDB(Model.Policy);

            if (!policyNetwork) return false;

            this.policyNetwork = await setModelState(this.policyNetwork, policyNetwork);

            policyNetwork.dispose();

            console.log('[PlayerAgent] Models loaded successfully');
            return true;
        } catch (error) {
            console.warn('[PlayerAgent] Could not load models, starting with new ones:', error);
            return false;
        }
    }
}
