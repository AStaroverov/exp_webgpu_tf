import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { predict } from '../train.ts';
import { InputArrays } from '../../Common/InputArrays.ts';
import { Model } from '../../Models/Transfer.ts';
import { disposeNetwork, getNetwork } from '../../Models/Utils.ts';

export class PlayerAgent {
    private policyNetwork?: tf.LayersModel;

    constructor() {
    }

    public static create() {
        return new PlayerAgent();
    }

    getVersion() {
        return this.policyNetwork?.optimizer.iterations ?? 0;
    }

    predict(state: InputArrays): { actions: Float32Array } {
        return predict(
            this.policyNetwork!,
            state,
        );
    }

    async sync() {
        return this.load();
    }

    private async load() {
        this.policyNetwork && disposeNetwork(this.policyNetwork);
        this.policyNetwork = await getNetwork(Model.Policy);
    }
}
