import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { createPolicyNetwork } from '../../Common/models.ts';
import { getStoreModelPath } from '../utils.ts';
import { getAgentState } from '../Database.ts';
import { predict } from '../Common/train.ts';
import { CONFIG } from '../Common/config.ts';
import { InputArrays } from '../../Common/prepareInputArrays.ts';

export class PlayerAgent {
    private version = 0;
    private policyNetwork!: tf.LayersModel;  // Сеть политики

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
        if (!(await this.load())) {
            this.policyNetwork = createPolicyNetwork();
        }

        return this;
    }

    private async load() {
        try {
            const [agentState, policyNetwork] = await Promise.all([
                getAgentState(),
                tf.loadLayersModel(getStoreModelPath('policy-model', CONFIG)),
            ]);

            if (!policyNetwork) {
                return false;
            }

            this.version = agentState?.version ?? 0;
            this.policyNetwork = policyNetwork;
            console.log('[PlayerAgent] Models loaded successfully');
            return true;
        } catch (error) {
            console.warn('[PlayerAgent] Could not load models, starting with new ones:', error);
            return false;
        }
    }
}
