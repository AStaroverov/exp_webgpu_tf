import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { AgentState, getAgentState } from '../Database.ts';
import { Memory } from '../../Common/Memory.ts';
import { getStoreModelPath } from '../utils.ts';
import { CONFIG } from '../../Common/config.ts';
import { act } from '../../Common/train.ts';

export class SlaveAgent {
    private reuse = -1;
    private version = -1;
    private memory: Memory;
    private policyNetwork!: tf.LayersModel;
    private valueNetwork!: tf.LayersModel;

    constructor() {
        this.memory = new Memory();
    }

    public static create() {
        return new SlaveAgent();
    }

    dispose() {
        this.policyNetwork?.dispose();
        this.valueNetwork?.dispose();
        this.disposeMemory();
    }

    rememberAction(tankId: number, state: Float32Array, action: Float32Array, logProb: number, value: number) {
        this.memory.addFirstPart(tankId, state, action, logProb, value);
    }

    rememberReward(tankId: number, reward: number, done: boolean, isLast = false) {
        this.memory.updateSecondPart(tankId, reward, done, isLast);
    }

    readMemory() {
        return {
            version: this.version,
            memories: this.memory.getBatch(CONFIG.gamma, CONFIG.lam),
        };
    }

    disposeMemory() {
        this.memory.dispose();
    }

    act(state: Float32Array): {
        actions: Float32Array,
        logProb: number,
        value: number
    } {
        return act(
            this.policyNetwork,
            this.valueNetwork,
            state,
        );
    }

    async sync() {
        try {
            const canReuse = this.reuse < CONFIG.reuseLimit;
            let agentState: undefined | AgentState;
            let start = Date.now();
            let isNewVersion = false;
            for (let i = 0; i < 1_000_000; i++) {
                if (i > 0) await new Promise(resolve => setTimeout(resolve, i * 100));
                agentState = await getAgentState();
                isNewVersion = (agentState?.version ?? -1) > this.version;
                if (agentState && (isNewVersion || canReuse)) break;
            }

            if (!isNewVersion) {
                await new Promise(resolve => setTimeout(resolve, 2_000));
            }

            const syncTime = Date.now() - start;
            if (syncTime > 1_000) {
                console.info('[SlaveAgent] Sync time:', syncTime);
            }

            const [valueNetwork, policyNetwork] = await Promise.all([
                tf.loadLayersModel(getStoreModelPath('value-model', CONFIG)),
                tf.loadLayersModel(getStoreModelPath('policy-model', CONFIG)),
            ]);

            if (agentState && valueNetwork && policyNetwork) {
                // we decay version to avoid reusing the same model more than N times
                this.reuse = this.version === agentState.version ? this.reuse + 1 : 0;
                this.version = agentState.version;
                this.valueNetwork = valueNetwork;
                this.policyNetwork = policyNetwork;
                return true;
            }

            return false;
        } catch (error) {
            console.warn('[SlaveAgent] Could not sync models:', error);
            return false;
        }
    }
}
