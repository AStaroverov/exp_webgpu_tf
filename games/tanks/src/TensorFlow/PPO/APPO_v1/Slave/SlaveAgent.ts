import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { ACTION_DIM } from '../../../Common/consts.ts';
import { getAgentState } from '../Database.ts';
import { computeLogProbTanh } from '../../../Common/computeLogProb.ts';
import { Memory } from '../../Common/Memory.ts';
import { getStoreModelPath } from '../utils.ts';
import { CONFIG } from '../../Common/config.ts';

export class SlaveAgent {
    private version = -1;
    private syncCountWithSameVersion = -1;

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
        return this.memory.getBatch(CONFIG.gamma, CONFIG.lam);
    }

    disposeMemory() {
        this.memory.dispose();
    }

    act(state: Float32Array): {
        rawActions: Float32Array,
        actions: Float32Array,
        logProb: number,
        value: number
    } {
        return tf.tidy(() => {
            const stateTensor = tf.tensor1d(state).expandDims(0);
            const predict = this.policyNetwork.predict(stateTensor) as tf.Tensor;
            const rawOutputSqueezed = predict.squeeze(); // [ACTION_DIM * 2] при batch=1
            const outMean = rawOutputSqueezed.slice([0], [ACTION_DIM]);   // ACTION_DIM штук
            const outLogStd = rawOutputSqueezed.slice([ACTION_DIM], [ACTION_DIM]);
            const clippedLogStd = outLogStd.clipByValue(-2, 0.2);
            const std = clippedLogStd.exp();
            const noise = tf.randomNormal([ACTION_DIM]).mul(std);
            const actions = outMean.add(noise);
            const logProb = computeLogProbTanh(actions, outMean, std);
            const value = this.valueNetwork.predict(stateTensor) as tf.Tensor;

            return {
                rawActions: actions.dataSync() as Float32Array,
                actions: actions.tanh().dataSync() as Float32Array,
                logProb: logProb.dataSync()[0],
                value: value.squeeze().dataSync()[0],
            };
        });
    }

    async sync() {
        try {
            let agentState;
            if (this.syncCountWithSameVersion >= 2) {
                const start = Date.now();
                agentState = await this.waitNewVersion();
                this.syncCountWithSameVersion = 0;
                console.log('[SlaveAgent] Awaiting', Date.now() - start);
            } else {
                agentState = await getAgentState();
            }

            const [valueNetwork, policyNetwork] = await Promise.all([
                tf.loadLayersModel(getStoreModelPath('value-model', CONFIG)),
                tf.loadLayersModel(getStoreModelPath('policy-model', CONFIG)),
            ]);

            if (agentState && valueNetwork && policyNetwork) {
                this.version = agentState.version;
                this.valueNetwork = valueNetwork;
                this.policyNetwork = policyNetwork;
                this.syncCountWithSameVersion++;
                return true;
            }

            return false;
        } catch (error) {
            console.warn('[SlaveAgent] Could not sync models:', error);
            return false;
        }
    }

    private async waitNewVersion() {
        while (true) {
            const agentState = await getAgentState();
            if ((agentState?.version ?? 0) > this.version) {
                return agentState;
            }

            await new Promise((resolve) => setTimeout(resolve, 100));
        }
    }
}
