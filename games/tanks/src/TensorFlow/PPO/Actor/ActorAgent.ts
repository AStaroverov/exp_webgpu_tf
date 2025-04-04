import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { Memory } from '../../Common/Memory.ts';
import { getStoreModelPath } from '../../Common/tfUtils.ts';
import { CONFIG } from '../config.ts';
import { act } from '../train.ts';
import { InputArrays } from '../../Common/prepareInputArrays.ts';
import { createPolicyNetwork, createValueNetwork } from '../../Common/models.ts';
import { setModelState } from '../../Common/modelsCopy.ts';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { policyAgentState, valueAgentState } from '../../Common/Database.ts';

export class ActorAgent {
    private reuse = Infinity; // cannot reuse on init
    private policyVersion = -1;
    private valueVersion = -1;
    private memory: Memory;
    private policyNetwork: tf.LayersModel = createPolicyNetwork();
    private valueNetwork: tf.LayersModel = createValueNetwork();

    constructor() {
        this.memory = new Memory();
    }

    public static create() {
        return new ActorAgent();
    }

    dispose() {
        // this.policyNetwork?.dispose();
        // this.valueNetwork?.dispose();
        this.disposeMemory();
    }

    rememberAction(tankId: number, state: InputArrays, action: Float32Array, logProb: number, value: number) {
        this.memory.addFirstPart(tankId, state, action, logProb, value);
    }

    rememberReward(tankId: number, reward: number, done: boolean, isLast = false) {
        this.memory.updateSecondPart(tankId, reward, done, isLast);
    }

    readMemory() {
        return {
            version: this.policyVersion,
            memories: this.memory.getBatch(CONFIG.gamma, CONFIG.lam),
        };
    }

    disposeMemory() {
        this.memory.dispose();
    }

    act(state: InputArrays): {
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
        while (!(await this.load())) {
            await new Promise((resolve) => macroTasks.addTimeout(resolve, 1000));
        }
    }

    private async load() {
        try {
            const start = Date.now();
            const canReuse = this.reuse < CONFIG.reuseLimit;
            let valueVersion = -1;
            let policyVersion = -1;
            let isNewVersion = false;

            for (let i = 0; i < 1_000_000; i++) {
                if (i > 0) await new Promise(resolve => macroTasks.addTimeout(resolve, i * 100));
                const agentStates = await Promise.all([policyAgentState.get(), valueAgentState.get()]);
                valueVersion = agentStates[1]?.version ?? -1;
                policyVersion = agentStates[0]?.version ?? -1;
                isNewVersion = policyVersion > this.policyVersion && valueVersion > this.valueVersion;
                if (isNewVersion || canReuse) break;
            }

            const syncTime = Date.now() - start;
            if (syncTime > 1_000) {
                console.info('Sync time:', syncTime);
            }

            if (isNewVersion) {
                const [valueNetwork, policyNetwork] = await Promise.all([
                    tf.loadLayersModel(getStoreModelPath('value-model', CONFIG)),
                    tf.loadLayersModel(getStoreModelPath('policy-model', CONFIG)),
                ]);

                if (!valueNetwork || !policyNetwork) {
                    console.warn('Could not load models');
                    return false;
                }

                this.reuse = 0;
                this.policyVersion = policyVersion;
                this.valueVersion = valueVersion;
                this.valueNetwork = await setModelState(this.valueNetwork, valueNetwork);
                this.policyNetwork = await setModelState(this.policyNetwork, policyNetwork);
                valueNetwork.dispose();
                policyNetwork.dispose();
                console.log('Models updated successfully');
                return true;
            } else if (canReuse) {
                this.reuse = this.reuse + 1;
                console.log('Models reused successfully');
                return true;
            }

            return false;
        } catch (error) {
            console.warn('Could not sync models:', error);
            return false;
        }
    }
}
