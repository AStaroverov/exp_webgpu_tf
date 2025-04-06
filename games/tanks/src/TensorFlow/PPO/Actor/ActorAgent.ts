import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { Memory } from '../../Common/Memory.ts';
import { getStoreModelPath } from '../../Common/tfUtils.ts';
import { CONFIG } from '../config.ts';
import { act } from '../train.ts';
import { InputArrays } from '../../Common/InputArrays.ts';
import { setModelState } from '../../Common/modelsCopy.ts';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { policyAgentState, valueAgentState } from '../../Common/Database.ts';
import { createPolicyNetwork, createValueNetwork } from '../../Common/models.ts';

export class ActorAgent {
    private reuse = 0;
    private policyVersion = -1;
    private valueVersion = -1;
    private memory: Memory;
    private policyNetwork?: tf.LayersModel;
    private valueNetwork?: tf.LayersModel;

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
            policyVersion: this.policyVersion,
            valueVersion: this.valueVersion,
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
        if (this.policyNetwork == null || this.valueNetwork == null) {
            throw new Error('Models not loaded');
        }

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
            await new Promise(resolve => macroTasks.addTimeout(resolve, this.reuse * 1_000));

            const agentStates = await Promise.all([policyAgentState.get(), valueAgentState.get()]);
            const policyVersion = agentStates[0]?.version ?? -1;
            const valueVersion = agentStates[1]?.version ?? -1;
            const isNewVersion = policyVersion > this.policyVersion && valueVersion > this.valueVersion;

            if (isNewVersion || this.policyNetwork == null || this.valueNetwork == null) {
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
                this.valueNetwork = await setModelState(this.valueNetwork ?? createValueNetwork(), valueNetwork);
                this.policyNetwork = await setModelState(this.policyNetwork ?? createPolicyNetwork(), policyNetwork);
                valueNetwork.dispose();
                policyNetwork.dispose();
                console.log('Models updated successfully');
                return true;
            } else if (this.reuse >= 0) {
                this.reuse += 1;
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
