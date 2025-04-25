// Буфер опыта для PPO
import { shuffle } from '../../../../../lib/shuffle.ts';
import { InputArrays } from './InputArrays.ts';
import { flatTypedArray } from './flat.ts';

export type Batch = {
    size: number,
    states: InputArrays[],
    actions: Float32Array[],
    rewards: Float32Array,
    dones: Float32Array,
    mean: Float32Array[],
    logStd: Float32Array[],
    logProbs: Float32Array,
}

export class Memory {
    private map = new Map<number, SubMemory>();

    constructor() {
    }

    size() {
        let size = 0;

        this.map.forEach(subMemory => {
            size += subMemory.size();
        });

        return size;
    }

    toArray() {
        return Array.from(this.map.values());
    }

    addFirstPart(
        id: number,
        state: InputArrays,
        stateReward: number,
        action: Float32Array,
        mean: Float32Array,
        logStd: Float32Array,
        logProb: number,
    ) {
        if (!this.map.has(id)) {
            this.map.set(id, new SubMemory());
        }
        this.map.get(id)!.addFirstPart(state, stateReward, action, mean, logStd, logProb);
    }

    updateSecondPart(id: number, reward: number, done: boolean) {
        if (!this.map.has(id)) {
            throw new Error('SubMemory not found');
        }
        this.map.get(id)!.updateSecondPart(reward, done);
    }

    getBatch(): Batch {
        const batches = this.getBatches();
        const values = shuffle(Array.from(batches.values()));

        return {
            size: values.reduce((acc, batch) => acc + batch.size, 0),
            states: (values.map(batch => batch.states)).flat(),
            actions: (values.map(batch => batch.actions)).flat(),
            mean: (values.map(batch => batch.mean)).flat(),
            logStd: (values.map(batch => batch.logStd)).flat(),
            logProbs: flatTypedArray(values.map(batch => batch.logProbs)),
            rewards: flatTypedArray(values.map(batch => batch.rewards)),
            dones: flatTypedArray(values.map(batch => batch.dones)),
        };
    }

    getBatches() {
        const batches = new Map<number, Batch>();

        this.map.forEach((subMemory, id) => {
            batches.set(id, subMemory.getBatch());
        });

        return batches;
    }

    dispose() {
        this.map.forEach(subMemory => subMemory.dispose());
        this.map.clear();
    }
}

export class SubMemory {
    private states: InputArrays[] = [];
    private stateRewards: number[] = [];
    private actions: Float32Array[] = [];
    private mean: Float32Array[] = [];
    private logStd: Float32Array[] = [];
    private logProbs: number[] = [];
    private actionRewards: number[] = [];
    private dones: boolean[] = [];

    constructor() {
    }

    size() {
        return this.states.length;
    }

    isDone() {
        return this.dones[this.dones.length - 1];
    }

    addFirstPart(
        state: InputArrays,
        stateReward: number,
        action: Float32Array,
        mean: Float32Array,
        logStd: Float32Array,
        logProb: number,
    ) {
        if (this.isDone()) {
            return;
        }
        this.states.push(state);
        this.stateRewards.push(stateReward);
        this.actions.push(action);
        this.mean.push(mean);
        this.logStd.push(logStd);
        this.logProbs.push(logProb);
    }

    updateSecondPart(reward: number, done: boolean) {
        if (this.isDone()) {
            return;
        }
        this.actionRewards.push(reward);
        this.dones.push(done);
    }

    getBatch(): Batch {
        if (this.states.length === 0) {
            throw new Error('Memory is empty');
        }
        if (this.states.length !== this.dones.length) {
            throw new Error('States and dones length mismatch');
        }

        const deltaReward = this.actionRewards.map((aR) => 0.1 * aR);
        const dones = this.dones.map(done => done ? 1.0 : 0.0);
        dones[dones.length - 1] = 1.0;

        return {
            size: this.states.length,
            states: (this.states),
            actions: (this.actions),
            mean: (this.mean),
            logStd: (this.logStd),
            logProbs: new Float32Array(this.logProbs),
            rewards: new Float32Array(deltaReward),
            dones: new Float32Array(dones),
        };
    }

    dispose() {
        this.states.length = 0;
        this.stateRewards.length = 0;
        this.actions.length = 0;
        this.mean.length = 0;
        this.logStd.length = 0;
        this.logProbs.length = 0;
        this.actionRewards.length = 0;
        this.dones.length = 0;
    }
}
