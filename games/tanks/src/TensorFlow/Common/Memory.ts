// Буфер опыта для PPO
import { shuffle } from '../../../../../lib/shuffle.ts';
import { InputArrays } from './InputArrays.ts';

export type Batch = {
    states: InputArrays[],
    actions: Float32Array[],
    logProbs: number[],
    // meta
    size: number,
    dones: number[],
    rewards: number[],
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

    addFirstPart(id: number, state: InputArrays, action: Float32Array, logProb: number) {
        if (!this.map.has(id)) {
            this.map.set(id, new SubMemory());
        }
        this.map.get(id)!.addFirstPart(state, action, logProb);
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
        const batch = {
            size: values.reduce((acc, batch) => acc + batch.size, 0),
            states: (values.map(batch => batch.states)).flat(),
            actions: (values.map(batch => batch.actions)).flat(),
            logProbs: (values.map(batch => batch.logProbs)).flat(),
            rewards: (values.map(batch => batch.rewards)).flat(),
            dones: (values.map(batch => batch.dones)).flat(),
        };

        return batch;
    }

    // Метод для получения батча для обучения
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
    private actions: Float32Array[] = [];
    private logProbs: number[] = [];
    private rewards: number[] = [];
    private dones: boolean[] = [];

    private tmpRewards: number[] = [];
    private tmpDones: boolean[] = [];

    constructor() {
    }

    size() {
        return this.states.length;
    }

    addFirstPart(state: InputArrays, action: Float32Array, logProb: number) {
        this.collapseTmpData();

        this.states.push(state);
        this.actions.push(action);
        this.logProbs.push(logProb);
    }

    updateSecondPart(reward: number, done: boolean) {
        this.tmpRewards.push(reward);
        this.tmpDones.push(done);
    }

    getBatch(): Batch {
        if (this.states.length === 0) {
            throw new Error('Memory is empty');
        }
        if (this.tmpDones.length > 0 || this.tmpRewards.length > 0) {
            this.collapseTmpData();
        }
        if (this.states.length !== this.rewards.length || this.states.length !== this.dones.length) {
            const minLen = Math.min(this.states.length, this.rewards.length, this.dones.length);
            this.states.length = minLen;
            this.actions.length = minLen;
            this.logProbs.length = minLen;
            this.rewards.length = minLen;
            this.dones.length = minLen;
        }

        const dones = (this.dones.map(done => done ? 1.0 : 0.0));
        dones[dones.length - 1] = 1.0;

        return {
            size: this.states.length,
            states: (this.states),
            actions: (this.actions),
            logProbs: (this.logProbs),
            rewards: (this.rewards),
            dones: (this.dones.map(done => done ? 1.0 : 0.0)),
        };
    }

    dispose() {
        this.states = [];
        this.actions = [];
        this.logProbs = [];
        this.rewards = [];
        this.dones = [];
        this.tmpRewards = [];
        this.tmpDones = [];
    }

    private collapseTmpData() {
        this.rewards.push(this.tmpRewards.reduce((acc, val) => acc + val, 0));
        this.dones.push(this.tmpDones.reduce((acc, val) => acc && val, true));
        this.tmpRewards = [];
        this.tmpDones = [];
    }
}
