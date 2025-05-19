import { InputArrays } from './InputArrays.ts';

export type AgentMemoryBatch = {
    size: number,
    states: InputArrays[],
    actions: Float32Array[],
    rewards: Float32Array,
    dones: Float32Array,
    mean: Float32Array[],
    logStd: Float32Array[],
    logProbs: Float32Array,
}

export class AgentMemory {
    public states: InputArrays[] = [];
    public actions: Float32Array[] = [];
    public mean: Float32Array[] = [];
    public logStd: Float32Array[] = [];
    public logProbs: number[] = [];
    public rewards: number[] = [];
    public dones: boolean[] = [];

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
        action: Float32Array,
        mean: Float32Array,
        logStd: Float32Array,
        logProb: number,
    ) {
        if (this.isDone()) {
            return;
        }
        this.states.push(state);
        this.actions.push(action);
        this.mean.push(mean);
        this.logStd.push(logStd);
        this.logProbs.push(logProb);
    }

    updateSecondPart(reward: number, done: boolean) {
        if (this.isDone()) {
            return;
        }
        this.rewards.push(reward);
        this.dones.push(done);
    }

    getBatch(): AgentMemoryBatch {
        if (this.states.length === 0) {
            throw new Error('Memory is empty');
        }

        this.setMinLength();

        const dones = this.dones.map(done => done ? 1.0 : 0.0);
        dones[dones.length - 1] = 1.0;

        return {
            size: this.states.length,
            states: (this.states),
            actions: (this.actions),
            mean: (this.mean),
            logStd: (this.logStd),
            logProbs: new Float32Array(this.logProbs),
            rewards: new Float32Array(this.rewards),
            dones: new Float32Array(dones),
        };
    }

    dispose() {
        this.states.length = 0;
        this.actions.length = 0;
        this.mean.length = 0;
        this.logStd.length = 0;
        this.logProbs.length = 0;
        this.rewards.length = 0;
        this.dones.length = 0;
    }

    private setMinLength() {
        const minLength = Math.min(
            this.states.length,
            this.actions.length,
            this.mean.length,
            this.logStd.length,
            this.logProbs.length,
            this.rewards.length,
            this.dones.length,
        );

        this.states.length = minLength;
        this.actions.length = minLength;
        this.mean.length = minLength;
        this.logStd.length = minLength;
        this.logProbs.length = minLength;
        this.rewards.length = minLength;
        this.dones.length = minLength;
    }
}
