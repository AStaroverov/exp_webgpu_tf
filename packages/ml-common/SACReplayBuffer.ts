// SAC Replay Buffer Manager
// Manages experience replay for SAC training

import { InputArrays } from './InputArrays.ts';
import { SACMemoryBatch } from './Memory.ts';
import { PrioritizedReplayBuffer } from './PrioritizedReplayBuffer.ts';
import { ReplayBuffer } from './ReplayBuffer.ts';

/**
 * Replay buffer for SAC that stores (s, a, r, s', done) transitions
 */
export class SACReplayBuffer {
    private states: InputArrays[] = [];
    private actions: Float32Array[] = [];
    private rewards: number[] = [];
    private nextStates: InputArrays[] = [];
    private dones: boolean[] = [];
    private perturbed: boolean[] = [];

    private maxSize: number;
    private currentSize: number = 0;
    private currentIndex: number = 0;

    private buffer?: ReplayBuffer | PrioritizedReplayBuffer;
    private usePrioritized: boolean;
    private tdErrors?: Float32Array;

    constructor(maxSize: number, usePrioritized: boolean = false) {
        this.maxSize = maxSize;
        this.usePrioritized = usePrioritized;
    }

    /**
     * Add a batch of transitions from SACMemoryBatch
     */
    addBatch(batch: SACMemoryBatch) {
        for (let i = 0; i < batch.size; i++) {
            this.addTransition(
                batch.states[i],
                batch.actions[i],
                batch.rewards[i],
                batch.nextStates[i],
                batch.dones[i] > 0.5,
                batch.perturbed[i] > 0.5,
            );
        }
    }

    /**
     * Add a single transition
     */
    addTransition(
        state: InputArrays,
        action: Float32Array,
        reward: number,
        nextState: InputArrays,
        done: boolean,
        perturbed: boolean = false,
    ) {
        if (this.currentSize < this.maxSize) {
            // Buffer not full yet, append
            this.states.push(state);
            this.actions.push(action);
            this.rewards.push(reward);
            this.nextStates.push(nextState);
            this.dones.push(done);
            this.perturbed.push(perturbed);
            this.currentSize++;
        } else {
            // Buffer full, overwrite oldest
            this.states[this.currentIndex] = state;
            this.actions[this.currentIndex] = action;
            this.rewards[this.currentIndex] = reward;
            this.nextStates[this.currentIndex] = nextState;
            this.dones[this.currentIndex] = done;
            this.perturbed[this.currentIndex] = perturbed;
        }

        this.currentIndex = (this.currentIndex + 1) % this.maxSize;

        // Invalidate buffer when new data is added
        this.buffer = undefined;
    }

    /**
     * Sample a batch of transitions
     */
    sample(
        batchSize: number,
        topPercent: number = 0.2,
        topPickChance: number = 0.2,
    ): SACMemoryBatch | undefined {
        if (this.currentSize === 0) {
            return undefined;
        }

        // Create or update buffer
        if (!this.buffer) {
            if (this.usePrioritized && this.tdErrors) {
                this.buffer = new PrioritizedReplayBuffer(this.tdErrors);
            } else {
                this.buffer = new ReplayBuffer(this.currentSize);
            }
        }

        // Get indices
        let indices: number[];
        if (this.buffer instanceof PrioritizedReplayBuffer) {
            indices = this.buffer.getSampleWithTop(
                batchSize,
                0,
                this.currentSize,
                topPercent,
                topPickChance,
            );
        } else {
            indices = this.buffer.getSample(batchSize, 0, this.currentSize);
        }

        // Gather batch
        const sampledStates: InputArrays[] = [];
        const sampledActions: Float32Array[] = [];
        const sampledRewards: number[] = [];
        const sampledNextStates: InputArrays[] = [];
        const sampledDones: boolean[] = [];
        const sampledPerturbed: boolean[] = [];

        for (const idx of indices) {
            sampledStates.push(this.states[idx]);
            sampledActions.push(this.actions[idx]);
            sampledRewards.push(this.rewards[idx]);
            sampledNextStates.push(this.nextStates[idx]);
            sampledDones.push(this.dones[idx]);
            sampledPerturbed.push(this.perturbed[idx]);
        }

        return {
            size: batchSize,
            states: sampledStates,
            actions: sampledActions,
            rewards: new Float32Array(sampledRewards),
            nextStates: sampledNextStates,
            // @ts-expect-error - js conversion
            dones: new Float32Array(sampledDones),
            // @ts-expect-error - js conversion
            perturbed: new Float32Array(sampledPerturbed),
        };
    }

    /**
     * Update TD errors for prioritized replay
     */
    updateTDErrors(tdErrors: Float32Array) {
        this.tdErrors = tdErrors;
        this.buffer = undefined; // Force buffer recreation
    }

    /**
     * Get current buffer size
     */
    size(): number {
        return this.currentSize;
    }

    /**
     * Check if buffer has enough samples
     */
    canSample(minSize: number): boolean {
        return this.currentSize >= minSize;
    }

    /**
     * Clear the buffer
     */
    clear() {
        this.states.length = 0;
        this.actions.length = 0;
        this.rewards.length = 0;
        this.nextStates.length = 0;
        this.dones.length = 0;
        this.perturbed.length = 0;
        this.currentSize = 0;
        this.currentIndex = 0;
        this.buffer = undefined;
        this.tdErrors = undefined;
    }
}
