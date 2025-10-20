// Quick test for SAC Replay Buffer
import { prepareRandomInputArrays } from './InputArrays.ts';
import { SACMemory } from './Memory.ts';
import { SACReplayBuffer } from './SACReplayBuffer.ts';

console.log('Testing SAC Replay Buffer...');

export function testSACReplayBuffer() {
    console.log('Creating SACReplayBuffer...');
    const buffer = new SACReplayBuffer(1000, false);

    console.log('Creating test transitions...');
    const memory = new SACMemory();

    // Add some test transitions
    for (let i = 0; i < 10; i++) {
        const state = prepareRandomInputArrays();
        const action = new Float32Array([0.1, 0.2, 0.3]);
        const reward = Math.random();
        const nextState = prepareRandomInputArrays();
        const done = i === 9;

        memory.addTransition(state, action, reward, nextState, done);
    }

    const batch = memory.getBatch();
    if (batch) {
        buffer.addBatch(batch);
        console.log('✅ Added batch to replay buffer');
        console.log('Buffer size:', buffer.size());

        // Test sampling
        const sample = buffer.sample(4);
        if (sample) {
            console.log('✅ Sampled batch successfully');
            console.log('Sample size:', sample.size);
            console.log('States:', sample.states.length);
            console.log('NextStates:', sample.nextStates.length);
        }
    }

    console.log('✅ Test completed');
}

// Uncomment to run test
// testSACReplayBuffer();
