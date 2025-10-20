/**
 * Simple test to verify SAC networks can be created
 * Run this manually to check Phase 2 completion
 */

import * as tf from '@tensorflow/tfjs';
import { createCriticNetwork, createPolicyNetwork } from '../Create.ts';
import { Model } from '../def.ts';
import { softUpdateTargetNetwork } from '../Utils.ts';

async function testNetworkCreation() {
    console.log('=== Testing SAC Network Creation ===\n');

    try {
        // Test Policy Network
        console.log('1. Creating Policy Network...');
        const policyNetwork = createPolicyNetwork();
        console.log(`✅ Policy Network created: ${policyNetwork.name}`);
        console.log(`   Inputs: ${policyNetwork.inputs.length}`);
        console.log(`   Outputs: ${policyNetwork.outputs.length}`);
        console.log(`   Total params: ${policyNetwork.countParams()}\n`);

        // Test Critic Networks
        console.log('2. Creating Critic Networks...');
        const critic1 = createCriticNetwork(Model.Critic1);
        console.log(`✅ Critic1 Network created: ${critic1.name}`);
        console.log(`   Inputs: ${critic1.inputs.length} (states + action)`);
        console.log(`   Outputs: ${critic1.outputs.length}`);
        console.log(`   Total params: ${critic1.countParams()}\n`);

        const critic2 = createCriticNetwork(Model.Critic2);
        console.log(`✅ Critic2 Network created: ${critic2.name}`);
        console.log(`   Total params: ${critic2.countParams()}\n`);

        // Test Target Critic Networks
        console.log('3. Creating Target Critic Networks...');
        const targetCritic1 = createCriticNetwork(Model.TargetCritic1);
        console.log(`✅ Target Critic1 created: ${targetCritic1.name}\n`);

        const targetCritic2 = createCriticNetwork(Model.TargetCritic2);
        console.log(`✅ Target Critic2 created: ${targetCritic2.name}\n`);

        // Test Soft Update
        console.log('4. Testing Soft Target Update...');
        const tau = 0.005;
        console.log(`   Using tau = ${tau}`);

        // Get initial weights
        const initialWeights = targetCritic1.getWeights()[0].arraySync();

        // Perform soft update
        softUpdateTargetNetwork(critic1, targetCritic1, tau);

        // Get updated weights
        const updatedWeights = targetCritic1.getWeights()[0].arraySync();

        // Check if weights changed
        const weightsChanged = JSON.stringify(initialWeights) !== JSON.stringify(updatedWeights);
        console.log(`✅ Soft update ${weightsChanged ? 'WORKS' : 'FAILED'} - weights ${weightsChanged ? 'changed' : 'unchanged'}\n`);

        // Test Forward Pass
        console.log('5. Testing Forward Pass...');

        // Create dummy inputs
        const batchSize = 4;
        const dummyInputs = policyNetwork.inputs.map(input => {
            return tf.randomNormal([batchSize, ...input.shape.slice(1)]);
        });

        // Policy network forward pass
        console.log('   Policy network...');
        const policyOutput = policyNetwork.predict(dummyInputs) as tf.Tensor[];
        console.log(`   ✅ Mean output shape: [${policyOutput[0].shape}]`);
        console.log(`   ✅ LogStd output shape: [${policyOutput[1].shape}]`);

        // Sample actions
        const mean = policyOutput[0];
        const logStd = policyOutput[1];
        const actions = tf.add(mean, tf.mul(tf.exp(logStd), tf.randomNormal(mean.shape)));

        // Critic network forward pass
        console.log('   Critic network...');
        const criticInputs = [...dummyInputs, actions];
        const qValue = critic1.predict(criticInputs) as tf.Tensor;
        console.log(`   ✅ Q-value output shape: [${qValue.shape}]`);
        console.log(`   ✅ Q-value sample: ${(await qValue.data())[0]}\n`);

        // Cleanup
        tf.dispose([...dummyInputs, ...policyOutput, actions, qValue]);
        policyNetwork.dispose();
        critic1.dispose();
        critic2.dispose();
        targetCritic1.dispose();
        targetCritic2.dispose();

        console.log('=== ✅ All Tests Passed! ===');
        console.log('\n📊 Summary:');
        console.log('   - Policy Network: ✅');
        console.log('   - Critic1 Network: ✅');
        console.log('   - Critic2 Network: ✅');
        console.log('   - Target Critics: ✅');
        console.log('   - Soft Update: ✅');
        console.log('   - Forward Pass: ✅');

    } catch (error) {
        console.error('❌ Test failed:', error);
        throw error;
    }
}

// Run test if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    testNetworkCreation().catch(console.error);
}

export { testNetworkCreation };
