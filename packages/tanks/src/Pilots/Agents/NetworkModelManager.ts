import * as tf from '@tensorflow/tfjs';

import { InputArrays, prepareInputArrays } from "../../../../ml-common/InputArrays";
import { patientAction } from "../../../../ml-common/utils";
import { disposeNetwork } from "../../../../ml/src/Models/Utils";
import { batchAct } from "../../../../ml/src/PPO/train";
import { TankAgent } from "./CurrentActorAgent";
import { random } from '../../../../../lib/random';

export type NetworkModelManager = ReturnType<typeof createNetworkModelManager>;

export const createNetworkModelManager = (getter: () => Promise<tf.LayersModel>) => {
    let network: undefined | tf.LayersModel = undefined;
    let promiseNetwork: undefined | Promise<tf.LayersModel> = undefined;
    let dateRequestNetwork: number = 0;

    let scheduledAgents: {width: number, height: number, agent: TankAgent}[] = []
    let computedAgents = new Map<TankAgent<unknown>, {
        state: InputArrays,
        actions: Float32Array,
        logits: Float32Array,
        logProb: number
    }>();

    const updateNetwork = async () => {
        const now = Date.now();
        const delta = now - dateRequestNetwork 
        
        if (delta > 10_000 || promiseNetwork == null) {
            network = undefined;
            promiseNetwork?.then(disposeNetwork);
            promiseNetwork = patientAction(getter).then((net) => {
                return (network = net);
                // return (network = random() < 0.9
                //     ? perturbWeights(net, 0.3)
                //     : net)
            });
            dateRequestNetwork = now;
        }

        return await promiseNetwork;
    }
    const getNetwork = () => network;
    
    const schedule = (width: number, height: number, agent: TankAgent<unknown>) => {
        if (computedAgents.size > 0) {
            computedAgents.clear();
        }

        scheduledAgents.push({
            width,
            height,
            agent,
        });
    }
    
    const get = (agent: TankAgent<unknown>, train: boolean) => {
        if (network == null) {
            scheduledAgents = [];
            return;
        }

        if (scheduledAgents.length > 0) {
            const states = scheduledAgents.map(({width, height, agent}) => prepareInputArrays(agent.tankEid, width, height));
            const noises = train
                ? scheduledAgents.map(({agent}) => agent.getNoise?.())
                : undefined;
            const options = train 
                ? { greedy: false, epsilon: 0.25, noises } 
                : { greedy: true };
            const result = batchAct(network, states, options);
            
            for (const [index, {agent}] of scheduledAgents.entries()) {
                computedAgents.set(agent, {
                    state: states[index],
                    ...result[index]
                });
            }

            scheduledAgents = [];
        }
        
        return computedAgents.get(agent);
    }

    return {
        updateNetwork,
        getNetwork,
        schedule,
        get,
    };
}

export function perturbWeights(model: tf.LayersModel, scale: number) {
    let count = 0;
    tf.tidy(() => {
        for (const v of model.trainableWeights) {
            if (!isPerturbable(v)) continue;
            if (random() < 0.95) continue;
            
            count += 1;
            const val = v.read() as tf.Tensor;

            let eps: tf.Tensor;
            if (val.shape.length === 2) {
                // Матрица весов [out_dim, in_dim]
                const [outDim, inDim] = val.shape;
                const epsOut = tf.randomNormal([outDim, 1]);
                const epsIn = tf.randomNormal([1, inDim]);
                // Двухфакторный шум (снижает рандом до более "структурированного" уровня)
                eps = epsOut.mul(epsIn);
            } else {
                eps = tf.randomNormal(val.shape);
            }

            const std = tf.moments(val).variance.sqrt();
            const perturbed = val.add(eps.mul(std).mul(scale));
            (val as tf.Variable).assign(perturbed);
        }
    });
    console.info(`Perturbing model weights with scale ${scale}, count ${count} from ${model.trainableWeights.length}`);

    return model;
}

function isPerturbable(v: tf.LayerVariable) {
    if (v.dtype !== 'float32') return false;

    const name = v.name.toLowerCase();

    if (
        name.includes('norm') ||
        name.endsWith('/gamma') ||
        name.endsWith('/beta') ||
        name.endsWith('/moving_mean') ||
        name.endsWith('/moving_variance')
    ) {
        return false;
    }

    return true;
}