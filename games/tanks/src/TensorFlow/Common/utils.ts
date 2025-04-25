import * as tf from '@tensorflow/tfjs';
import { macroTasks } from '../../../../../lib/TasksScheduler/macroTasks.ts';
import { throwingError } from '../../../../../lib/throwingError.ts';

export function getNetworkVersion(network: tf.LayersModel): number {
    const version = network.optimizer.iterations ?? throwingError('Network version is not defined');
    return version;
}

export function getNetworkLearningRate(network: tf.LayersModel): number {
    // @ts-expect-error
    return network.optimizer.learningRate;
}

export async function patientAction<T>(action: () => T | Promise<T>): Promise<T> {
    try {
        return await action();
    } catch (error) {
        await new Promise(resolve => macroTasks.addTimeout(resolve, 100));
        return patientAction(action);
    }
}

