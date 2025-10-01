import * as tf from '@tensorflow/tfjs';
import { macroTasks } from '../../../../lib/TasksScheduler/macroTasks.ts';
import { throwingError } from '../../../../lib/throwingError.ts';

export function getNetworkVersion(network: tf.LayersModel): number {
    return network.optimizer.iterations ?? throwingError('Network version is not defined');
}

export function getNetworkLearningRate(network: tf.LayersModel): number {
    // @ts-expect-error
    return network.optimizer.learningRate;
}

export async function patientAction<T>(action: () => T | Promise<T>, attempts: number = 100): Promise<T> {
    while (true) {
        attempts--;
        try {
            return await action();
        } catch (error) {
            if (attempts <= 0) {
                throw error;
            }

            await new Promise(resolve => macroTasks.addTimeout(resolve, 30));
        }
    }
}

