import * as tf from '@tensorflow/tfjs';
import { macroTasks } from '../../lib/TasksScheduler/macroTasks.ts';
import { CONFIG } from './config.ts';
import { CurriculumState, DEFAULT_CURRICULUM_STATE } from './Curriculum/types.ts';

export function setNetworkExpIteration(o: tf.LayersModel, it: number) {
    o.setUserDefinedMetadata({ ...o.getUserDefinedMetadata(), expIteration: it });
}

export function getNetworkExpIteration(network: tf.LayersModel): number {
    const meta = network.getUserDefinedMetadata() as undefined | { expIteration?: number };
    return meta?.expIteration ?? 0;
}

export function setNetworkLearningRate(o: tf.LayersModel, lr: number) {
    // @ts-expect-error
    o.optimizer.learningRate = lr;
}

export function getNetworkLearningRate(network: tf.LayersModel): number {
    // @ts-expect-error
    return network.optimizer.learningRate ?? CONFIG.lrConfig.initial;
}

export function setNetworkPerturbConfig(o: tf.LayersModel, chance: number, scale: number) {
    o.setUserDefinedMetadata({ ...o.getUserDefinedMetadata(), perturbScale: scale, perturbChance: chance });
}

export function getNetworkPerturbConfig(network: tf.LayersModel): { scale: number, chance: number } {
    const meta = network.getUserDefinedMetadata() as undefined | { perturbScale?: number, perturbChance?: number };
    return { scale: meta?.perturbScale ?? CONFIG.perturbWeightsConfig.initial, chance: meta?.perturbChance ?? CONFIG.perturbWeightsConfig.initial };
}

export function setNetworkCurriculumState(network: tf.LayersModel, curriculumState: CurriculumState) {
    network.setUserDefinedMetadata({
        ...network.getUserDefinedMetadata(),
        curriculumState,
    });
}

export function getNetworkCurriculumState(network: tf.LayersModel): CurriculumState {
    const meta = network.getUserDefinedMetadata() as undefined | { curriculumState?: CurriculumState };
    return cloneCurriculumState(meta?.curriculumState ?? DEFAULT_CURRICULUM_STATE);
}

function cloneCurriculumState(state: CurriculumState): CurriculumState {
    return {
        currentVersion: state.currentVersion,
        mapScenarioIndexToSuccessRatio: { ...state.mapScenarioIndexToSuccessRatio },
    };
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

