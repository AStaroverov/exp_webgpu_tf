/**
 * Persist the curriculum state inside the policy network's user-defined metadata,
 * so progress survives a learner restart (the network is reloaded from storage).
 * Verbatim port of `packages/ppo_tanks/src/curriculum/curriculumMeta.ts`.
 */

import * as tf from '@tensorflow/tfjs';
import { CurriculumState, DEFAULT_CURRICULUM_STATE } from './types.ts';

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
        iteration: state.iteration,
        mapScenarioIndexToSuccessRatio: { ...state.mapScenarioIndexToSuccessRatio },
    };
}
