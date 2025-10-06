import { NamedTensor } from '@tensorflow/tfjs-core/dist/tensor_types';
import * as tf from '../../../ml-common/tf';
import { AdamOptimizer, NamedTensorMap, Variable } from '../../../ml-common/tf';

const FM_POSTFIX = '/m';
const SM_POSTFIX = '/v';

export class PatchedAdamOptimizer extends AdamOptimizer {
    static className = 'PatchedAdamOptimizer';

    constructor(learningRate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon?: number) {
        super(learningRate, beta1, beta2, epsilon);
    }

    // NOTE: we key Adam moments by variable name instead of index
    // because TF.js layers can add variables after the first step.
    applyGradients(variableGradients: NamedTensor[] | NamedTensorMap) {
        // @ts-expect-error
        const accBeta1 = this.accBeta1;
        // @ts-expect-error
        const accBeta2 = this.accBeta2;
        // @ts-expect-error
        const accumulatedFirstMoment = this.accumulatedFirstMoment as { originalName: string, variable: Variable }[];
        // @ts-expect-error
        const accumulatedSecondMoment = this.accumulatedSecondMoment as { originalName: string, variable: Variable }[];

        const varNames = Array.isArray(variableGradients) ?
            variableGradients.map(v => v.name) :
            Object.keys(variableGradients);
        tf.tidy(() => {
            const oneMinusAccBeta1 = tf.sub(1, accBeta1);
            const oneMinusAccBeta2 = tf.sub(1, accBeta2);
            varNames.forEach((name, i) => {
                const value = tf.engine().registeredVariables[name];
                const trainable = false;

                let firstMomentIndex = accumulatedFirstMoment
                    .findIndex(({ originalName }) => originalName === `${name}${FM_POSTFIX}`);
                let secondMomentIndex = accumulatedSecondMoment
                    .findIndex(({ originalName }) => originalName === `${name}${SM_POSTFIX}`);

                if (firstMomentIndex === -1) {
                    firstMomentIndex = accumulatedFirstMoment.length;
                    accumulatedFirstMoment.push({
                        originalName: `${name}${FM_POSTFIX}`,
                        variable: tf.tidy(() => tf.zerosLike(value).variable(trainable)),
                    });
                }
                if (secondMomentIndex === -1) {
                    secondMomentIndex = accumulatedSecondMoment.length;
                    accumulatedSecondMoment.push({
                        originalName: `${name}${SM_POSTFIX}`,
                        variable: tf.tidy(() => tf.zerosLike(value).variable(trainable)),
                    });
                }

                const gradient = Array.isArray(variableGradients) ?
                    variableGradients[i].tensor :
                    variableGradients[name];

                if (gradient == null) {
                    return;
                }

                const firstMoment = accumulatedFirstMoment[firstMomentIndex].variable;
                const secondMoment = accumulatedSecondMoment[secondMomentIndex].variable;
                const newFirstMoment = tf.add(tf.mul(firstMoment, this.beta1), tf.mul(gradient, 1 - this.beta1));
                const newSecondMoment = tf.add(tf.mul(secondMoment, this.beta2), tf.mul(tf.square(gradient), 1 - this.beta2));
                const biasCorrectedFirstMoment = tf.div(newFirstMoment, oneMinusAccBeta1);
                const biasCorrectedSecondMoment = tf.div(newSecondMoment, oneMinusAccBeta2);
                firstMoment.assign(newFirstMoment);
                secondMoment.assign(newSecondMoment);
                const newValue = tf.add(tf.mul(tf.div(biasCorrectedFirstMoment, tf.add(tf.sqrt(biasCorrectedSecondMoment), this.epsilon)), -this.learningRate), value);
                value.assign(newValue);
            });
            accBeta1.assign(tf.mul(accBeta1, this.beta1));
            accBeta2.assign(tf.mul(accBeta2, this.beta2));
        });
        this.incrementIterations();
    }
}

tf.serialization.registerClass(PatchedAdamOptimizer);
