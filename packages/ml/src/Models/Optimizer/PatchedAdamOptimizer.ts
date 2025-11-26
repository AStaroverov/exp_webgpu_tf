import * as tf from '@tensorflow/tfjs';
import { AdamOptimizer, NamedTensorMap, Variable } from '@tensorflow/tfjs';
import { ENGINE } from '@tensorflow/tfjs-core/dist/engine';
import { tidy } from '@tensorflow/tfjs-core/dist/globals';
import { add } from '@tensorflow/tfjs-core/dist/ops/add';
import { div } from '@tensorflow/tfjs-core/dist/ops/div';
import { mul } from '@tensorflow/tfjs-core/dist/ops/mul';
import { sqrt } from '@tensorflow/tfjs-core/dist/ops/sqrt';
import { square } from '@tensorflow/tfjs-core/dist/ops/square';
import { sub } from '@tensorflow/tfjs-core/dist/ops/sub';
import { zerosLike } from '@tensorflow/tfjs-core/dist/ops/zeros_like';
import { NamedTensor } from '@tensorflow/tfjs-core/dist/tensor_types';

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

        tidy(() => {
            const oneMinusAccBeta1 = sub(1, accBeta1);
            const oneMinusAccBeta2 = sub(1, accBeta2);
            varNames.forEach((name, i) => {
                const value = ENGINE.registeredVariables[name];
                const trainable = false;

                let firstMomentIndex = accumulatedFirstMoment
                    .findIndex(({ originalName }) => originalName === `${name}${FM_POSTFIX}`);
                let secondMomentIndex = accumulatedSecondMoment
                    .findIndex(({ originalName }) => originalName === `${name}${SM_POSTFIX}`);

                if (firstMomentIndex === -1) {
                    firstMomentIndex = accumulatedFirstMoment.length;
                    accumulatedFirstMoment.push({
                        originalName: `${name}${FM_POSTFIX}`,
                        variable: tidy(() => zerosLike(value).variable(trainable)),
                    });
                }
                if (secondMomentIndex === -1) {
                    secondMomentIndex = accumulatedSecondMoment.length;
                    accumulatedSecondMoment.push({
                        originalName: `${name}${SM_POSTFIX}`,
                        variable: tidy(() => zerosLike(value).variable(trainable)),
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
                const newFirstMoment = add(mul(firstMoment, this.beta1), mul(gradient, 1 - this.beta1));
                const newSecondMoment = add(mul(secondMoment, this.beta2), mul(square(gradient), 1 - this.beta2));
                const biasCorrectedFirstMoment = div(newFirstMoment, oneMinusAccBeta1);
                const biasCorrectedSecondMoment = div(newSecondMoment, oneMinusAccBeta2);
                firstMoment.assign(newFirstMoment);
                secondMoment.assign(newSecondMoment);

                // Compute adaptive learning rate update
                const adaptiveUpdate = mul(
                    div(biasCorrectedFirstMoment, add(sqrt(biasCorrectedSecondMoment), this.epsilon)),
                    -this.learningRate
                );

                // Apply adaptive update
                let newValue = add(value, adaptiveUpdate);

                // Apply additional updates (e.g., weight decay in AdamW)
                const additionalUpdate = this.computeAdditionalUpdate(value, name);
                if (additionalUpdate !== null) {
                    newValue = add(newValue, additionalUpdate);
                }

                value.assign(newValue);
            });
            accBeta1.assign(mul(accBeta1, this.beta1));
            accBeta2.assign(mul(accBeta2, this.beta2));
        });
        this.incrementIterations();
    }

    protected computeAdditionalUpdate(_value: Variable, _name: string): tf.Tensor | null {
        return null;
    }
}

tf.serialization.registerClass(PatchedAdamOptimizer);
