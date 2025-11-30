import * as tf from '@tensorflow/tfjs';
import {
    applyCrossAttentionLayer,
    applyLaNLayer,
    convertInputsToTokens,
    createDenseLayer,
    createInputs,
    createNormalizationLayer
} from '../ApplyLayers.ts';
import { Model } from '../def.ts';

type NetworkConfig = {
    dim: number;
    heads: number;
    coreDepth: number;
    lanUnits: number;
};

const policyNetworkConfig: NetworkConfig = {
    dim: 32,
    heads: 4,
    coreDepth: 6,
    lanUnits: 32,
};

const valueNetworkConfig: NetworkConfig = {
    dim: 16,
    heads: 1,
    coreDepth: 2,
    lanUnits: 16,
};

/**
 * v8: Star-Topology Aggregation & Deep ResNet Core
 * 
 * Отличия от v6/v5:
 * 1. Не использует единый Transformer-стек для всех токенов.
 * 2. Использует топологию "Звезда": Танк является центром, который агрегирует информацию от окружения.
 * 3. Агрегация происходит через Cross-Attention (Tank -> Enemies, Tank -> Allies, etc).
 * 4. Основная вычислительная мощность сосредоточена в глубоком ResNet-подобном стеке из LaN слоев (Gated Linear Units) после агрегации.
 * 5. Работает с фиксированным вектором состояния после агрегации, а не с последовательностью.
 */
export function createNetwork(modelName: Model, config: NetworkConfig = modelName === Model.Policy ? policyNetworkConfig : valueNetworkConfig) {
    const inputs = createInputs(modelName);
    const tokens = convertInputsToTokens(inputs, config.dim);

    // --- 1. Context Aggregation (Star Topology) ---
    // Танк "смотрит" на каждую группу сущностей и извлекает релевантную информацию.
    // Результат всегда [batch, 1, dim]

    const tankQuery = tokens.tankTok; // [batch, 1, dim]

    const enemiesContext = applyCrossAttentionLayer({
        name: modelName + '_enemiesAgg',
        heads: config.heads,
        qTok: tankQuery,
        kvTok: tokens.enemiesTok,
        kvMask: inputs.enemiesMaskInput,
        preNorm: true,
    });

    const alliesContext = applyCrossAttentionLayer({
        name: modelName + '_alliesAgg',
        heads: config.heads,
        qTok: tankQuery,
        kvTok: tokens.alliesTok,
        kvMask: inputs.alliesMaskInput,
        preNorm: true,
    });

    const bulletsContext = applyCrossAttentionLayer({
        name: modelName + '_bulletsAgg',
        heads: config.heads,
        qTok: tankQuery,
        kvTok: tokens.bulletsTok,
        kvMask: inputs.bulletsMaskInput,
        preNorm: true,
    });

    // --- 2. Fusion ---
    // Объединяем собственное состояние танка и агрегированные контексты
    const fusedState = tf.layers.concatenate({ name: modelName + '_fusion', axis: -1 })
        .apply([
            tankQuery,      // [batch, 1, dim]
            enemiesContext, // [batch, 1, dim]
            alliesContext,  // [batch, 1, dim]
            bulletsContext  // [batch, 1, dim]
        ]) as tf.SymbolicTensor;
    
    // fusedState shape: [batch, 1, dim * 4]

    // Проецируем обратно в dim (или lanUnits) для обработки в Core
    let x = createDenseLayer({
        name: modelName + '_projection',
        units: config.lanUnits,
        useBias: false, // LaN layer usually handles bias or norm
        activation: 'linear'
    }).apply(fusedState) as tf.SymbolicTensor;

    // Flattening is safe here because we have [batch, 1, dim]
    x = tf.layers.flatten({ name: modelName + '_flattenCore' }).apply(x) as tf.SymbolicTensor;


    // --- 3. Deep ResNet-LaN Core ---
    // Глубокая обработка агрегированного состояния

    for (let i = 0; i < config.coreDepth; i++) {
        const residual = x;
        
        // LaN Layer: y = tanh(W1 x) * tanh(W2 x)
        // Реализует Hadamard Representation: произведение двух ветвей с tanh активацией.
        // Это заменяет стандартную активацию и первый слой MLP.
        const lanOut = applyLaNLayer({
            name: `${modelName}_core_block${i}`,
            units: config.lanUnits, // Internal width
            preNorm: true
        }, x);

        // Output Projection: z = W3 y
        // Смешиваем признаки после нелинейности (аналог второго слоя в MLP)
        const projected = createDenseLayer({
            name: `${modelName}_core_proj${i}`,
            units: config.lanUnits, // Project back to residual stream dimension
            useBias: false,
            activation: 'linear'
        }).apply(lanOut) as tf.SymbolicTensor;

        // Residual Connection
        x = tf.layers.add({ name: `${modelName}_core_res${i}` })
            .apply([residual, projected]) as tf.SymbolicTensor;
    }

    // Final Norm before heads
    x = createNormalizationLayer({ name: modelName + '_finalNorm' }).apply(x) as tf.SymbolicTensor;

    // --- 4. Heads ---
    const numHeads = modelName === Model.Policy ? 4 : 1;
    const heads = [];
    
    for (let i = 0; i < numHeads; i++) {
        // Simple projection for heads
        const head = applyLaNLayer({
            name: `${modelName}_head${i}`,
            units: config.lanUnits,
            preNorm: false // Already normed
        }, x);
        heads.push(head);
    }

    return { inputs, heads };
}
