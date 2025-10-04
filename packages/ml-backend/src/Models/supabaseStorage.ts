/**
 * Supabase Storage integration for model files
 */

import { createClient, RealtimeChannel, SupabaseClient } from '@supabase/supabase-js';
import * as tf from '@tensorflow/tfjs';
import { throwingError } from '../../../../lib/throwingError.ts';
import { AgentMemoryBatch } from '../../../ml-common/Memory.ts';

const SUPABASE_URL = process.env.SUPABASE_URL || throwingError('SUPABASE_URL not set');
const SUPABASE_KEY = process.env.SUPABASE_KEY || throwingError('SUPABASE_KEY not set');
const SUPABASE_MODELS_BUCKET = process.env.SUPABASE_MODELS_BUCKET || throwingError('SUPABASE_MODELS_BUCKET not set');
const SUPABASE_BUCKET_EXP_BATCHES = process.env.SUPABASE_BUCKET_EXP_BATCHES || throwingError('SUPABASE_BUCKET_EXP_BATCHES not set');

let supabase: SupabaseClient | null = null;

function getSupabaseClient(): SupabaseClient {
    if (!supabase) {
        supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
        console.info('✅ Supabase client initialized');
    }

    return supabase;
}

/**
 * Create TensorFlow.js IOHandler for saving models to Supabase Storage
 * @param modelName - name of the model (e.g., 'policy-model', 'value-model')
 * @param version - version number
 */
export function createSupabaseIOHandler(
    modelName: string,
    version: number
): tf.io.IOHandler {
    return {
        async save(modelArtifacts: tf.io.ModelArtifacts): Promise<tf.io.SaveResult> {
            const client = getSupabaseClient();
            if (!client) {
                throw new Error('Supabase client not configured');
            }

            try {
                // Prepare model.json content (everything except weightData)
                const { weightData, ...modelJSON } = modelArtifacts;

                // Convert weights to ArrayBuffer
                const weightsBuffer = tf.io.CompositeArrayBuffer.join(weightData);

                // Upload to Supabase: v{version}/{modelName}.json and .weights.bin
                const basePath = `v${version}-${modelName}`;

                const [jsonResult, weightsResult] = await Promise.all([
                    client.storage
                        .from(SUPABASE_MODELS_BUCKET)
                        .upload(`${basePath}/model.json`, JSON.stringify(modelJSON), {
                            contentType: 'application/json',
                            upsert: true,
                        }),
                    client.storage
                        .from(SUPABASE_MODELS_BUCKET)
                        .upload(`${basePath}/weights.bin`, weightsBuffer, {
                            contentType: 'application/octet-stream',
                            upsert: true,
                        }),
                ]);

                if (jsonResult.error) {
                    throw new Error(`Failed to upload ${modelName}.json: ${jsonResult.error.message}`);
                }

                if (weightsResult.error) {
                    throw new Error(`Failed to upload ${modelName}.weights.bin: ${weightsResult.error.message}`);
                }

                console.info(`📤 Uploaded ${modelName} v${version} to Supabase`);

                return { modelArtifactsInfo: tf.io.getModelArtifactsInfoForJSON(modelArtifacts) };
            } catch (error) {
                console.error(`❌ Failed to upload ${modelName} to Supabase:`, error);
                throw error;
            }
        },
    };
}

export function subscribeToExperienceBatches(
    onBatch: (batchInfo: {
        batchId: string;
        fileName: string;
        networkVersion: number;
        scenarioIndex: number;
        successRatio: number;
        timestamp: string;
    }) => void
): () => void {
    const client = getSupabaseClient();
    if (!client) {
        console.warn('⚠️  Cannot subscribe to experience batches - Supabase not configured');
        return () => { };
    }

    const channel: RealtimeChannel = client.channel('experience-notifications');

    channel
        .on('broadcast', { event: 'new-batch' }, (payload) => {
            console.info('📥 New experience batch notification:', payload.payload);
            onBatch(payload.payload);
        })
        .subscribe((status) => {
            if (status === 'SUBSCRIBED') {
                console.info('✅ Subscribed to experience batch notifications');
            }
        });

    return () => {
        channel.unsubscribe();
        console.info('❌ Unsubscribed from experience batch notifications');
    };
}

export async function downloadRecentExperienceBatches(count: number = 5): Promise<{
    batchId: string;
    networkVersion: number;
    scenarioIndex: number;
    successRatio: number;
    memoryBatch: AgentMemoryBatch;
    timestamp: string;
}[]> {
    const client = getSupabaseClient();
    if (!client) {
        console.warn('⚠️  Cannot download experience batches - Supabase not configured');
        return [];
    }

    try {
        const { data, error } = await client.storage
            .from(SUPABASE_BUCKET_EXP_BATCHES)
            .list('', { limit: count, sortBy: { column: 'created_at', order: 'desc' } });

        if (error) {
            throw error;
        }

        if (!data || data.length === 0) {
            console.info('ℹ️  No experience batches found in Supabase');
            return [];
        }

        const fileNames = data.map(file => file.name).filter((name): name is string => !!name);

        console.info(`✅ Found ${fileNames.length} recent experience batches`);

        return Promise.all(fileNames.map(downloadExperienceBatch)).then(batches =>
            batches.filter((b): b is NonNullable<typeof b> => b !== null)
        );
    } catch (error) {
        console.error('❌ Failed to list experience batches from Supabase:', error);
        return [];
    }
}

export async function downloadExperienceBatch(fileName: string): Promise<{
    batchId: string;
    networkVersion: number;
    scenarioIndex: number;
    successRatio: number;
    memoryBatch: AgentMemoryBatch;
    timestamp: string;
}> {
    const client = getSupabaseClient();

    try {
        const { data, error } = await client.storage
            .from(SUPABASE_BUCKET_EXP_BATCHES)
            .download(fileName);

        if (error) {
            throw error;
        }

        const text = await data.text();
        const batchData = JSON.parse(text);

        // Convert arrays back to typed arrays
        const memoryBatch: AgentMemoryBatch = {
            size: batchData.memoryBatch.size,
            states: batchData.memoryBatch.states,
            actions: batchData.memoryBatch.actions.map((a: number[]) => new Float32Array(a)),
            mean: batchData.memoryBatch.mean.map((m: number[]) => new Float32Array(m)),
            logStd: batchData.memoryBatch.logStd.map((ls: number[]) => new Float32Array(ls)),
            logProbs: new Float32Array(batchData.memoryBatch.logProbs),
            rewards: new Float32Array(batchData.memoryBatch.rewards),
            dones: new Float32Array(batchData.memoryBatch.dones),
        };

        console.info(`✅ Downloaded experience batch: ${batchData.batchId}`);

        return {
            memoryBatch,
            batchId: batchData.batchId,
            networkVersion: batchData.networkVersion,
            scenarioIndex: batchData.scenarioIndex,
            successRatio: batchData.successRatio,
            timestamp: batchData.timestamp,
        };
    } catch (error) {
        console.error(`❌ Failed to download batch ${fileName}:`, error);
        throw error;
    }
}

export async function deleteExperienceBatch(fileNames: string[]): Promise<void> {
    const client = getSupabaseClient();

    try {
        const { error } = await client.storage
            .from(SUPABASE_BUCKET_EXP_BATCHES)
            .remove(fileNames);

        if (error) {
            throw error;
        }

        console.info(`🗑️  Deleted processed batch: ${fileNames.join(', ')}`);
    } catch (error) {
        console.error(`❌ Failed to delete batch ${fileNames.join(', ')}:`, error);
    }
}

/**
 * Upload curriculum state to Supabase Storage
 * @param curriculumState - curriculum state data
 */
export async function uploadCurriculumState(curriculumState: {
    currentVersion: number;
    mapScenarioIndexToSuccessRatio: Record<number, number>;
}): Promise<void> {
    const client = getSupabaseClient();

    try {
        const fileName = 'curriculumState.json';
        const data = JSON.stringify(curriculumState);

        // Upload to Supabase Storage (upsert to overwrite existing file)
        const { error: uploadError } = await client.storage
            .from(SUPABASE_BUCKET_EXP_BATCHES)
            .upload(fileName, data, {
                contentType: 'application/json',
                upsert: true,
            });

        if (uploadError) {
            throw uploadError;
        }

        console.info(`✅ Uploaded curriculum state: version ${curriculumState.currentVersion}`);
    } catch (error) {
        console.error('❌ Failed to upload curriculum state:', error);
        throw error;
    }
}

/**
 * Download curriculum state from Supabase Storage
 * @returns curriculum state or default if not found
 */
export async function downloadCurriculumState(): Promise<{
    currentVersion: number;
    mapScenarioIndexToSuccessRatio: Record<number, number>;
}> {
    const client = getSupabaseClient();
    const defaultState = {
        currentVersion: 0,
        mapScenarioIndexToSuccessRatio: {},
    };

    try {
        const fileName = 'curriculumState.json';

        // Download from Supabase Storage
        const { data, error } = await client.storage
            .from(SUPABASE_BUCKET_EXP_BATCHES)
            .download(fileName);

        if (error) {
            console.warn('⚠️  Failed to download curriculum state, using default:', error.message);
            return defaultState;
        }

        if (!data) {
            console.warn('⚠️  No curriculum state found, using default');
            return defaultState;
        }

        const text = await data.text();
        const curriculumState = JSON.parse(text);

        console.info(`✅ Downloaded curriculum state: version ${curriculumState.currentVersion}`);
        return curriculumState;
    } catch (error) {
        console.error('❌ Failed to download curriculum state, using default:', error);
        return defaultState;
    }
}
