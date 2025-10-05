import { createClient, SupabaseClient } from '@supabase/supabase-js';
import * as tf from '@tensorflow/tfjs';
import { parse } from 'devalue';
import { throwingError } from '../../../../lib/throwingError.ts';
import { AgentMemoryBatch } from '../../../ml-common/Memory.ts';
import { DEFAULT_EXPERIMENT } from '../../../ml-common/config.ts';

const SUPABASE_URL = process.env.SUPABASE_URL || throwingError('SUPABASE_URL not set');
const SUPABASE_KEY = process.env.SUPABASE_KEY || throwingError('SUPABASE_KEY not set');
const SUPABASE_MODELS_BUCKET = process.env.SUPABASE_MODELS_BUCKET || throwingError('SUPABASE_MODELS_BUCKET not set');
const SUPABASE_BUCKET_EXP_BATCHES = process.env.SUPABASE_BUCKET_EXP_BATCHES || throwingError('SUPABASE_BUCKET_EXP_BATCHES not set');

let supabase: SupabaseClient | null = null;

export function getSupabaseClient(): SupabaseClient {
    if (!supabase) {
        supabase = createClient(SUPABASE_URL, SUPABASE_KEY, {
            global: {
                headers: {
                    apikey: SUPABASE_KEY,
                    Authorization: `Bearer ${SUPABASE_KEY}`,
                },
            },
            realtime: {
                params: {
                    apikey: SUPABASE_KEY,
                },
            },
        });
        console.info('✅ Supabase client initialized');
    }

    return supabase;
}

export function createSupabaseIOHandler(
    modelName: string,
    version: number
): tf.io.IOHandler {
    return {
        async save(modelArtifacts: tf.io.ModelArtifacts): Promise<tf.io.SaveResult> {
            const client = getSupabaseClient();

            try {
                // Prepare model.json content (everything except weightData)
                const { weightData, ...modelJSON } = modelArtifacts;

                // Convert weights to ArrayBuffer
                const weightsBuffer = tf.io.CompositeArrayBuffer.join(weightData);

                // Upload to Supabase: {expName}/v{version}/{modelName}.json and .weights.bin
                const basePath = `${DEFAULT_EXPERIMENT.expName}/v${version}-${modelName}`;

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

export async function downloadRecentExperienceBatches(count: number): Promise<{
    batchId: string;
    networkVersion: number;
    scenarioIndex: number;
    successRatio: number;
    memoryBatch: AgentMemoryBatch;
    timestamp: string;
}[]> {
    const client = getSupabaseClient();

    try {
        const { data, error } = await client.storage
            .from(SUPABASE_BUCKET_EXP_BATCHES)
            .list(DEFAULT_EXPERIMENT.expName, { limit: count, sortBy: { column: 'created_at', order: 'desc' } });

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
            .download(`${DEFAULT_EXPERIMENT.expName}/${fileName}`);

        if (error) {
            throw error;
        }

        const text = await data.text();
        const batchData = parse(text);

        console.info(`✅ Downloaded experience batch: ${batchData.batchId}`);

        return batchData;
    } catch (error) {
        console.error(`❌ Failed to download batch ${fileName}:`, error);
        throw error;
    }
}

export async function deleteExperienceBatch(fileNames: string[]): Promise<void> {
    const client = getSupabaseClient();

    try {
        const filePathsWithPrefix = fileNames.map(name => `${DEFAULT_EXPERIMENT.expName}/${name}`);
        const { error } = await client.storage
            .from(SUPABASE_BUCKET_EXP_BATCHES)
            .remove(filePathsWithPrefix);

        if (error) {
            throw error;
        }

        console.info(`🗑️  Deleted processed batch: ${fileNames.join(', ')}`);
    } catch (error) {
        console.error(`❌ Failed to delete batch ${fileNames.join(', ')}:`, error);
    }
}

export async function uploadCurriculumState(curriculumState: {
    currentVersion: number;
    mapScenarioIndexToSuccessRatio: Record<number, number>;
}): Promise<void> {
    const client = getSupabaseClient();

    try {
        const fileName = `${DEFAULT_EXPERIMENT.expName}/curriculumState.json`;
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
        const fileName = `${DEFAULT_EXPERIMENT.expName}/curriculumState.json`;

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
