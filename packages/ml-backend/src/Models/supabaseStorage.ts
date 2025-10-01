/**
 * Supabase Storage integration for model files
 */

import { createClient, SupabaseClient } from '@supabase/supabase-js';
import * as tf from '@tensorflow/tfjs';

const SUPABASE_URL = process.env.SUPABASE_URL || '';
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY || '';
const SUPABASE_BUCKET = process.env.SUPABASE_BUCKET || 'Models';

let supabase: SupabaseClient | null = null;

function getSupabaseClient(): SupabaseClient | null {
    if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
        console.warn('⚠️  Supabase credentials not set, model sync disabled');
        return null;
    }

    if (!supabase) {
        supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
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
                // Prepare model.json content
                const modelJSON = {
                    modelTopology: modelArtifacts.modelTopology,
                    weightsManifest: [
                        {
                            paths: [`${modelName}.weights.bin`],
                            weights: modelArtifacts.weightSpecs,
                        },
                    ],
                    format: modelArtifacts.format,
                    generatedBy: modelArtifacts.generatedBy,
                    convertedBy: modelArtifacts.convertedBy,
                };

                // Convert weights to ArrayBuffer
                const weightsBuffer = tf.io.CompositeArrayBuffer.join(modelArtifacts.weightData);

                // Upload to Supabase: v{version}/{modelName}.json and .weights.bin
                const basePath = `v${version}`;

                const [jsonResult, weightsResult] = await Promise.all([
                    client.storage
                        .from(SUPABASE_BUCKET)
                        .upload(`${basePath}/${modelName}.json`, JSON.stringify(modelJSON), {
                            contentType: 'application/json',
                            upsert: true,
                        }),
                    client.storage
                        .from(SUPABASE_BUCKET)
                        .upload(`${basePath}/${modelName}.weights.bin`, weightsBuffer, {
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

/**
 * Get public URL for model in Supabase Storage
 */
export function getModelPublicUrl(modelName: string, version: number): string | null {
    const client = getSupabaseClient();
    if (!client) return null;

    const { data } = client.storage
        .from(SUPABASE_BUCKET)
        .getPublicUrl(`v${version}/${modelName}.json`);

    return data.publicUrl;
}

/**
 * Check if Supabase is configured
 */
export function isSupabaseConfigured(): boolean {
    return Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);
}
