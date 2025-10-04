import * as tf from '@tensorflow/tfjs';
import { loadModelFromSupabase } from './supabaseStorage.ts';

export async function loadModel(
    modelName: string,
    version: number = 0
): Promise<tf.LayersModel> {
    // Try loading from Supabase
    console.info(`Loading ${modelName} v${version} from Supabase...`);
    return await loadModelFromSupabase(modelName, version);
}
