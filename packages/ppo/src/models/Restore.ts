import { loadLastNetworkFromDB, loadNetworkFromFS, saveNetworkToDB } from './Transfer.ts';
import { disposeNetwork } from './storage.ts';
import { Model } from './def.ts';

export function restoreModels(savePath: string, fsPath: string) {
    return Promise.all([
        loadLastNetworkFromDB(Model.Policy, savePath),
        loadLastNetworkFromDB(Model.Value, savePath),
    ]).then(([policyNetwork, valueNetwork]) => {
        if (!policyNetwork || !valueNetwork) {
            throw new Error('Failed to load models');
        }

        disposeNetwork(policyNetwork);
        disposeNetwork(valueNetwork);
    }).catch(error => {
        console.warn('Error loading models from DB', error);

        return upsertModels(savePath, fsPath);
    });
}

export function upsertModels(savePath: string, fsPath: string) {
    return Promise.all([
        loadNetworkFromFS(fsPath, Model.Policy),
        loadNetworkFromFS(fsPath, Model.Value),
    ]).then(([policyNetwork, valueNetwork]) => {
        if (!policyNetwork || !valueNetwork) {
            throw new Error('Failed to load models from FS');
        }

        return Promise.all([
            saveNetworkToDB(policyNetwork, Model.Policy, savePath),
            saveNetworkToDB(valueNetwork, Model.Value, savePath),
        ]).then(() => {
            disposeNetwork(policyNetwork);
            disposeNetwork(valueNetwork);
        }).catch(error => {
            console.error('Error saving models to DB', error);
        });
    }).catch(error => {
        console.error('Error loading models from FS', error);
    });
}
