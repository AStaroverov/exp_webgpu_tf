import { loadNetworkFromDB, loadNetworkFromFS, Model, saveNetworkToDB } from '../Transfer.ts';
import { disposeNetwork } from '../Utils.ts';

export function restoreModels(path: string) {
    return Promise.all([
        loadNetworkFromDB(Model.Policy),
        loadNetworkFromDB(Model.Value),
    ]).then(([policyNetwork, valueNetwork]) => {
        if (!policyNetwork || !valueNetwork) {
            throw new Error('Failed to load models');
        }

        disposeNetwork(policyNetwork);
        disposeNetwork(valueNetwork);
    }).catch(error => {
        console.warn('Error loading models from DB', error);

        return Promise.all([
            loadNetworkFromFS(path, Model.Policy),
            loadNetworkFromFS(path, Model.Value),
        ]).then(([policyNetwork, valueNetwork]) => {
            if (!policyNetwork || !valueNetwork) {
                throw new Error('Failed to load models from FS');
            }

            return Promise.all([
                saveNetworkToDB(policyNetwork, Model.Policy),
                saveNetworkToDB(valueNetwork, Model.Value),
            ]).then(() => {
                disposeNetwork(policyNetwork);
                disposeNetwork(valueNetwork);
            }).catch(error => {
                console.error('Error saving models to DB', error);
            });
        }).catch(error => {
            console.error('Error loading models from FS', error);
        });
    });
}