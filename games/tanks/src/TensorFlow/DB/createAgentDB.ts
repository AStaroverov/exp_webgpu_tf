import PouchDB from 'pouchdb';
import { upsert } from './utils.ts';

export function createAgentDB<State>(name: string) {
    const db = new PouchDB(name);

    async function set(state: Omit<State, '_id'>) {
        return upsert(db, '0', state);
    }

    async function get(): Promise<State | undefined> {
        try {
            return await db.get('0');
        } catch (error) {
            console.warn('Could not get agent state:', error);
            return undefined;
        }
    }

    return {
        set,
        get,
    };
}