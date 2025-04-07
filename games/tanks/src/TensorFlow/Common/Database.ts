import PouchDB from 'pouchdb';
import { Batch } from './Memory.ts';

// Создание баз данных
const policyMemoryDB = new PouchDB('policy-memory');
const valueMemoryDB = new PouchDB('value-memory');
const policyAgentStateDB = new PouchDB('policy-agent-state');
const valueAgentStateDB = new PouchDB('value-agent-state');
const agentLogDB = new PouchDB('agent-log');

// Типы
export type PolicyMemoryBatch = Omit<Batch, 'values' | 'returns'>;
export type ValueMemoryBatch = Pick<Batch, 'size' | 'states' | 'values' | 'returns'>;

export const policyMemory = createMemoryInterface<PolicyMemoryBatch>(policyMemoryDB);
export const valueMemory = createMemoryInterface<ValueMemoryBatch>(valueMemoryDB);

export type PolicyAgentState = {
    version: number;
    klHistory: number[];
    learningRate: number;
};
export const policyAgentState = createAgentState<PolicyAgentState>(policyAgentStateDB);

export type ValueAgentState = {
    version: number;
};
export const valueAgentState = createAgentState<ValueAgentState>(valueAgentStateDB);

// Agent log
type AgentLog = object;

export async function setAgentLog(state: Omit<AgentLog, '_id'>) {
    return upsert(agentLogDB, '0', state);
}

export async function getAgentLog(): Promise<AgentLog | undefined> {
    try {
        return await agentLogDB.get('0');
    } catch (error) {
        console.warn('Could not get agent log:', error);
        return undefined;
    }
}

// Утилиты
function createMemoryInterface<Batch>(db: PouchDB.Database) {
    async function addMemoryBatch(batch: { version: number; memories: Batch }) {
        const id = Date.now().toString();
        return db.put({ _id: id, ...batch });
    }

    async function getMemoryBatchCount() {
        const result = await db.allDocs();
        return result.total_rows;
    }

    async function getMemoryBatchList() {
        const result = await db.allDocs({ include_docs: true });
        return result.rows.map(r => r.doc as unknown as { _id: string; version: number; memories: Batch });
    }

    async function clearMemoryBatchList() {
        const docs = await db.allDocs({ include_docs: true });
        const deletions = docs.rows.map(row => ({
            ...row.doc,
            _deleted: true,
        }));
        return db.bulkDocs(deletions);
    }

    async function extractMemoryBatchList() {
        const list = await getMemoryBatchList();
        await clearMemoryBatchList();
        return list;
    }

    return {
        getMemoryBatchCount,
        addMemoryBatch,
        getMemoryBatchList,
        clearMemoryBatchList,
        extractMemoryBatchList,
    };
}

function createAgentState<State>(db: PouchDB.Database) {
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

async function upsert<T extends {}>(db: PouchDB.Database<T>, id: string, data: Partial<T>) {
    try {
        const existing = await db.get(id);
        return db.put({ ...(existing as any), ...data, _id: id, _rev: existing._rev });
    } catch (err: any) {
        if (err.status === 404) {
            return db.put({ ...(data as any), _id: id });
        } else {
            throw err;
        }
    }
}