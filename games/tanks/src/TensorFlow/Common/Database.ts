import Dexie, { Transaction } from 'dexie';
import { Batch } from './Memory.ts';

const db = new Dexie('tank-rl');

db.version(1).stores({
    'policy-memory': '++id',      // автоинкрементное поле id
    'policy-agent-state': '&id',    // фиксированный первичный ключ (уникальный)

    'value-memory': '++id',      // автоинкрементное поле id
    'value-agent-state': '&id',    // фиксированный первичный ключ (уникальный)

    'agent-log': '&id',    // фиксированный первичный ключ (уникальный)
});

export type PolicyMemoryBatch = Omit<Batch, 'values' | 'returns'>;
export const policyMemory = createMemoryInterface<PolicyMemoryBatch>('policy-memory');
export type ValueMemoryBatch = Pick<Batch, 'size' | 'states' | 'values' | 'returns'>;
export const valueMemory = createMemoryInterface<ValueMemoryBatch>('value-memory');

export type PolicyAgentState = {
    version: number;
    klHistory: number[];
    learningRate: number;
}
export const policyAgentState = createAgentState<PolicyAgentState>('policy-agent-state');

export type ValueAgentState = {
    version: number;
}
export const valueAgentState = createAgentState<ValueAgentState>('value-agent-state');

type AgentLog = object;

export function setAgentLog(state: Omit<AgentLog, 'id'>) {
    return db.table('agent-log').put({ id: 0, ...state });
}

export function getAgentLog(): Promise<undefined | AgentLog> {
    return db.table<AgentLog>('agent-log').get(0);
}

function createMemoryInterface<Batch>(name: string) {
    function addMemoryBatch(batch: { version: number, memories: Batch }) {
        return db.table(name).add(batch);
    }

    function getMemoryBatchCount() {
        return db.table(name).count();
    }

    function getMemoryBatchList(tx: Transaction | typeof db = db) {
        return (tx as Transaction).table<({ id: number, version: number, memories: Batch })>(name).toArray();
    }

    function clearMemoryBatchList(tx: Transaction | typeof db = db) {
        return (tx as Transaction).table(name).clear();
    }

    function extractMemoryBatchList() {
        return db.transaction('rw', db.table(name), async (tx) => {
            const list = await getMemoryBatchList(tx);
            await clearMemoryBatchList(tx);
            return list;
        });
    }


    return {
        getMemoryBatchCount,
        addMemoryBatch,
        getMemoryBatchList,
        clearMemoryBatchList,
        extractMemoryBatchList,
    };
}

function createAgentState<State>(name: string) {
    function set(state: Omit<State, 'id'>) {
        return db.table(name).put({ id: 0, ...state });
    }

    function get(): Promise<undefined | State> {
        return db.table<State>(name).get(0);
    }

    return {
        set,
        get,
    };
}