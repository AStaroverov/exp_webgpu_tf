import Dexie, { Transaction } from 'dexie';
import { Batch } from '../Common/Memory.ts';

const db = new Dexie('tank-rl');

db.version(1).stores({
    memory: '++id',      // автоинкрементное поле id
    'agent-state': '&id',    // фиксированный первичный ключ (уникальный)
    'agent-log': '&id',    // фиксированный первичный ключ (уникальный)
});

export function addMemoryBatch(batch: Batch) {
    return db.table('memory').add(batch);
}

export function getMemoryBatchCount() {
    return db.table('memory').count();
}

export function getMemoryBatchList(tx: Transaction | typeof db = db) {
    return (tx as Transaction).table<(Batch & { id: number })>('memory').toArray();
}

export function clearMemoryBatchList(tx: Transaction | typeof db = db) {
    return (tx as Transaction).table('memory').clear();
}

export function extractMemoryBatchList() {
    return db.transaction('rw', db.table('memory'), async (tx) => {
        const list = await getMemoryBatchList(tx);
        await clearMemoryBatchList(tx);
        return list;
    });
}

type AgentState = {
    version: number;
}

export function setAgentState(state: Omit<AgentState, 'id'>) {
    return db.table('agent-state').put({ id: 0, ...state });
}

export function getAgentState(): Promise<undefined | AgentState> {
    return db.table<AgentState>('agent-state').get(0);
}

type AgentLog = {
    logger: object,
}

export function setAgentLog(state: Omit<AgentLog, 'id'>) {
    return db.table('agent-log').put({ id: 0, ...state });
}

export function getAgentLog(): Promise<undefined | AgentLog> {
    return db.table<AgentLog>('agent-log').get(0);
}