import Dexie from 'dexie';
import { Batch } from '../Common/Memory.ts';

const db = new Dexie('tank-rl');

db.version(1).stores({
    memory: '++id',      // автоинкрементное поле id
    'agent-state': '&id',    // фиксированный первичный ключ (уникальный)
});

export function addMemoryBatch(batch: Batch) {
    return db.table('memory').add(batch);
}

export function getMemoryBatchCount() {
    return db.table('memory').count();
}

export function getMemoryBatchList() {
    return db.table<Batch>('memory').toArray();
}

export function clearMemoryBatchList() {
    return db.table('memory').clear();
}

type AgentState = {
    version: number;
    logger: object,
}

export function setAgentState(state: Omit<AgentState, 'id'>) {
    return db.table('agent-state').put({ id: 'state', ...state });
}

export function getAgentState(): Promise<undefined | AgentState> {
    return db.table<AgentState>('agent-state').get('state');
}