import Dexie from 'dexie';

const db = new Dexie('tank-rl');

db.version(1).stores({
    'agent-log': '&id',    // фиксированный первичный ключ (уникальный)
});

type AgentLog = object;

export function setAgentLog(state: Omit<AgentLog, 'id'>) {
    return db.table('agent-log').put({ id: 0, ...state });
}

export function getAgentLog(): Promise<undefined | AgentLog> {
    return db.table<AgentLog>('agent-log').get(0);
}
