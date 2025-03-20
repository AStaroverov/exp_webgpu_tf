import Dexie from 'dexie';
import { GradientsData } from './Slave/SlaveAgent.ts';
import { RLExperimentConfig } from './config.ts';

const db = new Dexie('tank-rl');

db.version(1).stores({
    gradients: '++id',      // автоинкрементное поле id
    'agent-state': '&id',    // фиксированный первичный ключ (уникальный)
});

export function addGradients(gradients: GradientsData) {
    return db.table('gradients').add(gradients);
}

export function getGradientsList() {
    return db.table<GradientsData>('gradients').toArray();
}

export function clearGradientsList() {
    return db.table('gradients').clear();
}

type AgentState = {
    config: RLExperimentConfig;
    logger: object,
}

export function setAgentState(state: Omit<AgentState, 'id'>) {
    return db.table('agent-state').put({ id: 'state', ...state });
}

export function getAgentState(): Promise<undefined | AgentState> {
    return db.table<AgentState>('agent-state').get('state');
}