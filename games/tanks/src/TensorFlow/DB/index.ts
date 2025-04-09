import { Batch } from '../Common/Memory.ts';
import { createAgentDB } from './createAgentDB.ts';
import { createMemoryBatchDB } from './createMemoryBatchDB.ts';


export type PolicyMemoryBatch = Omit<Batch, 'values' | 'returns'>;
export type ValueMemoryBatch = Pick<Batch, 'size' | 'states' | 'values' | 'returns'>;

export const policyMemory = createMemoryBatchDB<PolicyMemoryBatch>('policy-memory');
export const valueMemory = createMemoryBatchDB<ValueMemoryBatch>('value-memory');

export type PolicyAgentState = {
    version: number;
    klHistory: number[];
    learningRate: number;
};
export const policyAgentState = createAgentDB<PolicyAgentState>('policy-agent-state');

export type ValueAgentState = {
    version: number;
};
export const valueAgentState = createAgentDB<ValueAgentState>('value-agent-state');

