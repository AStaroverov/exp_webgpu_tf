import { frameInterval } from '../../../../../../lib/Rx/frameInterval.ts';
import { distinctUntilChanged, map } from 'rxjs';
import { destroyTank, getTankEngineLabel, getTankHealth } from '../../../Game/ECS/Entities/Tank/TankUtils.ts';
import { addTank, getTankEids, mapTankIdToAgent } from './engineMethods.ts';
import { dedobs, DEDOBS_REMOVE_DELAY, DEDOBS_RESET_DELAY } from '../../../../../../lib/Rx/dedobs.ts';
import { EntityId } from 'bitecs';
import { frameTasks } from '../../../../../../lib/TasksScheduler/frameTasks.ts';
import { getEngine } from './engine.ts';
import { initTensorFlow } from '../../../TensorFlow/Common/initTensorFlow.ts';
import { CurrentActorAgent } from '../../../TensorFlow/Common/Curriculum/Agents/CurrentActorAgent.ts';
import { hashArray } from '../../../../../../lib/hashArray.ts';

export const tankEids$ = frameInterval(10).pipe(
    map(getTankEids),
    distinctUntilChanged((a, b) => {
        if (a.length !== b.length) return false;
        return hashArray(a) === hashArray(b);
    }),
);

export const getTankState$ = dedobs(
    (id: number) => {
        return frameInterval(10).pipe(
            map(() => {
                return {
                    id,
                    health: (getTankHealth(id) * 100).toFixed(0),
                    engine: getTankEngineLabel(id),
                };
            }),
            distinctUntilChanged((a, b) => a.health === b.health && a.id === b.id),
        );
    },
    {
        removeDelay: DEDOBS_REMOVE_DELAY,
        resetDelay: DEDOBS_RESET_DELAY,
    },
);

export const finalizeGameState = async () => {
    const playerTeam = getTankEids();

    for (let i = 0; i < playerTeam.length; i++) {
        addTank(i, 1);
    }

    const allTanks = getTankEids();

    await initTensorFlow('wasm');
    await Promise.all(allTanks.map((tankId) => {
        const agent = new CurrentActorAgent(tankId, false);
        mapTankIdToAgent.set(tankId, agent);
        return agent.sync();
    }));
};

let stopAgents: undefined | VoidFunction = undefined;
export const activateBots = () => {
    let tankEids: EntityId[] = [];
    let frameIndex = 0;
    stopAgents = frameTasks.addInterval(() => {
        frameIndex++;
        if (tankEids.length === 0) {
            if (frameIndex < 10) return;
            tankEids = getTankEids().slice();
            frameIndex = 0;
        }

        const currentTankEids = getTankEids();
        let eid: EntityId | undefined = undefined;

        while (eid === undefined && tankEids.length > 0) {
            eid = tankEids.pop();
            if (eid === undefined) break;
            if (!currentTankEids.includes(eid)) eid = undefined;
        }

        eid && mapTankIdToAgent.get(eid)?.updateTankBehaviour(
            getEngine().width,
            getEngine().height,
        );
    }, 1);

    return stopAgents;
};
export const destroyBots = () => {
    const currentTankEids = getTankEids();
    stopAgents?.();

    for (const agent of mapTankIdToAgent.values()) {
        agent.dispose?.();
    }
    for (const tankEid of mapTankIdToAgent.keys()) {
        if (currentTankEids.includes(tankEid)) {
            destroyTank(tankEid);
        }
    }

    mapTankIdToAgent.clear();
    stopAgents = undefined;
};
