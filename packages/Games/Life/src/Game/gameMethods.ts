import { getEngine } from './engine.js';

export const finalizeGameState = async () => {
    // const playerTeamEids = Array.from(mapSlotToEid$.value.values())
    //     .filter((eid): eid is EntityId => eid != null);

    // for (let i = 0; i < playerTeamEids.length; i++) {
    //     const vehicleEid = addTank(i, 1, getVehicleType(playerTeamEids[i]) as TankVehicleType);
    //     const agent = getLoadedAgent(vehicleEid, '/assets/models/v1');
    //     setPilotAgent(vehicleEid, agent);
    // }

    // // Sync all AI agents to load TensorFlow models
    // const agents = getRegistratedAgents();
    // await Promise.all(agents.map(agent => agent?.sync ? agent.sync() : Promise.resolve())); 
};

export function activateBots() {
    getEngine().pilots.toggle(true);
}

export function deactivateBots() {
    getEngine().pilots.toggle(false);
}
