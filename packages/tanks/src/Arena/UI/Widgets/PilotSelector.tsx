import { ChangeEvent, useCallback } from 'react';
import { Select, SelectItem } from '../Components/Selector.tsx';
import { getLoadedAgent, LoadedAgent } from '../../../Plugins/Pilots/Agents/LoadedAgent.ts';
import { changeTankPilot, getPilotAgent$ } from '../../State/Game/gameMethods.ts';
import { useObservable } from 'react-use';
import { CurrentActorAgent, TankAgent } from '../../../Plugins/Pilots/Agents/CurrentActorAgent.ts';

const pilots = [
    { key: 'current', label: 'Pilot last' },
    { key: '/assets/models/v1', label: 'Pilot v1' },
    { key: '/assets/models/v2', label: 'Pilot v2' },
];

export function PilotSelector({ className, tankEid }: { className?: string, tankEid: number }) {
    const pilot = useObservable(getPilotAgent$(tankEid));
    const handleChangePilot = useCallback((event: ChangeEvent<{ value: string }>) => {
        if (event.target.value === 'current') {
            changeTankPilot(tankEid, new CurrentActorAgent(tankEid, false));
        }
        if (event.target.value.includes('/assets/models/')) {
            const agent = getLoadedAgent(tankEid, event.target.value);
            changeTankPilot(tankEid, agent);
        }
    }, [tankEid]);
    
    return (
        <div className={ `${ className } gap-2` }>
            <Select
                className="max-w-xs"
                label="Pilot"
                value={ getPilotKey(pilot) }
                onChange={ handleChangePilot }
            >
                { pilots.map((item) => <SelectItem key={ item.key }>{ item.label }</SelectItem>) }
            </Select>
        </div>
    );
}


function getPilotKey(agent: undefined | TankAgent | LoadedAgent) {
    if (agent instanceof LoadedAgent) {
        return agent.path;
    }
    if (agent instanceof CurrentActorAgent) {
        return 'current';
    }
    return 'player';
}