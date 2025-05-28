import { TankSelector } from './TankSelector.tsx';
import { PilotSelector } from './PilotSelector.tsx';
import { TankPreview } from './TankPreview.tsx';
import { Button } from '@heroui/react';
import { addTankToSlot, getTankEidBySlot$ } from '../State/Game/playerMethods.ts';
import { useCallback } from 'react';
import { TankType } from '../../Game/ECS/Components/Tank.ts';
import { useObservable } from '../../../../../lib/React/useSyncObservable.ts';

export function TankSlot({ className, slot }: { className: string, slot: number }) {
    const addTank = useCallback(() => addTankToSlot(TankType.Medium, slot), [slot]);
    const tankEid = useObservable(getTankEidBySlot$(slot));

    return (
        <div className={ `${ className } flex flex-col gap-2` }>
            { !tankEid && <Button onClick={ addTank }>Add tank</Button> }
            { tankEid && <>
                <div className="w-full grid grid-cols-2 gap-2">
                    <TankSelector className="" slot={ slot }/>
                    <PilotSelector className="" tankEid={ tankEid }/>
                </div>
                { tankEid && <TankPreview tankEid={ tankEid }/> }
            </> }
        </div>
    );
}
