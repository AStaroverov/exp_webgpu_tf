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
        <div className={ `${ className } gap-2` }>
            { !tankEid && <Button onClick={ addTank }>Add tank</Button> }
            { tankEid && <>
                <div className="w-full gap-2">
                    <TankSelector slot={ slot }/>
                    <PilotSelector tankEid={ tankEid }/>
                </div>
                { tankEid && <TankPreview tankEid={ tankEid }/> }
            </> }
        </div>
    );
}
