import { TankSelector } from './TankSelector.tsx';
import { TankPreview } from './TankPreview.tsx';
import { Button } from '@heroui/react';
import { getSlot$, openSlot, slotIsOpen$ } from '../../State/Game/modules/lobbySlots.ts';
import { useCallback } from 'react';
import { useObservable } from '../../../../../../lib/React/useSyncObservable.ts';
import { PilotSelector } from './PilotSelector.tsx';

export function TankSlot({ className, slot }: { className: string, slot: number }) {
    const isOpen = useObservable(slotIsOpen$(slot));
    const vehicleEid = useObservable(getSlot$(slot));
    
    const handleAddTank = useCallback(() => openSlot(slot), [slot]);

    return (
        <div className={ `${ className } flex flex-col gap-2` }>
            { !isOpen && <Button onPress={ handleAddTank }>Add tank</Button> }
            { isOpen && <TankSelector slot={ slot }/> }
            { vehicleEid && <TankPreview tankEid={ vehicleEid }/> }
            { vehicleEid && <PilotSelector tankEid={ vehicleEid } /> }
        </div>
    );
}
