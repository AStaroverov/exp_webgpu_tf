import { GameTankState } from '../Hooks/useGameAPI.ts';

export function TankPreview({ className, tank }: { className: string, tank: GameTankState }) {
    return (
        <div className={ className }>
            { tank.id } / { tank.health }
        </div>
    );
}
