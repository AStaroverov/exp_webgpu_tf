import { CSSProperties, useCallback, useEffect, useRef, useState } from 'react';
import { useLocalStorage } from 'react-use';
import { useObservable } from '../../../../../../lib/React/useSyncObservable.ts';
import { TestGameState$, incrementEnemyCount } from '../../State/GameState.ts';
import { startTestGame, spawnEnemy, exitTestGame, spawnPlayerVehicle } from '../../State/gameMethods.ts';
import { setRenderTarget } from '../../State/RenderTarget.ts';
import { Button } from '../../../Arena/UI/Components/Button.tsx';
import { Select, SelectItem } from '../../../Arena/UI/Components/Selector.tsx';
import { VehicleType } from '../../../Game/ECS/Components/Vehicle.ts';

const VEHICLE_OPTIONS: { key: VehicleType; label: string; emoji: string }[] = [
    { key: VehicleType.LightTank, label: 'Light Tank', emoji: '🏎️' },
    { key: VehicleType.MediumTank, label: 'Medium Tank', emoji: '🛡️' },
    { key: VehicleType.HeavyTank, label: 'Heavy Tank', emoji: '🦾' },
    { key: VehicleType.PlayerTank, label: 'Player Tank', emoji: '⚡' },
    { key: VehicleType.MeleeCar, label: 'Melee Car', emoji: '🚗' },
    { key: VehicleType.Harvester, label: 'Harvester', emoji: '🚜' },
];

export function BaseScreen({ className, style }: { className?: string, style?: CSSProperties }) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isSpawningEnemy, setIsSpawningEnemy] = useState(false);
    const [isSpawningPlayer, setIsSpawningPlayer] = useState(false);
    const [selectedPlayerType, setSelectedPlayerType] = useLocalStorage<VehicleType>('test-arena-player-type', VehicleType.PlayerTank);
    const [selectedEnemyType, setSelectedEnemyType] = useLocalStorage<VehicleType>('test-arena-enemy-type', VehicleType.MediumTank);
    const { isStarted, enemyCount } = useObservable(TestGameState$, TestGameState$.value);

    useEffect(() => {
        setRenderTarget(canvasRef.current);
        return () => setRenderTarget(null);
    }, [canvasRef.current]);

    useEffect(() => {
        if (!isStarted && canvasRef.current) {
            startTestGame();
        }
    }, [isStarted, canvasRef.current]);

    const handleStop = useCallback(() => {
        exitTestGame();
    }, []);

    const handleAddEnemy = useCallback(async () => {
        setIsSpawningEnemy(true);
        await spawnEnemy(selectedEnemyType ?? VehicleType.MediumTank);
        incrementEnemyCount();
        setIsSpawningEnemy(false);
    }, [selectedEnemyType]);

    const handleAddPlayer = useCallback(() => {
        setIsSpawningPlayer(true);
        spawnPlayerVehicle(selectedPlayerType ?? VehicleType.PlayerTank);
        setIsSpawningPlayer(false);
    }, [selectedPlayerType]);

    return (
        <div className={`${className} flex items-center justify-center relative`} style={style}>
            <canvas
                ref={canvasRef}
                className="absolute w-full h-full"
            />

            {/* Controls when game is running */}
            {isStarted && (
                <div className="absolute top-4 left-4 flex gap-6 items-start">
                    {/* Player Controls */}
                    <div className="flex gap-2 items-center bg-black/50 backdrop-blur-sm px-3 py-2 rounded-lg border border-emerald-500/30">
                        <span className="text-emerald-400 text-xs font-medium">Player:</span>
                        <Select
                            size="sm"
                            selectedKeys={[String(selectedPlayerType ?? VehicleType.PlayerTank)]}
                            onSelectionChange={(keys) => {
                                const key = Array.from(keys)[0];
                                if (key !== undefined) setSelectedPlayerType(Number(key) as VehicleType);
                            }}
                            className="w-40"
                            classNames={{
                                trigger: "bg-black/70 backdrop-blur-sm border-emerald-500/30 data-[hover=true]:bg-black/80 min-h-8 h-8",
                                value: "text-emerald-400 text-sm",
                                popoverContent: "bg-slate-900 border border-emerald-500/30",
                            }}
                            aria-label="Player vehicle type"
                        >
                            {VEHICLE_OPTIONS.map((opt) => (
                                <SelectItem key={String(opt.key)} textValue={opt.label}>
                                    <span className="text-emerald-400">{opt.emoji} {opt.label}</span>
                                </SelectItem>
                            ))}
                        </Select>
                        <Button 
                            color="success" 
                            size="sm" 
                            onPress={handleAddPlayer}
                            isLoading={isSpawningPlayer}
                        >
                            Spawn
                        </Button>
                    </div>

                    {/* Enemy Controls */}
                    <div className="flex gap-2 items-center bg-black/50 backdrop-blur-sm px-3 py-2 rounded-lg border border-amber-500/30">
                        <span className="text-amber-400 text-xs font-medium">Enemy:</span>
                        <div className="bg-black/70 px-2 py-1 rounded border border-amber-500/20">
                            <span className="text-amber-400 font-mono text-xs">{enemyCount}</span>
                        </div>
                        <Select
                            size="sm"
                            selectedKeys={[String(selectedEnemyType ?? VehicleType.MediumTank)]}
                            onSelectionChange={(keys) => {
                                const key = Array.from(keys)[0];
                                if (key !== undefined) setSelectedEnemyType(Number(key) as VehicleType);
                            }}
                            className="w-40"
                            classNames={{
                                trigger: "bg-black/70 backdrop-blur-sm border-amber-500/30 data-[hover=true]:bg-black/80 min-h-8 h-8",
                                value: "text-amber-400 text-sm",
                                popoverContent: "bg-slate-900 border border-amber-500/30",
                            }}
                            aria-label="Enemy vehicle type"
                        >
                            {VEHICLE_OPTIONS.map((opt) => (
                                <SelectItem key={String(opt.key)} textValue={opt.label}>
                                    <span className="text-amber-400">{opt.emoji} {opt.label}</span>
                                </SelectItem>
                            ))}
                        </Select>
                        <Button 
                            color="warning" 
                            size="sm" 
                            onPress={handleAddEnemy}
                            isLoading={isSpawningEnemy}
                        >
                            + Add
                        </Button>
                    </div>

                    {/* Exit Button */}
                    <Button color="danger" size="sm" onClick={handleStop}>
                        Exit
                    </Button>
                </div>
            )}
        </div>
    );
}

