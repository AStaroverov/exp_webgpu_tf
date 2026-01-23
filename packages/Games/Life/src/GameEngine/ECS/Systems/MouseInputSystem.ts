import { addEntity, removeEntity } from 'bitecs';
import { GameDI } from '../../DI/GameDI.js';
import { RenderDI } from '../../DI/RenderDI.js';
import { createSpawner } from '../Entities/Spawner.js';
import { CELL_SIZE, CELL_PADDING, worldToGrid, gridToWorld } from '../Entities/Cell.js';
import { GRID_SIZE } from '../Components/Cell.js';
import { cameraPosition } from 'renderer/src/ECS/Systems/ResizeSystem.js';
import { addTransformComponents, LocalTransform, setMatrixTranslate } from 'renderer/src/ECS/Components/Transform.js';
import { Color, Roundness } from 'renderer/src/ECS/Components/Common.js';
import { Shape, ShapeKind } from 'renderer/src/ECS/Components/Shape.js';

// Hover highlight color (semi-transparent white)
const HOVER_COLOR: [number, number, number, number] = [1, 1, 1, 0.3];

export function createMouseInputSystem({ world } = GameDI) {
    let isMouseDown = false;
    let lastClickGridX = -1;
    let lastClickGridY = -1;
    
    // Hover highlight entity
    let hoverEntityId: number | null = null;
    let hoverGridX = -1;
    let hoverGridY = -1;

    const screenToGrid = (e: MouseEvent): [number, number] | null => {
        const canvas = RenderDI.canvas;
        if (!canvas) return null;

        const rect = canvas.getBoundingClientRect();
        
        const normalizedX = (e.clientX - rect.left) / rect.width;
        const normalizedY = (e.clientY - rect.top) / rect.height;
        
        const worldWidth = rect.width;
        const worldHeight = rect.height;
        
        const worldX = cameraPosition.x - worldWidth / 2 + normalizedX * worldWidth;
        const worldY = cameraPosition.y - worldHeight / 2 + normalizedY * worldHeight;
        
        const [gridX, gridY] = worldToGrid(worldX, worldY);
        
        if (gridX >= 0 && gridX < GRID_SIZE && gridY >= 0 && gridY < GRID_SIZE) {
            return [gridX, gridY];
        }
        return null;
    };

    const createHoverEntity = () => {
        const eid = addEntity(world);
        addTransformComponents(world, eid);
        
        const size = CELL_SIZE - CELL_PADDING;
        Shape.addComponent(world, eid, ShapeKind.Rectangle, size, size);
        Color.addComponent(world, eid, HOVER_COLOR[0], HOVER_COLOR[1], HOVER_COLOR[2], HOVER_COLOR[3]);
        Roundness.addComponent(world, eid, 2);
        
        return eid;
    };

    const updateHoverPosition = (gridX: number, gridY: number) => {
        if (!hoverEntityId) {
            hoverEntityId = createHoverEntity();
        }
        
        const [worldX, worldY] = gridToWorld(gridX, gridY);
        setMatrixTranslate(LocalTransform.matrix.getBatch(hoverEntityId), worldX, worldY, 2); // Above spawners
        hoverGridX = gridX;
        hoverGridY = gridY;
    };

    const hideHover = () => {
        if (hoverEntityId) {
            // Move hover off-screen instead of removing (to avoid entity churn)
            setMatrixTranslate(LocalTransform.matrix.getBatch(hoverEntityId), -10000, -10000, 0);
        }
        hoverGridX = -1;
        hoverGridY = -1;
    };
    
    const handleClick = (e: MouseEvent) => {
        const gridPos = screenToGrid(e);
        if (!gridPos) return;
        
        const [gridX, gridY] = gridPos;
        
        // Only create spawner if position changed (avoid duplicates on same cell)
        if (gridX !== lastClickGridX || gridY !== lastClickGridY) {
            lastClickGridX = gridX;
            lastClickGridY = gridY;
            createSpawner(world, gridX, gridY, 500);
        }
    };

    const handleHover = (e: MouseEvent) => {
        const gridPos = screenToGrid(e);
        if (!gridPos) {
            hideHover();
            return;
        }
        
        const [gridX, gridY] = gridPos;
        
        if (gridX !== hoverGridX || gridY !== hoverGridY) {
            updateHoverPosition(gridX, gridY);
        }
    };
    
    const onMouseDown = (e: MouseEvent) => {
        isMouseDown = true;
        lastClickGridX = -1;
        lastClickGridY = -1;
        handleClick(e);
    };
    
    const onMouseMove = (e: MouseEvent) => {
        handleHover(e);
        if (isMouseDown) {
            handleClick(e);
        }
    };
    
    const onMouseUp = () => {
        isMouseDown = false;
        lastClickGridX = -1;
        lastClickGridY = -1;
    };

    const onMouseLeave = () => {
        hideHover();
    };

    // Setup event listeners when canvas is available
    let cleanup: VoidFunction | null = null;
    
    const setup = () => {
        const canvas = RenderDI.canvas;
        if (!canvas || cleanup) return;
        
        canvas.addEventListener('mousedown', onMouseDown);
        canvas.addEventListener('mousemove', onMouseMove);
        canvas.addEventListener('mouseleave', onMouseLeave);
        window.addEventListener('mouseup', onMouseUp);
        
        cleanup = () => {
            canvas.removeEventListener('mousedown', onMouseDown);
            canvas.removeEventListener('mousemove', onMouseMove);
            canvas.removeEventListener('mouseleave', onMouseLeave);
            window.removeEventListener('mouseup', onMouseUp);
        };
    };
    
    // System tick - just ensures event listeners are set up
    const tick = (_delta: number) => {
        if (RenderDI.canvas && !cleanup) {
            setup();
        }
    };
    
    const dispose = () => {
        cleanup?.();
        cleanup = null;
        if (hoverEntityId) {
            removeEntity(world, hoverEntityId);
            hoverEntityId = null;
        }
    };

    return { tick, dispose };
}

