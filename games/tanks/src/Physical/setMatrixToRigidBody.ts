import { RigidBody } from '@dimforge/rapier2d';
import { mat4 } from 'gl-matrix';

const position = { x: 0, y: 0 };

export function setMatrixToRigidBody(body: RigidBody, matrix: mat4): void {
    // Apply transformation to the rigid body
    position.x = matrix[12];
    position.y = matrix[13];
    body.setTranslation(position, true);
    body.setRotation(Math.atan2(matrix[4], matrix[0]), true);
}