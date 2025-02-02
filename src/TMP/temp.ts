import {Shader} from 'pixi.js';

import {hasBrokenVertexID} from '../../utils/detect';
import {buildShader, funcGLSL, glsl} from '../utils/shaders/buildShader';
import {
    computeRectVertexPosition,
    computeRoundJoinsVertexPosition,
    computeTrapezeVertexPosition,
    isValueNaN,
    prepareSecondPoint,
} from '../utils/shaders/computePositions';
import {MAX_CUMULATIVE_SUM, MAX_UNIFORM_BUFFER_LENGTH} from './def';

export const fragmentShader = buildShader({
    uniforms: {
        uColor: 'vec4',
    },
    // language=GLSL
    body: glsl`
        out vec4 fragColor;

        void main() {
            fragColor = uColor;
        }
    `,
});

const getPartIndex = funcGLSL(() => (name) => {
    // language=GLSL

    return glsl`
        int ${name}(float absVisIndex) {
            int partIndex = 0;
            while (absVisIndex >= uPartsSizeCumulativeSum[partIndex]) {
                partIndex += 1;
            }

            return partIndex;
        }
    `;
});

const isLastPoint = funcGLSL(() => (name) => {
    // language=GLSL
    return glsl`
        bool ${name}(int partIndex, float absVisIndex) {
            return uPartsSizeCumulativeSum[partIndex] - absVisIndex == 1.;
        }
    `;
});

const isRightestPart = funcGLSL(() => (name) => {
    // language=GLSL
    return glsl`
        bool ${name}(int partIndex) {
            return uPartsSizeCumulativeSum[partIndex] == float(${MAX_CUMULATIVE_SUM});
        }
    `;
});

const isLastPart = funcGLSL(() => (name) => {
    // language=GLSL
    return glsl`
        bool ${name}(int partIndex) {
            return uPartsSizeCumulativeSum[partIndex + 1] == float(${MAX_CUMULATIVE_SUM});
        }
    `;
});

const getPartPointIndex = funcGLSL(() => (name) => {
    // language=GLSL
    return glsl`
        float ${name}(int partIndex, float absVisIndex) {
            return absVisIndex - (partIndex == 0 ? 0. : uPartsSizeCumulativeSum[partIndex - 1]);
        }
    `;
});

const getPoint = funcGLSL(() => (name) => {
    // language=GLSL
    return glsl`
        // For stairs each second visual point consists from 2 real points (left and right)
        vec2 ${name}(float absVisIndex) {
            bool isStairs = uType == 2./*Stairs*/;
            int partIndex = ${getPartIndex()}(absVisIndex);
            float partPointIndex = ${getPartPointIndex()}(partIndex, absVisIndex);

            bool isFirstPoint = partPointIndex == 0.;
            bool isLeftestPart = partIndex == 0;

            if (isFirstPoint && isLeftestPart) {
                return uLeftestPoint;
            }

            bool isRightestPart = ${isRightestPart()}(partIndex);

            if (isFirstPoint && isRightestPart) {
                return uRightestPoint;
            }

            float texColOffsetY = 0.;

            if (isStairs) {
                texColOffsetY = -mod(partPointIndex, 2.);
                partPointIndex = ceil(partPointIndex / 2.);
            }

            bool isLastPoint = ${isLastPoint()}(partIndex, absVisIndex);
            vec2 offset = uPartsOffset[partIndex];
            float storeY = uPartsStoreY[partIndex];

            float x;
            float y;

            if (isStairs && isLastPoint && isLeftestPart) {
                y = uLeftestPoint.y;
            } else {
                float texColY = floor((partPointIndex + texColOffsetY) / 2.);
                int channelY = (int(partPointIndex + texColOffsetY) % 2) * 2 + 1;

                float yRelative = texelFetch(uPartsStore, ivec2(texColY, storeY), 0)[channelY];

                y = ${isValueNaN()}(yRelative) ? yRelative : offset.y + yRelative;
            }

            if (isStairs && isLastPoint) {
                bool isLastPart = ${isLastPart()}(partIndex);

                if (isLastPart) {
                    x = uRightestPoint.x;
                } else {
                    int rightPointPartIndex = ${getPartIndex()}(absVisIndex + 1.);
                    vec2 rightPointOffset = uPartsOffset[rightPointPartIndex];
                    float rightPointStoreY = uPartsStoreY[rightPointPartIndex];

                    x = rightPointOffset.x + texelFetch(uPartsStore, ivec2(0, rightPointStoreY), 0)[0];
                }
            } else {
                float texColX = floor(partPointIndex / 2.);
                int channelX = (int(partPointIndex) % 2) * 2;
                x = offset.x + texelFetch(uPartsStore, ivec2(texColX, storeY), 0)[channelX];
            }

            return vec2(x, y);
        }
    `;
});

const getCanvasPoint = funcGLSL(() => (name) => {
    // language=GLSL
    return glsl`
        vec2 ${name}(mat3 mat, vec2 relativeOffset) {
            return (mat * vec3(relativeOffset.x, -relativeOffset.y, 1.)).xy;
        }
    `;
});

const renderPoints =
    // language=GLSL
    glsl`
        vec2 rawPoint = ${getPoint()}(absVertexIndex);
        vec2 point = ${getCanvasPoint()}(uFinalMatrix, rawPoint);

        gl_Position = vec4(point, 0.0, 1.0);
        gl_PointSize = uWidth;
    `;

const renderArea =
    // language=GLSL
    glsl`
        // Be careful when updating following line, on some video cards mod function is buggy when using two variables, use % operation
        // https://stackoverflow.com/questions/16701342/glsl330-modulo-returns-unexpected-value
        int vertexIndex = int(absVertexIndex) % uInstanceSize;
        // Be careful when updating following line, on some video cards division is buggy
        float pointIndex = float(int(absVertexIndex) / uInstanceSize);

        vec2 pointA = ${getPoint()}(pointIndex);
        vec2 pointB = ${getPoint()}(pointIndex + 1.);

        vec2[2] absPoints = vec2[2](
        ${getCanvasPoint()}(uFinalMatrix, pointA),
        ${getCanvasPoint()}(uFinalMatrix, pointB)
        );

        vec2 point = ${computeTrapezeVertexPosition()}(
        absPoints,
        vertexIndex
        );

        gl_Position = vec4(point, 0., 1.);
    `;

const renderLines =
    // language=GLSL
    glsl`
        vec2 point;

        // Be careful when updating following line, on some video cards mod function is buggy when using two variables
        // https://stackoverflow.com/questions/16701342/glsl330-modulo-returns-unexpected-value
        int vertexIndex = int(absVertexIndex) % uInstanceSize;
        // Be careful when updating following line, on some video cards division is buggy
        float pointIndex = float(int(absVertexIndex) / uInstanceSize);

        vec2 pointA = ${getPoint()}(pointIndex);
        vec2 pointB = ${prepareSecondPoint()}(
        pointA,
        ${getPoint()}(pointIndex + 1.),
        uMinWidthTime
        );
        vec2[2] absPoints = vec2[2](
        ${getCanvasPoint()}(uFinalMatrix, pointA),
        ${getCanvasPoint()}(uFinalMatrix, pointB)
        );

        // compute vertex for lines
        if (vertexIndex < 6) {
            point = ${computeRectVertexPosition()}(
            absPoints,
            vertexIndex,
            uWorldWidth
            );
        } else { // compute vertex for joins
            point = ${computeRoundJoinsVertexPosition()}(
            absPoints[1],
            vertexIndex - 6,
            uWorldWidth
            );
        }

        gl_Position = vec4(point, 0.0, 1.0);
    `;

const renderDots =
    // language=GLSL
    glsl`
        vec2 point;

        // Be careful when updating following line, on some video cards mod function is buggy when using two variables
        // https://stackoverflow.com/questions/16701342/glsl330-modulo-returns-unexpected-value
        int vertexIndex = int(absVertexIndex) % uInstanceSize;
        // Be careful when updating following line, on some video cards division is buggy
        float pointIndex = float(int(absVertexIndex) / uInstanceSize);

        vec2 chartPoint = ${getPoint()}(pointIndex);

        vec2 absPoint = ${getCanvasPoint()}(uFinalMatrix, chartPoint);

        point = ${computeRoundJoinsVertexPosition()}(
        absPoint,
        vertexIndex,
        uWorldWidth / 2.
        );

        gl_Position = vec4(point, 0.0, 1.0);
    `;

export const vertexShader = buildShader({
    uniforms: {
        uInstanceSize: 'int',

        uType: 'float', // 0 - Points, 1 - Lines, 2 - Stairs, 3 - Area, 4 - Dots
        uWidth: 'float',
        uMinWidthTime: 'float',
        uWorldWidth: 'vec2',
        uRightestPoint: 'vec2',
        uLeftestPoint: 'vec2',
        uPartsStore: 'sampler2D',
        uPartsStoreY: `float[${MAX_UNIFORM_BUFFER_LENGTH}]`,
        uPartsOffset: `vec2[${MAX_UNIFORM_BUFFER_LENGTH / 2}]`,
        uPartsSizeCumulativeSum: `float[${MAX_UNIFORM_BUFFER_LENGTH}]`,
        uFinalMatrix: 'mat3',
    },
    attributes: {
        aVertexID: 'float',
    },
    // language=GLSL
    body: glsl`
        void main() {
            float absVertexIndex = ${hasBrokenVertexID ? 'aVertexID' : ' float(gl_VertexID)'};

            if (uType == 3./*Area*/) {
                ${renderArea}
            } else if (uType == 2./*Stairs*/ || uType == 1./*Lines*/) {
                ${renderLines}
            } else if (uType == 4./*Dots*/) {
                ${renderDots}
            } else {
                // Default render
                ${renderPoints}
            }
        }
    `,
});

export function getChartPartsShader() {
    return Shader.from(vertexShader.shader, fragmentShader.shader);
}
