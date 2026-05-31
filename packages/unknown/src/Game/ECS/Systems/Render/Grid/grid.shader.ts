import { ShaderMeta } from 'renderer/src/WGSL/ShaderMeta.ts';
import { VariableKind, VariableMeta } from 'renderer/src/Struct/VariableMeta.ts';
import { wgsl } from 'renderer/src/WGSL/wgsl.ts';

/** Max number of hex cells drawn per frame (instances). */
export const MAX_HEX_COUNT = 2048;

const WGSL_VERTEX = /* wgsl */`
    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) local_position: vec2f,
    };

    @vertex
    fn vs_main(
        @builtin(vertex_index) vertex_index: u32,
        @builtin(instance_index) instance_index: u32,
    ) -> VertexOutput {
        let center = uCells[instance_index].xy;
        let half = uParams.x; // circumradius — quad half-size that bounds the hex

        // Two-triangle quad centered on the hex.
        let local_pos = vec2f(
            select(-half, half, vertex_index > 0u && vertex_index < 4u),
            select(-half, half, vertex_index > 1u && vertex_index < 5u)
        );

        let world_pos = center + local_pos;
        let projected = (uProjection * vec4f(world_pos, 0.0, 1.0)).xy;
        // z = 0.0 matches the SDF shape pass; layering is by draw order.
        let position = vec4f(projected.x, -projected.y, 0.0, 1.0);

        return VertexOutput(position, local_pos);
    }
`;

const WGSL_FRAGMENT = /* wgsl */`
    const DEG60: f32 = 1.0471975512; // 60 degrees in radians

    // Signed distance to a POINTY-top regular hexagon, via 6 edge half-planes.
    // Edge normals point at 0,60,...,300 degrees; pointy hexes have vertical
    // left/right edges, so normal at 0deg is correct.
    fn sd_hex_pointy(p: vec2f, inradius: f32) -> f32 {
        var d = -1e9;
        for (var i = 0u; i < 6u; i++) {
            let a = f32(i) * DEG60;
            let n = vec2f(cos(a), sin(a));
            d = max(d, dot(p, n));
        }
        return d - inradius;
    }

    @fragment
    fn fs_main(
        @location(0) local_position: vec2f,
    ) -> @location(0) vec4f {
        let inradius = uParams.y;
        let lineWidth = uParams.z;
        let fillAlpha = uParams.w;

        let d = sd_hex_pointy(local_position, inradius);
        if (d > 0.0) {
            discard; // outside the hexagon
        }

        // Strong line near the border, faint fill inside.
        let edge = 1.0 - smoothstep(lineWidth * 0.5, lineWidth, abs(d));
        let alpha = max(fillAlpha, edge) * uColor.a;
        if (alpha < 0.01) {
            discard;
        }
        return vec4f(uColor.rgb, alpha);
    }
`;

export const shaderMeta = new ShaderMeta(
    {
        projection: new VariableMeta('uProjection', VariableKind.Uniform, `mat4x4<f32>`),
        params: new VariableMeta('uParams', VariableKind.Uniform, `vec4<f32>`),
        color: new VariableMeta('uColor', VariableKind.Uniform, `vec4<f32>`),
        cells: new VariableMeta('uCells', VariableKind.StorageRead, `array<vec4<f32>, ${MAX_HEX_COUNT}>`),
    },
    {},
    wgsl`
        ${WGSL_VERTEX}
        ${WGSL_FRAGMENT}
    `,
);
