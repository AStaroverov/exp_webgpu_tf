import { ShaderMeta } from '../../../../../../../../../src/WGSL/ShaderMeta.ts';
import { wgsl } from '../../../../../../../../../src/WGSL/wgsl.ts';
import { VariableKind, VariableMeta } from '../../../../../../../../../src/Struct/VariableMeta.ts';

export const shaderMeta = new ShaderMeta(
    {
        inputSampler: new VariableMeta('textureSampler', VariableKind.Sampler, `sampler`),
        inputTexture: new VariableMeta('inputTexture', VariableKind.Texture, `texture_2d<f32>`),
    },
    {},
    // language=WGSL
    wgsl`
const POSITION = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0),
    vec2f(1.0, 1.0),
    vec2f(-1.0, 1.0)
  );

  // Текстурные координаты для каждой вершины
const TEX_COORDS = array<vec2f, 6>(
    vec2f(0.0, 1.0),
    vec2f(1.0, 1.0),
    vec2f(1.0, 0.0),
    vec2f(0.0, 1.0),
    vec2f(1.0, 0.0),
    vec2f(0.0, 0.0)
  );


// Вершинный шейдер для полноэкранного квада
@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {var output: VertexOutput;
  output.position = vec4f(POSITION[vertexIndex], 0.0, 1.0);
  output.texCoord = TEX_COORDS[vertexIndex];
  return output;
}

// Структура для передачи данных из вершинного в фрагментный шейдер
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) texCoord: vec2f,
};

const PIXEL_SIZE: f32 = 3.0;          // Небольшие пиксели
const COLOR_DEPTH: f32 = 12.0;        // Ограниченная палитра
const EDGE_THRESHOLD: f32 = 0.2;
const CONTRAST_AMOUNT: f32 = 0.1;    // Высокий контраст
const VIGNETTE_STRENGTH: f32 = 0.3;   // Умеренное виньетирование
const VIGNETTE_SIZE: f32 = 0.6;
const ENABLE_DITHERING: bool = false; // Без дизеринга для четких границ
const ENABLE_SHARPENING: bool = true; // С повышением резкости
const SATURATION: f32 = 1.3;          // Высокая насыщенность

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let textureDimensions = vec2f(textureDimensions(inputTexture));
  let pixelSize = max(1.0, PIXEL_SIZE);
  
  // 1. Получаем координаты текущего пикселя
  let pixelCoordX = floor(input.texCoord.x * textureDimensions.x / pixelSize) * pixelSize / textureDimensions.x;
  let pixelCoordY = floor(input.texCoord.y * textureDimensions.y / pixelSize) * pixelSize / textureDimensions.y;
  let pixelCoord = vec2f(pixelCoordX, pixelCoordY);
  
  // 2. Получаем цвет из исходной текстуры
  var color = textureSample(inputTexture, textureSampler, pixelCoord);
  var rgbColor = color.rgb;
  
  // 3. Увеличение резкости (опционально)
  if (ENABLE_SHARPENING) {
    let pixelSize = vec2f(1.0 / textureDimensions.x, 1.0 / textureDimensions.y);
    let center = rgbColor;
    let top = textureSample(inputTexture, textureSampler, pixelCoord + vec2f(0.0, -pixelSize.y)).rgb;
    let right = textureSample(inputTexture, textureSampler, pixelCoord + vec2f(pixelSize.x, 0.0)).rgb;
    let bottom = textureSample(inputTexture, textureSampler, pixelCoord + vec2f(0.0, pixelSize.y)).rgb;
    let left = textureSample(inputTexture, textureSampler, pixelCoord + vec2f(-pixelSize.x, 0.0)).rgb;
    
    let sharpenAmount = 0.5;
    let neighbors = (top + right + bottom + left) * 0.25;
    rgbColor = center + (center - neighbors) * sharpenAmount;
  }
  
  // 4. Настройка насыщенности
  if (abs(SATURATION - 1.0) > 0.01) {
    let hsl = rgb_to_hsl(rgbColor);
    let adjustedHsl = vec3f(hsl.x, hsl.y * SATURATION, hsl.z);
    rgbColor = hsl_to_rgb(adjustedHsl);
  }
  
  // 5. Повышение контраста
  if (CONTRAST_AMOUNT != 0.0) {
    let contrastFactor = tan((CONTRAST_AMOUNT + 1.0) * 3.14159 / 4.0);
    rgbColor = (rgbColor - 0.5) * contrastFactor + 0.5;
    rgbColor = clamp(rgbColor, vec3f(0.0), vec3f(1.0));
  }
  
  // 6. Применение цветового квантования (уменьшение глубины цвета)
  if (COLOR_DEPTH < 255.0) {
    // Используем select для выбора между методами с дизерингом и без
    let ditherResult = dither8x8(input.position.xy, rgbColor);
    
    let levels = max(2.0, COLOR_DEPTH);
    let step = 1.0 / (levels - 1.0);
    let quantizedResult = floor(rgbColor / step + 0.5) * step;
    
    rgbColor = select(quantizedResult, ditherResult, ENABLE_DITHERING);
  }
  
  // 7. Эффект виньетирования
  if (VIGNETTE_STRENGTH > 0.0) {
    let center = vec2f(0.5);
    let dist = distance(input.texCoord, center);
    let vignetteSize = max(0.1, VIGNETTE_SIZE);
    let vignette = smoothstep(vignetteSize, vignetteSize * 1.8, dist);
    rgbColor = mix(rgbColor, rgbColor * (1.0 - vignette), VIGNETTE_STRENGTH);
  }
  
  return vec4f(rgbColor, color.a);
}

// Преобразование RGB в HSL
fn rgb_to_hsl(rgb: vec3f) -> vec3f {
  let r = rgb.r;
  let g = rgb.g;
  let b = rgb.b;
  
  let maxColor = max(max(r, g), b);
  let minColor = min(min(r, g), b);
  let delta = maxColor - minColor;
  
  var h: f32 = 0.0;
  var s: f32 = 0.0;
  let l = (maxColor + minColor) / 2.0;
  
  if (delta > 0.0) {
    // Используем select вместо тернарного оператора
    s = select(delta / (2.0 - maxColor - minColor), delta / (maxColor + minColor), l < 0.5);
    
    if (r == maxColor) {
      // Используем select вместо тернарного оператора
      h = (g - b) / delta + select(0.0, 6.0, g < b);
    } else if (g == maxColor) {
      h = (b - r) / delta + 2.0;
    } else {
      h = (r - g) / delta + 4.0;
    }
    
    h /= 6.0;
  }
  
  return vec3f(h, s, l);
}

// Преобразование HSL в RGB
fn hsl_to_rgb(hsl: vec3f) -> vec3f {
  let h = hsl.x;
  let s = hsl.y;
  let l = hsl.z;
  
  if (s == 0.0) {
    return vec3f(l);
  }
  
  // Используем select вместо тернарного оператора
  let q = select(l + s - l * s, l * (1.0 + s), l < 0.5);
  let p = 2.0 * l - q;
  
  let r = hue_to_rgb(p, q, h + 1.0/3.0);
  let g = hue_to_rgb(p, q, h);
  let b = hue_to_rgb(p, q, h - 1.0/3.0);
  
  return vec3f(r, g, b);
}

fn hue_to_rgb(p: f32, q: f32, t: f32) -> f32 {
  var tmod = t;
  if (tmod < 0.0) { tmod += 1.0; }
  if (tmod > 1.0) { tmod -= 1.0; }
  
  if (tmod < 1.0/6.0) { return p + (q - p) * 6.0 * tmod; }
  if (tmod < 1.0/2.0) { return q; }
  if (tmod < 2.0/3.0) { return p + (q - p) * (2.0/3.0 - tmod) * 6.0; }
  
  return p;
}

const DITHER_MATRIX = array<f32, 64>(
    0.0/64.0, 32.0/64.0, 8.0/64.0, 40.0/64.0, 2.0/64.0, 34.0/64.0, 10.0/64.0, 42.0/64.0,
    48.0/64.0, 16.0/64.0, 56.0/64.0, 24.0/64.0, 50.0/64.0, 18.0/64.0, 58.0/64.0, 26.0/64.0,
    12.0/64.0, 44.0/64.0, 4.0/64.0, 36.0/64.0, 14.0/64.0, 46.0/64.0, 6.0/64.0, 38.0/64.0,
    60.0/64.0, 28.0/64.0, 52.0/64.0, 20.0/64.0, 62.0/64.0, 30.0/64.0, 54.0/64.0, 22.0/64.0,
    3.0/64.0, 35.0/64.0, 11.0/64.0, 43.0/64.0, 1.0/64.0, 33.0/64.0, 9.0/64.0, 41.0/64.0,
    51.0/64.0, 19.0/64.0, 59.0/64.0, 27.0/64.0, 49.0/64.0, 17.0/64.0, 57.0/64.0, 25.0/64.0,
    15.0/64.0, 47.0/64.0, 7.0/64.0, 39.0/64.0, 13.0/64.0, 45.0/64.0, 5.0/64.0, 37.0/64.0,
    63.0/64.0, 31.0/64.0, 55.0/64.0, 23.0/64.0, 61.0/64.0, 29.0/64.0, 53.0/64.0, 21.0/64.0
);
// Функция для дизеринга
fn dither8x8(pos: vec2f, color: vec3f) -> vec3f {let x = u32(pos.x) % 8;
  let y = u32(pos.y) % 8;
  let threshold = DITHER_MATRIX[y * 8 + x];
  
  // Квантование цвета с диффузией ошибки
  let colorDepth = max(2.0, COLOR_DEPTH);
  let step = 1.0 / (colorDepth - 1.0);
  
  var result = vec3f(
    floor(color.r / step + threshold) * step,
    floor(color.g / step + threshold) * step,
    floor(color.b / step + threshold) * step
  );
  
  // Ограничить результат в диапазоне [0,1]
  result = clamp(result, vec3f(0.0), vec3f(1.0));
  
  return result;
}    
    `,
);
