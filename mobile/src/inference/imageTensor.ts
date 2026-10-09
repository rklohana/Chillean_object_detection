import type {Image, PixelFormat} from 'react-native-nitro-image';
import type {ImageSize} from '../domain/types';
import type {LetterboxTransform} from './yoloDecoder';

export type PreparedTensor = {
  data: Float32Array;
  transform: LetterboxTransform;
};

export function disposeImage(image: Image) {
  (image as Image & {dispose?: () => void}).dispose?.();
}

function channelIndexes(format: PixelFormat): [number, number, number, number] {
  switch (format) {
    case 'RGB':
      return [0, 1, 2, 3];
    case 'BGR':
      return [2, 1, 0, 3];
    case 'RGBA':
    case 'RGBX':
      return [0, 1, 2, 4];
    case 'BGRA':
    case 'BGRX':
      return [2, 1, 0, 4];
    case 'ARGB':
    case 'XRGB':
      return [1, 2, 3, 4];
    case 'ABGR':
    case 'XBGR':
      return [3, 2, 1, 4];
    default:
      throw new Error(`Unsupported image pixel format: ${format}`);
  }
}

export async function imageToLetterboxTensor(
  image: Image,
  inputSize = 640,
): Promise<PreparedTensor> {
  const source: ImageSize = {width: image.width, height: image.height};
  const scale = Math.min(inputSize / source.width, inputSize / source.height);
  const resizedWidth = Math.max(1, Math.round(source.width * scale));
  const resizedHeight = Math.max(1, Math.round(source.height * scale));
  const padX = Math.floor((inputSize - resizedWidth) / 2);
  const padY = Math.floor((inputSize - resizedHeight) / 2);
  const resized = await image.resizeAsync(resizedWidth, resizedHeight);
  try {
    const raw = await resized.toRawPixelDataAsync(false);
    const pixels = new Uint8Array(raw.buffer);
    const [rIndex, gIndex, bIndex, stride] = channelIndexes(raw.pixelFormat);
    const planeSize = inputSize * inputSize;
    const tensor = new Float32Array(planeSize * 3);
    tensor.fill(114 / 255);
    for (let y = 0; y < resizedHeight; y += 1) {
      for (let x = 0; x < resizedWidth; x += 1) {
        const sourceIndex = (y * resizedWidth + x) * stride;
        const targetIndex = (y + padY) * inputSize + x + padX;
        tensor[targetIndex] = pixels[sourceIndex + rIndex] / 255;
        tensor[planeSize + targetIndex] = pixels[sourceIndex + gIndex] / 255;
        tensor[planeSize * 2 + targetIndex] = pixels[sourceIndex + bIndex] / 255;
      }
    }
    return {data: tensor, transform: {source, scale, padX, padY, inputSize}};
  } finally {
    disposeImage(resized);
  }
}
