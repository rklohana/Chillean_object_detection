import {FileSystem, Dirs} from 'react-native-file-access';
import {Images, type Image} from 'react-native-nitro-image';
import * as ort from 'onnxruntime-react-native';
import type {Detection, ImageSize, PlateDetector} from '../domain/types';
import {disposeImage, imageToLetterboxTensor} from './imageTensor';
import {MODEL_CONFIG} from './modelConfig';
import {decodeYolo11, type LetterboxTransform} from './yoloDecoder';

export class OnnxPlateDetector implements PlateDetector {
  private session?: ort.InferenceSession;
  private lastTransform?: LetterboxTransform;

  async initialize(): Promise<void> {
    if (this.session) {
      return;
    }
    const modelPath = `${Dirs.CacheDir}/${MODEL_CONFIG.cacheName}`;
    if (!(await FileSystem.exists(modelPath))) {
      try {
        await FileSystem.cpAsset(MODEL_CONFIG.assetPath, modelPath);
      } catch (error) {
        throw new Error(
          `Detector model is missing. Copy plate_detector_fp32.onnx to ` +
            `mobile/android/app/src/main/assets/models/. ${String(error)}`,
        );
      }
    }
    try {
      this.session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['xnnpack', 'cpu'],
        graphOptimizationLevel: 'all',
        executionMode: 'sequential',
      });
    } catch {
      this.session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['cpu'],
        graphOptimizationLevel: 'all',
        executionMode: 'sequential',
      });
    }
  }

  async detectImage(uri: string): Promise<{detections: Detection[]; source: ImageSize}> {
    const image = await Images.loadFromFileAsync(uri.replace(/^file:\/\//, ''));
    try {
      const detections = await this.detectNativeImage(image);
      return {detections, source: {width: image.width, height: image.height}};
    } finally {
      disposeImage(image);
    }
  }

  async detectNativeImage(image: Image): Promise<Detection[]> {
    await this.initialize();
    const prepared = await imageToLetterboxTensor(image, MODEL_CONFIG.inputSize);
    this.lastTransform = prepared.transform;
    return this.run(prepared.data, prepared.transform);
  }

  async detectTensor(data: Float32Array): Promise<Detection[]> {
    await this.initialize();
    const transform: LetterboxTransform =
      this.lastTransform ?? {
        source: {width: MODEL_CONFIG.inputSize, height: MODEL_CONFIG.inputSize},
        scale: 1,
        padX: 0,
        padY: 0,
        inputSize: MODEL_CONFIG.inputSize,
      };
    return this.run(data, transform);
  }

  private async run(data: Float32Array, transform: LetterboxTransform): Promise<Detection[]> {
    const session = this.session;
    if (!session) {
      throw new Error('Detector session was not initialized');
    }
    if (data.length !== 3 * MODEL_CONFIG.inputSize * MODEL_CONFIG.inputSize) {
      throw new Error(`Detector input contains ${data.length} values; expected 1228800`);
    }
    const tensor = new ort.Tensor('float32', data, [1, 3, MODEL_CONFIG.inputSize, MODEL_CONFIG.inputSize]);
    try {
      const output = await session.run({[session.inputNames[0]]: tensor}, [session.outputNames[0]]);
      const result = output[session.outputNames[0]];
      if (!(result.data instanceof Float32Array)) {
        throw new Error(`Unexpected detector output type: ${result.type}`);
      }
      return decodeYolo11(
        result.data,
        result.dims,
        transform,
        MODEL_CONFIG.confidenceThreshold,
        MODEL_CONFIG.nmsIouThreshold,
      );
    } finally {
      tensor.dispose();
    }
  }

  async release(): Promise<void> {
    await this.session?.release();
    this.session = undefined;
  }
}
