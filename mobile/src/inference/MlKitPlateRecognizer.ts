import MlkitOcr from 'rn-mlkit-ocr';
import type {PlateRecognizer} from '../domain/types';
import {normalizePlateText} from '../domain/ocr';

export class MlKitPlateRecognizer implements PlateRecognizer {
  async recognize(uri: string) {
    const result = await MlkitOcr.recognizeText(uri, 'latin');
    return normalizePlateText(result.text);
  }
}
