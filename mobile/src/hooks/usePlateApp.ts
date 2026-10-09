import {useCallback, useEffect, useMemo, useState} from 'react';
import {addHistoryResults} from '../domain/history';
import type {HistoryEntry, PipelineOutput} from '../domain/types';
import {MlKitPlateRecognizer} from '../inference/MlKitPlateRecognizer';
import {OnnxPlateDetector} from '../inference/OnnxPlateDetector';
import {PlatePipeline} from '../inference/PlatePipeline';

export type ModelState = 'loading' | 'ready' | 'error';

export function usePlateApp() {
  const pipeline = useMemo(
    () => new PlatePipeline(new OnnxPlateDetector(), new MlKitPlateRecognizer()),
    [],
  );
  const [modelState, setModelState] = useState<ModelState>('loading');
  const [modelError, setModelError] = useState('');
  const [output, setOutput] = useState<PipelineOutput>();
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    let active = true;
    pipeline.initialize().then(() => {
      if (active) setModelState('ready');
    }).catch(error => {
      if (active) {
        setModelState('error');
        setModelError(error instanceof Error ? error.message : String(error));
      }
    });
    return () => {
      active = false;
      pipeline.release().catch(() => undefined);
    };
  }, [pipeline]);

  const commit = useCallback((result: PipelineOutput | null) => {
    if (!result) return;
    setOutput(result);
    setHistory(current => addHistoryResults(current, result.plates));
  }, []);

  const reset = useCallback(() => {
    pipeline.resetTracking();
    setOutput(undefined);
  }, [pipeline]);

  const clearHistory = useCallback(() => setHistory([]), []);

  return {pipeline, modelState, modelError, output, history, commit, reset, clearHistory};
}
