import React, {useCallback, useEffect, useRef, useState} from 'react';
import {
  ActivityIndicator,
  Image,
  LayoutChangeEvent,
  Linking,
  Pressable,
  SafeAreaView,
  StatusBar,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import {launchImageLibrary} from 'react-native-image-picker';
import {FileSystem} from 'react-native-file-access';
import {
  Camera,
  type CameraRef,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';
import {DetectionOverlay} from './src/components/DetectionOverlay';
import {HistoryPanel} from './src/components/HistoryPanel';
import type {ImageSize} from './src/domain/types';
import {usePlateApp} from './src/hooks/usePlateApp';
import {disposeImage} from './src/inference/imageTensor';

type Mode = 'camera' | 'gallery';

export default function App() {
  const [mode, setMode] = useState<Mode>('camera');
  const [previewReady, setPreviewReady] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [screenError, setScreenError] = useState('');
  const [galleryUri, setGalleryUri] = useState<string>();
  const [viewport, setViewport] = useState<ImageSize>({width: 0, height: 0});
  const cameraRef = useRef<CameraRef>(null);
  const camera = useCameraDevice('back');
  const permission = useCameraPermission();
  const {
    pipeline,
    modelState,
    modelError,
    output,
    history,
    commit,
    reset,
    clearHistory,
  } = usePlateApp();

  const changeMode = useCallback((next: Mode) => {
    reset();
    setScreenError('');
    setMode(next);
  }, [reset]);

  const handleLayout = useCallback((event: LayoutChangeEvent) => {
    const {width, height} = event.nativeEvent.layout;
    setViewport({width, height});
  }, []);

  useEffect(() => {
    if (mode !== 'camera' || !permission.hasPermission || !previewReady || modelState !== 'ready') {
      return;
    }
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      let path: string | undefined;
      try {
        const snapshot = await cameraRef.current?.takeSnapshot();
        if (!snapshot || cancelled) return;
        path = await snapshot.saveToTemporaryFileAsync('jpg', 92);
        disposeImage(snapshot);
        setProcessing(true);
        const result = await pipeline.processLive(`file://${path}`);
        if (!cancelled) {
          commit(result);
          setScreenError('');
        }
      } catch (caught) {
        if (!cancelled) setScreenError(caught instanceof Error ? caught.message : String(caught));
      } finally {
        setProcessing(false);
        if (path && (await FileSystem.exists(path))) await FileSystem.unlink(path);
        if (!cancelled) timer = setTimeout(tick, 750);
      }
    };
    timer = setTimeout(tick, 750);
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [
    commit,
    modelState,
    pipeline,
    mode,
    permission.hasPermission,
    previewReady,
  ]);

  const chooseImage = useCallback(async () => {
    if (modelState !== 'ready') {
      setScreenError(modelError || 'The detector model is not ready.');
      return;
    }
    const selection = await launchImageLibrary({mediaType: 'photo', selectionLimit: 1, quality: 1});
    const asset = selection.assets?.[0];
    if (!asset?.uri) return;
    setGalleryUri(asset.uri);
    setProcessing(true);
    setScreenError('');
    try {
      commit(await pipeline.processGallery(asset.uri));
    } catch (caught) {
      setScreenError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setProcessing(false);
    }
  }, [commit, modelError, modelState, pipeline]);

  const error = screenError || (modelState === 'error' ? modelError : '');

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" backgroundColor="#030712" />
      <View style={styles.header}>
        <View>
          <Text style={styles.title}>Chilean Plate Reader</Text>
          <Text style={styles.subtitle}>Offline YOLO11s + OCR</Text>
        </View>
        <Text style={styles.timing}>
          {processing ? 'Processing…' : output ? `${output.elapsedMs} ms` : 'Idle'}
        </Text>
      </View>

      <View style={styles.tabs}>
        {(['camera', 'gallery'] as Mode[]).map(value => (
          <Pressable
            accessibilityRole="button"
            key={value}
            onPress={() => changeMode(value)}
            style={[styles.tab, mode === value && styles.activeTab]}>
            <Text style={[styles.tabText, mode === value && styles.activeTabText]}>
              {value === 'camera' ? 'Camera' : 'Gallery'}
            </Text>
          </Pressable>
        ))}
      </View>

      <View style={styles.content} onLayout={handleLayout}>
        {mode === 'camera' ? (
          permission.hasPermission ? (
            camera ? (
              <Camera
                ref={cameraRef}
                style={StyleSheet.absoluteFill}
                device={camera}
                isActive
                resizeMode="cover"
                implementationMode="compatible"
                onPreviewStarted={() => setPreviewReady(true)}
                onPreviewStopped={() => setPreviewReady(false)}
                onError={cameraError => setScreenError(cameraError.message)}
              />
            ) : <CenteredMessage text="No back camera is available." />
          ) : (
            <View style={styles.centered}>
              <Text style={styles.message}>Camera permission is required for live detection.</Text>
              <ActionButton
                label={permission.canRequestPermission ? 'Allow camera' : 'Open Android settings'}
                onPress={() => {
                  if (permission.canRequestPermission) {
                    permission.requestPermission().catch(caught => setScreenError(String(caught)));
                  } else {
                    Linking.openSettings().catch(caught => setScreenError(String(caught)));
                  }
                }}
              />
            </View>
          )
        ) : galleryUri ? (
          <Image source={{uri: galleryUri}} resizeMode="contain" style={StyleSheet.absoluteFill} />
        ) : (
          <View style={styles.centered}>
            <Text style={styles.message}>Choose a vehicle photo to detect and read every plate.</Text>
            <ActionButton label="Choose image" onPress={() => chooseImage().catch(caught => setScreenError(String(caught)))} />
          </View>
        )}

        <DetectionOverlay
          plates={output?.plates ?? []}
          source={output?.source}
          view={viewport}
          resizeMode={mode === 'camera' ? 'cover' : 'contain'}
        />

        {mode === 'gallery' && galleryUri ? (
          <View style={styles.galleryButton}>
            <ActionButton label="Choose another" onPress={() => chooseImage().catch(caught => setScreenError(String(caught)))} />
          </View>
        ) : null}

        {modelState === 'loading' ? (
          <View style={styles.loading}>
            <ActivityIndicator color="#ffffff" />
            <Text style={styles.loadingText}>Loading offline model…</Text>
          </View>
        ) : null}
      </View>

      {error ? <Text style={styles.error}>{error}</Text> : null}
      <HistoryPanel entries={history} onClear={clearHistory} />
    </SafeAreaView>
  );
}

function ActionButton({label, onPress}: {label: string; onPress: () => void}) {
  return (
    <Pressable accessibilityRole="button" onPress={onPress} style={styles.actionButton}>
      <Text style={styles.actionText}>{label}</Text>
    </Pressable>
  );
}

function CenteredMessage({text}: {text: string}) {
  return <View style={styles.centered}><Text style={styles.message}>{text}</Text></View>;
}

const styles = StyleSheet.create({
  safeArea: {flex: 1, backgroundColor: '#030712'},
  header: {paddingHorizontal: 16, paddingVertical: 12, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center'},
  title: {color: '#f9fafb', fontSize: 20, fontWeight: '800'},
  subtitle: {color: '#9ca3af', fontSize: 12, marginTop: 2},
  timing: {color: '#34d399', fontVariant: ['tabular-nums'], fontWeight: '700'},
  tabs: {flexDirection: 'row', marginHorizontal: 16, marginBottom: 10, backgroundColor: '#111827', borderRadius: 10},
  tab: {flex: 1, alignItems: 'center', paddingVertical: 9, borderRadius: 10},
  activeTab: {backgroundColor: '#2563eb'},
  tabText: {color: '#9ca3af', fontWeight: '700'},
  activeTabText: {color: '#ffffff'},
  content: {flex: 1, backgroundColor: '#000000', overflow: 'hidden'},
  centered: {flex: 1, alignItems: 'center', justifyContent: 'center', padding: 28, gap: 18},
  message: {color: '#d1d5db', textAlign: 'center', fontSize: 16, lineHeight: 23},
  actionButton: {backgroundColor: '#2563eb', borderRadius: 8, paddingHorizontal: 16, paddingVertical: 10},
  actionText: {color: '#ffffff', fontWeight: '800'},
  galleryButton: {position: 'absolute', right: 12, bottom: 12},
  loading: {position: 'absolute', alignSelf: 'center', top: 20, backgroundColor: '#111827dd', borderRadius: 8, paddingHorizontal: 14, paddingVertical: 10, flexDirection: 'row', gap: 8},
  loadingText: {color: '#ffffff'},
  error: {color: '#fecaca', backgroundColor: '#7f1d1d', paddingHorizontal: 12, paddingVertical: 8, fontSize: 12},
});
