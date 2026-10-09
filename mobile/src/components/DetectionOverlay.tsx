import React from 'react';
import {StyleSheet, Text, View} from 'react-native';
import {mapBoxToView} from '../domain/geometry';
import type {ImageSize, PlateResult} from '../domain/types';

type Props = {
  plates: PlateResult[];
  source?: ImageSize;
  view: ImageSize;
  resizeMode: 'contain' | 'cover';
};

export function DetectionOverlay({plates, source, view, resizeMode}: Props) {
  if (!source || view.width <= 0 || view.height <= 0) {
    return null;
  }
  return (
    <View pointerEvents="none" style={StyleSheet.absoluteFill}>
      {plates.map((plate, index) => {
        const box = mapBoxToView(plate.box, source, view, resizeMode);
        const label = plate.ocr?.normalizedText || 'reading…';
        const uncertain = plate.ocr?.format === 'uncertain';
        return (
          <View
            key={`${index}-${plate.box.x}-${plate.box.y}`}
            style={[styles.box, {left: box.x, top: box.y, width: box.width, height: box.height}]}>
            <Text style={[styles.label, uncertain && styles.uncertain]}>
              {label} · {Math.round(plate.confidence * 100)}%
            </Text>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  box: {position: 'absolute', borderColor: '#34d399', borderWidth: 2, borderRadius: 4},
  label: {
    position: 'absolute', left: -2, top: -25, paddingHorizontal: 6, paddingVertical: 3,
    borderRadius: 4, backgroundColor: '#047857', color: '#ffffff', fontSize: 12, fontWeight: '700',
  },
  uncertain: {backgroundColor: '#b45309'},
});
