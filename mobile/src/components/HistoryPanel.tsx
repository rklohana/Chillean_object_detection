import Clipboard from '@react-native-clipboard/clipboard';
import React from 'react';
import {Pressable, ScrollView, StyleSheet, Text, View} from 'react-native';
import type {HistoryEntry} from '../domain/types';

type Props = {entries: HistoryEntry[]; onClear: () => void};

export function HistoryPanel({entries, onClear}: Props) {
  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Session history</Text>
        <Pressable accessibilityRole="button" onPress={onClear} disabled={entries.length === 0}>
          <Text style={[styles.clear, entries.length === 0 && styles.disabled]}>Clear</Text>
        </Pressable>
      </View>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.list}>
        {entries.length === 0 ? (
          <Text style={styles.empty}>Recognized plates will appear here.</Text>
        ) : entries.map(entry => (
          <Pressable
            accessibilityLabel={`Copy ${entry.text}`}
            accessibilityRole="button"
            key={entry.id}
            onPress={() => Clipboard.setString(entry.text)}
            style={styles.item}>
            <Text style={styles.plate}>{entry.text}</Text>
            <Text style={styles.confidence}>{Math.round(entry.confidence * 100)}% · tap to copy</Text>
          </Pressable>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {backgroundColor: '#111827', paddingTop: 10, paddingBottom: 12},
  header: {flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 16},
  title: {color: '#f9fafb', fontSize: 14, fontWeight: '700'},
  clear: {color: '#60a5fa', fontWeight: '600'},
  disabled: {opacity: 0.4},
  list: {minHeight: 58, alignItems: 'center', paddingHorizontal: 12, gap: 8},
  empty: {color: '#9ca3af', paddingHorizontal: 4},
  item: {backgroundColor: '#1f2937', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 8},
  plate: {color: '#f9fafb', fontSize: 17, fontWeight: '800', letterSpacing: 1.5},
  confidence: {color: '#9ca3af', fontSize: 11, marginTop: 2},
});
