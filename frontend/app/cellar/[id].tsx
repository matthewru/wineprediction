import React, { useEffect, useMemo, useState } from 'react';
import { View, StyleSheet, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { ThemedText } from '../../components/ThemedText';
import { WineButton } from '../../components/WineButton';
import { wineTheme } from '../../constants/Colors';
import { fonts } from '../../constants/Fonts';
import * as SecureStore from 'expo-secure-store';
import { API_BASE_URL } from '../../services/api';
import * as Progress from 'react-native-progress';

export default function CellarDetailsScreen() {
  const router = useRouter();
  const params = useLocalSearchParams();
  const [userName, setUserName] = useState<string>('');

  const item = useMemo(() => {
    try {
      if (typeof params.item === 'string') return JSON.parse(decodeURIComponent(params.item));
      return null;
    } catch {
      return null;
    }
  }, [params.item]);
  const isCatalog = !!(item && item.source === 'catalog');

  useEffect(() => {
    (async () => {
      try {
        const token = await SecureStore.getItemAsync('auth_token');
        if (!token) return;
        const res = await fetch(`${API_BASE_URL}/me`, { headers: { Authorization: `Bearer ${token}` } });
        if (!res.ok) return;
        const data = await res.json();
        setUserName(data?.user?.name ?? '');
      } catch {}
    })();
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.headerRow}>
        <ThemedText type="title">Wine Details</ThemedText>
        <WineButton title="Back" variant="secondary" onPress={() => router.back()} />
      </View>

      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.section}>
          <ThemedText type="subtitle" style={styles.sectionTitle}>{item?.name ?? 'Bottle'}</ThemedText>
          {!isCatalog && (
            <View style={styles.row}><ThemedText>Owner</ThemedText><ThemedText style={styles.value}>{userName || 'You'}</ThemedText></View>
          )}
          <View style={styles.row}><ThemedText>Variety</ThemedText><ThemedText style={styles.value}>{item?.variety ?? '—'}</ThemedText></View>
          <View style={styles.row}><ThemedText>Country</ThemedText><ThemedText style={styles.value}>{item?.country ?? '—'}</ThemedText></View>
          <View style={styles.row}><ThemedText>Region</ThemedText><ThemedText style={styles.value}>{[item?.region1, item?.region2].filter(Boolean).join(', ') || '—'}</ThemedText></View>
          <View style={styles.row}><ThemedText>Age</ThemedText><ThemedText style={styles.value}>{typeof item?.age === 'number' ? `${item.age} yrs` : (item?.predicted?.age_bucket ?? '—')}</ThemedText></View>
        </View>

        {isCatalog ? (
          <View style={styles.section}>
            <ThemedText type="subtitle" style={styles.sectionTitle}>Real Wine Characteristics</ThemedText>
            <View style={styles.row}><ThemedText>Price</ThemedText><ThemedText style={styles.value}>{item?.price != null ? `$${Number(item.price).toFixed(0)}` : '—'}</ThemedText></View>
            <View style={styles.row}><ThemedText>Rating</ThemedText><ThemedText style={styles.value}>{item?.rating != null ? Number(item.rating).toFixed(1) : '—'}</ThemedText></View>
            {(Array.isArray(item?.predicted?.flavors) && item.predicted.flavors.length > 0) && (
              <View style={{ marginTop: 8 }}>
                <ThemedText style={{ fontFamily: fonts.spaceGrotesk, marginBottom: 6 }}>Key Flavors</ThemedText>
                <View style={styles.tagsContainer}>
                  {item.predicted.flavors.map((f: any, i: number) => (
                    <View key={`fchip-${i}`} style={styles.tag}><ThemedText style={styles.tagText}>{f.flavor}</ThemedText></View>
                  ))}
                </View>
              </View>
            )}
            {(Array.isArray(item?.predicted?.mouthfeel) && item.predicted.mouthfeel.length > 0) && (
              <View style={{ marginTop: 8 }}>
                <ThemedText style={{ fontFamily: fonts.spaceGrotesk, marginBottom: 6 }}>Mouthfeel</ThemedText>
                <View style={styles.tagsContainer}>
                  {item.predicted.mouthfeel.map((m: any, i: number) => (
                    <View key={`mchip-${i}`} style={styles.tag}><ThemedText style={styles.tagText}>{m.mouthfeel}</ThemedText></View>
                  ))}
                </View>
              </View>
            )}
          </View>
        ) : (
          item?.predicted && (
            <View style={styles.section}>
              <ThemedText type="subtitle" style={styles.sectionTitle}>Model Predictions</ThemedText>
              {item.predicted.rating && (
                <View style={styles.row}><ThemedText>Rating</ThemedText><ThemedText style={styles.value}>{item.predicted.rating?.predicted_rating?.toFixed?.(1) ?? '—'}</ThemedText></View>
              )}
              {item.predicted.price && (
                <View style={styles.row}><ThemedText>Price</ThemedText><ThemedText style={styles.value}>{`$${Number(item.predicted.price.weighted_lower ?? 0).toFixed(0)} - $${Number(item.predicted.price.weighted_upper ?? 0).toFixed(0)}`}</ThemedText></View>
              )}
              {Array.isArray(item.predicted.flavors) && item.predicted.flavors.length > 0 && (
                <View style={{ marginTop: 8 }}>
                  <ThemedText style={{ fontFamily: fonts.spaceGrotesk, marginBottom: 6 }}>Top Flavors</ThemedText>
                  {item.predicted.flavors.slice(0, 5).map((f: any, i: number) => (
                    <View key={`f-${i}`} style={styles.progressItem}>
                      <ThemedText style={styles.progressLabel}>{f.flavor}</ThemedText>
                      <Progress.Bar
                        progress={Math.max(0, Math.min(1, Number(f.confidence ?? 0)))}
                        height={8}
                        width={null}
                        color={wineTheme.colors.primary}
                        unfilledColor={`${wineTheme.colors.text}22`}
                        borderWidth={2}
                        borderColor={`${wineTheme.colors.primary}55`}
                        borderRadius={8}
                        style={styles.progressBar}
                      />
                    </View>
                  ))}
                  <View style={styles.tagsContainer}>
                    {item.predicted.flavors.map((f: any, i: number) => (
                      <View key={`fchip-${i}`} style={styles.tag}><ThemedText style={styles.tagText}>{f.flavor}</ThemedText></View>
                    ))}
                  </View>
                </View>
              )}
              {Array.isArray(item.predicted.mouthfeel) && item.predicted.mouthfeel.length > 0 && (
                <View style={{ marginTop: 12 }}>
                  <ThemedText style={{ fontFamily: fonts.spaceGrotesk, marginBottom: 6 }}>Top Mouthfeel</ThemedText>
                  {item.predicted.mouthfeel.slice(0, 5).map((m: any, i: number) => (
                    <View key={`m-${i}`} style={styles.progressItem}>
                      <ThemedText style={styles.progressLabel}>{m.mouthfeel}</ThemedText>
                      <Progress.Bar
                        progress={Math.max(0, Math.min(1, Number(m.confidence ?? 0)))}
                        height={8}
                        width={null}
                        color={wineTheme.colors.primary}
                        unfilledColor={`${wineTheme.colors.text}22`}
                        borderWidth={2}
                        borderColor={`${wineTheme.colors.primary}55`}
                        borderRadius={8}
                        style={styles.progressBar}
                      />
                    </View>
                  ))}
                  <View style={styles.tagsContainer}>
                    {item.predicted.mouthfeel.map((m: any, i: number) => (
                      <View key={`mchip-${i}`} style={styles.tag}><ThemedText style={styles.tagText}>{m.mouthfeel}</ThemedText></View>
                    ))}
                  </View>
                </View>
              )}
            </View>
          )
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: wineTheme.colors.background,
    paddingHorizontal: 20,
    paddingTop: 12,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  content: {
    paddingBottom: 24,
  },
  section: {
    backgroundColor: wineTheme.colors.surface,
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: `${wineTheme.colors.primary}33`,
    marginBottom: 12,
  },
  sectionTitle: {
    marginBottom: 8,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  value: {
    fontFamily: fonts.spaceGrotesk,
    color: wineTheme.colors.primary,
  },
  progressItem: {
    marginBottom: 6,
  },
  progressLabel: {
    fontSize: 12,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text,
    marginBottom: 3,
  },
  progressBar: {
    alignSelf: 'stretch',
  },
  tagsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 8,
  },
  tag: {
    backgroundColor: wineTheme.colors.background,
    borderRadius: 16,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderWidth: 1,
    borderColor: `${wineTheme.colors.primary}33`,
  },
  tagText: {
    fontFamily: fonts.outfit,
  },
}); 