import React, { useEffect, useState } from 'react';
import { View, ActivityIndicator, FlatList, StyleSheet, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import * as SecureStore from 'expo-secure-store';
import { API_BASE_URL, wineAPI, type RecommendResponse, type MatchItem } from '../services/api';
import { wineTheme } from '../constants/Colors';
import { ThemedText } from '../components/ThemedText';
import { WineButton } from '../components/WineButton';

interface CellarItem {
  _id?: string;
  name?: string;
  age?: number;
  [key: string]: any;
}

export default function CellarScreen() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<'personal' | 'global' | 'foryou'>('personal');
  const [loading, setLoading] = useState(true);
  const [items, setItems] = useState<CellarItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [publicLoading, setPublicLoading] = useState(false);
  const [publicItems, setPublicItems] = useState<CellarItem[]>([]);
  const [publicError, setPublicError] = useState<string | null>(null);
  const [recLoading, setRecLoading] = useState(false);
  const [recError, setRecError] = useState<string | null>(null);
  const [recItems, setRecItems] = useState<MatchItem[] | null>(null);

  const fetchCellar = async () => {
    try {
      const token = await SecureStore.getItemAsync('auth_token');
      if (!token) {
        router.replace('/');
        return;
      }
      const res = await fetch(`${API_BASE_URL}/cellar`, {
        method: 'GET',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setItems(Array.isArray(data.items) ? data.items : []);
      setError(null);
    } catch (e: any) {
      setError(e?.message ?? 'Failed to load cellar');
    } finally {
      setLoading(false);
    }
  };

  const fetchPublic = async () => {
    try {
      setPublicLoading(true);
      const res = await fetch(`${API_BASE_URL}/cellar/public`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setPublicItems(Array.isArray(data.items) ? data.items : []);
      setPublicError(null);
    } catch (e: any) {
      setPublicError(e?.message ?? 'Failed to load global wines');
    } finally {
      setPublicLoading(false);
    }
  };

  useEffect(() => {
    fetchCellar();
  }, []);

  useEffect(() => {
    if (activeTab === 'global' && publicItems.length === 0 && !publicLoading) {
      fetchPublic();
    }
    if (activeTab === 'foryou' && !recItems && !recLoading) {
      (async () => {
        try {
          setRecLoading(true);
          setRecError(null);
          const res: RecommendResponse = await wineAPI.recommend({ top_k: 10, diversity_lambda: 0.2, source: 'both', blend: { ratio_catalog: 0.7 } });
          setRecItems(res.matches || []);
        } catch (e: any) {
          setRecError(e?.message ?? 'Failed to load recommendations');
        } finally {
          setRecLoading(false);
        }
      })();
    }
  }, [activeTab]);

  const handleOpenDetails = (item: CellarItem, idx: number) => {
    const id = item._id ?? item.bottle_id ?? String(idx);
    const param = encodeURIComponent(JSON.stringify(item));
    router.push(`/cellar/${id}?item=${param}` as any);
  };

  const scoreBucket = (score: number | undefined): 'High' | 'Medium' | 'Low' => {
    const s = typeof score === 'number' ? score : 0;
    if (s >= 0.6) return 'High';
    if (s >= 0.45) return 'Medium';
    return 'Low';
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.headerRow}>
        <ThemedText type="title">Cellar</ThemedText>
        <WineButton title="Add wines" onPress={() => router.push('/country')} />
      </View>

      <View style={styles.segments}>
        <TouchableOpacity
          onPress={() => setActiveTab('personal')}
          style={[styles.segment, activeTab === 'personal' && styles.segmentActive]}
        >
          <ThemedText style={[styles.segmentText, activeTab === 'personal' && styles.segmentTextActive]}>My Cellar</ThemedText>
        </TouchableOpacity>
        <TouchableOpacity
          onPress={() => setActiveTab('global')}
          style={[styles.segment, activeTab === 'global' && styles.segmentActive]}
        >
          <ThemedText style={[styles.segmentText, activeTab === 'global' && styles.segmentTextActive]}>Global</ThemedText>
        </TouchableOpacity>
        <TouchableOpacity
          onPress={() => setActiveTab('foryou')}
          style={[styles.segment, activeTab === 'foryou' && styles.segmentActive]}
        >
          <ThemedText style={[styles.segmentText, activeTab === 'foryou' && styles.segmentTextActive]}>For You</ThemedText>
        </TouchableOpacity>
      </View>

      {activeTab === 'personal' && (loading ? (
        <View style={styles.center}>
          <ActivityIndicator color={wineTheme.colors.primary} />
        </View>
      ) : error ? (
        <View style={styles.errorBox}>
          <ThemedText type="subtitle" style={styles.errorText}>Unable to load</ThemedText>
          <ThemedText>{error}</ThemedText>
          <WineButton title="Retry" onPress={() => { setLoading(true); fetchCellar(); }} style={{ marginTop: 12 }} />
        </View>
      ) : items.length === 0 ? (
        <View style={styles.center}>
          <ThemedText>No bottles yet.</ThemedText>
          <WineButton title="Add wines" onPress={() => router.push('/country')} style={{ marginTop: 16 }} />
        </View>
      ) : (
        <FlatList
          contentContainerStyle={{ paddingBottom: 24 }}
          data={items}
          keyExtractor={(it, idx) => (it._id ?? it.bottle_id ?? String(idx))}
          showsVerticalScrollIndicator={false}
          renderItem={({ item, index }) => (
            <TouchableOpacity style={styles.card} onPress={() => handleOpenDetails(item, index)}>
              <View style={styles.cardHeader}>
                <ThemedText type="subtitle" style={{ flexShrink: 1 }}>
                  {item.name ?? 'Bottle'}
                </ThemedText>
                <View style={styles.ageBadge}>
                  <ThemedText style={styles.ageBadgeText}>
                    {typeof item.age === 'number' ? `${item.age} yr${item.age === 1 ? '' : 's'}` : 'N/A'}
                  </ThemedText>
                </View>
              </View>
              {/* Add more fields when available */}
            </TouchableOpacity>
          )}
        />
      ))}

      {activeTab === 'global' && (publicLoading ? (
        <View style={styles.center}>
          <ActivityIndicator color={wineTheme.colors.primary} />
        </View>
      ) : publicError ? (
        <View style={styles.errorBox}>
          <ThemedText type="subtitle" style={styles.errorText}>Unable to load</ThemedText>
          <ThemedText>{publicError}</ThemedText>
          <WineButton title="Retry" onPress={() => fetchPublic()} style={{ marginTop: 12 }} />
        </View>
      ) : publicItems.length === 0 ? (
        <View style={styles.center}>
          <ThemedText>No global wines yet.</ThemedText>
        </View>
      ) : (
        <FlatList
          contentContainerStyle={{ paddingBottom: 24 }}
          data={publicItems}
          keyExtractor={(it, idx) => (it._id ?? String(idx))}
          showsVerticalScrollIndicator={false}
          renderItem={({ item, index }) => (
            <TouchableOpacity style={styles.card} onPress={() => handleOpenDetails(item, index)}>
              <View style={styles.cardHeader}>
                <ThemedText type="subtitle" style={{ flexShrink: 1 }}>
                  {item.name ?? 'Bottle'}
                </ThemedText>
                <View style={styles.ageBadge}>
                  <ThemedText style={styles.ageBadgeText}>
                    {typeof item.age === 'number' ? `${item.age} yr${item.age === 1 ? '' : 's'}` : 'N/A'}
                  </ThemedText>
                </View>
              </View>
            </TouchableOpacity>
          )}
        />
      ))}

      {activeTab === 'foryou' && (recLoading ? (
        <View style={styles.center}>
          <ActivityIndicator color={wineTheme.colors.primary} />
        </View>
      ) : recError ? (
        <View style={styles.errorBox}>
          <ThemedText type="subtitle" style={styles.errorText}>Unable to load</ThemedText>
          <ThemedText>{recError}</ThemedText>
          <WineButton title="Retry" onPress={() => { setRecLoading(true); setRecItems(null); setActiveTab('foryou'); }} style={{ marginTop: 12 }} />
        </View>
      ) : !recItems || recItems.length === 0 ? (
        <View style={styles.center}>
          <ThemedText>No recommendations yet. Save a few bottles to build your profile.</ThemedText>
        </View>
      ) : (
        <FlatList
          contentContainerStyle={{ paddingBottom: 24 }}
          data={recItems}
          keyExtractor={(it, idx) => `${it.index}-${idx}`}
          showsVerticalScrollIndicator={false}
          renderItem={({ item, index }) => (
            <TouchableOpacity style={styles.card} onPress={() => handleOpenDetails(item as any, index)}>
              <View style={styles.cardHeader}>
                <ThemedText type="subtitle" style={{ flexShrink: 1 }}>
                  {item.source === 'catalog' ? '🏷️ ' : ''}{item.name ?? 'Wine'}
                </ThemedText>
                <View style={styles.ageBadge}>
                  <ThemedText style={styles.ageBadgeText}>
                    {typeof item.age === 'number' ? `${item.age} yr${item.age === 1 ? '' : 's'}` : (item?.predicted?.age_bucket ?? 'N/A')}
                  </ThemedText>
                </View>
              </View>
              <ThemedText>
                {(item.variety ? item.variety + ' · ' : '') + (item.country || '') + (item.region1 ? ', ' + item.region1 : '')}
              </ThemedText>
              <ThemedText>
                {(item.price != null ? `$${Number(item.price).toFixed(0)}` : '—')}
                {item.rating != null ? ` · ${Number(item.rating).toFixed(0)}/100` : ''}
                {` · ${(typeof item.score === 'number' ? (item.score * 100).toFixed(0) : '0')}% match`}
                {item.source ? ` · ${item.source}` : ''}
              </ThemedText>
              <ThemedText style={{ opacity: 0.7, marginTop: 4 }}>{scoreBucket(item.score)} match</ThemedText>
            </TouchableOpacity>
          )}
        />
      ))}
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
  segments: {
    flexDirection: 'row',
    backgroundColor: wineTheme.colors.surface,
    borderRadius: 10,
    padding: 4,
    gap: 6,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: `${wineTheme.colors.primary}33`,
  },
  segment: {
    flex: 1,
    borderRadius: 8,
    alignItems: 'center',
    paddingVertical: 8,
  },
  segmentActive: {
    backgroundColor: wineTheme.colors.primary,
  },
  segmentText: {
    color: wineTheme.colors.text,
  },
  segmentTextActive: {
    color: 'white',
  },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  errorBox: {
    backgroundColor: wineTheme.colors.surface,
    borderRadius: 12,
    padding: 16,
  },
  errorText: {
    color: wineTheme.colors.primary,
    marginBottom: 6,
  },
  card: {
    backgroundColor: wineTheme.colors.surface,
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: `${wineTheme.colors.primary}33`,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  ageBadge: {
    backgroundColor: wineTheme.colors.primary,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
  },
  ageBadgeText: {
    color: 'white',
    fontSize: 12,
  },
}); 