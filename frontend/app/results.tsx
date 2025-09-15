import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, TouchableOpacity, Alert, FlatList, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useWine } from '../context/WineContext';
import { wineAPI, type AllPredictionsResponse, type WinePredictionInput, type MatchItem, type MatchRealInput, API_BASE_URL } from '../services/api';
import { wineTheme } from '../constants/Colors';
import { fonts } from '../constants/Fonts';
import * as Progress from 'react-native-progress';
import Bottle from '../components/Bottle';
import * as SecureStore from 'expo-secure-store';

export default function ResultsScreen() {
  const router = useRouter();
  const { wineData } = useWine();
  const [predictions, setPredictions] = useState<AllPredictionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [flavorProgress, setFlavorProgress] = useState<number[]>([]);
  const [mouthfeelProgress, setMouthfeelProgress] = useState<number[]>([]);
  const [isPublic, setIsPublic] = useState<boolean>(false);
  const [matches, setMatches] = useState<MatchItem[] | null>(null);
  const [matchesLoading, setMatchesLoading] = useState<boolean>(false);

  const fetchPredictions = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Validate that we have all required data
      if (!wineData.variety || !wineData.country || !wineData.region1 || !wineData.age) {
        setError('Missing required wine information. Please go back and complete all fields.');
        setLoading(false);
        return;
      }

      // Prepare the input for the API
      const input: WinePredictionInput = {
        variety: wineData.variety,
        country: wineData.country,
        province: wineData.region1, // Using region1 as province
        age: wineData.age,
        region_hierarchy: `${wineData.country} > ${wineData.region1}${wineData.region2 ? ` > ${wineData.region2}` : ''}`
      };

      console.log('Making API request with:', input);
      
      // Call the API to get all predictions
      const response = await wineAPI.predictAll(input);
      setPredictions(response);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch predictions:', err);
      setError('Failed to get wine predictions. Please check your connection and try again.');
    } finally {
      setLoading(false);
    }
  };

  // Only fetch predictions when component mounts (when user navigates to this screen)
  useEffect(() => {
    fetchPredictions();
  }, []); // Empty dependency array - only runs once when component mounts

  // Animate progress bars when predictions load/update
  useEffect(() => {
    if (!predictions) return;
    const flavorItems = predictions.flavors.slice(0, 5);
    const mouthfeelItems = predictions.mouthfeel.slice(0, 5);
    setFlavorProgress(new Array(flavorItems.length).fill(0));
    setMouthfeelProgress(new Array(mouthfeelItems.length).fill(0));
    const t = setTimeout(() => {
      setFlavorProgress(
        flavorItems.map(f => Math.max(0, Math.min(1, f.confidence)))
      );
      setMouthfeelProgress(
        mouthfeelItems.map(m => Math.max(0, Math.min(1, m.confidence)))
      );
    }, 50);
    return () => clearTimeout(t);
  }, [predictions]);

  // Fetch real-wine matches once predictions are ready
  useEffect(() => {
    const runMatch = async () => {
      if (!predictions) return;
      try {
        setMatchesLoading(true);
        const input: MatchRealInput = {
          variety: wineData.variety ?? null,
          country: wineData.country ?? null,
          region1: wineData.region1 ?? null,
          region2: wineData.region2 ?? null,
          age: wineData.age ?? null,
          price: predictions.price ? (predictions.price.weighted_lower + predictions.price.weighted_upper) / 2 : null,
          rating: predictions.rating?.predicted_rating ?? null,
          predicted: {
            flavors: predictions.flavors.slice(0, 10).map(f => ({ flavor: f.flavor, confidence: f.confidence })),
            mouthfeel: predictions.mouthfeel.slice(0, 10).map(m => ({ mouthfeel: m.mouthfeel, confidence: m.confidence })),
          },
          top_k: 5,
        };
        const res = await wineAPI.matchReal(input);
        setMatches(res.matches ?? []);
      } catch (e) {
        console.warn('match-real failed', e);
        setMatches(null);
      } finally {
        setMatchesLoading(false);
      }
    };
    runMatch();
  }, [predictions]);

  const handleRetry = () => {
    fetchPredictions();
  };

  const handleBack = () => {
    router.back();
  };

  const handleStartOver = () => {
    router.push('/');
  };

  const handleSaveToCellar = async () => {
    try {
      if (!predictions) return;
      const token = await SecureStore.getItemAsync('auth_token');
      if (!token) {
        Alert.alert('Not signed in', 'Please sign in again.');
        router.replace('/');
        return;
      }
      const payload = {
        name: `${wineData.variety} (${wineData.country}${wineData.region1 ? ', ' + wineData.region1 : ''})`,
        variety: wineData.variety,
        country: wineData.country,
        region1: wineData.region1,
        region2: wineData.region2,
        age: wineData.age,
        predicted: {
          price: predictions.price,
          rating: predictions.rating,
          flavors: predictions.flavors.slice(0, 10),
          mouthfeel: predictions.mouthfeel.slice(0, 10),
        },
        created_via: 'results',
        public: isPublic,
      };
      const res = await fetch(`${API_BASE_URL}/cellar`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      Alert.alert('Saved', isPublic ? 'Wine saved to your cellar and public list.' : 'Wine saved to your cellar.');
      router.replace('/cellar');
    } catch (e: any) {
      Alert.alert('Save failed', e?.message ?? 'Unknown error');
    }
  };

  const renderMatch = ({ item, index }: { item: MatchItem; index: number }) => {
    return (
      <View style={styles.matchCard}>
        <Text style={styles.matchRank}>#{index + 1}</Text>
        <View style={{ flex: 1 }}>
          <Text style={styles.matchName}>{item.name ?? 'Unknown wine'}</Text>
          <Text style={styles.matchMeta}>
            {item.variety ? `${item.variety} · ` : ''}
            {item.country || ''}{item.region1 ? `, ${item.region1}` : ''}{item.region2 ? `, ${item.region2}` : ''}
          </Text>
          <Text style={styles.matchMeta}>
            {item.price != null ? `$${Number(item.price).toFixed(0)}` : '—'}
            {item.rating != null ? ` · ${Number(item.rating).toFixed(0)}/100` : ''}
            {` · sim ${(item.score * 100).toFixed(1)}%`}
          </Text>
        </View>
      </View>
    );
  };

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={wineTheme.colors.primary} />
          <Text style={styles.loadingText}>Analyzing your wine...</Text>
          <Text style={styles.subLoadingText}>This may take a few moments</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (error) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.errorContainer}>
          <Text style={styles.errorTitle}>Oops!</Text>
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity style={styles.retryButton} onPress={handleRetry}>
            <Text style={styles.retryButtonText}>Try Again</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.backButton} onPress={handleBack}>
            <Text style={styles.backButtonText}>Go Back</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  if (!predictions) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>No predictions available</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <ScrollView style={{ flex: 1 }} contentContainerStyle={[styles.scrollContent, { paddingBottom: 160 }]} showsVerticalScrollIndicator={false}>
          <View style={styles.topSection}>
            <View style={styles.header}>
              <Text style={styles.title}>Wine Analysis Results</Text>
              <Text style={styles.subtitle}>
                {wineData.variety} from {wineData.country}
              </Text>
            </View>

            <View style={styles.infoGrid}>
              <View style={styles.infoCard}>
                <Text style={styles.infoLabel}>Age</Text>
                <Text style={styles.infoValue}>{wineData.age} years</Text>
              </View>
              <View style={styles.infoCard}>
                <Text style={styles.infoLabel}>Rating</Text>
                <Text style={styles.infoValue}>{predictions.rating.predicted_rating.toFixed(1)}/100</Text>
              </View>
              <View style={styles.infoCard}>
                <Text style={styles.infoLabel}>Price Range</Text>
                <Text style={styles.infoValue}>
                  ${predictions.price.weighted_lower.toFixed(0)} - ${predictions.price.weighted_upper.toFixed(0)}
                </Text>
                <Text style={styles.confidence}>{predictions.price.confidence_interval}</Text>
              </View>
              <View style={styles.infoCard}>
                <Text style={styles.infoLabel}>Location</Text>
                <Text style={styles.infoValue}>
                  {wineData.country}{wineData.region1 ? `, ${wineData.region1}` : ''}{wineData.region2 ? `, ${wineData.region2}` : ''}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.dividerHorizontal} />

          <View style={styles.bottomSection}>
            <View style={styles.bottomSplit}>
              <View style={styles.leftPane}>
                <View style={[styles.panelBox, { flex: 1 }]}> 
                  <Text style={styles.panelTitle}>Flavor Profile</Text>
                  {predictions.flavors.slice(0, 5).map((flavor, i) => (
                    <View key={`flavor-${i}`} style={styles.progressItem}>
                      <Text style={styles.progressLabel}>{flavor.flavor}</Text>
                      <Progress.Bar
                        progress={flavorProgress[i] ?? 0}
                        animated
                        animationType="timing"
                        animationConfig={{ duration: 600 }}
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
                </View>
                <View style={[styles.panelBox, { flex: 1 }]}> 
                  <Text style={styles.panelTitle}>Mouthfeel</Text>
                  {predictions.mouthfeel.slice(0, 5).map((feel, i) => (
                    <View key={`mouthfeel-${i}`} style={styles.progressItem}>
                      <Text style={styles.progressLabel}>{feel.mouthfeel}</Text>
                      <Progress.Bar
                        progress={mouthfeelProgress[i] ?? 0}
                        animated
                        animationType="timing"
                        animationConfig={{ duration: 600 }}
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
                </View>
              </View>

              <View style={styles.dividerVertical} />

              <View style={styles.rightPane}>
                <View style={[styles.panelBox, { flex: 1, alignItems: 'center', justifyContent: 'center' }]}> 
                  <Bottle color={wineTheme.colors.primary} scale={1.3} />
                </View>
              </View>
            </View>

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Closest Real Wines</Text>
              {matchesLoading && (
                <View style={{ paddingVertical: 8 }}>
                  <ActivityIndicator color={wineTheme.colors.primary} />
                </View>
              )}
              {!matchesLoading && (!matches || matches.length === 0) && (
                <Text style={styles.noMatches}>No close matches found.</Text>
              )}
              {!!matches && matches.length > 0 && (
                <FlatList
                  data={matches}
                  keyExtractor={(item, i) => `${item.index}-${i}`}
                  renderItem={renderMatch}
                  ItemSeparatorComponent={() => <View style={{ height: 8 }} />}
                  scrollEnabled={false}
                />
              )}
            </View>
          </View>
        </ScrollView>

        <View style={styles.footer}>
          <TouchableOpacity style={styles.primaryButton} onPress={handleStartOver}>
            <Text style={styles.primaryButtonText}>Analyze Another Wine</Text>
          </TouchableOpacity>
          <View style={{ flexDirection: 'row', gap: 8 }}>
            <TouchableOpacity style={[styles.secondaryButton, { flex: 1 }]} onPress={handleBack}>
              <Text style={styles.secondaryButtonText}>Go Back</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.toggleButton, { flex: 1 }]}
              onPress={() => setIsPublic((v) => !v)}
            >
              <Text style={[styles.toggleButtonText, isPublic && styles.toggleButtonTextActive]}>
                {isPublic ? 'Public' : 'Private'}
              </Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.primaryButton, { flex: 1 }]} onPress={handleSaveToCellar}>
              <Text style={styles.primaryButtonText}>{isPublic ? 'Save Public' : 'Save Private'}</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: wineTheme.colors.background,
  },
  content: {
    flex: 1,
    padding: 12,
    paddingBottom: 12,
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
    flexGrow: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  loadingText: {
    fontSize: 18,
    fontWeight: '600',
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text,
    marginTop: 16,
    textAlign: 'center',
  },
  subLoadingText: {
    fontSize: 14,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text,
    opacity: 0.7,
    marginTop: 8,
    textAlign: 'center',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorTitle: {
    fontSize: 24,
    fontWeight: '700',
    fontFamily: fonts.spaceGrotesk,
    color: wineTheme.colors.primary,
    marginBottom: 12,
  },
  errorText: {
    fontSize: 16,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text,
    textAlign: 'center',
    marginBottom: 24,
    lineHeight: 24,
  },
  retryButton: {
    backgroundColor: wineTheme.colors.primary,
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 12,
    marginBottom: 12,
  },
  retryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    fontFamily: fonts.outfit,
  },
  backButton: {
    paddingHorizontal: 24,
    paddingVertical: 12,
  },
  backButtonText: {
    color: wineTheme.colors.primary,
    fontSize: 16,
    fontWeight: '600',
    fontFamily: fonts.outfit,
  },
  header: {
    marginBottom: 10,
    alignItems: 'center',
  },
  title: {
    fontSize: 22,
    fontWeight: '700',
    fontFamily: fonts.spaceGrotesk,
    color: wineTheme.colors.text,
    textAlign: 'center',
    marginBottom: 6,
  },
  subtitle: {
    fontSize: 15,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text,
    opacity: 0.8,
    textAlign: 'center',
  },
  topSection: {
    justifyContent: 'flex-start',
  },
  dividerHorizontal: {
    height: 1,
    backgroundColor: `${wineTheme.colors.text}22`,
    marginVertical: 4,
  },
  infoGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    justifyContent: 'space-between',
  },
  infoCard: {
    backgroundColor: wineTheme.colors.surface,
    borderRadius: 12,
    padding: 10,
    flexGrow: 1,
    flexBasis: '48%',
    borderWidth: 1,
    borderColor: `${wineTheme.colors.primary}33`,
  },
  infoLabel: {
    fontSize: 12,
    color: wineTheme.colors.text,
    opacity: 0.7,
    fontFamily: fonts.outfit,
    marginBottom: 4,
  },
  infoValue: {
    fontSize: 15,
    fontWeight: '700',
    fontFamily: fonts.spaceGrotesk,
    color: wineTheme.colors.primary,
    lineHeight: 20,
    flexShrink: 1,
  },
  confidence: {
    fontSize: 14,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text,
    opacity: 0.7,
  },
  bottomSection: {
    flex: 1,
  },
  bottomSplit: {
    flex: 1,
    flexDirection: 'row',
  },
  leftPane: {
    flex: 1,
    gap: 8,
    paddingRight: 8,
  },
  rightPane: {
    flex: 1,
    paddingLeft: 8,
  },
  panelBox: {
    backgroundColor: wineTheme.colors.surface,
    borderRadius: 12,
    padding: 8,
    borderWidth: 1,
    borderColor: `${wineTheme.colors.primary}33`,
  },
  panelTitle: {
    fontSize: 14,
    fontWeight: '700',
    fontFamily: fonts.spaceGrotesk,
    color: wineTheme.colors.text,
    marginBottom: 4,
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
  dividerVertical: {
    width: 1,
    backgroundColor: `${wineTheme.colors.text}22`,
  },
  placeholderBox: {
    backgroundColor: wineTheme.colors.surface,
    borderRadius: 12,
    padding: 10,
    borderWidth: 1,
    borderColor: `${wineTheme.colors.primary}33`,
    alignItems: 'center',
    justifyContent: 'center',
  },
  placeholderTitle: {
    fontSize: 14,
    fontWeight: '700',
    fontFamily: fonts.spaceGrotesk,
    color: wineTheme.colors.text,
    marginBottom: 4,
  },
  placeholderSubtitle: {
    fontSize: 12,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text,
    opacity: 0.7,
  },
  section: {
    backgroundColor: wineTheme.colors.surface,
    borderRadius: 16,
    padding: 20,
    marginTop: 12,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    fontFamily: fonts.spaceGrotesk,
    color: wineTheme.colors.text,
    marginBottom: 16,
  },
  noMatches: {
    fontSize: 14,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text,
    opacity: 0.7,
  },
  matchCard: {
    flexDirection: 'row',
    gap: 10,
    backgroundColor: wineTheme.colors.background,
    borderRadius: 12,
    padding: 10,
    borderWidth: 1,
    borderColor: `${wineTheme.colors.primary}33`,
  },
  matchRank: {
    fontSize: 16,
    fontWeight: '700',
    color: wineTheme.colors.primary,
    fontFamily: fonts.spaceGrotesk,
    width: 28,
  },
  matchName: {
    fontSize: 15,
    fontWeight: '700',
    fontFamily: fonts.spaceGrotesk,
    color: wineTheme.colors.text,
  },
  matchMeta: {
    fontSize: 12,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text,
    opacity: 0.8,
    marginTop: 2,
  },
  actionsContainer: {
    marginTop: 6,
    gap: 6,
  },
  primaryButton: {
    backgroundColor: wineTheme.colors.primary,
    paddingVertical: 10,
    borderRadius: 12,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    fontFamily: fonts.outfit,
  },
  secondaryButton: {
    borderWidth: 2,
    borderColor: wineTheme.colors.primary,
    paddingVertical: 10,
    borderRadius: 12,
    alignItems: 'center',
    backgroundColor: 'transparent',
  },
  secondaryButtonText: {
    color: wineTheme.colors.primary,
    fontSize: 16,
    fontWeight: '600',
    fontFamily: fonts.outfit,
  },
  toggleButton: {
    borderWidth: 2,
    borderColor: wineTheme.colors.primary,
    paddingVertical: 10,
    borderRadius: 12,
    alignItems: 'center',
    backgroundColor: wineTheme.colors.surface,
  },
  toggleButtonText: {
    color: wineTheme.colors.text,
    fontSize: 16,
    fontWeight: '600',
    fontFamily: fonts.outfit,
  },
  toggleButtonTextActive: {
    color: wineTheme.colors.primary,
  },
  footer: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: wineTheme.colors.surface,
    padding: 12,
    borderTopWidth: 1,
    borderTopColor: `${wineTheme.colors.primary}33`,
    gap: 8,
  },
}); 