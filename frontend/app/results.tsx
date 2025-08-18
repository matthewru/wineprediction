import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useWine } from '../context/WineContext';
import { wineAPI, type AllPredictionsResponse, type WinePredictionInput } from '../services/api';
import { wineTheme } from '../constants/Colors';
import { fonts } from '../constants/Fonts';
import * as Progress from 'react-native-progress';

export default function ResultsScreen() {
  const router = useRouter();
  const { wineData } = useWine();
  const [predictions, setPredictions] = useState<AllPredictionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  const handleRetry = () => {
    fetchPredictions();
  };

  const handleBack = () => {
    router.back();
  };

  const handleStartOver = () => {
    router.push('/');
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
                      progress={Math.max(0, Math.min(1, flavor.confidence))}
                      height={8}
                      width={null}
                      color={wineTheme.colors.primary}
                      unfilledColor={`${wineTheme.colors.text}22`}
                      borderWidth={0}
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
                      progress={Math.max(0, Math.min(1, feel.confidence))}
                      height={8}
                      width={null}
                      color={wineTheme.colors.primary}
                      unfilledColor={`${wineTheme.colors.text}22`}
                      borderWidth={0}
                      borderRadius={8}
                      style={styles.progressBar}
                    />
                  </View>
                ))}
              </View>
            </View>

            <View style={styles.dividerVertical} />

            <View style={styles.rightPane}>
              <View style={[styles.placeholderBox, { flex: 1 }]}>
                <Text style={styles.placeholderTitle}>Wine Visual</Text>
                <Text style={styles.placeholderSubtitle}>Bottle/Label preview</Text>
              </View>
            </View>
          </View>

          <View style={styles.actionsContainer}>
            <TouchableOpacity style={styles.primaryButton} onPress={handleStartOver}>
              <Text style={styles.primaryButtonText}>Analyze Another Wine</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.secondaryButton} onPress={handleBack}>
              <Text style={styles.secondaryButtonText}>Go Back</Text>
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
    padding: 10,
    borderWidth: 1,
    borderColor: `${wineTheme.colors.primary}33`,
  },
  panelTitle: {
    fontSize: 14,
    fontWeight: '700',
    fontFamily: fonts.spaceGrotesk,
    color: wineTheme.colors.text,
    marginBottom: 6,
  },
  progressItem: {
    marginBottom: 8,
  },
  progressLabel: {
    fontSize: 12,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text,
    marginBottom: 4,
  },
  progressBar: {
    width: '100%',
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
  priceContainer: {
    alignItems: 'center',
  },
  priceRange: {
    fontSize: 24,
    fontWeight: '700',
    fontFamily: fonts.spaceGrotesk,
    color: wineTheme.colors.primary,
    marginBottom: 4,
  },
  confidence: {
    fontSize: 14,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text,
    opacity: 0.7,
  },
  ratingContainer: {
    alignItems: 'center',
  },
  rating: {
    fontSize: 32,
    fontWeight: '700',
    fontFamily: fonts.spaceGrotesk,
    color: wineTheme.colors.primary,
    marginBottom: 4,
  },
  ratingDescription: {
    fontSize: 16,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text,
    fontWeight: '500',
  },
  tagsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  tag: {
    backgroundColor: wineTheme.colors.background,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    borderWidth: 1,
    borderColor: `${wineTheme.colors.primary}33`, // Add 33 for 20% opacity
  },
  tagText: {
    fontSize: 14,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text,
    fontWeight: '500',
  },
  tagConfidence: {
    fontSize: 12,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.primary,
    fontWeight: '600',
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
}); 