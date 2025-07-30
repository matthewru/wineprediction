import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, ActivityIndicator, Alert, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useWine } from '../context/WineContext';
import { wineAPI, type AllPredictionsResponse, type WinePredictionInput } from '../services/api';
import { wineTheme } from '../constants/Colors';
import { fonts } from '../constants/Fonts';

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
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.title}>Wine Analysis Results</Text>
          <Text style={styles.subtitle}>
            {wineData.variety} from {wineData.country}
          </Text>
        </View>

        {/* Price Prediction */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>💰 Price Range</Text>
          <View style={styles.priceContainer}>
            <Text style={styles.priceRange}>
              ${predictions.price.weighted_lower.toFixed(0)} - ${predictions.price.weighted_upper.toFixed(0)}
            </Text>
            <Text style={styles.confidence}>{predictions.price.confidence_interval}</Text>
          </View>
        </View>

        {/* Rating Prediction */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>⭐ Quality Rating</Text>
          <View style={styles.ratingContainer}>
            <Text style={styles.rating}>{predictions.rating.predicted_rating.toFixed(1)}/100</Text>
            <Text style={styles.ratingDescription}>
              {predictions.rating.predicted_rating >= 90 ? 'Exceptional' :
               predictions.rating.predicted_rating >= 85 ? 'Very Good' :
               predictions.rating.predicted_rating >= 80 ? 'Good' : 'Average'}
            </Text>
          </View>
        </View>

        {/* Flavor Profile */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🍇 Flavor Profile</Text>
          <View style={styles.tagsContainer}>
            {predictions.flavors.map((flavor, index) => (
              <View key={index} style={styles.tag}>
                <Text style={styles.tagText}>{flavor.flavor}</Text>
                <Text style={styles.tagConfidence}>{(flavor.confidence * 100).toFixed(0)}%</Text>
              </View>
            ))}
          </View>
        </View>

        {/* Mouthfeel */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>👄 Mouthfeel</Text>
          <View style={styles.tagsContainer}>
            {predictions.mouthfeel.map((feel, index) => (
              <View key={index} style={styles.tag}>
                <Text style={styles.tagText}>{feel.mouthfeel}</Text>
                <Text style={styles.tagConfidence}>{(feel.confidence * 100).toFixed(0)}%</Text>
              </View>
            ))}
          </View>
        </View>

        {/* Actions */}
        <View style={styles.actionsContainer}>
          <TouchableOpacity style={styles.primaryButton} onPress={handleStartOver}>
            <Text style={styles.primaryButtonText}>Analyze Another Wine</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.secondaryButton} onPress={handleBack}>
            <Text style={styles.secondaryButtonText}>Go Back</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: wineTheme.colors.background,
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
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
    marginBottom: 32,
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    fontFamily: fonts.spaceGrotesk,
    color: wineTheme.colors.text,
    textAlign: 'center',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text,
    opacity: 0.8,
    textAlign: 'center',
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
    marginTop: 24,
    gap: 12,
  },
  primaryButton: {
    backgroundColor: wineTheme.colors.primary,
    paddingVertical: 16,
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
    paddingVertical: 16,
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