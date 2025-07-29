import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useWine } from '../context/WineContext';
import { WineButton } from '../components/WineButton';
import { wineTheme } from '../constants/Colors';
import { fonts } from '../constants/Fonts';

export default function ResultsScreen() {
  const router = useRouter();
  const { wineData } = useWine();
  const { country, region1, region2, variety, age } = wineData;

  const handleBack = () => {
    router.push('/BubbleScroller');
  };

  const handleStartOver = () => {
    router.push('/country');
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.title}>Wine Selection Results</Text>
          <Text style={styles.subtitle}>Your customized wine profile</Text>
        </View>

        <View style={styles.content}>
          <View style={styles.resultsCard}>
            <Text style={styles.sectionTitle}>Selected Features:</Text>
            
            <View style={styles.featureRow}>
              <Text style={styles.featureLabel}>Country:</Text>
              <Text style={styles.featureValue}>{country || 'Not selected'}</Text>
            </View>
            
            <View style={styles.featureRow}>
              <Text style={styles.featureLabel}>Primary Region:</Text>
              <Text style={styles.featureValue}>{region1 || 'Not selected'}</Text>
            </View>
            
            <View style={styles.featureRow}>
              <Text style={styles.featureLabel}>Secondary Region:</Text>
              <Text style={styles.featureValue}>{region2 || 'Not selected'}</Text>
            </View>
            
            <View style={styles.featureRow}>
              <Text style={styles.featureLabel}>Grape Variety:</Text>
              <Text style={styles.featureValue}>{variety || 'Not selected'}</Text>
            </View>
            
            <View style={styles.featureRow}>
              <Text style={styles.featureLabel}>Wine Age:</Text>
              <Text style={styles.featureValue}>{age ? `${age} years` : 'Not selected'}</Text>
            </View>
          </View>

          <View style={styles.placeholder}>
            <Text style={styles.placeholderText}>Wine Predictions</Text>
            <Text style={styles.placeholderSubtext}>Coming soon...</Text>
            <Text style={styles.placeholderDetail}>
              This is where we'll show predicted wine characteristics like flavor profiles, 
              mouthfeel, pricing, and ratings based on your selections.
            </Text>
          </View>
        </View>
      </ScrollView>

      <View style={styles.actions}>
        <WineButton 
          title="← Back to Grape Selection" 
          onPress={handleBack} 
          variant="secondary" 
          style={styles.actionButton} 
        />
        <WineButton 
          title="Start Over" 
          onPress={handleStartOver} 
          variant="primary" 
          style={styles.actionButton} 
        />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: wineTheme.colors.background,
  },
  scrollContent: {
    flexGrow: 1,
    padding: 20,
    paddingBottom: 120, // Extra padding for the action buttons
  },
  header: {
    alignItems: 'center',
    marginBottom: 30,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    fontFamily: fonts.spaceGrotesk,
    color: wineTheme.colors.text,
    marginBottom: 8,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text,
    textAlign: 'center',
  },
  content: {
    flex: 1,
  },
  resultsCard: {
    backgroundColor: wineTheme.colors.surface,
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    fontFamily: fonts.spaceGrotesk,
    color: wineTheme.colors.text,
    marginBottom: 15,
  },
  featureRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: wineTheme.colors.background,
  },
  featureLabel: {
    fontSize: 16,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text,
    fontWeight: '500',
  },
  featureValue: {
    fontSize: 16,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.primary,
    fontWeight: '600',
  },
  placeholder: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: wineTheme.colors.surface,
    borderRadius: 12,
    padding: 40,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  placeholderText: {
    fontSize: 24,
    fontWeight: '600',
    fontFamily: fonts.spaceGrotesk,
    color: wineTheme.colors.text,
    marginBottom: 12,
  },
  placeholderSubtext: {
    fontSize: 16,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text,
    textAlign: 'center',
    marginBottom: 16,
  },
  placeholderDetail: {
    fontSize: 14,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text,
    textAlign: 'center',
    lineHeight: 20,
    opacity: 0.8,
  },
  actions: {
    position: 'absolute',
    bottom: 50,
    left: 20,
    right: 20,
    flexDirection: 'row',
    gap: 10,
  },
  actionButton: {
    flex: 1,
    height: 50,
    borderRadius: 12,
  },
}); 