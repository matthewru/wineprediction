import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useWine } from '../context/WineContext';
import { WineButton } from '../components/WineButton';

export default function WineFeaturesScreen() {
  const router = useRouter();
  const { wineData } = useWine();
  const { country, region1, region2, variety, age } = wineData;

  const handleBack = () => {
    router.push('/country');
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.title}>Wine Features</Text>
          <Text style={styles.subtitle}>Select additional wine characteristics</Text>
        </View>

        <View style={styles.content}>
          <Text style={styles.sectionTitle}>Selected Values:</Text>
          <View style={styles.selectedValues}>
            <Text style={styles.valueText}>Country: {country || 'Not selected'}</Text>
            <Text style={styles.valueText}>Primary Region: {region1 || 'Not selected'}</Text>
            <Text style={styles.valueText}>Secondary Region: {region2 || 'Not selected'}</Text>
            <Text style={styles.valueText}>Variety: {variety || 'Not selected'}</Text>
            <Text style={styles.valueText}>Age: {age || 'Not selected'}</Text>
          </View>

          <View style={styles.placeholder}>
            <Text style={styles.placeholderText}>Wine Features Selection</Text>
            <Text style={styles.placeholderSubtext}>Coming soon...</Text>
          </View>
        </View>
      </ScrollView>

      <View style={styles.actions}>
        <WineButton 
          title="Back to Geography" 
          onPress={handleBack} 
          variant="secondary" 
          style={styles.backButton} 
        />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  scrollContent: {
    flexGrow: 1,
    padding: 20,
    paddingBottom: 120, // Extra padding for the back button
  },
  header: {
    alignItems: 'center',
    marginBottom: 30,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#7f8c8d',
    textAlign: 'center',
  },
  content: {
    flex: 1,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: 15,
  },
  selectedValues: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  valueText: {
    fontSize: 16,
    color: '#34495e',
    marginBottom: 8,
  },
  placeholder: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#ffffff',
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
    color: '#2c3e50',
    marginBottom: 12,
  },
  placeholderSubtext: {
    fontSize: 16,
    color: '#7f8c8d',
    textAlign: 'center',
  },
  actions: {
    position: 'absolute',
    bottom: 50,
    left: 20,
    right: 20,
    alignItems: 'center',
  },
  backButton: {
    width: '100%',
    height: 60,
    borderRadius: 12,
  },
});
