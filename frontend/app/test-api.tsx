import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Alert } from 'react-native';
import { wineAPI } from '../services/api';

export default function TestApiScreen() {
  const [testing, setTesting] = useState(false);
  const [results, setResults] = useState<string[]>([]);

  const addResult = (message: string) => {
    setResults(prev => [...prev, `${new Date().toLocaleTimeString()}: ${message}`]);
  };

  const testHealthCheck = async () => {
    try {
      addResult('Testing health check...');
      const health = await wineAPI.healthCheck();
      addResult(`✅ Health check successful: ${JSON.stringify(health)}`);
    } catch (error) {
      addResult(`❌ Health check failed: ${error}`);
    }
  };

  const testPrediction = async () => {
    try {
      addResult('Testing wine prediction...');
      const testInput = {
        variety: 'Pinot Noir',
        country: 'US',
        province: 'California',
        age: 5,
        region_hierarchy: 'US > California > Napa Valley'
      };
      
      const prediction = await wineAPI.predictAll(testInput);
      addResult(`✅ Prediction successful!`);
      addResult(`Price: $${prediction.price.weighted_lower}-$${prediction.price.weighted_upper}`);
      addResult(`Rating: ${prediction.rating.predicted_rating.toFixed(1)}/100`);
      addResult(`Flavors: ${prediction.flavors.map(f => f.flavor).join(', ')}`);
    } catch (error) {
      addResult(`❌ Prediction failed: ${error}`);
    }
  };

  const runAllTests = async () => {
    setTesting(true);
    setResults([]);
    
    addResult('🔄 Starting API tests...');
    
    await testHealthCheck();
    await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
    await testPrediction();
    
    addResult('🏁 Tests completed!');
    setTesting(false);
  };

  const clearResults = () => {
    setResults([]);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>API Connection Test</Text>
      
      <View style={styles.buttonContainer}>
        <TouchableOpacity 
          style={[styles.button, styles.primaryButton]} 
          onPress={runAllTests}
          disabled={testing}
        >
          <Text style={styles.buttonText}>
            {testing ? '🔄 Testing...' : '🧪 Run All Tests'}
          </Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.button, styles.secondaryButton]} 
          onPress={testHealthCheck}
          disabled={testing}
        >
          <Text style={styles.secondaryButtonText}>❤️ Health Check</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.button, styles.secondaryButton]} 
          onPress={clearResults}
        >
          <Text style={styles.secondaryButtonText}>🗑️ Clear Results</Text>
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.resultsContainer}>
        <Text style={styles.resultsTitle}>Test Results:</Text>
        {results.length === 0 ? (
          <Text style={styles.noResults}>No tests run yet. Tap "Run All Tests" to start.</Text>
        ) : (
          results.map((result, index) => (
            <Text key={index} style={styles.resultText}>
              {result}
            </Text>
          ))
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#F5F5F5',
    paddingTop: 60, // Account for status bar
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    color: '#333',
    textAlign: 'center',
    marginBottom: 30,
  },
  buttonContainer: {
    gap: 12,
    marginBottom: 20,
  },
  button: {
    paddingVertical: 16,
    paddingHorizontal: 20,
    borderRadius: 12,
    alignItems: 'center',
  },
  primaryButton: {
    backgroundColor: '#8B4A6B',
  },
  secondaryButton: {
    backgroundColor: 'white',
    borderWidth: 2,
    borderColor: '#8B4A6B',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  secondaryButtonText: {
    color: '#8B4A6B',
    fontSize: 16,
    fontWeight: '600',
  },
  resultsContainer: {
    flex: 1,
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    maxHeight: '60%',
  },
  resultsTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  noResults: {
    fontSize: 14,
    color: '#666',
    fontStyle: 'italic',
    textAlign: 'center',
    marginTop: 20,
  },
  resultText: {
    fontSize: 12,
    color: '#333',
    marginBottom: 6,
    lineHeight: 18,
    fontFamily: 'monospace',
  },
}); 