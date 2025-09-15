import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Alert } from 'react-native';
import { wineAPI, API_BASE_URL } from '../services/api';
import * as SecureStore from 'expo-secure-store';

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

  const testProfile = async () => {
    try {
      addResult('Testing profile fetch...');
      const token = await SecureStore.getItemAsync('auth_token');
      if (!token) {
        addResult('❌ No auth_token found. Sign in first.');
        return;
      }
      const res = await fetch(`${API_BASE_URL}/me/profile`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const txt = await res.text();
      addResult(`Profile status ${res.status}: ${txt}`);
    } catch (error) {
      addResult(`❌ Profile fetch failed: ${error}`);
    }
  };

  const testProfileFull = async () => {
    try {
      addResult('Testing full profile vector...');
      const token = await SecureStore.getItemAsync('auth_token');
      if (!token) {
        addResult('❌ No auth_token found. Sign in first.');
        return;
      }
      const res = await fetch(`${API_BASE_URL}/me/profile?full=1`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      const vecLen = Array.isArray(data.profile_vec) ? data.profile_vec.length : 0;
      addResult(`Full profile: has_profile=${data.has_profile} dim=${data.profile_dim} len=${vecLen} norm=${data.norm}`);
      if (vecLen > 0) {
        addResult(`First 8 vals: ${(data.profile_vec as number[]).slice(0,8).map((v:number)=>v.toFixed(4)).join(', ')}`);
      }
    } catch (error) {
      addResult(`❌ Full profile fetch failed: ${error}`);
    }
  };

  const testProfileWithBottles = async () => {
    try {
      addResult('Testing profile with bottles...');
      const token = await SecureStore.getItemAsync('auth_token');
      if (!token) {
        addResult('❌ No auth_token found. Sign in first.');
        return;
      }
      const res = await fetch(`${API_BASE_URL}/me/profile?full=1&include_bottles=1&limit=50`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      const count = Array.isArray(data.bottles) ? data.bottles.length : 0;
      addResult(`Bottles fetched: ${count}. has_profile=${data.has_profile} dim=${data.profile_dim}`);
      if (count > 0) {
        const names = (data.bottles as any[]).slice(0, 3).map(b => b.name || 'Bottle').join(' | ');
        addResult(`Sample bottles: ${names}`);
      }
    } catch (error) {
      addResult(`❌ Profile+bottles fetch failed: ${error}`);
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
          onPress={testProfile}
          disabled={testing}
        >
          <Text style={styles.secondaryButtonText}>👤 Check Profile</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={[styles.button, styles.secondaryButton]} 
          onPress={testProfileFull}
          disabled={testing}
        >
          <Text style={styles.secondaryButtonText}>📈 Profile (Full Vector)</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={[styles.button, styles.secondaryButton]} 
          onPress={testProfileWithBottles}
          disabled={testing}
        >
          <Text style={styles.secondaryButtonText}>🍷 Profile + Bottles</Text>
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