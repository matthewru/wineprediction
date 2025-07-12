import { useState } from 'react';
import { View, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { ThemedText } from '../components/ThemedText';
import { WineButton } from '../components/WineButton';
import GestureRecognizer from 'react-native-swipe-gestures';
import ScreenLayout from '../components/ScreenLayout';
import CountryMap from '@/components/CountryMap';

export default function CountryScreen() {
  const router = useRouter();
  const [selectedIndex, setSelectedIndex] = useState(0);

  const countryList = [
    'US', 'Italy', 'France', 'Spain', 'Portugal',
    'Chile', 'Argentina', 'Austria', 'Germany', 'Australia'
  ];

  const handlePrev = () => {
    setSelectedIndex((prev) =>
      prev > 0 ? prev - 1 : countryList.length - 1
    );
  };

  const handleNext = () => {
    setSelectedIndex((prev) =>
      prev < countryList.length - 1 ? prev + 1 : 0
    );
  };

  const handleContinue = () => {
    router.push('/primary-region');
  };

  return (
    <ScreenLayout onContinue={handleContinue}>
      <GestureRecognizer
        onSwipeLeft={handleNext}
        onSwipeRight={handlePrev}
        config={{
          velocityThreshold: 0.3,
          directionalOffsetThreshold: 80,
        }}
      >
        <View style={styles.pickerWrapper}>
          <View style={styles.pickerRow}>
            <WineButton
              title="←"
              onPress={handlePrev}
              variant="secondary"
              style={styles.navButton}
            />
            <ThemedText type="subtitle">
              {countryList[selectedIndex]}
            </ThemedText>
            <WineButton
              title="→"
              onPress={handleNext}
              variant="secondary"
              style={styles.navButton}
            />
          </View>
        </View>
      </GestureRecognizer>

      <View style={styles.mapContainer}>
          <CountryMap country={countryList[selectedIndex]} />
      </View>
    </ScreenLayout>
  );
}

const styles = StyleSheet.create({
  pickerWrapper: {
    height: 60,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
  },
  pickerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    width: '100%',
  },
  navButton: {
    minWidth: 50,
    minHeight: 50,
  },
  mapContainer: {
    alignItems: 'center',
  },
  mapPlaceholder: {
    width: '100%',
    height: 250, // same height as the map for layout consistency
  },
});
