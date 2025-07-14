import { useRef, useState, useEffect } from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';
import { useRouter } from 'expo-router';
import { ThemedText } from '../components/ThemedText';
import { WineButton } from '../components/WineButton';
import ScreenLayout from '../components/ScreenLayout';
import CountryMap from '@/components/CountryMap';
import { useWine } from '../context/WineContext';
import type { ICarouselInstance } from 'react-native-reanimated-carousel';
import Carousel from 'react-native-reanimated-carousel';

const { width: screenWidth } = Dimensions.get('window');

// Import regions data
const regionsData = require('../data/regions.json');

const countryList = [
  'US', 'Italy', 'France', 'Spain', 'Portugal',
  'Chile', 'Argentina', 'Austria', 'Germany', 'Australia',
];

// Country flag emojis mapping
const countryFlags: Record<string, string> = {
  'US': '🇺🇸',
  'Italy': '🇮🇹',
  'France': '🇫🇷',
  'Spain': '🇪🇸',
  'Portugal': '🇵🇹',
  'Chile': '🇨🇱',
  'Argentina': '🇦🇷',
  'Austria': '🇦🇹',
  'Germany': '🇩🇪',
  'Australia': '🇦🇺',
};

export default function PrimaryRegionScreen() {
  const router = useRouter();
  const { wineData, setWineData } = useWine();
  const [selectedRegionIndex, setSelectedRegionIndex] = useState(0);
  const carouselRef = useRef<ICarouselInstance>(null);

  // Get the current country from context
  const currentCountry = wineData.country || 'US';
  const countryIndex = countryList.indexOf(currentCountry);
  
  // Get regions for the current country
  const regionsForCountry = regionsData[currentCountry] ? Object.keys(regionsData[currentCountry]) : [];
  const regionList = regionsForCountry.length > 0 ? regionsForCountry : ['No regions available'];

  const handlePrev = () => {
    carouselRef.current?.prev();
  };
  
  const handleNext = () => {
    carouselRef.current?.next();
  };

  const handleContinue = () => {
    // Save the selected primary region to the context
    setWineData(prev => ({
      ...prev,
      region1: regionList[selectedRegionIndex]
    }));
    router.push('/secondary-region');
  };

  // Reset region index when country changes
  useEffect(() => {
    setSelectedRegionIndex(0);
  }, [currentCountry]);

  return (
    <ScreenLayout onContinue={handleContinue}>
      {/* Country Picker (Read-only) */}
      <View style={styles.pickerWrapper}>
        <View style={styles.pickerRow}>
          <ThemedText type="subtitle" style={styles.flagEmoji}>{countryFlags[currentCountry]}</ThemedText>
          <ThemedText type="subtitle">{currentCountry}</ThemedText>
          <ThemedText type="subtitle" style={styles.flagEmoji}>{countryFlags[currentCountry]}</ThemedText>
        </View>
      </View>

      {/* Primary Region Picker */}
      <View style={styles.pickerWrapper}>
        <View style={styles.pickerRow}>
          <WineButton title="←" onPress={handlePrev} variant="secondary" style={styles.navButton} />
          <ThemedText type="subtitle">{regionList[selectedRegionIndex]}</ThemedText>
          <WineButton title="→" onPress={handleNext} variant="secondary" style={styles.navButton} />
        </View>
      </View>

      <Carousel
        ref={carouselRef}
        loop
        width={screenWidth}
        height={600} // 300 map + spacing
        data={regionList}
        scrollAnimationDuration={250}
        onSnapToItem={(index) => setSelectedRegionIndex(index)}
        renderItem={({ item }) => (
          <View style={styles.regionSlide}>
            <CountryMap country={currentCountry} highlightedRegion={regionList[selectedRegionIndex]} />
          </View>
        )}
      />
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
  disabledButton: {
    opacity: 0.5,
  },
  flagEmoji: {
    fontSize: 24,
  },
  regionSlide: {
    width: screenWidth,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
