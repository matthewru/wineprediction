import React, { useRef, useState, useEffect } from 'react';
import { View, StyleSheet, Dimensions, SafeAreaView } from 'react-native';
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

type Step = 'country' | 'primary-region' | 'secondary-region';

export default function CountryScreen() {
  const router = useRouter();
  const { wineData, setWineData } = useWine();
  
  // Determine initial step based on existing data
  const getInitialStep = (): Step => {
    if (wineData.region2) return 'secondary-region';
    if (wineData.region1) {
      // Check if the selected primary region has real secondary regions
      const country = wineData.country;
      const primaryRegion = wineData.region1;
      if (country && primaryRegion && regionsData[country] && regionsData[country][primaryRegion]) {
        const secondaryRegions = regionsData[country][primaryRegion];
        const hasRealSecondaryRegions = secondaryRegions.filter((region: string) => region !== 'Unknown').length > 0;
        return hasRealSecondaryRegions ? 'secondary-region' : 'primary-region';
      }
      return 'primary-region';
    }
    if (wineData.country) return 'primary-region';
    return 'country';
  };

  const [currentStep, setCurrentStep] = useState<Step>(getInitialStep());
  const [selectedCountryIndex, setSelectedCountryIndex] = useState(
    wineData.country ? countryList.indexOf(wineData.country) : 0
  );
  const [selectedPrimaryRegionIndex, setSelectedPrimaryRegionIndex] = useState(0);
  const [selectedSecondaryRegionIndex, setSelectedSecondaryRegionIndex] = useState(0);
  
  const countryCarouselRef = useRef<ICarouselInstance>(null);
  const primaryRegionCarouselRef = useRef<ICarouselInstance>(null);
  const secondaryRegionCarouselRef = useRef<ICarouselInstance>(null);

  // Get current selections
  const currentCountry = countryList[selectedCountryIndex];
  const regionsForCountry = regionsData[currentCountry] ? Object.keys(regionsData[currentCountry]) : [];
  const primaryRegionList = regionsForCountry.length > 0 ? regionsForCountry : ['No regions available'];
  const currentPrimaryRegion = primaryRegionList[selectedPrimaryRegionIndex];
  
  // Get secondary regions for the selected primary region
  const secondaryRegionsForPrimary = regionsData[currentCountry] && regionsData[currentCountry][currentPrimaryRegion] 
    ? regionsData[currentCountry][currentPrimaryRegion] 
    : [];
  
  // Filter out "Unknown" secondary regions and check if there are real secondary regions
  const realSecondaryRegions = secondaryRegionsForPrimary.filter((region: string) => region !== 'Unknown');
  const hasRealSecondaryRegions = realSecondaryRegions.length > 0;
  
  // Use real secondary regions if available, otherwise use filtered list
  const secondaryRegionList = hasRealSecondaryRegions ? realSecondaryRegions : secondaryRegionsForPrimary;
  const currentSecondaryRegion = secondaryRegionList[selectedSecondaryRegionIndex];

  // Initialize region indices based on existing data
  useEffect(() => {
    if (wineData.region1 && primaryRegionList.length > 0) {
      const primaryIndex = primaryRegionList.indexOf(wineData.region1);
      if (primaryIndex !== -1) {
        setSelectedPrimaryRegionIndex(primaryIndex);
      }
    }
    if (wineData.region2 && secondaryRegionList.length > 0) {
      const secondaryIndex = secondaryRegionList.indexOf(wineData.region2);
      if (secondaryIndex !== -1) {
        setSelectedSecondaryRegionIndex(secondaryIndex);
      }
    }
  }, [wineData.region1, wineData.region2, primaryRegionList, secondaryRegionList]);

  // Navigation handlers for country step
  const handleCountryPrev = () => {
    countryCarouselRef.current?.prev();
  };
  
  const handleCountryNext = () => {
    countryCarouselRef.current?.next();
  };

  // Navigation handlers for primary region step (manual cycling, no carousel)
  const handlePrimaryRegionPrev = () => {
    setSelectedPrimaryRegionIndex((prev) => (prev - 1 + primaryRegionList.length) % primaryRegionList.length);
  };
  
  const handlePrimaryRegionNext = () => {
    setSelectedPrimaryRegionIndex((prev) => (prev + 1) % primaryRegionList.length);
  };

  // Navigation handlers for secondary region step
  const handleSecondaryRegionPrev = () => {
    secondaryRegionCarouselRef.current?.prev();
  };
  
  const handleSecondaryRegionNext = () => {
    secondaryRegionCarouselRef.current?.next();
  };

  // Step navigation handlers
  const handleContinue = () => {
    if (currentStep === 'country') {
      // Save country and move to primary region step
      setWineData(prev => ({
        ...prev,
        country: currentCountry
      }));
      setCurrentStep('primary-region');
    } else if (currentStep === 'primary-region') {
      // Save primary region
      setWineData(prev => ({
        ...prev,
        region1: currentPrimaryRegion
      }));
      
      // Check if there are real secondary regions for this primary region
      if (hasRealSecondaryRegions) {
        setCurrentStep('secondary-region');
      } else {
        // Skip secondary region step and go directly to grape selection
        router.push('/BubbleScroller');
      }
    } else if (currentStep === 'secondary-region') {
      // Save secondary region and navigate to grape selection
      setWineData(prev => ({
        ...prev,
        region2: currentSecondaryRegion
      }));
      router.push('/BubbleScroller');
    }
  };

  const handleSkip = () => {
    router.push('/BubbleScroller');
  };

  const handleBack = () => {
    if (currentStep === 'primary-region') {
      setCurrentStep('country');
    } else if (currentStep === 'secondary-region') {
      setCurrentStep('primary-region');
    }
  };

  // Reset region indices when country or primary region changes
  useEffect(() => {
    if (currentStep === 'primary-region') {
      setSelectedPrimaryRegionIndex(0);
    }
    if (currentStep === 'secondary-region') {
      setSelectedSecondaryRegionIndex(0);
    }
  }, [currentCountry, currentStep]);

  // Reset secondary region index when primary region changes
  useEffect(() => {
    setSelectedSecondaryRegionIndex(0);
  }, [currentPrimaryRegion]);

  // Sync carousels when step changes or when going back
  useEffect(() => {
    if (currentStep === 'country' && countryCarouselRef.current) {
      countryCarouselRef.current.scrollTo({ index: selectedCountryIndex, animated: false });
    }
  }, [currentStep, selectedCountryIndex]);

  useEffect(() => {
    if (currentStep === 'primary-region' && primaryRegionCarouselRef.current) {
      primaryRegionCarouselRef.current.scrollTo({ index: selectedPrimaryRegionIndex, animated: false });
    }
  }, [currentStep, selectedPrimaryRegionIndex]);

  useEffect(() => {
    if (currentStep === 'secondary-region' && secondaryRegionCarouselRef.current) {
      secondaryRegionCarouselRef.current.scrollTo({ index: selectedSecondaryRegionIndex, animated: false });
    }
  }, [currentStep, selectedSecondaryRegionIndex]);

  // Render country step
  const renderCountryStep = () => (
    <>
      <View style={styles.pickerWrapper}>
        <View style={styles.pickerRow}>
          <WineButton title="←" onPress={handleCountryPrev} variant="secondary" style={styles.navButton} />
          <ThemedText type="subtitle">{currentCountry}</ThemedText>
          <WineButton title="→" onPress={handleCountryNext} variant="secondary" style={styles.navButton} />
        </View>
      </View>

      <Carousel
        ref={countryCarouselRef}
        loop
        width={screenWidth}
        height={600}
        data={countryList}
        scrollAnimationDuration={250}
        onSnapToItem={(index) => setSelectedCountryIndex(index)}
        renderItem={({ item }) => (
          <View style={styles.countrySlide}>
            <CountryMap country={item} />
          </View>
        )}
      />
    </>
  );

  // Render primary region step (no carousel, with pulse animation)
  const renderPrimaryRegionStep = () => (
    <>
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
          <WineButton title="←" onPress={handlePrimaryRegionPrev} variant="secondary" style={styles.navButton} />
          <ThemedText type="subtitle">{currentPrimaryRegion}</ThemedText>
          <WineButton title="→" onPress={handlePrimaryRegionNext} variant="secondary" style={styles.navButton} />
        </View>
      </View>

      <View style={styles.regionSlide}>
        <CountryMap country={currentCountry} highlightedRegion={currentPrimaryRegion} />
      </View>
    </>
  );

  // Render secondary region step
  const renderSecondaryRegionStep = () => (
    <>
      {/* Country Picker (Read-only) */}
      <View style={styles.pickerWrapper}>
        <View style={styles.pickerRow}>
          <ThemedText type="subtitle" style={styles.flagEmoji}>{countryFlags[currentCountry]}</ThemedText>
          <ThemedText type="subtitle">{currentCountry}</ThemedText>
          <ThemedText type="subtitle" style={styles.flagEmoji}>{countryFlags[currentCountry]}</ThemedText>
        </View>
      </View>

      {/* Primary Region Picker (Read-only) */}
      <View style={styles.pickerWrapper}>
        <View style={styles.pickerRow}>
          <ThemedText type="subtitle" style={styles.flagEmoji}>📍</ThemedText>
          <ThemedText type="subtitle">{currentPrimaryRegion}</ThemedText>
          <ThemedText type="subtitle" style={styles.flagEmoji}>📍</ThemedText>
        </View>
      </View>

      {/* Secondary Region Picker */}
      <View style={styles.pickerWrapper}>
        <View style={styles.pickerRow}>
          <WineButton title="←" onPress={handleSecondaryRegionPrev} variant="secondary" style={styles.navButton} />
          <ThemedText type="subtitle">{currentSecondaryRegion}</ThemedText>
          <WineButton title="→" onPress={handleSecondaryRegionNext} variant="secondary" style={styles.navButton} />
        </View>
      </View>

      <Carousel
        ref={secondaryRegionCarouselRef}
        loop
        width={screenWidth}
        height={600}
        data={secondaryRegionList}
        scrollAnimationDuration={250}
        onSnapToItem={(index) => setSelectedSecondaryRegionIndex(index)}
        renderItem={({ item }) => (
          <View style={styles.regionSlide}>
            <CountryMap country={currentCountry} highlightedRegion={currentSecondaryRegion} />
          </View>
        )}
      />
    </>
  );

  // Custom ScreenLayout with dynamic buttons
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        {currentStep === 'country' && renderCountryStep()}
        {currentStep === 'primary-region' && renderPrimaryRegionStep()}
        {currentStep === 'secondary-region' && renderSecondaryRegionStep()}
      </View>
      
      <View style={styles.actions}>
        {currentStep === 'country' && (
          <WineButton title="Continue" onPress={handleContinue} style={styles.continueButton} />
        )}
        {currentStep === 'primary-region' && (
          <View style={styles.buttonRow}>
            <WineButton title="Back" onPress={handleBack} variant="secondary" style={styles.backButton} />
            <WineButton title="Skip" onPress={handleSkip} variant="secondary" style={styles.skipButton} />
            <WineButton title="Continue" onPress={handleContinue} style={styles.continueButton} />
          </View>
        )}
        {currentStep === 'secondary-region' && (
          <View style={styles.buttonRow}>
            <WineButton title="Back" onPress={handleBack} variant="secondary" style={styles.backButton} />
            <WineButton title="Skip" onPress={handleSkip} variant="secondary" style={styles.skipButton} />
            <WineButton title="Continue" onPress={handleContinue} style={styles.continueButton} />
          </View>
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
    paddingHorizontal: 40,
    paddingTop: 20,
  },
  content: {
    flex: 1,
  },
  pickerWrapper: {
    height: 60,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 10,
  },
  pickerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    width: '100%',
  },
  navButton: {
    minWidth: 60,
    minHeight: 40,
    paddingHorizontal: 12,
    paddingVertical: 8,
    marginHorizontal: 15,
  },
  flagEmoji: {
    fontSize: 28,
    marginHorizontal: 15,
    minWidth: 60,
    minHeight: 50,
    textAlign: 'center',
    textAlignVertical: 'center',
    backgroundColor: '#D2D4C8',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#8B2635',
    paddingHorizontal: 12,
    paddingVertical: 8,
    overflow: 'hidden',
  },
  countrySlide: {
    width: screenWidth,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 20,
  },
  regionSlide: {
    width: screenWidth,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 20,
  },
  actions: {
    position: 'absolute',
    bottom: 50,
    left: 20,
    right: 20,
    alignItems: 'center',
  },
  buttonRow: {
    flexDirection: 'row',
    width: '100%',
    gap: 10,
  },
  backButton: {
    flex: 1,
    height: 60,
    borderRadius: 12,
  },
  skipButton: {
    flex: 1,
    height: 60,
    borderRadius: 12,
  },
  continueButton: {
    flex: 1,
    height: 60,
    borderRadius: 12,
  },
});
