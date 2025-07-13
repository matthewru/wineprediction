import { useRef, useState } from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';
import { useRouter } from 'expo-router';
import { ThemedText } from '../components/ThemedText';
import { WineButton } from '../components/WineButton';
import ScreenLayout from '../components/ScreenLayout';
import CountryMap from '@/components/CountryMap';
import type { ICarouselInstance } from 'react-native-reanimated-carousel';

import Carousel from 'react-native-reanimated-carousel';

const { width: screenWidth } = Dimensions.get('window');
const carouselRef = useRef<ICarouselInstance>(null);

const countryList = [
  'US', 'Italy', 'France', 'Spain', 'Portugal',
  'Chile', 'Argentina', 'Austria', 'Germany', 'Australia',
];

export default function CountryScreen() {
  const router = useRouter();
  const [selectedIndex, setSelectedIndex] = useState(0);

  const handlePrev = () => {
    carouselRef.current?.prev();
  };
  
  const handleNext = () => {
    carouselRef.current?.next();
  };
  

  const handleContinue = () => {
    router.push('/primary-region');
  };

  return (
    <ScreenLayout onContinue={handleContinue}>
      <View style={styles.pickerWrapper}>
        <View style={styles.pickerRow}>
          <WineButton title="←" onPress={handlePrev} variant="secondary" style={styles.navButton} />
          <ThemedText type="subtitle">{countryList[selectedIndex]}</ThemedText>
          <WineButton title="→" onPress={handleNext} variant="secondary" style={styles.navButton} />
        </View>
      </View>

      <Carousel
        ref={carouselRef}
        loop
        width={screenWidth}
        height={600} // 300 map + spacing
        data={countryList}
        scrollAnimationDuration={250}
        onSnapToItem={(index) => setSelectedIndex(index)}
        renderItem={({ item }) => (
          <View style={styles.countrySlide}>
            <CountryMap country={item} />
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
  countrySlide: {
    width: screenWidth,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
