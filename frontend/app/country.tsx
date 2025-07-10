import { useState } from 'react';
import { View, TouchableOpacity, Image, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { wineTheme } from '../constants/Colors';
import { WineButton } from '../components/WineButton';
import { ThemedText } from '../components/ThemedText';

export default function CountryScreen() {
  const router = useRouter();
  const [selectedIndex, setSelectedIndex] = useState(0);

  const countryList = ['France', 'Italy', 'Spain', 'United States', 'Argentina'];
  //   const countryImages: { [key: number]: any } = {
  //     0: require('../assets/images/france.jpg'),
  //     1: require('../assets/images/italy.jpg'),
  //     2: require('../assets/images/spain.jpg'),
  //     3: require('../assets/images/usa.jpg'),
  //     4: require('../assets/images/argentina.jpg'),
  //   };

  const handlePrev = () => {
    setSelectedIndex((prev) => (prev > 0 ? prev - 1 : countryList.length - 1));
  };

  const handleNext = () => {
    setSelectedIndex((prev) => (prev < countryList.length - 1 ? prev + 1 : 0));
  };

  const handleContinue = () => {
    router.push('/primary-region');
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <WineButton 
          title="←" 
          onPress={handlePrev}
          variant="secondary"
          style={styles.navButton}
        />

        <ThemedText type="subtitle">{countryList[selectedIndex]}</ThemedText>

        <WineButton 
          title="→" 
          onPress={handleNext}
          variant="secondary"
          style={styles.navButton}
        />
      </View>

      {/* <Image source={countryImages[selectedIndex]} style={styles.image} /> */}

      <View style={styles.actions}>
        <WineButton title="Continue" onPress={handleContinue} />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: wineTheme.colors.background,
    padding: 20,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  navButton: {
    minWidth: 50,
    minHeight: 50,
  },
  image: {
    width: '100%',
    height: 200,
    borderRadius: 10,
    marginBottom: 20,
  },
  actions: {
    alignItems: 'center',
    gap: 15,
  },
});
