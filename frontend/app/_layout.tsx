import { Stack } from 'expo-router';
import { WineProvider } from '../context/WineContext';
import { useFonts as useGrotesk, SpaceGrotesk_500Medium } from '@expo-google-fonts/space-grotesk';
import { useFonts as useOutfit, Outfit_400Regular } from '@expo-google-fonts/outfit';
import * as SplashScreen from 'expo-splash-screen';
import { useEffect } from 'react';
import { View, Text } from 'react-native';
import { useFonts } from 'expo-font';
import { GestureHandlerRootView } from 'react-native-gesture-handler';

// Keep the splash screen visible while we fetch resources
SplashScreen.preventAutoHideAsync();

export default function Layout() {
  const [groteskLoaded] = useGrotesk({
    SpaceGrotesk_500Medium,
  });

  const [outfitLoaded] = useOutfit({
    Outfit_400Regular,
  });

  const [localFontsLoaded] = useFonts({
    'SpaceMono-Regular': require('../assets/fonts/SpaceMono-Regular.ttf'),
  });

  useEffect(() => {
    if (groteskLoaded && outfitLoaded && localFontsLoaded) {
      SplashScreen.hideAsync();
    }
  }, [groteskLoaded, outfitLoaded, localFontsLoaded]);

  if (!groteskLoaded || !outfitLoaded || !localFontsLoaded) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <Text>Loading fonts...</Text>
      </View>
    );
  }
  
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <WineProvider>
        <Stack screenOptions={{ headerShown: false }} />
      </WineProvider>
    </GestureHandlerRootView>
  );
}
