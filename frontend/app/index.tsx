import { View } from 'react-native';
import { useRouter } from 'expo-router';
import { wineTheme } from '../constants/Colors';
import { WineButton } from '../components/WineButton';
import { ThemedText } from '../components/ThemedText';

export default function HomeScreen() {
  const router = useRouter();

  return (
    <View style={{ 
      flex: 1, 
      justifyContent: 'center', 
      alignItems: 'center',
      backgroundColor: wineTheme.colors.background
    }}>
      <ThemedText type="title">
        Wine Customizer 🍷
      </ThemedText>
      <WineButton
        title="Start Creating"
        onPress={() => router.push('/country')}
        style={{ marginTop: 30 }}
      />
    </View>
  );
}
