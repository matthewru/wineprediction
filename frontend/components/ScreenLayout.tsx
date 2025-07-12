import { View, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { WineButton } from './WineButton';
import { wineTheme } from '@/constants/Colors';

export default function ScreenLayout({
  children,
  onContinue,
}: {
  children: React.ReactNode;
  onContinue: () => void;
}) {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>{children}</View>
      <View style={styles.actions}>
        <WineButton title="Continue" onPress={onContinue} style={styles.continueButton} />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: wineTheme.colors.background,
    paddingHorizontal: 20,
    paddingTop: 20,
  },
  content: {
    flex: 1,
  },
  actions: {
    position: 'absolute',
    bottom: 50,
    left: 20,
    right: 20,
    alignItems: 'center',
  },
  continueButton: {
    width: '100%',
    height: 60,
    borderRadius: 12,
  },
});
