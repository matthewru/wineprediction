import { View, Alert, StyleSheet, Platform } from 'react-native';
import { useRouter } from 'expo-router';
import * as SecureStore from 'expo-secure-store';
import * as Google from 'expo-auth-session/providers/google';
import { makeRedirectUri } from 'expo-auth-session';
import * as WebBrowser from 'expo-web-browser';
import Constants from 'expo-constants';
import { useEffect } from 'react';
import { API_BASE_URL } from '../services/api';
import { SafeAreaView } from 'react-native-safe-area-context';
import { wineTheme } from '../constants/Colors';
import { ThemedText } from '../components/ThemedText';
import { WineButton } from '../components/WineButton';

WebBrowser.maybeCompleteAuthSession();

export default function HomeScreen() {
  const router = useRouter();

  const extra = (Constants.expoConfig?.extra as any) || {};
  const IOS_NATIVE_REDIRECT =
  `com.googleusercontent.apps.${extra.googleIosClientId.replace('.apps.googleusercontent.com','')}:/oauthredirect`;

  const redirectUri = Platform.select({
    ios: makeRedirectUri({ native: IOS_NATIVE_REDIRECT }),
    android: makeRedirectUri({ scheme: 'frontend', path: 'redirect' })
  })!;

  const [request, response, promptAsync] = Google.useIdTokenAuthRequest({
    iosClientId: extra.googleIosClientId,
    androidClientId: extra.googleAndroidClientId,
    redirectUri,
    scopes: ['openid','email','profile'],
    responseType: 'id_token',
  });


  useEffect(() => {
    if (request?.url) console.log('AUTH URL =>', decodeURIComponent(request.url));
  }, [request]);

  useEffect(() => {
    (async () => {
      if (response?.type !== 'success') return;
      const idToken =
        (response as any).authentication?.idToken ||
        (response as any).params?.id_token;
      if (!idToken) return Alert.alert('Login error', 'No id_token returned');

      try {
        const res = await fetch(`${API_BASE_URL}/auth/google`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ idToken }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        await SecureStore.setItemAsync('auth_token', data.token);
        router.replace('/cellar');
      } catch (e: any) {
        Alert.alert('Login error', e?.message ?? 'Unknown error');
      }
    })();
  }, [response]);

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.centerWrap}>
        <ThemedText type="title" style={styles.title}>Wine Customizer</ThemedText>
        <ThemedText type="subtitle" style={styles.subtitle}>Sign in to build your cellar and generate wines</ThemedText>
        <WineButton
          title="Sign in with Google"
          onPress={() => promptAsync({ useProxy: true } as any)}
          style={styles.signIn}
          disabled={!request}
        />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: wineTheme.colors.background,
    paddingHorizontal: 20,
    paddingTop: 12,
  },
  centerWrap: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
  },
  title: {
    textAlign: 'center',
  },
  subtitle: {
    textAlign: 'center',
    opacity: 0.8,
  },
  signIn: {
    marginTop: 8,
    alignSelf: 'stretch',
    height: 52,
    borderRadius: 12,
  },
});
