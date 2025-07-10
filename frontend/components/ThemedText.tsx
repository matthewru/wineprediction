import { StyleSheet, Text, type TextProps } from 'react-native';
import { wineTheme } from '../constants/Colors';
import { fonts } from '../constants/Fonts';

export type ThemedTextProps = TextProps & {
  lightColor?: string;
  darkColor?: string;
  type?: 'default' | 'title' | 'defaultSemiBold' | 'subtitle' | 'link';
  fontFamily?: keyof typeof fonts;
};

export function ThemedText({
  style,
  lightColor,
  darkColor,
  type = 'default',
  fontFamily,
  ...rest
}: ThemedTextProps) {
  const color = wineTheme.colors.text;
  
  // Determine font family based on type if not explicitly provided
  let defaultFontFamily: string = fonts.outfit; // Default to Outfit for body text
  if (type === 'title' || type === 'subtitle') {
    defaultFontFamily = fonts.spaceGrotesk; // Use SpaceGrotesk for headers
  }
  
  const finalFontFamily = fontFamily ? fonts[fontFamily] : defaultFontFamily;

  return (
    <Text
      style={[
        { color, fontFamily: finalFontFamily },
        type === 'default' ? styles.default : undefined,
        type === 'title' ? styles.title : undefined,
        type === 'defaultSemiBold' ? styles.defaultSemiBold : undefined,
        type === 'subtitle' ? styles.subtitle : undefined,
        type === 'link' ? styles.link : undefined,
        style,
      ]}
      {...rest}
    />
  );
}

const styles = StyleSheet.create({
  default: {
    fontSize: 16,
    lineHeight: 24,
  },
  defaultSemiBold: {
    fontSize: 16,
    lineHeight: 24,
    fontWeight: '600',
  },
  title: {
    fontSize: 32,
    fontWeight: '900',
    lineHeight: 32,
  },
  subtitle: {
    fontSize: 20,
    fontWeight: '800',
  },
  link: {
    lineHeight: 30,
    fontSize: 16,
    color: wineTheme.colors.primary,
  },
});
