import { View, type ViewProps } from 'react-native';
import { wineTheme } from '../constants/Colors';

export type ThemedViewProps = ViewProps & {
  lightColor?: string;
  darkColor?: string;
};

export function ThemedView({ style, lightColor, darkColor, ...otherProps }: ThemedViewProps) {
  const backgroundColor = wineTheme.colors.background;

  return <View style={[{ backgroundColor }, style]} {...otherProps} />;
}
