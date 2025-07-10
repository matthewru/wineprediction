import React from 'react';
import { TouchableOpacity, Text, StyleSheet, TouchableOpacityProps } from 'react-native';
import { wineTheme } from '../constants/Colors';
import { fonts } from '../constants/Fonts';

interface WineButtonProps extends TouchableOpacityProps {
  title: string;
  variant?: 'primary' | 'secondary';
}

export const WineButton: React.FC<WineButtonProps> = ({ 
  title, 
  variant = 'primary', 
  style, 
  ...props 
}) => {
  return (
    <TouchableOpacity
      style={[
        styles.button,
        variant === 'primary' ? styles.primary : styles.secondary,
        style
      ]}
      {...props}
    >
      <Text style={[
        styles.text,
        variant === 'primary' ? styles.primaryText : styles.secondaryText
      ]}>
        {title}
      </Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 44,
  },
  primary: {
    backgroundColor: wineTheme.colors.primary,
  },
  secondary: {
    backgroundColor: wineTheme.colors.surface,
    borderWidth: 1,
    borderColor: wineTheme.colors.primary,
  },
  text: {
    fontSize: 16,
    fontWeight: '600',
    fontFamily: fonts.outfit,
  },
  primaryText: {
    color: 'white',
  },
  secondaryText: {
    color: wineTheme.colors.primary,
  },
}); 