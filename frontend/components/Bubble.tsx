import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, ImageSourcePropType } from 'react-native';
import { fonts } from '../constants/Fonts';
import { wineTheme } from '../constants/Colors';

export interface BubbleData {
  id: string;
  text: string;
  color: string;
  size: 'small' | 'medium' | 'medium-large' | 'large';
  icon?: ImageSourcePropType;
}

interface BubbleProps {
  data: BubbleData;
  selectedVariety: string | null;
  onPress?: () => void;
}

const Bubble: React.FC<BubbleProps> = ({ data, selectedVariety, onPress }) => {
  const [isSelected, setIsSelected] = useState(false);

  // Only update selection state when it actually changes for this specific bubble
  useEffect(() => {
    const shouldBeSelected = selectedVariety === data.text;
    if (shouldBeSelected !== isSelected) {
      setIsSelected(shouldBeSelected);
    }
  }, [selectedVariety, data.text, isSelected]);

  // Memoize size styles to prevent recreation
  const sizeStyles = React.useMemo(() => {
    switch (data.size) {
      case 'small':
        return { width: 80, height: 80, borderRadius: 40 };
      case 'medium':
        return { width: 120, height: 120, borderRadius: 60 };
      case 'medium-large':
        return { width: 140, height: 140, borderRadius: 70 };
      case 'large':
        return { width: 160, height: 160, borderRadius: 80 };
      default:
        return { width: 120, height: 120, borderRadius: 60 };
    }
  }, [data.size]);

  // Memoize icon size
  const iconSize = React.useMemo(() => {
    switch (data.size) {
      case 'small':
        return { width: 64, height: 64 };
      case 'medium':
        return { width: 96, height: 96 };
      case 'medium-large':
        return { width: 112, height: 112 };
      case 'large':
        return { width: 128, height: 128 };
      default:
        return { width: 96, height: 96 };
    }
  }, [data.size]);

  const BubbleContent = (
    <View style={[
      styles.bubble, 
      sizeStyles, 
      {
        backgroundColor: wineTheme.colors.background,
        borderWidth: isSelected ? 4 : 2,
        borderColor: isSelected ? wineTheme.colors.primary : wineTheme.colors.surface,
      }
    ]}>
      {data.icon ? (
        <Image 
          source={data.icon} 
          style={[styles.grapeIcon, iconSize]}
          resizeMode="contain"
        />
      ) : (
        <View style={[styles.grapeIcon, iconSize, { backgroundColor: 'rgba(255,255,255,0.3)' }]} />
      )}
      <Text 
        style={[
          styles.text, 
          { 
            color: isSelected ? wineTheme.colors.primary : wineTheme.colors.text,
            fontWeight: isSelected ? '700' : '600'
          }
        ]} 
        numberOfLines={2} 
        ellipsizeMode="tail"
      >
        {data.text}
      </Text>
    </View>
  );

  if (onPress) {
    return (
      <TouchableOpacity onPress={onPress} activeOpacity={0.7}>
        {BubbleContent}
      </TouchableOpacity>
    );
  }

  return BubbleContent;
};

Bubble.displayName = 'Bubble';

const styles = StyleSheet.create({
  bubble: {
    justifyContent: 'center',
    alignItems: 'center',
    marginHorizontal: 10,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    padding: 8,
  },
  grapeIcon: {
    marginBottom: 4,
  },
  text: {
    fontSize: 12,
    fontWeight: '600',
    fontFamily: fonts.outfit,
    textAlign: 'center',
    paddingHorizontal: 4,
    textShadowColor: 'rgba(0, 0, 0, 0.75)',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 2,
  },
});

export default Bubble;
