import React, { useCallback } from 'react';
import { View, StyleSheet, Text, Dimensions } from 'react-native';
import { PanGestureHandler, PanGestureHandlerGestureEvent } from 'react-native-gesture-handler';
import Animated, {
  useAnimatedGestureHandler,
  useAnimatedStyle,
  useSharedValue,
  runOnJS,
  clamp,
} from 'react-native-reanimated';
import { wineTheme } from '../constants/Colors';
import { fonts } from '../constants/Fonts';

const { width: screenWidth } = Dimensions.get('window');

interface SliderProps {
  min?: number;
  max?: number;
  step?: number;
  value?: number;
  onValueChange?: (value: number) => void;
  disabled?: boolean;
  label?: string;
  width?: number;
}

const Slider: React.FC<SliderProps> = ({
  min = 0,
  max = 100,
  step = 1,
  value = 50,
  onValueChange,
  disabled = false,
  label,
  width = screenWidth - 40,
}) => {
  const SLIDER_WIDTH = width - 64;
  const THUMB_SIZE = 20;
  
  // Safe initialization with validation
  const safeValue = React.useMemo(() => {
    if (typeof value !== 'number' || isNaN(value)) return min;
    return clamp(value, min, max);
  }, [value, min, max]);
  
  // Initialize position based on the safe value
  const position = useSharedValue(
    ((safeValue - min) / (max - min)) * SLIDER_WIDTH
  );

  const updateValue = useCallback((newPosition: number) => {
    'worklet';
    if (isNaN(newPosition) || !isFinite(newPosition)) return;
    
    const percentage = newPosition / SLIDER_WIDTH;
    const rawValue = min + percentage * (max - min);
    const steppedValue = Math.round(rawValue / step) * step;
    const clampedValue = clamp(steppedValue, min, max);
    
    if (onValueChange && !isNaN(clampedValue) && isFinite(clampedValue)) {
      runOnJS(onValueChange)(clampedValue);
    }
  }, [min, max, step, SLIDER_WIDTH, onValueChange]);

  const gestureHandler = useAnimatedGestureHandler<PanGestureHandlerGestureEvent, { startX: number }>({
    onStart: (_, context) => {
      context.startX = position.value;
    },
    onActive: (event, context) => {
      if (disabled) return;
      
      const newPosition = clamp(
        context.startX + event.translationX,
        0,
        SLIDER_WIDTH
      );
      position.value = newPosition;
      updateValue(newPosition);
    },
  });

  const trackFillStyle = useAnimatedStyle(() => {
    return {
      width: Math.max(0, Math.min(position.value, SLIDER_WIDTH)),
    };
  });

  const thumbStyle = useAnimatedStyle(() => {
    return {
      transform: [{ translateX: clamp(position.value, 0, SLIDER_WIDTH) }],
    };
  });

  // Update position when value prop changes with validation
  React.useEffect(() => {
    const validValue = typeof safeValue === 'number' && !isNaN(safeValue) ? safeValue : min;
    const newPosition = ((validValue - min) / (max - min)) * SLIDER_WIDTH;
    
    if (!isNaN(newPosition) && isFinite(newPosition)) {
      position.value = clamp(newPosition, 0, SLIDER_WIDTH);
    }
  }, [safeValue, min, max, SLIDER_WIDTH, position]);

  return (
    <View style={[styles.container, { width }]}>
      {label && (
        <View style={[styles.labelContainer, { width: SLIDER_WIDTH + 32 }]}>
          <Text style={styles.minMaxLabel}>{min}</Text>
          <Text style={styles.label}>{label}</Text>
          <Text style={styles.minMaxLabel}>{max}</Text>
        </View>
      )}
      
      <View style={[styles.sliderContainer, { width: SLIDER_WIDTH + 32 }]}>
        <View style={[styles.trackContainer, { width: SLIDER_WIDTH }]}>
          <View style={[styles.track, { width: SLIDER_WIDTH }]}>
            <Animated.View style={[styles.trackFill, trackFillStyle]} />
            <PanGestureHandler onGestureEvent={gestureHandler} enabled={!disabled}>
              <Animated.View style={[styles.thumb, thumbStyle]} />
            </PanGestureHandler>
          </View>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    paddingVertical: 12,
    paddingHorizontal: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    fontFamily: fonts.spaceGrotesk,
    color: wineTheme.colors.text,
    textAlign: 'center',
    flex: 1,
  },
  labelContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
    paddingHorizontal: 16,
  },
  minMaxLabel: {
    fontSize: 11,
    fontFamily: fonts.spaceMono,
    color: wineTheme.colors.text,
    fontWeight: '500',
  },
  sliderContainer: {
    height: 32,
    justifyContent: 'center',
    paddingHorizontal: 16,
    alignSelf: 'center',
  },
  trackContainer: {
    height: 32,
    justifyContent: 'center',
    alignSelf: 'center',
  },
  track: {
    height: 4,
    backgroundColor: wineTheme.colors.surface,
    borderRadius: 2,
    justifyContent: 'center',
    alignItems: 'flex-start',
  },
  trackFill: {
    height: '100%',
    backgroundColor: wineTheme.colors.primary,
    borderRadius: 2,
  },
  thumb: {
    position: 'absolute',
    width: 20,
    height: 20,
    backgroundColor: wineTheme.colors.primary,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#FFFFFF',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
    top: -8,
    left: -10,
  },
});

export default Slider; 