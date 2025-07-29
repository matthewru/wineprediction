import React, { useRef, useCallback, useState } from 'react';
import { View, StyleSheet, Dimensions, SafeAreaView, ScrollView, NativeScrollEvent, NativeSyntheticEvent, Text, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';
import { useWine } from '../context/WineContext';
import { WineButton } from '../components/WineButton';
import Bubble, { BubbleData } from '../components/Bubble';
import Slider from '../components/Slider';
import { wineTheme } from '../constants/Colors';
import { fonts } from '../constants/Fonts';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

// Wine-themed color palette based on the theme
const wineColors = {
  deepRed: wineTheme.colors.primary, // Use the primary wine color
  burgundy: '#722F37',
  darkRed: '#6B1F3A',
  mediumRed: '#A02C2F',
  lightRed: '#B73E56',
  rosé: '#E8A4A0',
  champagne: '#F4E287',
  chardonnay: '#E8E288',
  sauvignonBlanc: '#D4E157',
  riesling: '#F0F4C3',
  darkPurple: '#4A1A2C',
  lightPurple: '#8E6B8B',
  oliveGreen: '#A9DFBF',
  goldAccent: '#F7DC6F',
  neutral: wineTheme.colors.surface,
};

// Import all grape icons statically for instant loading
const grapeIcons = {
  pinotNoir: require('../assets/grapes/pinot_noir.png'),
  chardonnay: require('../assets/grapes/chardonnay.png'),
  cabernetSauvignon: require('../assets/grapes/cabernet_sauvignon.png'),
  riesling: require('../assets/grapes/riesling.png'),
  syrah: require('../assets/grapes/syrah.png'),
  sauvignonBlanc: require('../assets/grapes/sauvignon_blanc.png'),
  zinfandel: require('../assets/grapes/zinfandel.png'),
  malbec: require('../assets/grapes/malbec.png'),
  nebbiolo: require('../assets/grapes/nebbiolo.png'),
  merlot: require('../assets/grapes/merlot.png'),
  tempranillo: require('../assets/grapes/tempranillo.png'),
  sangiovese: require('../assets/grapes/sangiovese.png'),
  pinotGris: require('../assets/grapes/pinot_gris.png'),
  grunerVeltliner: require('../assets/grapes/gruner_veltliner.png'),
  cabernetFranc: require('../assets/grapes/cabernet_franc.png'),
  gewurztraminer: require('../assets/grapes/gewurztraminer.png'),
  gamay: require('../assets/grapes/gamay.png'),
  viognier: require('../assets/grapes/viognier.png'),
  carmenere: require('../assets/grapes/carmenere.png'),
  grenache: require('../assets/grapes/grenache.png'),
  petiteSirah: require('../assets/grapes/petite_sirah.png'),
  barbera: require('../assets/grapes/barbera.png'),
  cheninBlanc: require('../assets/grapes/chenin_blanc.png'),
  glera: require('../assets/grapes/glera.png'),
  sangioveseGrosso: require('../assets/grapes/sangiovese_grosso.png'),
  neroDAvola: require('../assets/grapes/nero_d\'avola.png'),
  melon: require('../assets/grapes/melon.png'),
};

// Load the cleaned grape varieties data with wine-themed colors and icons
const wineGrapes: BubbleData[] = [
  { id: '1', text: 'Pinot Noir', color: wineColors.burgundy, size: 'medium-large', icon: grapeIcons.pinotNoir },
  { id: '2', text: 'Chardonnay', color: wineColors.champagne, size: 'medium-large', icon: grapeIcons.chardonnay },
  { id: '3', text: 'Cabernet Sauvignon', color: wineColors.deepRed, size: 'medium-large', icon: grapeIcons.cabernetSauvignon },
  { id: '4', text: 'Riesling', color: wineColors.riesling, size: 'medium-large', icon: grapeIcons.riesling },
  { id: '5', text: 'Syrah', color: wineColors.darkPurple, size: 'medium-large', icon: grapeIcons.syrah },
  { id: '6', text: 'Sauvignon Blanc', color: wineColors.sauvignonBlanc, size: 'medium-large', icon: grapeIcons.sauvignonBlanc },
  { id: '7', text: 'Zinfandel', color: wineColors.mediumRed, size: 'medium-large', icon: grapeIcons.zinfandel },
  { id: '8', text: 'Malbec', color: wineColors.darkRed, size: 'medium-large', icon: grapeIcons.malbec },
  { id: '9', text: 'Nebbiolo', color: wineColors.burgundy, size: 'medium-large', icon: grapeIcons.nebbiolo },
  { id: '10', text: 'Merlot', color: wineColors.deepRed, size: 'medium-large', icon: grapeIcons.merlot },
  { id: '11', text: 'Tempranillo', color: wineColors.mediumRed, size: 'medium-large', icon: grapeIcons.tempranillo },
  { id: '12', text: 'Sangiovese', color: wineColors.lightRed, size: 'medium-large', icon: grapeIcons.sangiovese },
  { id: '13', text: 'Pinot Gris', color: wineColors.rosé, size: 'medium-large', icon: grapeIcons.pinotGris },
  { id: '14', text: 'Grüner Veltliner', color: wineColors.oliveGreen, size: 'medium-large', icon: grapeIcons.grunerVeltliner },
  { id: '15', text: 'Cabernet Franc', color: wineColors.darkRed, size: 'medium-large', icon: grapeIcons.cabernetFranc },
  { id: '16', text: 'Gewürztraminer', color: wineColors.goldAccent, size: 'medium-large', icon: grapeIcons.gewurztraminer },
  { id: '17', text: 'Gamay', color: wineColors.lightRed, size: 'medium-large', icon: grapeIcons.gamay },
  { id: '18', text: 'Viognier', color: wineColors.chardonnay, size: 'medium-large', icon: grapeIcons.viognier },
  { id: '19', text: 'Carmenère', color: wineColors.darkPurple, size: 'medium-large', icon: grapeIcons.carmenere },
  { id: '20', text: 'Grenache', color: wineColors.mediumRed, size: 'medium-large', icon: grapeIcons.grenache },
  { id: '21', text: 'Petite Sirah', color: wineColors.darkPurple, size: 'medium-large', icon: grapeIcons.petiteSirah },
  { id: '22', text: 'Barbera', color: wineColors.burgundy, size: 'medium-large', icon: grapeIcons.barbera },
  { id: '23', text: 'Chenin Blanc', color: wineColors.champagne, size: 'medium-large', icon: grapeIcons.cheninBlanc },
  { id: '24', text: 'Glera', color: wineColors.neutral, size: 'medium-large', icon: grapeIcons.glera },
  { id: '25', text: 'Sangiovese Grosso', color: wineColors.lightRed, size: 'medium-large', icon: grapeIcons.sangioveseGrosso },
  { id: '26', text: 'Nero d\'Avola', color: wineColors.darkPurple, size: 'medium-large', icon: grapeIcons.neroDAvola },
  { id: '27', text: 'Melon', color: wineColors.oliveGreen, size: 'medium-large', icon: grapeIcons.melon }
];

// Diagonal column positioning system
// Each column cascades down and to the right
const CELL_SIZE = 130;
const GRID_PADDING = 30;
const COLUMN_OFFSET_X = 180; // Horizontal spacing between diagonal columns
const COLUMN_OFFSET_Y = 150;  // Vertical offset for each item in a diagonal column
const SEAMLESS_OFFSET_X = 240;

// Define diagonal columns - each object has bubbles array, starting X and Y offsets
// Bubbles in each column will be positioned diagonally down and to the right
const diagonalColumns = [
  { 
    bubbles: [1, 2, 3], 
    startX: 0,    // Column 0: standard position
    startY: 0   // Column 0: starts higher up
  },
  { 
    bubbles: [4, 5, 6], 
    startX: -20,   // Column 1: slight right offset
    startY: 50   // Column 1: offset down a bit
  },
  { 
    bubbles: [7, 8, 9], 
    startX: -70,  // Column 2: slight left offset
    startY: -50    // Column 2: slightly higher than column 1
  },
  { 
    bubbles: [10, 11, 12], 
    startX: -90,   // Column 3: more right offset
    startY: 0   // Column 3: lower
  },
  { 
    bubbles: [13, 14, 15], 
    startX: -100,    // Column 4: back to standard
    startY: 50   // Column 4: back up higher
  },
  { 
    bubbles: [16, 17, 18], 
    startX: -150,  // Column 5: left offset
    startY: -50    // Column 5: offset down a bit
  },
  { 
    bubbles: [19, 20, 21],  
    startX: -170,   // Column 6: small right offset
    startY: 0   // Column 6: offset down a bit
  },
  { 
    bubbles: [22, 23, 24],  
    startX: -180,   // Column 7: tiny left offset
    startY: 50   // Column 7: offset down a bit
  },
  { 
    bubbles: [25, 26, 27],  
    startX: -230,   // Column 8: right offset
    startY: -50    // Column 8: offset down a bit
  }
];

const getScatteredPosition = (index: number) => {
  const bubbleId = index + 1; // Convert 0-based index to 1-based ID
  
  // Find the bubble in the diagonal columns
  for (let col = 0; col < diagonalColumns.length; col++) {
    const columnData = diagonalColumns[col];
    const positionInColumn = columnData.bubbles.indexOf(bubbleId);
    
    if (positionInColumn !== -1) {
      // Position based on column and position within column
      return {
        left: GRID_PADDING + (col * COLUMN_OFFSET_X) + columnData.startX + (positionInColumn * 40), // Diagonal offset + custom X
        top: GRID_PADDING + columnData.startY + (positionInColumn * COLUMN_OFFSET_Y),
      };
    }
  }
  
  // Fallback position if not found
  return { left: GRID_PADDING, top: GRID_PADDING };
};

// Calculate total width needed for diagonal column layout
// Make it so the pattern repeats seamlessly
const getTotalWidth = () => {
  return diagonalColumns.length * COLUMN_OFFSET_X - SEAMLESS_OFFSET_X;
};

// Calculate exact height needed for diagonal column layout
const getContainerHeight = () => {
  const maxItemsInColumn = Math.max(...diagonalColumns.map(col => col.bubbles.length));
  const maxStartY = Math.max(...diagonalColumns.map(col => col.startY));
  return GRID_PADDING * 2 + maxStartY + (maxItemsInColumn * COLUMN_OFFSET_Y) + CELL_SIZE;
};

const BubbleScroller = () => {
  const router = useRouter();
  const { wineData, setWineData } = useWine();
  const { country, region1, region2, variety, age } = wineData;
  
  const scrollViewRef = useRef<ScrollView>(null);
  const [sliderValue, setSliderValue] = useState(age || 10); // Use age from wine context or default to 10
  const totalWidth = getTotalWidth();
  const containerHeight = getContainerHeight();

  // Sync slider value with wine context age
  React.useEffect(() => {
    if (age && age !== sliderValue) {
      setSliderValue(age);
    }
  }, [age]);
  
  // Create double array for infinite looping effect using original stable data
  const tripleData = React.useMemo(() => {
    return [...wineGrapes, ...wineGrapes];
  }, []);
  
  const handleBack = () => {
    router.push('/country');
  };

  const handleContinue = () => {
    router.push('/results');
  };

  const handleGrapeSelection = useCallback((grapeVariety: string) => {
    // Update the wine context with the selected variety
    setWineData(prev => ({
      ...prev,
      variety: grapeVariety,
      age: sliderValue
    }));
  }, [setWineData, sliderValue]);

  const handleScroll = useCallback((event: NativeSyntheticEvent<NativeScrollEvent>) => {
    const { contentOffset } = event.nativeEvent;
    const { x } = contentOffset;
    
    // Infinite scroll logic for 2x duplication
    if (x >= totalWidth * 1.5) {
      scrollViewRef.current?.scrollTo({ x: x - totalWidth, animated: false });
    } else if (x <= totalWidth * 0.2) {
      scrollViewRef.current?.scrollTo({ x: x + totalWidth, animated: false });
    }
  }, [totalWidth]);
  
  React.useEffect(() => {
    // Start in the middle of the first set to allow infinite scrolling in both directions
    setTimeout(() => {
      scrollViewRef.current?.scrollTo({ x: totalWidth * 0.5, animated: false });
    }, 100);
  }, [totalWidth]);

  const handleSliderChange = useCallback((value: number) => {
    // Add validation to ensure we only set valid numbers
    if (typeof value === 'number' && !isNaN(value) && isFinite(value)) {
      setSliderValue(value);
      // Update wine context with new age
      setWineData(prev => ({
        ...prev,
        age: value
      }));
    }
  }, [setWineData]);

  return (
    <SafeAreaView style={styles.container}>
      {/* Header Section */}
      <View style={styles.headerContainer}>
        <View style={styles.headerTextContainer}>
          <Text style={styles.title}>Pick a Grape</Text>
        </View>
        
        {/* Navigation buttons in top right */}
        <View style={styles.navigationButtons}>
          <TouchableOpacity onPress={handleBack} style={styles.backButton}>
            <Text style={styles.backButtonText}>← Back</Text>
          </TouchableOpacity>
          
          <TouchableOpacity onPress={handleContinue} style={styles.continueButton}>
            <Text style={styles.continueButtonText}>Continue →</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Wine Context Display - Compact */}
      <View style={styles.contextContainer}>

        <View style={styles.selectionChips}>
          {country && (
            <View style={styles.chip}>
              <Text style={styles.chipText}>{country}</Text>
            </View>
          )}
          {region1 && (
            <View style={styles.chip}>
              <Text style={styles.chipText}>{region1}</Text>
            </View>
          )}
          {region2 && (
            <View style={styles.chip}>
              <Text style={styles.chipText}>{region2}</Text>
            </View>
          )}
          {variety && (
            <View style={[styles.chip, styles.varietyChip]}>
              <Text style={[styles.chipText, styles.varietyChipText]}>{variety}</Text>
            </View>
          )}
          <View style={styles.chip}>
            <Text style={styles.chipText}>{age || sliderValue} years</Text>
          </View>
        </View>
      </View>
      
      {/* Bubble Scroller - takes remaining space */}
      <View style={{ flex: 1 }}>
        <View style={[styles.scrollViewWrapper, { height: containerHeight - 80 }]}>
          <ScrollView 
            ref={scrollViewRef}
            style={styles.scrollView}
            contentContainerStyle={[styles.scrollContent, { width: totalWidth * 2, height: containerHeight - 80 }]}
            horizontal
            showsHorizontalScrollIndicator={false}
            showsVerticalScrollIndicator={false}
            onScroll={handleScroll}
            scrollEventThrottle={16}
            scrollEnabled={true}
          >
            <View style={[styles.bubbleContainer, { height: containerHeight - 80 }]}>
              {tripleData.map((bubble, index) => {
                const originalIndex = index % wineGrapes.length;
                const setIndex = Math.floor(index / wineGrapes.length);
                const position = getScatteredPosition(originalIndex);
                
                return (
                  <View
                    key={`${bubble.id}-${setIndex}`}
                    style={[
                      styles.bubbleWrapper,
                      {
                        left: position.left + (setIndex * totalWidth),
                        top: position.top,
                      }
                    ]}
                  >
                    <Bubble 
                      data={bubble}
                      selectedVariety={variety}
                      onPress={() => handleGrapeSelection(bubble.text)} 
                    />
                  </View>
                );
              })}
            </View>
          </ScrollView>
        </View>
      </View>
      
      {/* Wine Age Slider */}
      <View style={styles.sliderContainer}>
        <Slider 
          min={8}
          max={30}
          step={1}
          value={sliderValue}
          onValueChange={handleSliderChange}
          label={`Wine Age: ${sliderValue} years`}
        />
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: wineTheme.colors.background,
  },
  scrollViewWrapper: {
    alignSelf: 'center',
  },
  scrollView: {
    // Removed flex: 1 to prevent taking all available space
  },
  scrollContent: {
    // Height is set dynamically
  },
  bubbleContainer: {
    position: 'relative',
    // Height is set dynamically
  },
  bubbleWrapper: {
    position: 'absolute',
  },
  headerContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between', // Added to space out back button
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 10,
  },
  headerTextContainer: {
    marginLeft: 10,
    flex: 1,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    fontFamily: fonts.spaceGrotesk,
    color: wineTheme.colors.text,
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 16,
    fontFamily: fonts.outfit,
    color: wineTheme.colors.text, // Use text color since subtext doesn't exist
    lineHeight: 22,
  },
  contextContainer: {
    marginHorizontal: 20,
    marginBottom: 10,
  },
  contextTitle: {
    fontSize: 16,
    fontWeight: '600',
    fontFamily: fonts.spaceGrotesk,
    color: wineTheme.colors.text,
    marginBottom: 8,
  },
  selectionChips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    padding: 12,
    backgroundColor: wineTheme.colors.surface, // Use surface as card background
    borderRadius: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  chip: {
    backgroundColor: wineTheme.colors.primary,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 15,
  },
  chipText: {
    color: '#ffffff', // Use white text on colored chips
    fontSize: 12,
    fontWeight: '600',
    fontFamily: fonts.outfit,
  },
  varietyChip: {
    backgroundColor: wineTheme.colors.primary, // Use a darker red for variety
    borderColor: wineTheme.colors.primary,
  },
  varietyChipText: {
    color: '#ffffff', // Use white text
    fontFamily: fonts.outfit,
  },
  sliderContainer: {
    backgroundColor: wineTheme.colors.surface, // Use surface as card background
    marginTop: 15,
    marginHorizontal: 20,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  backButton: {
    backgroundColor: wineTheme.colors.surface,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: wineTheme.colors.primary,
  },
  backButtonText: {
    color: wineTheme.colors.text,
    fontSize: 14,
    fontWeight: '600',
    fontFamily: fonts.outfit,
  },
  continueButton: {
    backgroundColor: wineTheme.colors.primary,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: wineTheme.colors.primary,
  },
  continueButtonText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '600',
    fontFamily: fonts.outfit,
  },
  navigationButtons: {
    flexDirection: 'row',
    gap: 10,
  },
});

export default BubbleScroller;
