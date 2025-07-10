export const fonts = {
  spaceGrotesk: 'SpaceGrotesk_500Medium',
  outfit: 'Outfit_400Regular',
  spaceMono: 'SpaceMono-Regular',
} as const;

export type FontFamily = typeof fonts[keyof typeof fonts]; 