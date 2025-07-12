// components/CountryMap.tsx
import { View } from 'react-native';
import USMap from './maps/USMap';
import { ThemedText } from './ThemedText';
import ItalyMap from './maps/ItalyMap';
import FranceMap from './maps/FranceMap';
import SpainMap from './maps/SpainMap';
import PortugalMap from './maps/PortugalMap';
import ChileMap from './maps/ChileMap';
import ArgentinaMap from './maps/ArgentinaMap';
import GermanyMap from './maps/GermanyMap';
import AustriaMap from './maps/AustriaMap';
import AustraliaMap from './maps/AustraliaMap';
export default function CountryMap({ country }: { country: string }) {
  switch (country) {
    case 'US':
        return <USMap onRegionClick={() => {}} highlightedRegion="" />;
    case 'Italy':
        return <ItalyMap onRegionClick={() => {}} highlightedRegion="" />;
    case 'France':
        return <FranceMap onRegionClick={() => {}} highlightedRegion="" />;
    case 'Spain':
        return <SpainMap onRegionClick={() => {}} highlightedRegion="" />;
    case 'Portugal':
        return <PortugalMap onRegionClick={() => {}} highlightedRegion="" />;
    case 'Chile':
        return <ChileMap onRegionClick={() => {}} highlightedRegion="" />;
    case 'Argentina':
        return <ArgentinaMap onRegionClick={() => {}} highlightedRegion="" />;
    case 'Austria':
        return <AustriaMap onRegionClick={() => {}} highlightedRegion="" />;
    case 'Germany':
        return <GermanyMap onRegionClick={() => {}} highlightedRegion="" />;
    case 'Australia':
        return <AustraliaMap onRegionClick={() => {}} highlightedRegion="" />;
    default:
      return (
        <PlaceholderMap country={country} />
      );
  }
}

function PlaceholderMap({ country }: { country: string }) {
  return (
    <View
      style={{
        width: '90%',
        height: 250,
        backgroundColor: '#eee',
        justifyContent: 'center',
        alignItems: 'center',
        borderRadius: 12,
      }}
    >
      <ThemedText type="default">Map for {country} coming soon…</ThemedText>
    </View>
  );
}
