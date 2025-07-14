// components/CountryMap.tsx
import { View } from 'react-native';
import { ThemedText } from './ThemedText';

import USMap from './maps/USMap';
import ItalyMap from './maps/ItalyMap';
import FranceMap from './maps/FranceMap';
import SpainMap from './maps/SpainMap';
import PortugalMap from './maps/PortugalMap';
import ChileMap from './maps/ChileMap';
import ArgentinaMap from './maps/ArgentinaMap';
import GermanyMap from './maps/GermanyMap';
import AustriaMap from './maps/AustriaMap';
import AustraliaMap from './maps/AustraliaMap';

export default function CountryMap({ country, highlightedRegion = "" }: { country: string; highlightedRegion?: string }) {
  const mapComponents: Record<string, React.ReactNode> = {
    US: <USMap onRegionClick={() => {}} highlightedRegion={highlightedRegion} />,
    Italy: <ItalyMap onRegionClick={() => {}} highlightedRegion={highlightedRegion} />,
    France: <FranceMap onRegionClick={() => {}} highlightedRegion={highlightedRegion} />,
    Spain: <SpainMap onRegionClick={() => {}} highlightedRegion={highlightedRegion} />,
    Portugal: <PortugalMap onRegionClick={() => {}} highlightedRegion={highlightedRegion} />,
    Chile: <ChileMap onRegionClick={() => {}} highlightedRegion={highlightedRegion} />,
    Argentina: <ArgentinaMap onRegionClick={() => {}} highlightedRegion={highlightedRegion} />,
    Austria: <AustriaMap onRegionClick={() => {}} highlightedRegion={highlightedRegion} />,
    Germany: <GermanyMap onRegionClick={() => {}} highlightedRegion={highlightedRegion} />,
    Australia: <AustraliaMap onRegionClick={() => {}} highlightedRegion={highlightedRegion} />,
  };

  const MapComponent = mapComponents[country];
  return (
    <MapWrapper>
      {MapComponent || <PlaceholderMap country={country} />}
    </MapWrapper>
  );
}

const MapWrapper = ({ children }: { children: React.ReactNode }) => (
  <View
    style={{
      width: '100%',
      height: 500,
      alignItems: 'center',
      justifyContent: 'center',
      // transform: removed for now
      // borderWidth: 1,
      // borderColor: 'red',
    }}
  >
    {children}
  </View>
);

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
