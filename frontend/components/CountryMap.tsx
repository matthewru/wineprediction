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

const mapComponents: Record<string, React.ReactNode> = {
  US: <USMap onRegionClick={() => {}} highlightedRegion="" />,
  Italy: <ItalyMap onRegionClick={() => {}} highlightedRegion="" />,
  France: <FranceMap onRegionClick={() => {}} highlightedRegion="" />,
  Spain: <SpainMap onRegionClick={() => {}} highlightedRegion="" />,
  Portugal: <PortugalMap onRegionClick={() => {}} highlightedRegion="" />,
  Chile: <ChileMap onRegionClick={() => {}} highlightedRegion="" />,
  Argentina: <ArgentinaMap onRegionClick={() => {}} highlightedRegion="" />,
  Austria: <AustriaMap onRegionClick={() => {}} highlightedRegion="" />,
  Germany: <GermanyMap onRegionClick={() => {}} highlightedRegion="" />,
  Australia: <AustraliaMap onRegionClick={() => {}} highlightedRegion="" />,
};

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

export default function CountryMap({ country }: { country: string }) {
  const MapComponent = mapComponents[country];
  return (
    <MapWrapper>
      {MapComponent || <PlaceholderMap country={country} />}
    </MapWrapper>
  );
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
