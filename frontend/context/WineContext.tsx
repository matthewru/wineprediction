import React, { createContext, useContext, useState, ReactNode } from 'react';

interface WineData {
  country: string | null;
  region1: string | null;
  region2: string | null;
  variety: string | null;
  age: number | null;
}

interface WineContextType {
  wineData: WineData;
  setWineData: React.Dispatch<React.SetStateAction<WineData>>;
}

const WineContext = createContext<WineContextType | null>(null);

interface WineProviderProps {
  children: ReactNode;
}

export const WineProvider = ({ children }: WineProviderProps) => {
  const [wineData, setWineData] = useState<WineData>({
    country: null,
    region1: null,
    region2: null,
    variety: null,
    age: null,
  });

  return (
    <WineContext.Provider value={{ wineData, setWineData }}>
      {children}
    </WineContext.Provider>
  );
};

export const useWine = () => {
  const context = useContext(WineContext);
  if (!context) {
    throw new Error('useWine must be used within a WineProvider');
  }
  return context;
};
