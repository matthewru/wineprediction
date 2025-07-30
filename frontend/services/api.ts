// Wine prediction API service

// Get the appropriate API URL based on environment
const getApiUrl = () => {
  if (__DEV__) {
    // Development - use your computer's IP address
    // Replace this IP with your computer's actual IP if it changes
    const COMPUTER_IP = '10.0.0.140'; // Your Mac's IP address
    
    return `http://${COMPUTER_IP}:5001`;
  } else {
    // Production - replace with your actual deployed URL
    return 'https://your-deployed-backend.com';
  }
};

const API_BASE_URL = getApiUrl();

// Rating curve function to fix rating inflation
const adjustRating = (rawRating: number): number => {
  // Current model outputs 85-95, we want more realistic 75-92 spread
  // Apply a curve that brings down inflated ratings
  
  if (rawRating >= 95) return 92;  // Cap exceptional wines at 92
  if (rawRating >= 90) return 88 + (rawRating - 90) * 0.8;  // 90-95 → 88-92
  if (rawRating >= 85) return 82 + (rawRating - 85) * 1.2;  // 85-90 → 82-88
  if (rawRating >= 80) return 76 + (rawRating - 80) * 1.2;  // 80-85 → 76-82
  
  return Math.max(72, rawRating * 0.9);  // Below 80 → scale down, min 72
};

export interface WinePredictionInput {
  variety: string;
  country: string;
  province: string;
  age: number;
  region_hierarchy: string;
}

export interface PricePrediction {
  weighted_lower: number;
  weighted_upper: number;
  confidence_interval: string;
}

export interface RatingPrediction {
  predicted_rating: number;
  confidence_interval: [number, number];
}

export interface FlavorPrediction {
  flavor: string;
  confidence: number;
}

export interface MouthfeelPrediction {
  mouthfeel: string;
  confidence: number;
}

export interface AllPredictionsResponse {
  price: PricePrediction;
  rating: RatingPrediction;
  flavors: FlavorPrediction[];
  mouthfeel: MouthfeelPrediction[];
}

// Helper function to create a timeout promise
const createTimeoutPromise = (timeoutMs: number) => {
  return new Promise((_, reject) => {
    setTimeout(() => reject(new Error('Request timed out')), timeoutMs);
  });
};

class WineAPI {
  private async makeRequest<T>(endpoint: string, data: any): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    console.log(`Making request to: ${url}`);
    console.log('Request data:', data);
    
    try {
      // Create fetch promise
      const fetchPromise = fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      // Race between fetch and timeout
      const response = await Promise.race([
        fetchPromise,
        createTimeoutPromise(30000), // 30 second timeout
      ]) as Response;

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      console.log(`Success response from ${endpoint}:`, result);
      return result;
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      
      // Provide more specific error messages
      if (error instanceof TypeError && error.message.includes('Network request failed')) {
        throw new Error(`Cannot connect to server at ${API_BASE_URL}. Make sure the Flask server is running and your phone is on the same WiFi network.`);
      } else if (error instanceof Error && error.message === 'Request timed out') {
        throw new Error('Request timed out. The server might be overloaded.');
      }
      
      throw error;
    }
  }

  async predictPrice(input: WinePredictionInput): Promise<PricePrediction> {
    return this.makeRequest<PricePrediction>('/predict-price-lite', input);
  }

  async predictRating(input: WinePredictionInput): Promise<RatingPrediction> {
    const result = await this.makeRequest<RatingPrediction>('/predict-rating-lite', input);
    
    // Apply rating curve to fix inflation
    const adjustedRating = adjustRating(result.predicted_rating);
    
    return {
      ...result,
      predicted_rating: adjustedRating
    };
  }

  async predictFlavors(input: WinePredictionInput): Promise<FlavorPrediction[]> {
    // Add minimal confidence threshold and max results to get all flavors
    const requestData = {
      ...input,
      confidence_threshold: 0.1, // Very low threshold to get almost all results
      top_k: 20 // Get up to 20 flavors
    };
    return this.makeRequest<FlavorPrediction[]>('/predict-flavor', requestData);
  }

  async predictMouthfeel(input: WinePredictionInput): Promise<MouthfeelPrediction[]> {
    // Add minimal confidence threshold and max results to get all mouthfeel
    const requestData = {
      ...input,
      confidence_threshold: 0.1, // Very low threshold to get almost all results
      top_k: 20 // Get up to 20 mouthfeel characteristics
    };
    return this.makeRequest<MouthfeelPrediction[]>('/predict-mouthfeel', requestData);
  }

  async predictAll(input: WinePredictionInput): Promise<AllPredictionsResponse> {
    // Add minimal confidence threshold and max results to get comprehensive results
    const requestData = {
      ...input,
      confidence_threshold: 0.1, // Very low threshold to get almost all results
      top_k: 20 // Get up to 20 of each type
    };
    
    const result = await this.makeRequest<AllPredictionsResponse>('/predict-all', requestData);
    
    // Apply rating curve to fix inflation
    const adjustedRating = adjustRating(result.rating.predicted_rating);
    
    return {
      ...result,
      rating: {
        ...result.rating,
        predicted_rating: adjustedRating
      }
    };
  }

  async healthCheck(): Promise<{ status: string; models_loaded?: boolean }> {
    const url = `${API_BASE_URL}/health`;
    console.log(`Health check to: ${url}`);
    
    try {
      // Create fetch promise
      const fetchPromise = fetch(url);

      // Race between fetch and timeout
      const response = await Promise.race([
        fetchPromise,
        createTimeoutPromise(10000), // 10 second timeout for health check
      ]) as Response;
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      return response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }
}

export const wineAPI = new WineAPI(); 