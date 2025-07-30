# Wine Prediction App - API Integration Setup

## Quick Start Guide

### 1. Start the Backend API

```bash
cd backend
python app.py
```

The Flask server will start on `http://localhost:5001` and warm up all models.

### 2. Update API URL for Production

In `frontend/services/api.ts`, replace the production URL:

```typescript
const API_BASE_URL = __DEV__ 
  ? 'http://localhost:5001'  // Development
  : 'https://your-deployed-backend.com';  // Update this!
```

### 3. Test the Connection

1. Start your Expo app:
```bash
cd frontend
npx expo start
```

2. Go through the wine selection flow
3. On the results screen, you should see API calls in the backend console
4. The app will show predicted price, rating, flavors, and mouthfeel

### 4. API Endpoints Available

- `POST /predict-price-lite` - Wine price range prediction
- `POST /predict-rating-lite` - Wine quality rating prediction  
- `POST /predict-flavor` - Wine flavor profile prediction
- `POST /predict-mouthfeel` - Wine mouthfeel characteristics
- `POST /predict-all` - All predictions in one call (recommended)
- `GET /health` - Health check and model status

### 5. Troubleshooting

**"Failed to get wine predictions"**
- Check that Flask server is running on port 5001
- Ensure all models are trained (run training scripts if needed)
- Check network connectivity

**Missing model files**
- Run the training scripts in `backend/services/`:
  - `train_flavor_predictor.py`
  - `train_mouthfeel_predictor.py` 
  - `train_lite_price_predictor.py`
  - `train_lite_rating_predictor.py`

**Network issues on mobile**
- For iOS simulator: Use `http://localhost:5001`
- For Android emulator: Use `http://10.0.2.2:5001`
- For physical device: Use your computer's IP address

### 6. Production Deployment

For production, you'll want to deploy your Flask API to a cloud service like:
- Railway
- Heroku  
- Google Cloud Run
- AWS Lambda
- DigitalOcean App Platform

Then update the production URL in `api.ts`.

## Why This Approach Works

✅ **Easy to implement** - Just HTTP requests, no complex mobile ML setup
✅ **Fast performance** - Server-grade hardware runs predictions quickly  
✅ **Easy updates** - Improve models without app store updates
✅ **Better debugging** - Server logs make troubleshooting easier
✅ **Scalable** - Can handle multiple users and add caching/optimization

## Future: Local Models (Optional)

Later, if you want to explore local models for offline use:

1. **Convert models** to mobile formats (Core ML for iOS, TensorFlow Lite for Android)
2. **Quantize models** to reduce size (can reduce from 37MB to ~5-10MB)
3. **Use Expo ML** or React Native libraries for inference
4. **Implement fallback** - try local first, fall back to API if local fails

But for now, the API approach will get you up and running immediately with excellent performance! 