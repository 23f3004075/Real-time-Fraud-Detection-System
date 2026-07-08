import os
import joblib
import pandas as pd
import numpy as np
import shap
from datetime import datetime

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/best_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "../model/scaler.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "../model/encoder.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    explainer = shap.TreeExplainer(model)
except Exception as e:
    print(f"Warning: Model assets missing. Error: {e}")
    model, scaler, encoder, explainer = None, None, None, None

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1)/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def process_and_predict(data: dict, cache=None):
    if not all([model, scaler, encoder, explainer]):
        return 0.0, ["Model offline"]

    
    trans_time = pd.to_datetime(data.get('trans_date_trans_time', datetime.utcnow()))
    dob = pd.to_datetime(data.get('dob', '1980-01-01'))
    
    hour = trans_time.hour
    day_of_week = trans_time.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0
    age = trans_time.year - dob.year
    dist_km = calculate_haversine_distance(
        data['lat'], data['long'], data['merch_lat'], data['merch_long']
    )

    
    user_key = f"user_profile:{data['user_id']}:avg_amt"
    
   
    if cache and cache.get(user_key):
        user_hist_avg = float(cache.get(user_key))
    else:
        user_hist_avg = 50.00
        
    
    true_avg_amt_last_5 = user_hist_avg
    true_amt_ratio = data['amount'] / (user_hist_avg + 1e-6)
    
   
    # Rule A: Hard safety threshold 
    if data['amount'] > 10000.00:
        return 0.99, ["Rule Block: Absolute amount exceeds $10,000 hard safety limit"]
        
    # Rule B: Extreme Velocity (Spending 50x their normal baseline)
    if true_amt_ratio > 50.0 and data['amount'] > 500.00:
        return 0.98, [f"Rule Block: Velocity Spike ({true_amt_ratio:.1f}x normal spending)"]
        
    # Rule C: Impossible Travel / High-Risk Offshore (Very far distance + high amount)
    if dist_km > 5000 and data['amount'] > 1000.00:
        return 0.97, [f"Rule Block: High amount at extreme offshore distance ({dist_km:.0f}km)"]
        
    # Rule D: Late Night Online Shopping (1 AM - 5 AM) for suspicious amounts
    if (1 <= hour <= 5) and data['category'] == 'shopping_net' and data['amount'] > 500.00:
        return 0.96, ["Rule Block: Late night high-value online transaction"]
    
    if cache:
        new_avg = (user_hist_avg * 0.9) + (data['amount'] * 0.1)
        cache.setex(user_key, 2592000, str(new_avg))
    
    merch_freq_24h = 1

    raw_features = {
        'merchant': data['merchant'], 
        'category': data['category'], 
        'amt': data['amount'], 
        'gender': data['gender'], 
        'city': data['city'], 
        'state': data['state'],
        'lat': data['lat'], 
        'long': data['long'], 
        'city_pop': data['city_pop'], 
        'job': data['job'], 
        'merch_lat': data['merch_lat'], 
        'merch_long': data['merch_long'],
        'hour': hour, 
        'day_of_week': day_of_week, 
        'is_weekend': is_weekend,
        'age': age, 
        'dist_km': dist_km, 
        'avg_amt_last_5': true_avg_amt_last_5,
        'amt_ratio': true_amt_ratio,
        'merch_freq_24h': merch_freq_24h
    }
    
    df = pd.DataFrame([raw_features])

    cat_cols = ['merchant', 'category', 'gender', 'city', 'state', 'job']
    num_cols = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 
                'age', 'dist_km', 'avg_amt_last_5', 'amt_ratio', 'merch_freq_24h']
                
    df[cat_cols] = encoder.transform(df[cat_cols])
    df[num_cols] = scaler.transform(df[num_cols])

    probability = float(model.predict_proba(df)[0][1])
    shap_values = explainer.shap_values(df)
    
   
    feature_impacts = dict(zip(df.columns, shap_values[0]))
    
    
    top_features = sorted(feature_impacts.items(), key=lambda item: abs(item[1]), reverse=True)
    
    reasons = []
    for feat, impact in top_features[:2]:
        direction = "Increased" if impact > 0 else "Decreased"
        if abs(impact) > 0.05:
            reasons.append(f"{feat} ({direction} fraud risk)")

    if not reasons:
        reasons.append("Standard profile")

    return probability, reasons