import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

def train_and_save(df, target_col='price'):
    X = df.drop(columns=[target_col, 'car_id', 'zip_prefix'], errors='ignore')
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # We'll use RF for the primary app demo
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds)
    }
    
    joblib.dump(model, 'models/car_price_model.pkl')
    return metrics