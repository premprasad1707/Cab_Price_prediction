def engineer_features(df):
    current_year = 2024
    df['age'] = current_year - df['year']
    df['mileage_per_year'] = df['mileage'] / (df['age'] + 1)
    
    # Simple Brand Premium Logic
    luxury = ['BMW', 'Audi', 'Tesla', 'Mercedes']
    df['is_luxury'] = df['brand'].apply(lambda x: 1 if x in luxury else 0)
    
    # Price Intelligence Score (Mock logic for demonstration)
    df['pis_score'] = (df['is_luxury'] * 20) - (df['age'] * 2) - (df['mileage']/10000)
    return df