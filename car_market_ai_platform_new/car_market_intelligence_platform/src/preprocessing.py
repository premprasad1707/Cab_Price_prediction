import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataProcessor:
    def __init__(self):
        self.encoders = {}
        self.scaler = StandardScaler()
        
    def clean_data(self, df):
        df = df.copy()
        # Fix logic errors
        df = df[df['mileage'] >= 0]
        df = df[df['year'] <= 2024]
        df = df[df['price'] > 500]
        # Handle missing
        df = df.dropna()
        return df

    def transform(self, df, training=False):
        df = df.copy()
        categorical_cols = ['brand', 'model', 'fuel_type', 'transmission']
        numerical_cols = ['year', 'mileage', 'sale_month']
        
        for col in categorical_cols:
            if training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.encoders[col] = le
            else:
                le = self.encoders[col]
                # Handle unknown categories by mapping to first seen
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])
        
        if training:
            self.scaler.fit(df[numerical_cols])
        
        df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        return df