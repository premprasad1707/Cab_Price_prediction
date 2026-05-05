# AI Car Market Intelligence Platform

A Streamlit-based analytics and price intelligence app for exploring used-car listings, estimating market value, and identifying suspicious listings. The project combines data preprocessing, feature engineering, interactive Plotly charts, and machine-learning-style workflows in a modern dashboard UI.

## What This App Does

The platform helps users understand car market behavior through four main views:

- **Market Dashboard**: Tracks listing volume, average price, top brand, mileage trends, price distribution, and price-vs-mileage patterns.
- **Price Prediction**: Estimates a vehicle's market value from brand, model, year, mileage, fuel type, and transmission.
- **Anomaly Detection**: Uses Isolation Forest to flag suspicious or unusual listings based on price, mileage, and year.
- **Market Insights**: Shows brand-level price retention and highlights key market observations.

## UI Highlights

The app includes a customized Streamlit interface with:

- Dark emerald premium theme
- Smooth CSS transitions and hover states
- JavaScript-powered reveal animations
- Animated metric values
- Enhanced sidebar styling
- Improved price prediction layout
- Styled anomaly review table with risk scoring
- Responsive dashboard cards and Plotly visualizations

## Tech Stack

- **Python**
- **Streamlit** for the web interface
- **Pandas** and **NumPy** for data handling
- **Scikit-learn** for anomaly detection and model workflows
- **Plotly** for interactive charts
- **Joblib** for model serialization support
- **XGBoost / LightGBM** listed for advanced modeling experiments

## Project Structure

```text
car_market_intelligence_platform/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── car_prices.csv
└── src/
    ├── preprocessing.py
    ├── feature_engineering.py
    └── train_models.py
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The App

Start the Streamlit server:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Data

The app expects the dataset at:

```text
data/car_prices.csv
```

Expected fields include vehicle details such as brand, model, year, mileage, fuel type, transmission, and price.

## Notes

The current price prediction flow uses a simple estimation formula inside `app.py` for demonstration. The project already includes training-related modules in `src/`, so this can be extended to load a trained model with `joblib` for production-style predictions.

## Future Improvements

- Connect the prediction screen to a saved trained model
- Add model performance metrics
- Add filters for brand, fuel type, year, and price range
- Add export options for anomaly reports
- Add authentication for admin or analyst workflows
- Improve mobile-specific layout polish
