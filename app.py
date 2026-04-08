import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Page Configuration
st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title('📈 Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below.')

@st.cache_resource
def download_data(symbol, start_date, end_date):
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        return df
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

# Sidebar Inputs
symbol = st.sidebar.text_input('Enter a Stock Symbol', value='SPY').upper()
duration = st.sidebar.number_input('Enter the duration (days of history)', value=3000)

today = datetime.date.today()
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)

# Fetch Data
data = download_data(symbol, start_date, end_date)
scaler = StandardScaler()

def dataframe():
    st.header(f'Recent Data for {symbol}')
    if data is not None and not data.empty:
        st.dataframe(data.tail(10))
    else:
        st.error("No data found for this symbol.")

def model_engine(model, num):
    if data is None or data.empty:
        st.error("No data available.")
        return

    # 1. Prepare features and target in one DataFrame
    df = data[['Close']].copy()
    df['target'] = df['Close'].shift(-num)
    
    # 2. DROP ALL NaNs HERE
    # This removes the last 'num' rows where 'target' is NaN
    # AND any rows where 'Close' might be NaN
    df_clean = df.dropna()
    
    if df_clean.empty:
        st.error("Not enough data to train after removing empty rows.")
        return

    # 3. Prepare Training Features (X) and Target (y)
    # Use double brackets [['Close']] to keep it as a 2D array for the scaler
    X = df_clean[['Close']].values
    y = df_clean['target'].values

    # 4. Scaling
    # Fit the scaler ONLY on training data to avoid data leakage
    X_scaled = scaler.fit_transform(X)

    # 5. Split and Train
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=7)
    
    model.fit(X_train, y_train)
    
    # 6. Metrics
    test_preds = model.predict(X_test)
    st.subheader('Model Performance')
    st.write(f"R2 Score: {r2_score(y_test, test_preds):.4f}")

    # 7. Forecasting the Future
    # To predict the future, we take the last 'num' days of the ORIGINAL data
    # (These are the days that didn't have a 'target' yet)
    last_days_raw = df[['Close']].tail(num).values
    last_days_scaled = scaler.transform(last_days_raw)
    
    forecast_pred = model.predict(last_days_scaled)
    
    st.subheader(f'Forecast for next {num} days')
    for i, price in enumerate(forecast_pred, 1):
        st.write(f"Day {i}: ${price:.2f}")

def predict():
    st.header("Predict Future Prices")
    model_choice = st.radio('Choose a model', ['Linear Regression', 'Random Forest', 'XGBoost'])
    num_days = st.number_input('How many days forecast?', min_value=1, value=5)
    
    if st.button('Run Prediction'):
        if model_choice == 'Linear Regression':
            engine = RandomForestRegressor(n_estimators=1000)
        elif model_choice == 'Random Forest':
            engine = RandomForestRegressor(n_estimators=100)
        else:
            engine = XGBRegressor()
        
        model_engine(engine, int(num_days))

def main():
    option = st.sidebar.selectbox('Make a choice', ['Recent Data', 'Predict'])
    
    if option == 'Recent Data':
        dataframe()
    elif option == 'Predict':
        predict()

if __name__ == '__main__':
    main()