# etl_forecast_pipeline.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import DateOffset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from generate_report import create_html_report

st.set_page_config(page_title="Automated Business Forecast & Report", layout="wide")

# ---------------------- Upload & Business Metadata ---------------------- #
st.title("📊 Automated Business Financial Forecast & ETL Pipeline")
st.markdown("Upload your monthly financial data and let us forecast and analyze your performance.")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# ---------------------- ETL + Feature Engineering ---------------------- #
@st.cache_data
def load_and_transform(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Month'] = df['Date'].dt.month
    df['Month_Year'] = df['Date'].dt.to_period('M').astype(str)
    df['Season'] = df['Date'].dt.month % 12 // 3 + 1
    df['Season'] = df['Season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})

    required_cols = ['Marketing_Spend', 'Food_Costs', 'Labor_Costs', 'Rent', 'Utilities']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"🚨 Your file is missing required columns: {missing}")
        st.stop()

    df['Total_Expenses'] = df[required_cols].sum(axis=1)
    return df

# ---------------------- Forecasting Functions ---------------------- #
def forecast_sarima(series, steps=36):
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=steps)
    return forecast.predicted_mean

# ---------------------- Model Evaluation ---------------------- #
def evaluate_model(df):
    features = ['Customer_Footfall', 'Marketing_Spend', 'Food_Costs', 'Labor_Costs',
                'Rent', 'Utilities', 'Revenue', 'Delivery_Ratio', 'DineIn_Ratio']
    target = 'Profit'
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'MAE': round(mean_absolute_error(y_test, y_pred), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        'R²': round(r2_score(y_test, y_pred), 2)
    }

# ---------------------- Visualization + Report ---------------------- #
if uploaded_file is not None:
    business_name = st.text_input("🏢 Business Name", "Cafe Mocha")
    business_sector = st.selectbox("📊 Business Sector", ["Food & Beverage", "Retail", "Technology", "Healthcare", "Other"])

    df = load_and_transform(uploaded_file)
    st.success(f"Uploaded and processed data for **{business_name}** ({business_sector})")

    st.header("📈 Revenue vs Expense Forecast (36 Months)")
    df_monthly = df.resample('MS', on='Date').sum()
    revenue_series = df_monthly['Revenue']
    expense_series = df_monthly['Total_Expenses']

    revenue_forecast = forecast_sarima(revenue_series)
    expense_forecast = forecast_sarima(expense_series)
    future_dates = pd.date_range(start=revenue_series.index[-1] + DateOffset(months=1), periods=36, freq='MS')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=revenue_series.index, y=revenue_series.values,
                             mode='lines+markers', name='Revenue', line=dict(color='deepskyblue', width=2)))
    fig.add_trace(go.Scatter(x=expense_series.index, y=expense_series.values,
                             mode='lines+markers', name='Expenses', line=dict(color='dodgerblue', width=2)))
    fig.add_trace(go.Scatter(x=future_dates, y=revenue_forecast.values,
                             mode='lines+markers', name='Forecasted Revenue', line=dict(color='deepskyblue', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=future_dates, y=expense_forecast.values,
                             mode='lines+markers', name='Forecasted Expenses', line=dict(color='dodgerblue', width=2, dash='dash')))

    fig.update_layout(
        title="📊 Monthly Revenue vs Expenses (Historical + 36-Month Forecast)",
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        template="plotly_white",
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
        font=dict(size=14),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.header("📊 Monthly & Seasonal Summary")
    month_summary = df.groupby('Month_Year')[['Revenue', 'Total_Expenses', 'Profit']].sum().reset_index()
    season_summary = df.groupby('Season')[['Revenue', 'Total_Expenses', 'Profit']].mean().reset_index()

    st.subheader("Monthly Summary Table")
    st.dataframe(month_summary.tail(12))

    st.subheader("Seasonal Averages")
    st.dataframe(season_summary)

    st.header("📋 Model Performance Report")
    metrics = evaluate_model(df)
    st.markdown(f"""
    **Mean Absolute Error (MAE)**: ${metrics['MAE']}  
    **Root Mean Squared Error (RMSE)**: ${metrics['RMSE']}  
    **R² Score**: {metrics['R²']}  

    Forecasting model is suitable for directional insights and near-future planning.
    """)

    st.download_button("📥 Download Cleaned Dataset", data=df.to_csv(index=False).encode('utf-8'), file_name=f"{business_name}_Cleaned.csv", key="cleaned_csv")

    html_path = create_html_report(df, business_name=business_name, business_sector=business_sector)
    with open(html_path, "r", encoding="utf-8") as f:
        html_data = f.read()

    st.download_button(
        label="📄 Download Financial Report (HTML)",
        data=html_data,
        file_name="Cafe_Mocha_Report.html",
        mime="text/html",
        key="html_report"
    )

    st.info("💡 Open the report in a browser and use **Print → Save as PDF** to export.")