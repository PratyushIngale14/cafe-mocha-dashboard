# CM_Dashboard.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import DateOffset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib

st.set_page_config(page_title="Cafe Mocha Financial Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Cafe_Mocha_Cleaned.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month_Year'] = df['Date'].dt.to_period('M').astype(str)
    df['Season'] = df['Date'].dt.month % 12 // 3 + 1
    df['Season'] = df['Season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})
    df['Month'] = df['Date'].dt.month
    return df

df = load_data()

page = st.sidebar.radio("Navigate", ["Home", "Financial Insights", "Forecast & Prediction"])

if page == "Home":
    st.title("Welcome to Cafe Mocha")
    st.markdown("""
    **Cafe Mocha**, founded in 2019, is a cozy destination that has served over **{} customers** and counting. 
    Owned and passionately run by **Mrs. Harsha Ingale**, the café continues to grow thanks to its loyal customers and consistent vision.

    Navigate through this dashboard to view performance metrics, trends, and our financial forecast.
    """.format(int(df['Customer_Footfall'].sum())))
    st.image("cafe_mocha_home.png", use_container_width=True)

elif page == "Financial Insights":
    st.title("Cafe Mocha Financial Insights")

    total_customers = int(df['Customer_Footfall'].sum())
    avg_profit = df['Profit'].mean()
    min_profit = df['Profit'].min()

    if avg_profit < 0:
        summary = f"Business is operating at a **loss**, average monthly loss is ${abs(avg_profit):.2f}."
    elif avg_profit < 2000:
        summary = f"Profit margins are **low**, with average monthly profit of ${avg_profit:.2f}."
    else:
        summary = f"Cafe Mocha is **profitable**, average monthly profit is ${avg_profit:.2f}."

    st.metric("👣 Total Customers Served", f"{total_customers}")
    st.metric("📉 Lowest Monthly Profit", f"${min_profit}")
    st.success(summary)

    st.subheader("Expense Distribution")
    expense_cols = ['Marketing_Spend', 'Food_Costs', 'Labor_Costs', 'Rent', 'Utilities']
    expense_sum = df[expense_cols].sum().reset_index()
    expense_sum.columns = ['Expense', 'Total']
    fig1 = px.pie(expense_sum, names='Expense', values='Total', title="Total Expense Breakdown", hole=0.4)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Profit Distribution")
    fig2 = px.histogram(df, x='Profit', nbins=30, title="Distribution of Monthly Profit", marginal='box')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Monthly Profit Trend")
    profit_monthly = df.groupby('Month_Year')['Profit'].sum().reset_index()
    fig3 = px.line(profit_monthly, x='Month_Year', y='Profit', markers=True, title="Monthly Profit Over Time")
    fig3.update_traces(line=dict(color='green'))
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Average Profit by Month")
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    df['Month_Name'] = df['Month'].map(month_names)
    avg_profit_month = df.groupby('Month_Name')['Profit'].mean().reindex(list(month_names.values())).reset_index()
    fig4 = px.bar(avg_profit_month, x='Month_Name', y='Profit', title="Good vs Bad Months", color='Profit', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Revenue vs Total Expenses")
    rev_exp = df.groupby('Month_Year')[['Revenue', 'Total_Expenses']].sum().reset_index()
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=rev_exp['Month_Year'], y=rev_exp['Revenue'], mode='lines+markers', name='Revenue'))
    fig5.add_trace(go.Scatter(x=rev_exp['Month_Year'], y=rev_exp['Total_Expenses'], mode='lines+markers', name='Expenses'))
    fig5.update_layout(title='Monthly Revenue vs Expenses', xaxis_title='Month', yaxis_title='Amount ($)')
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Seasonal Profit Trends")
    seasonal = df.groupby('Season')['Profit'].mean().reset_index()
    fig6 = px.bar(seasonal, x='Season', y='Profit', color='Profit', title="Average Profit by Season", color_continuous_scale='Bluered')
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Customer Footfall Trend")
    footfall = df.groupby('Month_Year')['Customer_Footfall'].sum().reset_index()
    fig7 = px.area(footfall, x='Month_Year', y='Customer_Footfall', title="Monthly Customer Footfall", line_shape='spline')
    st.plotly_chart(fig7, use_container_width=True)

    st.subheader("Key Takeaways")
    st.markdown("""
    - Profitability is seasonal; winter months perform best.
    - Food and labor are the highest contributors to total expenses.
    - Revenue outpaces expenses in most months, but fluctuations remain.
    - Some months show unusually high footfall with lower profits, indicating potential inefficiencies.
    - Marketing spend peaks correlate with higher profits in following months.
    """)

elif page == "Forecast & Prediction":
    st.title("Financial Forecast & Model Performance")

    df_monthly = df.resample('MS', on='Date').sum()
    profit_series = df_monthly['Profit']
    revenue_series = df_monthly['Revenue']
    expense_series = df_monthly['Total_Expenses']

    def forecast_sarima(series, steps=36):
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=steps)
        return forecast.predicted_mean

    revenue_forecast = forecast_sarima(revenue_series)
    expense_forecast = forecast_sarima(expense_series)
    forecast = SARIMAX(profit_series, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False).get_forecast(steps=36)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    future_dates = pd.date_range(start=profit_series.index[-1] + DateOffset(months=1), periods=36, freq='MS')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=revenue_series.index, y=revenue_series.values, mode='lines+markers', name='Revenue', line=dict(color='deepskyblue', width=2)))
    fig.add_trace(go.Scatter(x=expense_series.index, y=expense_series.values, mode='lines+markers', name='Expenses', line=dict(color='dodgerblue', width=2)))
    fig.add_trace(go.Scatter(x=future_dates, y=revenue_forecast.values, mode='lines+markers', name='Forecasted Revenue', line=dict(color='deepskyblue', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=future_dates, y=expense_forecast.values, mode='lines+markers', name='Forecasted Expenses', line=dict(color='dodgerblue', width=2, dash='dash')))
    fig.update_layout(title="Monthly Revenue vs Expenses (Historical + 36-Month Forecast)", xaxis_title="Date", yaxis_title="Amount ($)", template="plotly_dark", hovermode='x unified', legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'), font=dict(size=14), margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Performance (XGBoost)")
    features = ['Customer_Footfall', 'Marketing_Spend', 'Food_Costs', 'Labor_Costs', 'Rent', 'Utilities', 'Revenue', 'Delivery_Ratio', 'DineIn_Ratio']
    X = df[features]
    y = df['Profit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(random_state=42)
    param_grid = {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [3]}
    grid = GridSearchCV(model, param_grid, cv=5, scoring='r2')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.markdown(f"""
    - **MAE**: ${mae:,.2f}  
    - **RMSE**: ${rmse:,.2f}  
    - **R² Score**: {r2:.2f}  

    The predictive model shows reasonable performance. Lower RMSE and MAE suggest accurate short-term projections. The R² score indicates the proportion of variance explained by the model.
    """)

    st.subheader("Strategic Recommendations")
    st.markdown("""
    - Optimize food and labor costs to improve profit margins.
    - Focus marketing budget on underperforming months.
    - Monitor customer trends to enhance engagement in slow seasons.
    - Plan staffing and inventory based on seasonality and forecasts.
    """)

# CM_dashboard.py (Updated)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import DateOffset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from generate_report import create_html_report

st.set_page_config(page_title="Cafe Mocha Dashboard", layout="wide")

# Sidebar Navigation
page = st.sidebar.radio("Navigate", ["Home", "Financial Insights", "Forecasting", "Sentiment Insights"])

@st.cache_data
def load_reviews():
    df = pd.read_csv("CafeMocha_Sentiment_Annotated.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Clean_Review'] = df['Clean_Review'].astype(str)
    return df

# Common keywords
dish_keywords = ['pasta', 'burger', 'coffee', 'pizza', 'mocha', 'sandwich', 'wrap', 'brownie', 'smoothie']
aspects = ['food', 'service', 'ambience', 'pricing', 'staff', 'menu', 'cleanliness', 'waiting time', 'music', 'lighting']

if page == "Sentiment Insights":
    st.title("💬 Customer Sentiment Dashboard: Cafe Mocha")
    df = load_reviews()

    for word in dish_keywords + aspects:
        df[word] = df['Clean_Review'].str.contains(fr'\b{word}\b', case=False)

    # WordCloud
    def wordcloud_plot(sentiment='LABEL_2'):
        text = ' '.join(df[df['Predicted_Label'] == sentiment]['Clean_Review'])
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        return wc

    # Bubble Chart
    def bubble_chart():
        results = []
        for asp in aspects:
            for label in df['Predicted_Label'].unique():
                count = df[(df[asp]) & (df['Predicted_Label'] == label)].shape[0]
                results.append({'Aspect': asp, 'Sentiment': label, 'Count': count})
        df_bubble = pd.DataFrame(results)
        return px.scatter(df_bubble, x='Aspect', y='Sentiment', size='Count', color='Sentiment', size_max=60,
                          title="Sentiment Distribution by Business Aspect")

    def dish_sentiment_bar():
        results = []
        for dish in dish_keywords:
            for label in df['Predicted_Label'].unique():
                count = df[(df[dish]) & (df['Predicted_Label'] == label)].shape[0]
                results.append({'Dish': dish, 'Sentiment': label, 'Count': count})
        return px.bar(pd.DataFrame(results), x='Dish', y='Count', color='Sentiment', barmode='group',
                      title='Sentiment Breakdown by Dish')

    def heatmap():
        matrix = []
        for dish in dish_keywords:
            row = [df[(df[dish]) & (df['Predicted_Label'] == label)].shape[0] for label in ['LABEL_0', 'LABEL_1', 'LABEL_2']]
            matrix.append(row)
        heatmap_df = pd.DataFrame(matrix, index=dish_keywords, columns=['Negative', 'Neutral', 'Positive'])
        return px.imshow(heatmap_df, text_auto=True, title='Dish vs Sentiment Heatmap')

    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    monthly = df.groupby(['Month', 'Predicted_Label']).size().reset_index(name='Count')
    fig_monthly = px.line(monthly, x='Month', y='Count', color='Predicted_Label', title='📆 Monthly Sentiment Trend')

    def radar_chart():
        radar_data = []
        for asp in aspects:
            score = df[df[asp]].groupby('Predicted_Label').size().to_dict()
            radar_data.append({
                'Aspect': asp,
                'Positive': score.get('LABEL_2', 0),
                'Negative': score.get('LABEL_0', 0)
            })
        radar_df = pd.DataFrame(radar_data)
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=radar_df['Positive'], theta=radar_df['Aspect'], fill='toself', name='Positive'))
        fig.add_trace(go.Scatterpolar(r=radar_df['Negative'], theta=radar_df['Aspect'], fill='toself', name='Negative'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title='Radar Chart: Avg Sentiment per Aspect')
        return fig

    def donut():
        rating_count = df['Rating'].value_counts().sort_index()
        return px.pie(names=rating_count.index, values=rating_count.values, hole=0.5, title='⭐ Rating Distribution')

    def gap_chart():
        results = []
        for dish in dish_keywords:
            total = df[dish].sum()
            pos = df[(df[dish]) & (df['Predicted_Label'] == 'LABEL_2')].shape[0]
            results.append({'Dish': dish, 'Total Mentions': total, 'Positive Mentions': pos})
        df_gap = pd.DataFrame(results)
        df_gap['Gap'] = df_gap['Total Mentions'] - df_gap['Positive Mentions']
        return px.bar(df_gap, x='Dish', y='Gap', title='🚨 Gap: Mentions vs Positives', color='Gap')

    st.subheader("Sentiment Word Clouds")
    col1, col2 = st.columns(2)
    with col1:
        st.image(wordcloud_plot('LABEL_2').to_array(), caption='Positive Reviews')
    with col2:
        st.image(wordcloud_plot('LABEL_0').to_array(), caption='Negative Reviews')

    st.plotly_chart(bubble_chart(), use_container_width=True)
    st.plotly_chart(dish_sentiment_bar(), use_container_width=True)
    st.plotly_chart(heatmap(), use_container_width=True)
    st.plotly_chart(fig_monthly, use_container_width=True)
    st.plotly_chart(radar_chart(), use_container_width=True)
    st.plotly_chart(donut(), use_container_width=True)
    st.plotly_chart(gap_chart(), use_container_width=True)

    st.info("These insights help identify strong and weak points across food, service, ambience, and dishes based on real feedback.")
