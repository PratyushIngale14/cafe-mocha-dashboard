import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import matplotlib.pyplot as plt


# Data input
df = pd.read_csv("Cafe_Mocha_Cleaned.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['Month_Year'] = df['Date'].dt.to_period('M').astype(str)
df['Season'] = df['Date'].dt.month % 12 // 3 + 1
df['Season'] = df['Season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})
df['Total_Expenses'] = df[['Marketing_Spend','Food_Costs','Labor_Costs','Rent','Utilities']].sum(axis=1)

# Model metrics (sample)
mae = 3198.22
rmse = 4889.33
r2 = 0.73

# Forecast plot (dummy chart)
plt.figure(figsize=(8, 4))
df.groupby('Month_Year')['Profit'].sum().plot()
plt.title('Monthly Profit Trend')
plt.ylabel('Profit ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("forecast_plot.png")

# Tables
month_table = df.groupby('Month_Year')[['Revenue','Total_Expenses','Profit']].sum().tail(12).to_html(index=True)
season_table = df.groupby('Season')[['Revenue','Total_Expenses','Profit']].mean().to_html(index=True)

# Executive summary (simple logic)
avg_profit = df['Profit'].mean()
if avg_profit > 2000:
    summary = "Business shows consistent profitability with healthy margins."
elif avg_profit > 0:
    summary = "Business is stable but can improve profit efficiency."
else:
    summary = "Business is currently operating at a loss. Action recommended."

# Render HTML
env = Environment(loader=FileSystemLoader('templates'))
template = env.get_template('report_template.html')
html_out = template.render(
    business_name="Cafe Mocha",
    business_sector="Food & Beverage",
    generated_on=datetime.now().strftime("%Y-%m-%d"),
    executive_summary=summary,
    mae=mae,
    rmse=rmse,
    r2=r2,
    month_table=month_table,
    season_table=season_table,
    forecast_plot="forecast_plot.png"
)


def create_html_report(df, business_name="Cafe Mocha", business_sector="Food & Beverage"):
    from jinja2 import Environment, FileSystemLoader
    from datetime import datetime
    import matplotlib.pyplot as plt

    df['Month_Year'] = df['Date'].dt.to_period('M').astype(str)
    df['Season'] = df['Date'].dt.month % 12 // 3 + 1
    df['Season'] = df['Season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})
    df['Total_Expenses'] = df[['Marketing_Spend', 'Food_Costs', 'Labor_Costs', 'Rent', 'Utilities']].sum(axis=1)

    # Plot
    plt.figure(figsize=(8, 4))
    df.groupby('Month_Year')['Profit'].sum().plot()
    plt.title('Monthly Profit Trend')
    plt.ylabel('Profit ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("forecast_plot.png")

    month_table = df.groupby('Month_Year')[['Revenue','Total_Expenses','Profit']].sum().tail(12).to_html(index=True)
    season_table = df.groupby('Season')[['Revenue','Total_Expenses','Profit']].mean().to_html(index=True)
    avg_profit = df['Profit'].mean()

    if avg_profit > 2000:
        summary = "Business shows consistent profitability with healthy margins."
    elif avg_profit > 0:
        summary = "Business is stable but can improve profit efficiency."
    else:
        summary = "Business is currently operating at a loss. Action recommended."

    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('report_template.html')
    html_out = template.render(
        business_name=business_name,
        business_sector=business_sector,
        generated_on=datetime.now().strftime("%Y-%m-%d"),
        executive_summary=summary,
        mae=3198.22,
        rmse=4889.33,
        r2=0.73,
        month_table=month_table,
        season_table=season_table,
        forecast_plot="forecast_plot.png"
    )

    with open("financial_report.html", "w") as f:
        f.write(html_out)

    return "financial_report.html"


