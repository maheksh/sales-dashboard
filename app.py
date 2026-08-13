import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Sales Data Analysis Dashboard", layout="wide")
st.title("📊 Sales Data Analysis Dashboard")

# ---------------- Load & clean data ----------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, encoding="latin1")
    df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]
    df = df.drop_duplicates()
    df = df.dropna(subset=["Sales", "Profit", "Discount", "Quantity"])

    # Guard against divide-by-zero / inf leaking into Profit_Margin
    df["Profit_Margin"] = df["Profit"] / df["Sales"]
    df["Profit_Margin"] = df["Profit_Margin"].replace([np.inf, -np.inf], np.nan)

    # Convert date if exists
    if "Order_Date" in df.columns:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"])
        df["Year"] = df["Order_Date"].dt.year
        df["Month"] = df["Order_Date"].dt.month
        df["Quarter"] = df["Order_Date"].dt.quarter

    return df

# Load data - use relative path for cloud deployment
df = load_data("dsproject1.csv")

# Resolve the sub-category column name defensively — CSV headers vary
SUBCAT_CANDIDATES = ["Sub_Category", "SubCategory", "Sub_category", "Subcategory"]
subcat_col = next((c for c in SUBCAT_CANDIDATES if c in df.columns), None)

# Check for customer segment column
segment_candidates = ["Customer_Segment", "Segment", "CustomerSegment"]
segment_col = next((c for c in segment_candidates if c in df.columns), None)

# Check for customer columns
customer_candidates = ["Customer_ID", "CustomerID", "Customer_Name", "CustomerName"]
customer_col = next((c for c in customer_candidates if c in df.columns), None)

# ---------------- Sidebar filters ----------------
st.sidebar.subheader("🔍 Filters")
region_filter = st.sidebar.multiselect(
    "Filter by Region", options=df["Region"].unique(), default=list(df["Region"].unique())
)
category_filter = st.sidebar.multiselect(
    "Filter by Category", options=df["Category"].unique(), default=list(df["Category"].unique())
)

# Year filter if date exists
year_filter = None
if "Year" in df.columns:
    year_filter = st.sidebar.multiselect(
        "Filter by Year", 
        options=sorted(df["Year"].unique()), 
        default=sorted(df["Year"].unique())
    )

zscore_threshold = st.sidebar.slider(
    "Anomaly Sensitivity (Z-score threshold)", min_value=1.5, max_value=4.0, value=3.0, step=0.25
)

# .copy() avoids the earlier bug where df_filtered was a view and new
# column assignments (Profit_Zscore) didn't reliably register on rerun.
df_filtered = df[df["Region"].isin(region_filter) & df["Category"].isin(category_filter)].copy()

# Apply year filter if exists
if year_filter is not None:
    df_filtered = df_filtered[df_filtered["Year"].isin(year_filter)].copy()

# ---------------- Data Quality Metrics in Sidebar ----------------
with st.sidebar.expander("📊 Data Quality"):
    st.write(f"Total rows loaded: {len(df):,}")
    st.write(f"Filtered rows: {len(df_filtered):,}")
    st.write(f"Columns: {len(df.columns)}")
    missing = df_filtered.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        st.write("Missing values:")
        st.write(missing)
    else:
        st.write("✅ No missing values")

# ---------------- Export Functionality ----------------
@st.cache_data
def convert_df_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode('utf-8')

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Export Data")

csv = convert_df_to_csv(df_filtered)
st.sidebar.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name="filtered_sales_data.csv",
    mime="text/csv",
)

if df_filtered.empty:
    st.warning("No data matches the current filters.")
    st.stop()

# ---------------- KPIs ----------------
st.markdown("### 📈 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", f"${df_filtered['Sales'].sum():,.0f}", 
            delta=f"{len(df_filtered):,} orders")
col2.metric("Total Profit", f"${df_filtered['Profit'].sum():,.0f}",
            delta=f"{df_filtered['Profit_Margin'].mean():.1%} avg margin")
col3.metric("Avg Discount", f"{df_filtered['Discount'].mean():.2%}")
col4.metric("Avg Order Value", f"${df_filtered['Sales'].mean():,.0f}")

# ---------------- Performance Alerts ----------------
st.markdown("### 🚨 Performance Alerts")
alerts_shown = False

# Profit alert
if df_filtered["Profit"].sum() < 0:
    st.error(f"⚠️ Overall profit is negative (${df_filtered['Profit'].sum():,.0f}) for current filters!")
    alerts_shown = True

# Discount alert
high_disc_avg = df_filtered[df_filtered["Discount"] > 0.4]["Profit"].mean()
if len(df_filtered[df_filtered["Discount"] > 0.4]) > 0 and high_disc_avg < 0:
    st.warning(f"⚠️ Average profit is negative (${high_disc_avg:,.2f}) for orders with >40% discount")
    alerts_shown = True

# Low margin alert
low_margin_cats = df_filtered.groupby("Category")["Profit_Margin"].mean()
low_margin_cats = low_margin_cats[low_margin_cats < 0.05]
if len(low_margin_cats) > 0:
    st.info(f"ℹ️ Categories with <5% margin: {', '.join(low_margin_cats.index)}")
    alerts_shown = True

if not alerts_shown:
    st.success("✅ No performance alerts for current filters")

# ---------------- Create Tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualizations", "🔍 Anomaly Detection", "📊 Data Explorer", "📋 Summary Stats"])

with tab1:
    # ---------------- Chart 1: Profit by Category ----------------
    st.subheader("Profit by Category")
    cat_profit = df_filtered.groupby("Category")["Profit"].sum()
    st.bar_chart(cat_profit)

    # ---------------- Chart 2: Sales by Region ----------------
    st.subheader("Sales by Region")
    region_sales = df_filtered.groupby("Region")["Sales"].sum()
    st.bar_chart(region_sales)

    # ---------------- Chart 3: Discount vs Profit (Interactive) ----------------
    st.subheader("Discount vs Profit (Interactive)")
    fig = px.scatter(
        df_filtered, 
        x="Discount", 
        y="Profit", 
        color="Category" if len(category_filter) > 1 else None,
        hover_data=["Sales", "Quantity"] if subcat_col is None else [subcat_col, "Sales"],
        title="Discount vs Profit Relationship",
        opacity=0.6
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Correlation Heatmap ----------------
    st.subheader("🔥 Feature Correlation")
    corr_cols = ["Sales", "Quantity", "Discount", "Profit", "Profit_Margin"]
    corr_matrix = df_filtered[corr_cols].corr()

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, cmap="Blues", ax=ax2)
    st.pyplot(fig2)

    # ---------------- Profit Margin Distribution ----------------
    st.subheader("📊 Profit Margin Distribution")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    df_filtered["Profit_Margin"].hist(bins=50, ax=ax3, color="#4CAF50", alpha=0.7)
    ax3.axvline(0, color="red", linestyle="--", label="Break-even")
    ax3.set_xlabel("Profit Margin")
    ax3.set_ylabel("Frequency")
    ax3.legend()
    st.pyplot(fig3)

    # ---------------- Time Series Analysis ----------------
    if "Order_Date" in df.columns:
        st.subheader("📅 Sales Trend Over Time")
        time_sales = df_filtered.groupby(df_filtered["Order_Date"].dt.to_period("M"))["Sales"].sum()
        time_sales.index = time_sales.index.astype(str)
        st.line_chart(time_sales)

    # ---------------- Customer Segment Analysis ----------------
    if segment_col:
        st.subheader("👥 Customer Segment Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Profit by Segment**")
            segment_profit = df_filtered.groupby(segment_col)["Profit"].sum()
            st.bar_chart(segment_profit)
        
        with col2:
            st.write("**Sales by Segment**")
            segment_sales = df_filtered.groupby(segment_col)["Sales"].sum()
            st.bar_chart(segment_sales)

    # ---------------- Top Products ----------------
    if subcat_col:
        st.subheader("🏅 Top 10 Products by Profit")
        top_products = df_filtered.groupby(subcat_col).agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Profit_Margin=("Profit_Margin", "mean"),
            Order_Count=("Profit", "count")
        ).sort_values("Total_Profit", ascending=False).head(10)
        
        st.dataframe(top_products.style.format({
            "Total_Sales": "${:,.0f}",
            "Total_Profit": "${:,.0f}",
            "Profit_Margin": "{:.1%}"
        }))

    # ---------------- Top Customers ----------------
    if customer_col:
        st.subheader("👤 Top 10 Customers by Revenue")
        top_customers = df_filtered.groupby(customer_col).agg(
            Total_Spent=("Sales", "sum"),
            Orders=("Profit", "count"),
            Total_Profit=("Profit", "sum")
        ).sort_values("Total_Spent", ascending=False).head(10)
        st.dataframe(top_customers)

with tab2:
    # ---------------- Anomaly Detection (Z-score) ----------------
    st.subheader("Anomaly Detection (Z-score on Profit)")

    profit_mean = df_filtered["Profit"].mean()
    profit_std = df_filtered["Profit"].std()

    if not profit_std or np.isnan(profit_std) or profit_std == 0:
        st.info("Not enough variation in Profit for the current filters to compute z-scores.")
        df_filtered["Profit_Zscore"] = np.nan
        anomalies = df_filtered.iloc[0:0]
    else:
        df_filtered["Profit_Zscore"] = (df_filtered["Profit"] - profit_mean) / profit_std
        anomalies = df_filtered[np.abs(df_filtered["Profit_Zscore"]) > zscore_threshold]

    with st.expander("Debug: Profit z-score distribution"):
        st.write(df_filtered["Profit_Zscore"].describe())

    st.write(f"Detected **{len(anomalies)}** anomalous transactions (threshold: {zscore_threshold})")

    # Export anomalies
    if len(anomalies) > 0:
        csv_anomalies = convert_df_to_csv(anomalies)
        st.download_button(
            label="📥 Download Anomalies as CSV",
            data=csv_anomalies,
            file_name="anomalies.csv",
            mime="text/csv",
        )

    display_cols = ["Category", "Sales", "Discount", "Profit"]
    if subcat_col:
        display_cols.insert(1, subcat_col)
    else:
        st.caption("⚠️ No sub-category column found in this file — showing without it.")
    st.dataframe(anomalies[display_cols])

with tab3:
    # ---------------- Data Explorer ----------------
    st.subheader("🔍 Explore Filtered Data")
    
    # Column selector
    all_columns = df_filtered.columns.tolist()
    default_columns = ["Category", "Sales", "Profit", "Discount", "Quantity"]
    if subcat_col and subcat_col in all_columns:
        default_columns.insert(1, subcat_col)
    
    selected_columns = st.multiselect(
        "Select columns to display",
        options=all_columns,
        default=[c for c in default_columns if c in all_columns]
    )
    
    # Sort options
    sort_col = st.selectbox("Sort by", options=selected_columns)
    sort_order = st.radio("Sort order", ["Descending", "Ascending"])
    
    # Display data
    if selected_columns:
        display_df = df_filtered[selected_columns].sort_values(
            by=sort_col, 
            ascending=(sort_order == "Ascending")
        )
        st.dataframe(display_df.head(100))
        st.caption(f"Showing first 100 rows out of {len(display_df):,} total rows")
    else:
        st.info("Please select at least one column to display")

with tab4:
    # ---------------- Summary Statistics ----------------
    st.subheader("Summary Statistics")
    
    numeric_cols = ["Sales", "Profit", "Discount", "Quantity", "Profit_Margin"]
    available_numeric = [c for c in numeric_cols if c in df_filtered.columns]
    
    if available_numeric:
        st.dataframe(df_filtered[available_numeric].describe())
        
        # Additional statistics
        st.subheader("Additional Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Median Profit", f"${df_filtered['Profit'].median():,.2f}")
            st.metric("Std Dev Profit", f"${df_filtered['Profit'].std():,.2f}")
        
        with col2:
            st.metric("75th Percentile Sales", f"${df_filtered['Sales'].quantile(0.75):,.2f}")
            st.metric("Max Discount", f"{df_filtered['Discount'].max():.1%}")
        
        with col3:
            st.metric("Negative Profit Orders", f"{(df_filtered['Profit'] < 0).sum():,}")
            st.metric("% Negative Profit", f"{(df_filtered['Profit'] < 0).mean():.1%}")

# ---------------- Predictive Model: Profit Prediction ----------------
st.markdown("---")
st.subheader("📈 Profit Prediction Model")

model_df = df_filtered[["Sales", "Discount", "Quantity", "Profit"]].dropna()

MIN_ROWS_FOR_MODEL = 10
if len(model_df) < MIN_ROWS_FOR_MODEL:
    st.info(
        f"Not enough rows ({len(model_df)}) under the current filters to train a reliable model. "
        f"Need at least {MIN_ROWS_FOR_MODEL}."
    )
else:
    X = model_df[["Sales", "Discount", "Quantity"]]
    y = model_df["Profit"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(X_test) == 0:
        st.info("Filtered dataset too small to hold out a test set for evaluation.")
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)

        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Model R² Score", f"{r2:.2f}")
        mcol2.metric("Mean Absolute Error", f"${mae:,.2f}")

        st.caption("Linear Regression model predicting Profit from Sales, Discount, and Quantity.")

        coef_df = pd.DataFrame({
            "Feature": ["Sales", "Discount", "Quantity"],
            "Impact on Profit": model.coef_
        })
        st.write("**Feature Impact on Profit:**")
        st.dataframe(coef_df)

# ---------------- Key Insights ----------------
st.markdown("---")
st.subheader("💡 Key Insights")

# Calculate dynamic insights from filtered data
insights = []

# Insight 1: Discount analysis
high_discount_threshold = 0.3
high_discount_mask = df_filtered["Discount"] >= high_discount_threshold
if high_discount_mask.any():
    high_discount_df = df_filtered[high_discount_mask]
    neg_profit_pct = (high_discount_df["Profit"] < 0).mean() * 100
    avg_profit_high_discount = high_discount_df["Profit"].mean()
    avg_profit_low_discount = df_filtered[~high_discount_mask]["Profit"].mean()
    
    insights.append(
        f"📉 **High Discount Impact:** Transactions with discounts ≥ {high_discount_threshold:.0%} "
        f"({len(high_discount_df):,} orders) have {neg_profit_pct:.0f}% negative profit rate. "
        f"Average profit drops from ${avg_profit_low_discount:,.2f} to ${avg_profit_high_discount:,.2f} "
        f"when discounts exceed this threshold."
    )
else:
    insights.append("✅ **Discount Levels:** No transactions with discounts ≥ 30% in current selection.")

# Insight 2: Category performance
if len(category_filter) > 1:
    cat_stats = df_filtered.groupby("Category").agg(
        total_profit=("Profit", "sum"),
        avg_margin=("Profit_Margin", "mean"),
        order_count=("Profit", "count")
    ).sort_values("total_profit", ascending=False)
    
    best_cat = cat_stats.index[0]
    worst_cat = cat_stats.index[-1]
    best_profit = cat_stats.loc[best_cat, "total_profit"]
    worst_profit = cat_stats.loc[worst_cat, "total_profit"]
    
    insights.append(
        f"🏆 **Category Performance:** **{best_cat}** leads with ${best_profit:,.0f} total profit "
        f"(avg margin: {cat_stats.loc[best_cat, 'avg_margin']:.1%}), while **{worst_cat}** "
        f"shows ${worst_profit:,.0f} (avg margin: {cat_stats.loc[worst_cat, 'avg_margin']:.1%})."
    )
elif len(category_filter) == 1:
    cat_name = category_filter[0]
    avg_margin = df_filtered["Profit_Margin"].mean()
    total_profit = df_filtered["Profit"].sum()
    insights.append(
        f"📊 **Single Category View:** {cat_name} shows average profit margin of {avg_margin:.1%} "
        f"with total profit of ${total_profit:,.0f} across {len(df_filtered):,} orders."
    )

# Insight 3: Anomaly impact
if len(anomalies) > 0:
    total_profit = df_filtered["Profit"].sum()
    anomaly_profit = anomalies["Profit"].sum()
    anomaly_pct = (anomaly_profit / total_profit * 100) if total_profit != 0 else 0
    
    if anomaly_profit < 0:
        insights.append(
            f"⚠️ **Loss Concentration:** {len(anomalies)} anomalous transactions represent "
            f"${anomaly_profit:,.0f} in losses ({abs(anomaly_pct):.1f}% of total profit), "
            f"suggesting targeted review could significantly improve profitability."
        )
    else:
        insights.append(
            f"✨ **Positive Outliers:** {len(anomalies)} exceptional transactions contribute "
            f"${anomaly_profit:,.0f} in profit ({anomaly_pct:.1f}% of total)."
        )
else:
    insights.append("✅ **Data Quality:** No statistical anomalies detected with current sensitivity settings.")

# Insight 4: Regional performance
if len(region_filter) > 1:
    region_profit = df_filtered.groupby("Region")["Profit"].sum().sort_values(ascending=False)
    top_region = region_profit.index[0]
    bottom_region = region_profit.index[-1]
    
    insights.append(
        f"🌍 **Regional Spread:** **{top_region}** generates the highest profit "
        f"(${region_profit[top_region]:,.0f}), while **{bottom_region}** shows "
        f"${region_profit[bottom_region]:,.0f} — a gap of ${region_profit[top_region] - region_profit[bottom_region]:,.0f}."
    )

# Insight 5: Model performance (if model was trained)
if len(model_df) >= MIN_ROWS_FOR_MODEL and len(X_test) > 0:
    if r2 > 0.7:
        model_quality = "strongly predicts"
    elif r2 > 0.4:
        model_quality = "moderately predicts"
    else:
        model_quality = "weakly predicts"
    
    top_feature = coef_df.loc[coef_df["Impact on Profit"].abs().idxmax(), "Feature"]
    insights.append(
        f"🤖 **Model Insight:** Linear regression {model_quality} profit (R²={r2:.2f}). "
        f"**{top_feature}** has the strongest impact on profit prediction."
    )

# Display insights or fallback message
if insights:
    for i, insight in enumerate(insights, 1):
        st.markdown(f"{i}. {insight}")
        if i < len(insights):
            st.markdown("---")
else:
    st.info("Select more data points to generate insights.")

# Footer
st.markdown("---")
st.markdown("📊 **Sales Data Analysis Dashboard** | Built with Streamlit")
