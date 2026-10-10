"""
E-Commerce Sales Analytics - Streamlit Frontend UI
Interactive dashboard for EDA and real-time sales predictions.
"""

import streamlit as st
import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json

API_URL  = "http://localhost:5000"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="E-Commerce Sales Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding-bottom: 0.3rem;
    }
    .metric-card {
        background: #f0f4ff; border-radius: 10px;
        padding: 16px; text-align: center;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.6rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=80)
    st.markdown("## E-Commerce Analytics")
    st.markdown("**ML-powered sales forecasting**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔮 Predict Sales", "📈 EDA Plots", "ℹ️ Model Info"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Dataset: E-Commerce Sales Analytics (150k orders)")
    st.caption("Backend: Flask API | Model: Random Forest")

# ── Helper: call API ───────────────────────────────────────────────────────────
def api_get(endpoint):
    try:
        r = requests.get(f"{API_URL}/{endpoint}", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown('<div class="main-header">🛒 E-Commerce Sales Analytics</div>',
                unsafe_allow_html=True)
    st.markdown("Real-time insights from 138K+ orders (2021–2025)")
    st.markdown("---")

    stats = api_get("dataset_stats")
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Orders",       f"{stats.get('total_records', 0):,}")
        c2.metric("Total Revenue",      f"${stats.get('total_revenue', 0)/1e6:.1f}M")
        c3.metric("Total Profit",       f"${stats.get('total_profit', 0)/1e6:.1f}M")
        c4.metric("Avg Order Value",    f"${stats.get('avg_order_value', 0):,.2f}")

        st.markdown("---")
        sub1, sub2 = st.columns(2)
        sub1.metric("Date Range",   stats.get("date_range", "N/A"))
        sub2.metric("Avg Rating",   stats.get("avg_rating", "N/A"))
    else:
        st.warning("Backend API is not running. Start it with: `python app.py`")

    st.markdown("---")
    st.subheader("Key EDA Charts")

    assets_dir = os.path.join(BASE_DIR, "assets")
    charts = [
        ("eda_monthly_trend.png",   "Monthly Revenue Trend"),
        ("eda_net_sales_dist.png",  "Net Sales Distribution"),
        ("eda_sales_channel.png",   "Revenue by Sales Channel"),
        ("eda_region.png",          "Revenue by Region"),
    ]
    cols = st.columns(2)
    for i, (fname, title) in enumerate(charts):
        fp = os.path.join(assets_dir, fname)
        if os.path.exists(fp):
            cols[i % 2].image(fp, caption=title, use_container_width=True)
        else:
            cols[i % 2].info(f"Run `train_model.py` to generate: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT SALES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Sales":
    st.markdown("## 🔮 Predict Net Sales")
    st.markdown("Enter order details to get a real-time sales prediction.")
    st.markdown("---")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Order Details")
            quantity       = st.slider("Quantity",         1, 20, 3)
            gross_sales    = st.number_input("Gross Sales ($)", 10.0, 20000.0, 500.0, step=10.0)
            discount_amount= st.number_input("Discount Amount ($)", 0.0, 5000.0, 50.0, step=5.0)
            tax_amount     = st.number_input("Tax Amount ($)", 0.0, 2000.0, 30.0, step=1.0)
            shipping_cost  = st.number_input("Shipping Cost ($)", 0.0, 500.0, 10.0, step=0.5)
            product_cost   = st.number_input("Product Cost ($)", 1.0, 10000.0, 250.0, step=10.0)

        with col2:
            st.subheader("Delivery & Customer")
            delivery_days     = st.slider("Delivery Days",    1, 30, 5)
            est_delivery_days = st.slider("Est. Delivery Days", 1, 30, 5)
            customer_age      = st.slider("Customer Age",    18, 90, 35)
            loyalty_earned    = st.slider("Loyalty Points Earned",   0, 500, 50)
            loyalty_redeemed  = st.slider("Loyalty Points Redeemed", 0, 500, 10)
            order_count       = st.slider("Customer Order Count",    1, 50, 5)

        with col3:
            st.subheader("Categorical Attributes")
            sales_channel    = st.selectbox("Sales Channel",
                ["Website", "Mobile App", "Social Media", "In-Store", "Marketplace"])
            payment_method   = st.selectbox("Payment Method",
                ["Credit Card", "Debit Card", "Digital Wallet", "Cash on Delivery",
                 "Bank Transfer", "PayPal"])
            shipping_method  = st.selectbox("Shipping Method",
                ["Standard", "Express", "Economy", "Overnight", "Two-Day"])
            customer_segment = st.selectbox("Customer Segment",
                ["Consumer", "Premium", "VIP", "Corporate"])
            region           = st.selectbox("Region",
                ["North", "South", "East", "West", "Central"])
            order_status     = st.selectbox("Order Status",
                ["Completed", "Pending", "Shipped", "Cancelled", "Returned"])

        submitted = st.form_submit_button("🚀 Predict Net Sales", use_container_width=True)

    if submitted:
        discount_ratio   = discount_amount / (gross_sales + 1e-9)
        revenue_per_unit = gross_sales / (quantity + 1e-9)
        is_delayed       = 1 if delivery_days > est_delivery_days else 0
        year, month, dow = 2025, 6, 2

        payload = {
            "quantity": quantity,
            "gross_sales": gross_sales,
            "discount_amount": discount_amount,
            "tax_amount": tax_amount,
            "shipping_cost": shipping_cost,
            "product_cost": product_cost,
            "delivery_days": delivery_days,
            "estimated_delivery_days": est_delivery_days,
            "customer_age": customer_age,
            "loyalty_points_earned": loyalty_earned,
            "loyalty_points_redeemed": loyalty_redeemed,
            "customer_order_count": order_count,
            "discount_ratio": discount_ratio,
            "revenue_per_unit": revenue_per_unit,
            "is_delayed": is_delayed,
            "year": year, "month": month, "dow": dow,
            "sales_channel": sales_channel,
            "payment_method": payment_method,
            "shipping_method": shipping_method,
            "customer_segment": customer_segment,
            "region": region,
            "order_status": order_status,
            "return_status": "No Return",
        }

        with st.spinner("Predicting ..."):
            try:
                r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                result = r.json()

                st.markdown("---")
                st.success("Prediction Complete!")
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted Net Sales",  f"${result['predicted_net_sales']:,.2f}")
                m2.metric("Estimated Profit",     f"${result['estimated_profit']:,.2f}")
                m3.metric("Model Used",            result['model_used'])

            except Exception as e:
                st.error(f"API Error: {e}. Make sure `python app.py` is running.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA PLOTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA Plots":
    st.markdown("## 📈 Exploratory Data Analysis")
    st.markdown("---")

    assets_dir = os.path.join(BASE_DIR, "assets")
    all_charts = [
        ("eda_net_sales_dist.png",        "Net Sales Distribution"),
        ("eda_monthly_trend.png",          "Monthly Revenue Trend"),
        ("eda_sales_channel.png",          "Revenue by Sales Channel"),
        ("eda_region.png",                 "Revenue by Region"),
        ("eda_correlation.png",            "Feature Correlation Heatmap"),
        ("eda_profit_margin.png",          "Profit Margin Distribution"),
        ("model_actual_vs_predicted.png",  "Actual vs Predicted"),
        ("model_feature_importance.png",   "Feature Importance"),
        ("model_comparison.png",           "Model Comparison"),
    ]

    if not os.path.exists(assets_dir) or not os.listdir(assets_dir):
        st.warning("No plots found. Run `python train_model.py` first.")
    else:
        for fname, title in all_charts:
            fp = os.path.join(assets_dir, fname)
            if os.path.exists(fp):
                st.image(fp, caption=title, use_container_width=True)
                st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ Model Info":
    st.markdown("## ℹ️ Model Information & Performance")
    st.markdown("---")

    info = api_get("model_info")
    if info:
        st.markdown(f"**Best Model:** `{info['best_model']}`")
        st.markdown(f"**Prediction Target:** `{info['target']}`")

        st.subheader("Model Performance Metrics")
        metrics_data = []
        for m in [info['rf_metrics'], info['gb_metrics']]:
            metrics_data.append({
                "Model": m['model'],
                "MAE ($)": f"{m['MAE']:,}",
                "RMSE ($)": f"{m['RMSE']:,}",
                "R² Score": m['R2'],
                "MAPE (%)": m['MAPE'],
            })
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

        st.subheader("Feature List")
        feat_df = pd.DataFrame({"Feature": info['features']})
        feat_df.index += 1
        st.dataframe(feat_df, use_container_width=True)
    else:
        st.warning("Backend API not reachable. Start with: `python app.py`")
