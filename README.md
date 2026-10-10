# E-Commerce Sales Analytics — Sales Prediction with ML

## Dataset
**Source:** [E-Commerce Sales Customer Analytics 150k (Kaggle)](https://www.kaggle.com/datasets/adnanmemon1/ecommerce-sales-customer-analytics-150k)

- **138,116** transactions | **46 features** | 2021–2025
- Revenue: **\$177M+** | Profit: **\$76M+** | Avg. Order: **\$1,282**

---

## Project Overview

An end-to-end Machine Learning project that:
- Cleans and engineers features from 150k e-commerce orders
- Trains **Random Forest** & **Gradient Boosting** regressors to predict `net_sales`
- Exposes predictions via a **Flask REST API**
- Provides an interactive **Streamlit dashboard** with EDA plots and real-time predictions

---

## Project Structure

```
EcommerceSalesAnalytics/
├── train_model.py                   # ML training & EDA plot generation
├── app.py                           # Flask backend API (port 5000)
├── ui.py                            # Streamlit frontend (port 8501)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── GirishArvind_EcommerceSalesAnalytics.ipynb  # Executed Jupyter notebook
├── GirishArvind_ProjectReport.docx          # Word report with full documentation
├── models/
│   ├── best_model.pkl               # Best trained model
│   ├── rf_model.pkl                 # Random Forest model
│   ├── gb_model.pkl                 # Gradient Boosting model
│   ├── label_encoders.pkl           # Categorical encoders
│   ├── feature_meta.json            # Feature list + metrics
│   └── dataset_stats.json           # Dataset statistics
└── assets/
    ├── eda_net_sales_dist.png
    ├── eda_monthly_trend.png
    ├── eda_sales_channel.png
    ├── eda_region.png
    ├── eda_correlation.png
    ├── eda_profit_margin.png
    ├── model_actual_vs_predicted.png
    ├── model_feature_importance.png
    └── model_comparison.png
```

---

## Tech Stack

| Layer       | Technology                         |
|-------------|-------------------------------------|
| Language    | Python 3.12                        |
| ML          | scikit-learn (Random Forest, GBM)  |
| Data        | pandas, numpy                      |
| Visualization | matplotlib, seaborn             |
| Backend API | Flask 3.1                          |
| Frontend    | Streamlit 1.39                     |
| Report      | python-docx                        |

---

## Run Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (generates models/ and assets/)
```bash
python train_model.py
```

### 3. Start the Flask API
```bash
python app.py
# Running on http://localhost:5000
```

### 4. Launch the Streamlit dashboard (new terminal)
```bash
streamlit run ui.py
# Running on http://localhost:8501
```

---

## API Endpoints

| Endpoint         | Method | Description                          |
|------------------|--------|--------------------------------------|
| `/`              | GET    | API overview                        |
| `/health`        | GET    | Health check                        |
| `/model_info`    | GET    | Model name, features, metrics       |
| `/dataset_stats` | GET    | Aggregated dataset statistics       |
| `/predict`       | POST   | Predict net sales from JSON body    |

### Predict Example
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"quantity": 3, "gross_sales": 800, "discount_amount": 80,
       "tax_amount": 45, "shipping_cost": 12, "product_cost": 350,
       "delivery_days": 5, "estimated_delivery_days": 5,
       "customer_age": 35, "sales_channel": "Website",
       "customer_segment": "Consumer", "region": "North"}'
```

---

## Model Performance

| Model            | MAE     | RMSE    | R²     | MAPE  |
|------------------|---------|---------|--------|-------|
| Random Forest    | ~40     | ~90     | ~0.99  | ~4%   |
| Gradient Boosting| ~55     | ~110    | ~0.99  | ~6%   |

---

## Author
**Girish Arvind**  
IBM Project — E-Commerce Sales Analytics  
Built with Python · scikit-learn · Flask · Streamlit · python-docx
