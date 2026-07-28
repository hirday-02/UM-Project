import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from train_models import train_and_save_pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="EduPro Predictive Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #3b82f6;
        color: white;
    }
    .stMetric label {
        font-weight: bold;
    }
    .badge-success {
        background-color: #065f46;
        color: #34d399;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-warning {
        background-color: #78350f;
        color: #fbbf24;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-danger {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data & Joblib ML Artifacts ---
@st.cache_data
def load_data():
    excel_file = 'EduPro Online Platform.xlsx'
    if not os.path.exists(excel_file):
        st.error(f"Dataset '{excel_file}' not found in workspace.")
        st.stop()
        
    excel_wb = pd.ExcelFile(excel_file)
    df_courses = pd.read_excel(excel_wb, 'Courses')
    df_teachers = pd.read_excel(excel_wb, 'Teachers')
    df_transactions = pd.read_excel(excel_wb, 'Transactions')
    
    course_metrics = df_transactions.groupby('CourseID').agg(
        EnrollmentCount=('TransactionID', 'count'),
        TotalRevenue=('Amount', 'sum')
    ).reset_index()
    
    teacher_map = df_transactions.groupby('CourseID')['TeacherID'].agg(
        lambda x: x.mode()[0] if not x.empty else np.nan
    ).reset_index()
    
    df_summary = df_courses.merge(course_metrics, on='CourseID', how='left').merge(
        teacher_map, on='CourseID', how='left'
    ).merge(
        df_teachers, on='TeacherID', how='left', suffixes=('_course', '_teacher')
    )
    
    df_summary['EnrollmentCount'] = df_summary['EnrollmentCount'].fillna(0).astype(int)
    df_summary['TotalRevenue'] = df_summary['TotalRevenue'].fillna(0.0)
    
    cat_benchmarks = df_summary.groupby('CourseCategory').agg(
        CategoryAvgRevenue=('TotalRevenue', 'mean'),
        CategoryAvgEnrollment=('EnrollmentCount', 'mean'),
        CategoryAvgPrice=('CoursePrice', 'mean')
    ).reset_index()
    
    df_summary = df_summary.merge(cat_benchmarks, on='CourseCategory', how='left')
    return df_summary

@st.cache_resource
def load_ml_artifacts():
    # If joblib artifacts do not exist, run training pipeline
    if not (os.path.exists('demand_model.joblib') and os.path.exists('revenue_model.joblib') and os.path.exists('scaler.joblib') and os.path.exists('model_metadata.joblib')):
        st.info("Training ML models and building Joblib artifacts for first-time launch...")
        train_and_save_pipeline()
        
    demand_model = joblib.load('demand_model.joblib')
    revenue_model = joblib.load('revenue_model.joblib')
    scaler = joblib.load('scaler.joblib')
    metadata = joblib.load('model_metadata.joblib')
    
    return demand_model, revenue_model, scaler, metadata

try:
    df_summary = load_data()
    demand_model, revenue_model, scaler, metadata = load_ml_artifacts()
except Exception as e:
    st.error(f"Error initializing dashboard artifacts: {e}")
    st.stop()

# --- Title Header ---
st.title("🎓 EduPro Predictive Analytics & Revenue Forecasting Dashboard")
st.markdown("##### *Empowering Course Demand Forecasting, Pricing Optimization, and Strategic Launch Planning (in ₹ INR)*")
st.divider()

# --- Sidebar Filters ---
st.sidebar.header("🔍 Dashboard Filters")
selected_categories = st.sidebar.multiselect(
    "Course Category", 
    options=list(df_summary['CourseCategory'].unique()),
    default=list(df_summary['CourseCategory'].unique())
)

selected_levels = st.sidebar.multiselect(
    "Course Level", 
    options=list(df_summary['CourseLevel'].unique()),
    default=list(df_summary['CourseLevel'].unique())
)

selected_types = st.sidebar.multiselect(
    "Course Type", 
    options=list(df_summary['CourseType'].unique()),
    default=list(df_summary['CourseType'].unique())
)

price_min = float(df_summary['CoursePrice'].min())
price_max = float(df_summary['CoursePrice'].max())
price_range = st.sidebar.slider("Price Range (₹)", price_min, price_max, (price_min, price_max))

filtered_df = df_summary[
    (df_summary['CourseCategory'].isin(selected_categories)) &
    (df_summary['CourseLevel'].isin(selected_levels)) &
    (df_summary['CourseType'].isin(selected_types)) &
    (df_summary['CoursePrice'] >= price_range[0]) &
    (df_summary['CoursePrice'] <= price_range[1])
]

# --- Executive Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"₹{filtered_df['TotalRevenue'].sum():,.2f}")
col2.metric("Total Enrollments", f"{filtered_df['EnrollmentCount'].sum():,} students")
avg_p = filtered_df['CoursePrice'].mean() if len(filtered_df) > 0 else 0
col3.metric("Avg Course Price", f"₹{avg_p:,.2f}")
col4.metric("Active Courses", f"{len(filtered_df)}")

st.divider()

# --- Dashboard Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Market Overview & Categories", 
    "👨‍🏫 Instructor Performance", 
    "🔮 AI Demand & Revenue Predictor",
    "📈 Model Benchmarks & Feature Importance"
])

# --- TAB 1: MARKET OVERVIEW ---
with tab1:
    st.subheader("Category Performance Benchmarks")
    cat_summary = filtered_df.groupby('CourseCategory').agg(
        TotalRev=('TotalRevenue', 'sum'),
        TotalEnr=('EnrollmentCount', 'sum')
    ).reset_index().sort_values(by='TotalRev', ascending=False)
    
    cA, cB = st.columns(2)
    with cA:
        fig_rev = px.bar(cat_summary, x='TotalRev', y='CourseCategory', orientation='h', title="Total Revenue by Category (₹)", color='TotalRev', color_continuous_scale='Viridis')
        st.plotly_chart(fig_rev, use_container_width=True)
    with cB:
        fig_enr = px.bar(cat_summary, x='TotalEnr', y='CourseCategory', orientation='h', title="Total Enrollments by Category", color='TotalEnr', color_continuous_scale='Cividis')
        st.plotly_chart(fig_enr, use_container_width=True)

# --- TAB 2: INSTRUCTOR PERFORMANCE ---
with tab2:
    st.subheader("Instructor Experience & Rating Influence")
    fig_inst = px.scatter(
        filtered_df, x='YearsOfExperience', y='TotalRevenue', size='EnrollmentCount',
        color='CourseCategory', hover_data=['TeacherName', 'CourseName'],
        title="Instructor Experience vs Revenue Generation (₹)"
    )
    st.plotly_chart(fig_inst, use_container_width=True)

# --- TAB 3: PREDICTION ENGINE (TRAINED ML MODELS) ---
with tab3:
    st.subheader("🔮 Machine Learning Simulator: Forecast Course Demand & Revenue")
    st.markdown("Enter proposed course specs below. Prediction is powered by trained **Lasso** and **Gradient Boosting** models (`.joblib`).")
    
    p1, p2 = st.columns(2)
    with p1:
        input_category = st.selectbox("Target Category", df_summary['CourseCategory'].unique())
        input_level = st.selectbox("Course Level", ["Beginner", "Intermediate", "Advanced"])
        input_price = st.number_input("Proposed Course Price (₹)", min_value=0.0, max_value=1000.0, value=250.0, step=10.0)
        input_duration = st.number_input("Proposed Course Duration (Hours)", min_value=1.0, max_value=100.0, value=25.0)
    with p2:
        input_rating = st.slider("Expected Course Rating", 1.0, 5.0, 4.2)
        input_exp = st.slider("Assigned Instructor Experience (Years)", 1, 30, 8)
        input_trating = st.slider("Instructor Rating", 1.0, 5.0, 4.5)
        domain_match = st.checkbox("Instructor Expertise Matches Category Exactly?", value=True)

    if st.button("🚀 Run ML Predictive Simulation", use_container_width=True):
        # 1. Look up category benchmarks
        cat_row = df_summary[df_summary['CourseCategory'] == input_category]
        if not cat_row.empty:
            cat_avg_rev = cat_row['CategoryAvgRevenue'].values[0]
            cat_avg_enr = cat_row['CategoryAvgEnrollment'].values[0]
            cat_avg_price = cat_row['CategoryAvgPrice'].values[0]
        else:
            cat_avg_rev, cat_avg_enr, cat_avg_price = 35000.0, 165.0, 200.0

        is_paid = 1 if input_price > 0 else 0
        inst_quality = round(input_trating * np.log1p(input_exp), 2)
        exp_match = 1 if domain_match else 0
        level_map = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
        level_encoded = level_map.get(input_level, 1)

        # Categorical bucket helpers
        def get_price_band(p):
            if p == 0: return 'Free'
            elif p <= 100: return 'Budget'
            elif p <= 300: return 'Mid-Tier'
            else: return 'Premium'

        def get_duration_bucket(d):
            if d < 15: return 'Short (<15h)'
            elif d < 15: return 'Medium (15-30h)'
            elif d < 45: return 'Long (30-45h)'
            else: return 'Extensive (45h+)'

        def get_rating_tier(r):
            if r < 2.5: return 'Below Average'
            elif r < 3.5: return 'Average'
            elif r < 4.5: return 'Good'
            else: return 'Top Rated'

        def get_exp_bucket(e):
            if e <= 4: return 'Junior (1-4y)'
            elif e <= 8: return 'Mid-Level (5-8y)'
            else: return 'Senior (9y+)'

        price_band = get_price_band(input_price)
        duration_bucket = get_duration_bucket(input_duration)
        rating_tier = get_rating_tier(input_rating)
        exp_bucket = get_exp_bucket(input_exp)

        # Build feature vector matching metadata['feature_names']
        feature_names = metadata['feature_names']
        X_input = pd.DataFrame(0, index=[0], columns=feature_names)

        # Numerical fields
        num_dict = {
            'CoursePrice': input_price,
            'CourseDuration': input_duration,
            'CourseRating': input_rating,
            'YearsOfExperience': input_exp,
            'TeacherRating': input_trating,
            'CategoryAvgRevenue': cat_avg_rev,
            'CategoryAvgEnrollment': cat_avg_enr,
            'CategoryAvgPrice': cat_avg_price,
            'InstructorQualityScore': inst_quality,
            'ExpertiseMatchScore': exp_match,
            'IsPaid': is_paid,
            'CourseLevel_Encoded': level_encoded
        }

        for col, val in num_dict.items():
            if col in X_input.columns:
                X_input.loc[0, col] = val

        # Categorical one-hot flags
        cat_flags = [
            f"CourseCategory_{input_category}",
            f"PriceBand_{price_band}",
            f"DurationBucket_{duration_bucket}",
            f"RatingTier_{rating_tier}",
            f"ExperienceBucket_{exp_bucket}",
            f"CourseType_{'Paid' if is_paid else 'Free'}"
        ]

        for flag in cat_flags:
            if flag in X_input.columns:
                X_input.loc[0, flag] = 1

        # Scale numerical features using saved StandardScaler
        X_input_scaled = X_input.copy()
        num_to_scale = metadata['num_cols_to_scale']
        X_input_scaled[num_to_scale] = scaler.transform(X_input[num_to_scale])

        # Run Model Predictions (.predict)
        raw_demand_pred = demand_model.predict(X_input_scaled)[0]
        raw_rev_pred = revenue_model.predict(X_input_scaled)[0]

        pred_demand = int(np.clip(round(raw_demand_pred), 100, 220))
        pred_revenue = float(np.maximum(raw_rev_pred, 0.0)) if is_paid else 0.0

        # --- Business Decision Logic (Task 7 Requirements) ---
        # 1. Demand Level
        if pred_demand >= 175:
            demand_level = "High Demand"
            demand_badge = "badge-success"
        elif pred_demand >= 150:
            demand_level = "Medium Demand"
            demand_badge = "badge-warning"
        else:
            demand_level = "Low Demand"
            demand_badge = "badge-danger"

        # 2. Revenue Category
        if pred_revenue >= 60000:
            rev_category = "High Yield"
            rev_badge = "badge-success"
        elif pred_revenue >= 20000:
            rev_category = "Medium Yield"
            rev_badge = "badge-warning"
        else:
            rev_category = "Low Yield"
            rev_badge = "badge-danger"

        # 3. Launch Recommendation & Risk
        if demand_level == "High Demand" and rev_category == "High Yield":
            recommendation = "🚀 Highly Recommended Launch"
            risk_level = "Low Risk"
            risk_badge = "badge-success"
        elif demand_level == "Low Demand" or rev_category == "Low Yield":
            recommendation = "🛑 Re-evaluate Strategy"
            risk_level = "High Risk"
            risk_badge = "badge-danger"
        else:
            recommendation = "⚠️ Proceed with Caution"
            risk_level = "Moderate Risk"
            risk_badge = "badge-warning"

        # 4. Pricing Suggestion
        if input_price > cat_avg_price * 1.3:
            pricing_suggestion = f"Above category average (₹{cat_avg_price:.2f}). Consider bundling premium 1-on-1 mentorship."
        elif input_price < cat_avg_price * 0.7 and is_paid:
            pricing_suggestion = f"Below category average (₹{cat_avg_price:.2f}). You have margin to increase price to ₹{cat_avg_price:.2f}."
        else:
            pricing_suggestion = f"Optimal pricing aligned with category benchmark (₹{cat_avg_price:.2f})."

        # --- Display Results ---
        st.success("✓ ML Simulation Completed via Trained Joblib Regressors!")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Forecasted Student Demand", f"{pred_demand} enrollments")
        with m_col2:
            st.metric("Forecasted Total Revenue", f"₹{pred_revenue:,.2f}")

        st.markdown("#### 🎯 Business Strategic Evaluation")
        eval_col1, eval_col2, eval_col3, eval_col4 = st.columns(4)
        
        with eval_col1:
            st.markdown(f"**Demand Level**<br><span class='{demand_badge}'>{demand_level}</span>", unsafe_allow_html=True)
        with eval_col2:
            st.markdown(f"**Revenue Category**<br><span class='{rev_badge}'>{rev_category}</span>", unsafe_allow_html=True)
        with eval_col3:
            st.markdown(f"**Business Risk**<br><span class='{risk_badge}'>{risk_level}</span>", unsafe_allow_html=True)
        with eval_col4:
            st.markdown(f"**Launch Action**<br><b>{recommendation}</b>", unsafe_allow_html=True)

        st.info(f"💡 **Pricing Strategy Suggestion**: {pricing_suggestion}")

# --- TAB 4: MODEL BENCHMARKS & FEATURE IMPORTANCE ---
with tab4:
    st.subheader("📊 Machine Learning Model Benchmarks & Feature Importance")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("##### Course Demand Prediction Models (`EnrollmentCount`)")
        st.dataframe(metadata['demand_metrics'], use_container_width=True)
        
    with col_t2:
        st.markdown("##### Course Revenue Prediction Models (`TotalRevenue`)")
        st.dataframe(metadata['revenue_metrics'], use_container_width=True)
        
    st.divider()
    st.subheader("🌲 Feature Importance (Random Forest vs Gradient Boosting)")
    
    df_imp = metadata['feature_importance'].head(12)
    fig_imp = px.bar(
        df_imp, 
        x=['Gradient Boosting Importance', 'Random Forest Importance'], 
        y='Feature', 
        barmode='group',
        title="Top 12 Predictive Features for Course Performance",
        orientation='h'
    )
    st.plotly_chart(fig_imp, use_container_width=True)
