import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def train_and_save_pipeline():
    excel_file = 'EduPro Online Platform.xlsx'
    if not os.path.exists(excel_file):
        raise FileNotFoundError(f"Dataset file '{excel_file}' not found.")

    excel_wb = pd.ExcelFile(excel_file)
    df_courses = pd.read_excel(excel_wb, 'Courses')
    df_teachers = pd.read_excel(excel_wb, 'Teachers')
    df_transactions = pd.read_excel(excel_wb, 'Transactions')

    # Aggregations
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

    # Feature Engineering
    df_summary['IsPaid'] = (df_summary['CourseType'] == 'Paid').astype(int)
    df_summary['InstructorQualityScore'] = (df_summary['TeacherRating'] * np.log1p(df_summary['YearsOfExperience'])).round(2)
    df_summary['ExpertiseMatchScore'] = (
        df_summary['Expertise'].str.strip().str.lower() == df_summary['CourseCategory'].str.strip().str.lower()
    ).astype(int)

    cat_benchmarks = df_summary.groupby('CourseCategory').agg(
        CategoryAvgRevenue=('TotalRevenue', 'mean'),
        CategoryAvgEnrollment=('EnrollmentCount', 'mean'),
        CategoryAvgPrice=('CoursePrice', 'mean')
    ).reset_index()

    df_summary = df_summary.merge(cat_benchmarks, on='CourseCategory', how='left')

    level_map = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
    df_summary['CourseLevel_Encoded'] = df_summary['CourseLevel'].map(level_map).fillna(1)

    # Categorical Buckets
    def get_price_band(price):
        if price == 0: return 'Free'
        elif price <= 100: return 'Budget'
        elif price <= 300: return 'Mid-Tier'
        else: return 'Premium'

    def get_duration_bucket(d):
        if d < 15: return 'Short (<15h)'
        elif d < 30: return 'Medium (15-30h)'
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

    df_summary['PriceBand'] = df_summary['CoursePrice'].apply(get_price_band)
    df_summary['DurationBucket'] = df_summary['CourseDuration'].apply(get_duration_bucket)
    df_summary['RatingTier'] = df_summary['CourseRating'].apply(get_rating_tier)
    df_summary['ExperienceBucket'] = df_summary['YearsOfExperience'].apply(get_exp_bucket)

    # Prepare ML matrix X and targets y
    cols_to_drop = ['CourseID', 'CourseName', 'TeacherID', 'TeacherName', 'EnrollmentCount', 'TotalRevenue', 'Expertise', 'CourseLevel', 'Gender']
    X_raw = df_summary.drop(columns=[c for c in cols_to_drop if c in df_summary.columns])
    cat_cols = ['CourseCategory', 'PriceBand', 'DurationBucket', 'RatingTier', 'ExperienceBucket', 'CourseType']
    X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)

    # Ensure all columns are numeric
    bool_cols = X_encoded.select_dtypes(include=['bool']).columns
    X_encoded[bool_cols] = X_encoded[bool_cols].astype(int)

    y_demand = df_summary['EnrollmentCount']
    y_revenue = df_summary['TotalRevenue']

    # Train / Test split
    X_train, X_test, y_train_enr, y_test_enr = train_test_split(X_encoded, y_demand, test_size=0.20, random_state=42)
    _, _, y_train_rev, y_test_rev = train_test_split(X_encoded, y_revenue, test_size=0.20, random_state=42)

    # Scale numerical features
    num_cols_to_scale = ['CoursePrice', 'CourseDuration', 'CourseRating', 'YearsOfExperience', 'TeacherRating', 'CategoryAvgRevenue', 'CategoryAvgEnrollment', 'CategoryAvgPrice', 'InstructorQualityScore']
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[num_cols_to_scale] = scaler.fit_transform(X_train[num_cols_to_scale])
    X_test_scaled[num_cols_to_scale] = scaler.transform(X_test[num_cols_to_scale])

    # Model Dictionary
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=10.0, random_state=42),
        'Lasso': Lasso(alpha=1.0, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    }

    def evaluate_all(y_tr, y_te, target_name):
        res = []
        trained_dict = {}
        for name, m in models.items():
            m_copy = sklearn_clone(m)
            m_copy.fit(X_train_scaled, y_tr)
            trained_dict[name] = m_copy
            
            p_te = m_copy.predict(X_test_scaled)
            mae = mean_absolute_error(y_te, p_te)
            rmse = np.sqrt(mean_squared_error(y_te, p_te))
            r2 = r2_score(y_te, p_te)
            
            res.append({
                'Model': name,
                'MAE': round(mae, 2),
                'RMSE': round(rmse, 2),
                'R²': round(r2, 4)
            })
        df_res = pd.DataFrame(res).sort_values(by='R²', ascending=False)
        return df_res, trained_dict

    def sklearn_clone(model):
        from sklearn.base import clone
        return clone(model)

    df_demand_metrics, trained_demand_models = evaluate_all(y_train_enr, y_test_enr, 'Demand')
    df_revenue_metrics, trained_rev_models = evaluate_all(y_train_rev, y_test_rev, 'Revenue')

    # Select best models
    best_demand_model = trained_demand_models['Lasso']
    best_revenue_model = trained_rev_models['Gradient Boosting']

    # Feature Importance Dataframes
    rev_rf = trained_rev_models['Random Forest'].feature_importances_
    rev_gb = trained_rev_models['Gradient Boosting'].feature_importances_
    
    df_importance = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Random Forest Importance': rev_rf,
        'Gradient Boosting Importance': rev_gb
    }).sort_values(by='Gradient Boosting Importance', ascending=False)

    # Save Joblib Artifacts
    joblib.dump(best_demand_model, 'demand_model.joblib')
    joblib.dump(best_revenue_model, 'revenue_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    metadata = {
        'feature_names': list(X_encoded.columns),
        'num_cols_to_scale': num_cols_to_scale,
        'cat_benchmarks': cat_benchmarks.to_dict(orient='records'),
        'demand_metrics': df_demand_metrics,
        'revenue_metrics': df_revenue_metrics,
        'feature_importance': df_importance
    }
    joblib.dump(metadata, 'model_metadata.joblib')

    print("[SUCCESS] Models and pipeline metadata trained & saved successfully with Joblib!")
    return metadata

if __name__ == '__main__':
    train_and_save_pipeline()
