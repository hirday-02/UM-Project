# 🎓 EduPro: Predictive Modeling for Course Demand & Revenue Forecasting

An end-to-end Machine Learning project and interactive Streamlit web dashboard built to forecast course enrollments and revenue performance for the EduPro online learning platform.

---

## 📌 Project Overview & Problem Statement
EduPro is an online learning platform looking to optimize course catalogue management and maximize top-line revenue. Executive leadership requires data-driven answers to key business questions:

* **Demand Forecasting**: Which proposed courses will become popular among students?
* **Category Yield**: Which subject categories generate the highest total revenue?
* **Instructor Impact**: How do instructor credentials and experience drive conversion rates?
* **Pricing & Launch Strategy**: How should new courses be priced to maximize total revenue without penalizing student demand?

---

## 📊 Dataset Details
The project utilizes relational data from `EduPro Online Platform.xlsx`:
* **Users**: Demographics (`UserID`, `Age`, `Gender`, `Email`).
* **Teachers**: Instructor metadata (`TeacherID`, `Expertise`, `YearsOfExperience`, `TeacherRating`).
* **Courses**: Catalogue specs (`CourseID`, `CourseCategory`, `CourseType`, `CourseLevel`, `CoursePrice`, `CourseDuration`, `CourseRating`).
* **Transactions**: Purchase logs (`TransactionID`, `UserID`, `CourseID`, `TransactionDate`, `Amount`, `PaymentMethod`, `TeacherID`).

---

## 🏗️ Project Structure
```text
EduPro-ML-Project/
├── EduPro Online Platform.xlsx   # Relational Excel Dataset
├── train_models.py               # ML Training Script (Dumps Joblib Artifacts)
├── demand_model.joblib           # Trained Lasso Demand Model
├── revenue_model.joblib          # Trained Gradient Boosting Revenue Model
├── scaler.joblib                 # Fitted StandardScaler Pipeline
├── model_metadata.joblib         # Feature names, benchmarks & metrics
├── app.py                        # Streamlit Interactive Web Application
├── requirements.txt              # Cloud Deployment Dependencies
└── README.md                     # Comprehensive Project Documentation
```

---

## 🤖 Machine Learning Model Benchmarks

Multiple regression models were trained and evaluated using 80/20 Train/Test splits:

### 1. Course Demand Models (`EnrollmentCount`)
| Model | MAE (Test) | RMSE (Test) | R² (Test) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Lasso Regression** | **9.56** | **12.92** | **0.0833** | **Selected Model** |
| Ridge Regression | 9.84 | 13.18 | 0.0462 | Evaluated |
| Random Forest Regressor | 10.67 | 13.63 | -0.0188 | Overfitted |
| Gradient Boosting Regressor | 11.48 | 14.88 | -0.2149 | Overfitted |
| Linear Regression | 16.88 | 20.08 | -1.2133 | Overfitted |

### 2. Course Revenue Models (`TotalRevenue` in ₹)
| Model | MAE (Test) | RMSE (Test) | R² (Test) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting Regressor** | **₹3,084.19** | **₹4,688.83** | **0.9717** | **Selected Model** |
| Lasso Regression | ₹3,958.72 | ₹5,019.86 | 0.9675 | Evaluated |
| Linear Regression | ₹3,978.78 | ₹5,055.14 | 0.9671 | Evaluated |
| Random Forest Regressor | ₹3,437.62 | ₹5,116.81 | 0.9663 | Evaluated |
| Ridge Regression | ₹8,153.98 | ₹9,110.85 | 0.8931 | Evaluated |

---

## 🚀 How to Run Locally

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/hirday-02/UM-Project.git
   cd UM-Project
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train Models & Dump Artifacts (Optional)**:
   ```bash
   python train_models.py
   ```

4. **Launch Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

## ☁️ Deployment Instructions for Streamlit Community Cloud

1. Log in to **[Streamlit Community Cloud](https://share.streamlit.io/)** using your GitHub account.
2. Click **"New app"**.
3. Select Repository: `hirday-02/UM-Project`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Click **"Deploy!"**.

---

## 💡 Key Business Insights & Strategic Recommendations

1. **Focus Content Investment in High-Yield Tiers**:
   * **Artificial Intelligence**, **Cybersecurity**, and **Project Management** generate 3x to 4x higher revenue per course than entry-level utility skills.
2. **Capitalize on Unexploited Pricing Power**:
   * Premium subject categories maintain strong enrollment volume even at price points between ₹350 and ₹490.
3. **Enforce Instructor Domain Matching**:
   * Courses taught by instructors with matching domain expertise (`ExpertiseMatchScore = 1`) achieve higher student rating tiers, directly boosting course demand.
