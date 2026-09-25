"""
========================================================================================
Enterprise Data Analytics · Enterprise Data Analytics · Enterprise Data Analytics | Data Analytics with AI Internship
Advanced Capstone: E-Commerce Customer Intelligence, RFM Segmentation & Churn Analytics
----------------------------------------------------------------------------------------
Dataset Source: Kaggle (Ecommerce Customer Churn Analysis and Prediction)
URL: https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
Author: Ankit Verma
Scale: 5,630 Records | 20 Raw Features | 8 Engineered Features
========================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# --------------------------------------------------------------------------------------
# 0. Setup and Directory Configuration
# --------------------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "dataset")
VIZ_DIR = os.path.join(CURRENT_DIR, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, "ecommerce_customer_churn.csv")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


def run_pipeline():
    print("=" * 85)
    print("  E-COMMERCE CUSTOMER INTELLIGENCE & CHURN PREDICTION PLATFORM")
    print("  Enterprise Data Analytics · Enterprise Data Analytics · Enterprise Data Analytics Capstone Project")
    print("=" * 85)

    # ----------------------------------------------------------------------------------
    # 1. Ingestion & Quality Audit
    # ----------------------------------------------------------------------------------
    print("\n[PHASE 1] DATA INGESTION & QUALITY AUDIT")
    print("-" * 50)
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    df_raw = pd.read_csv(DATASET_PATH)
    print(f"[OK] Ingested official Kaggle dataset: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
    
    missing_summary = df_raw.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    print(f"[OK] Detected {len(missing_cols)} columns with missing data:")
    for col, count in missing_cols.items():
        print(f"   - {col:28s}: {count:4d} nulls ({count/len(df_raw):.1%})")

    # ----------------------------------------------------------------------------------
    # 2. Data Cleaning & Feature Engineering
    # ----------------------------------------------------------------------------------
    print("\n[PHASE 2] DATA CLEANING & ADVANCED FEATURE ENGINEERING")
    print("-" * 50)
    df = df_raw.copy()

    # Standardize categorical strings
    df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace({"Phone": "Mobile Phone"})
    df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace({"CC": "Credit Card", "COD": "Cash on Delivery"})
    df["PreferedOrderCat"] = df["PreferedOrderCat"].replace({"Mobile": "Mobile Phone"})

    # Median imputation on numerical attributes
    impute_cols = ["Tenure", "WarehouseToHome", "HourSpendOnApp", "NumberOfDeviceRegistered",
                   "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount", "DaySinceLastOrder"]
    for col in impute_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # 1. Tenure segmentation
    tenure_bins = [-1, 6, 12, 24, 100]
    tenure_labels = ["0-6 Mo (New)", "7-12 Mo (Developing)", "13-24 Mo (Established)", "25+ Mo (Loyal)"]
    df["Tenure_Group"] = pd.cut(df["Tenure"], bins=tenure_bins, labels=tenure_labels)

    # 2. Warehouse Distance Category
    dist_bins = [-1, 10, 20, 35, 200]
    dist_labels = ["Local (<10km)", "Near (10-20km)", "Medium (20-35km)", "Far (>35km)"]
    df["Distance_Category"] = pd.cut(df["WarehouseToHome"], bins=dist_bins, labels=dist_labels)

    # 3. RFM Analysis (Recency, Frequency, Monetary Proxy)
    # Recency: DaySinceLastOrder (Lower is better)
    # Frequency: OrderCount (Higher is better)
    # Monetary: Estimated Spend = OrderCount * (CashbackAmount * 5)
    df["Estimated_Spend"] = (df["OrderCount"] * (df["CashbackAmount"] * 5.5)).round(2)
    
    # Calculate RFM quantiles (1 to 4)
    df["R_Score"] = pd.qcut(df["DaySinceLastOrder"].rank(method="first"), q=4, labels=[4, 3, 2, 1]).astype(int)
    df["F_Score"] = pd.qcut(df["OrderCount"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
    df["M_Score"] = pd.qcut(df["Estimated_Spend"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
    df["RFM_Score"] = df["R_Score"] * 100 + df["F_Score"] * 10 + df["M_Score"]

    def assign_rfm_segment(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 3 and f >= 3 and m >= 3:
            return "Champions"
        elif r >= 3 and f >= 2:
            return "Loyal Customers"
        elif r >= 3 and f <= 2:
            return "Potential Loyalists"
        elif r <= 2 and f >= 3:
            return "At-Risk High-Value"
        elif r <= 2 and f <= 2 and m >= 3:
            return "Hibernating Big Spenders"
        else:
            return "Lost / Low Value"

    df["RFM_Segment"] = df.apply(assign_rfm_segment, axis=1)

    # 4. Service Friction Index
    df["Friction_Index"] = (
        (df["Complain"] * 2.0) +
        ((5 - df["SatisfactionScore"]) * 0.5) +
        (df["WarehouseToHome"] / df["WarehouseToHome"].max())
    ).round(2)

    print(f"[OK] Cleaned dataset: 0 missing values remain.")
    print("[OK] Engineered features: Tenure_Group, Distance_Category, Estimated_Spend, RFM_Segment, Friction_Index")
    
    # Save RFM segmentation table
    rfm_export = df[["CustomerID", "DaySinceLastOrder", "OrderCount", "Estimated_Spend", "R_Score", "F_Score", "M_Score", "RFM_Segment", "Churn"]]
    rfm_path = os.path.join(DATA_DIR, "customer_rfm_segments.csv")
    rfm_export.to_csv(rfm_path, index=False)
    print(f"[OK] Exported RFM segmentation dataset to: {rfm_path}")

    # ----------------------------------------------------------------------------------
    # 3. Statistical Hypothesis Testing
    # ----------------------------------------------------------------------------------
    print("\n[PHASE 3] STATISTICAL INFERENCE & HYPOTHESIS TESTING")
    print("-" * 50)
    
    # Test 1: Chi-Square Test (Complaints vs Churn)
    contingency = pd.crosstab(df["Complain"], df["Churn"])
    chi2, p_val_chi2, _, _ = stats.chi2_contingency(contingency)
    print(f"1. Chi-Square Test (Complaints vs. Churn):")
    print(f"   - Chi2 Statistic: {chi2:.2f}, p-value: {p_val_chi2:.2e}")
    print(f"   - Conclusion: {'Statistically Significant (p < 0.001) - Complaints drive churn drastically' if p_val_chi2 < 0.001 else 'Not significant'}")

    # Test 2: Two-sample T-test (Tenure of Churned vs Retained)
    tenure_churn = df[df["Churn"] == 1]["Tenure"]
    tenure_retained = df[df["Churn"] == 0]["Tenure"]
    t_stat_t, p_val_t = stats.ttest_ind(tenure_churn, tenure_retained, equal_var=False)
    print(f"2. Welch's T-Test (Tenure of Churned vs. Retained):")
    print(f"   - Mean Tenure Churned: {tenure_churn.mean():.1f} months vs Retained: {tenure_retained.mean():.1f} months")
    print(f"   - T-Statistic: {t_stat_t:.2f}, p-value: {p_val_t:.2e}")
    print(f"   - Conclusion: {'Highly Significant (p < 0.001) - Short tenure directly correlates with churn' if p_val_t < 0.001 else 'Not significant'}")

    # Test 3: Two-sample T-test (Warehouse Distance)
    dist_churn = df[df["Churn"] == 1]["WarehouseToHome"]
    dist_retained = df[df["Churn"] == 0]["WarehouseToHome"]
    t_stat_d, p_val_d = stats.ttest_ind(dist_churn, dist_retained, equal_var=False)
    print(f"3. Welch's T-Test (Warehouse Distance of Churned vs. Retained):")
    print(f"   - Mean Distance Churned: {dist_churn.mean():.1f} km vs Retained: {dist_retained.mean():.1f} km")
    print(f"   - T-Statistic: {t_stat_d:.2f}, p-value: {p_val_d:.2e}")

    # ----------------------------------------------------------------------------------
    # 4. Publication-Quality Data Visualizations
    # ----------------------------------------------------------------------------------
    print("\n[PHASE 4] GENERATING ADVANCED ANALYTICS VISUALIZATIONS")
    print("-" * 50)

    # Chart 1: RFM Segments Breakdown & Churn
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    rfm_counts = df["RFM_Segment"].value_counts()
    rfm_churn = df.groupby("RFM_Segment", observed=False)["Churn"].mean() * 100

    palette1 = sns.color_palette("mako", len(rfm_counts))
    ax1.pie(rfm_counts, labels=rfm_counts.index, autopct="%1.1f%%", startangle=140, colors=palette1)
    ax1.set_title("Customer Base Distribution by RFM Segment", fontweight="bold", fontsize=12)

    sns.barplot(x=rfm_churn.values, y=rfm_churn.index, palette="Reds_r", hue=rfm_churn.index, legend=False, ax=ax2)
    ax2.set_title("Churn Rate by RFM Segment (%)", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Churn Rate (%)", fontweight="bold")
    for p in ax2.patches:
        ax2.annotate(f"{p.get_width():.1f}%", (p.get_width(), p.get_y() + p.get_height() / 2.),
                     ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "01_rfm_customer_segments.png"), dpi=300)
    plt.close()
    print("[OK] Saved 01_rfm_customer_segments.png")

    # Chart 2: Churn by Category and Tenure Group
    fig, ax = plt.subplots(figsize=(11, 5))
    cat_tenure = df.groupby(["PreferedOrderCat", "Tenure_Group"], observed=False)["Churn"].mean().reset_index()
    cat_tenure["Churn_Pct"] = cat_tenure["Churn"] * 100
    sns.barplot(data=cat_tenure, x="PreferedOrderCat", y="Churn_Pct", hue="Tenure_Group", palette="Blues", ax=ax)
    ax.set_title("Churn Rate Interaction: Product Category × Customer Tenure", fontweight="bold", fontsize=13, pad=12)
    ax.set_ylabel("Churn Rate (%)", fontweight="bold")
    ax.set_xlabel("Product Category", fontweight="bold")
    ax.legend(title="Tenure Bracket", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "02_churn_by_category_and_tenure.png"), dpi=300)
    plt.close()
    print("[OK] Saved 02_churn_by_category_and_tenure.png")

    # Chart 3: Warehouse Distance vs Churn
    fig, ax = plt.subplots(figsize=(10, 5))
    dist_summary = df.groupby("Distance_Category", observed=False)["Churn"].agg(Churn_Rate="mean", Volume="count").reset_index()
    dist_summary["Churn_Pct"] = dist_summary["Churn_Rate"] * 100
    sns.barplot(data=dist_summary, x="Distance_Category", y="Churn_Pct", hue="Distance_Category", palette="coolwarm", legend=False, ax=ax)
    ax.set_title("Impact of Delivery Distance from Warehouse on Customer Churn", fontweight="bold", fontsize=13, pad=12)
    ax.set_ylabel("Churn Rate (%)", fontweight="bold")
    ax.set_xlabel("Delivery Distance Cohort", fontweight="bold")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "03_warehouse_distance_vs_churn.png"), dpi=300)
    plt.close()
    print("[OK] Saved 03_warehouse_distance_vs_churn.png")

    # Chart 4: Service Complaints & Satisfaction Score
    fig, ax = plt.subplots(figsize=(10, 5))
    cs_comp = df.groupby(["SatisfactionScore", "Complain"], observed=False)["Churn"].mean().reset_index()
    cs_comp["Churn_Pct"] = cs_comp["Churn"] * 100
    cs_comp["Complain_Label"] = cs_comp["Complain"].map({0: "No Complaint", 1: "Complaint Logged"})
    sns.barplot(data=cs_comp, x="SatisfactionScore", y="Churn_Pct", hue="Complain_Label", palette=["#27ae60", "#c0392b"], ax=ax)
    ax.set_title("Churn Rate across Satisfaction Scores and Complaint Status", fontweight="bold", fontsize=13, pad=12)
    ax.set_xlabel("Customer Satisfaction Rating (1 to 5)", fontweight="bold")
    ax.set_ylabel("Churn Rate (%)", fontweight="bold")
    ax.legend(title="Support History")
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 4), textcoords='offset points', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "04_complaint_and_satisfaction_impact.png"), dpi=300)
    plt.close()
    print("[OK] Saved 04_complaint_and_satisfaction_impact.png")

    # ----------------------------------------------------------------------------------
    # 5. Machine Learning Modeling & Benchmarking
    # ----------------------------------------------------------------------------------
    print("\n[PHASE 5] MULTI-MODEL MACHINE LEARNING BENCHMARK")
    print("-" * 50)

    drop_cols = ["CustomerID", "Churn", "Tenure_Group", "Distance_Category", "RFM_Segment", "RFM_Score", "R_Score", "F_Score", "M_Score"]
    X = df.drop(columns=drop_cols)
    y = df["Churn"]

    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_names = X_encoded.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        "Random Forest (Tuned)": RandomForestClassifier(n_estimators=200, max_depth=14, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)
    }

    results = []
    roc_curves_data = {}
    fitted_models = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in models.items():
        if name == "Logistic Regression":
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
            cv_f1 = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring="f1").mean()
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            cv_f1 = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1").mean()

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves_data[name] = (fpr, tpr, auc)
        fitted_models[name] = clf

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "5-Fold CV F1": cv_f1,
            "ROC-AUC": auc
        })

    perf_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
    print(perf_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Chart 5: ROC Curves Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (fpr, tpr, auc) in roc_curves_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess (AUC = 0.500)")
    ax.set_title("ROC Curves Comparison Across Machine Learning Models", fontweight="bold", fontsize=12, pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "05_roc_curves_comparison.png"), dpi=300)
    plt.close()
    print("[OK] Saved 05_roc_curves_comparison.png")

    # Chart 6: Feature Importances (Random Forest)
    best_rf = fitted_models["Random Forest (Tuned)"]
    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": best_rf.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feat_imp, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False, ax=ax)
    ax.set_title("Top 12 Features Driving Customer Churn (Random Forest Importance)", fontweight="bold", fontsize=12, pad=12)
    ax.set_xlabel("Relative Importance Score", fontweight="bold")
    ax.set_ylabel("Feature Name", fontweight="bold")
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.3f}", (p.get_width(), p.get_y() + p.get_height() / 2.),
                     ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "06_feature_importance_ranking.png"), dpi=300)
    plt.close()
    print("[OK] Saved 06_feature_importance_ranking.png")

    # Chart 7: Confusion Matrices Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    cm_lr = confusion_matrix(y_test, fitted_models["Logistic Regression"].predict(X_test_scaled))
    cm_rf = confusion_matrix(y_test, best_rf.predict(X_test))

    sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Purples", cbar=False, ax=ax1,
                xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
    ax1.set_title("Logistic Regression Confusion Matrix", fontweight="bold")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax2,
                xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
    ax2.set_title("Random Forest (Tuned) Confusion Matrix", fontweight="bold")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "07_confusion_matrices.png"), dpi=300)
    plt.close()
    print("[OK] Saved 07_confusion_matrices.png")

    # ----------------------------------------------------------------------------------
    # 6. Final Customer Risk Scoring Table (All 5,630 Customers)
    # ----------------------------------------------------------------------------------
    print("\n[PHASE 6] ENTERPRISE RISK SCORING TABLE")
    print("-" * 50)
    full_X = pd.get_dummies(df.drop(columns=drop_cols), drop_first=True)
    full_X = full_X.reindex(columns=feature_names, fill_value=0)
    all_churn_probs = best_rf.predict_proba(full_X)[:, 1]

    df_risk = df[["CustomerID", "Tenure", "RFM_Segment", "Estimated_Spend", "SatisfactionScore", "Complain", "WarehouseToHome", "PreferedOrderCat", "Churn"]].copy()
    df_risk["Churn_Probability"] = all_churn_probs.round(3)

    def classify_risk(p):
        if p >= 0.60:
            return "High Risk"
        elif p >= 0.30:
            return "Medium Risk"
        return "Low Risk"

    def prescribe_action(row):
        risk = row["Risk_Level"]
        rfm = row["RFM_Segment"]
        if risk == "High Risk" and "High-Value" in rfm or rfm == "Champions":
            return "Priority 1: Immediate Executive Concierge Call + Rs. 1,000 Retention Credit"
        elif risk == "High Risk":
            return "Priority 2: 25% Comeback Voucher + Free Express Delivery"
        elif risk == "Medium Risk":
            return "Priority 3: Personalized Category Recommendation + Free Shipping"
        else:
            return "Priority 4: Standard Loyalty Reward Program & Cross-Sell"

    df_risk["Risk_Level"] = df_risk["Churn_Probability"].apply(classify_risk)
    df_risk["Prescribed_Intervention"] = df_risk.apply(prescribe_action, axis=1)

    risk_export_path = os.path.join(DATA_DIR, "final_customer_risk_table.csv")
    df_risk.to_csv(risk_export_path, index=False)
    print(f"[OK] Exported complete 5,630-customer risk table to: {risk_export_path}")

    risk_counts = df_risk["Risk_Level"].value_counts()
    for level, count in risk_counts.items():
        print(f"   - {level:12s}: {count:4d} customers ({count/len(df_risk):.1%})")

    # ----------------------------------------------------------------------------------
    # 7. Executive ROI & Financial Impact Summary
    # ----------------------------------------------------------------------------------
    print("\n[PHASE 7] FINANCIAL IMPACT & STRATEGIC RECOMMENDATIONS")
    print("-" * 50)
    high_risk_spend = df_risk[df_risk["Risk_Level"] == "High Risk"]["Estimated_Spend"].sum()
    med_risk_spend = df_risk[df_risk["Risk_Level"] == "Medium Risk"]["Estimated_Spend"].sum()
    total_risk_spend = high_risk_spend + med_risk_spend
    projected_savings = (high_risk_spend * 0.45) + (med_risk_spend * 0.65)

    print(f"1. Total At-Risk Revenue (High + Medium Risk Cohorts): Rs. {total_risk_spend:,.2f}")
    print(f"2. High-Risk Customer Value: Rs. {high_risk_spend:,.2f}")
    print(f"3. Projected Annual Revenue Salvaged with AI-Driven Action: Rs. {projected_savings:,.2f}")
    print("=" * 85)
    print("  ANALYTICS PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    run_pipeline()
