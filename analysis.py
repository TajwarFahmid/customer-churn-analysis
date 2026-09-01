"""Customer missing-data and churn analysis"""
from pathlib import Path

DATA_PATH = Path(__file__).resolve().with_name("customer_missing_data.csv")

# Extracted from notebook cell 29
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# loading data
df = pd.read_csv(DATA_PATH)

# finding missingness rates
missing_counts = df.isnull().sum()
missing_pct = (missing_counts / len(df)) * 100
missing_summary = pd.DataFrame(
    {"Missing_Count": missing_counts, "Missing_Pct": missing_pct}
).query("Missing_Count > 0")
print("--- Missingness Summary ---")
print(missing_summary.sort_values(by="Missing_Pct", ascending=False))

# missingness Correlation Matrix
missing_indicators = df[missing_summary.index].isnull().astype(int)
missing_corr = missing_indicators.corr()

plt.figure(figsize=(9, 7))
sns.heatmap(
    missing_corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, vmin=-0.1, vmax=0.1
)
plt.title("Pairwise Missingness Correlation Matrix")
plt.tight_layout()
plt.show()

# tests for MAR data:
print("\n--- Diagnostic Evidence (MAR Conditioning) ---")
# Income missingness conditioned on Segment
print(
    pd.crosstab(
        df["segment"], df["annual_income"].isnull(), normalize="index"
    ).round(4)
)

# Deposit Balance missingness conditioned on Region
print(
    pd.crosstab(
        df["region"], df["avg_deposit_balance"].isnull(), normalize="index"
    ).round(4)
)

# Credit Score missingness conditioned on Tenure Quintiles
df["tenure_quintile"] = pd.qcut(df["tenure_years"], q=5)
print(
    pd.crosstab(
        df["tenure_quintile"], df["credit_score"].isnull(), normalize="index"
    ).round(4)
)

# Extracted from notebook cell 32
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

# loading dataset and separate target/identifiers
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["customer_id", "churned"]).copy()
y = df["churned"].copy()

# imputing categorical mode for region (MCAR, 2.67% missing) then one-hot encode
X["region"] = X["region"].fillna(X["region"].mode()[0])
X_encoded = pd.get_dummies(X, columns=["region", "segment"], drop_first=True)
feature_names = X_encoded.columns.tolist()

# tuning k for KNNImputer on the complete cases subset
complete_subset = X_encoded.dropna().copy()
scaler_tune = StandardScaler()
complete_scaled = scaler_tune.fit_transform(complete_subset)

# Artificially mask 10% of complete cases to evaluate recovery RMSE across k
rng = np.random.RandomState(42)
mask = rng.rand(*complete_scaled.shape) < 0.10
corrupted_scaled = complete_scaled.copy()
corrupted_scaled[mask] = np.nan

k_candidates = [3, 5, 7, 9, 11, 15]
k_errors = {}

for k in k_candidates:
    knn = KNNImputer(n_neighbors=k)
    imputed_tune = knn.fit_transform(corrupted_scaled)
    rmse = np.sqrt(np.mean((complete_scaled[mask] - imputed_tune[mask]) ** 2))
    k_errors[k] = rmse

best_k = min(k_errors, key=k_errors.get)
print("--- KNN Hyperparameter Tuning (Recovery RMSE) ---")
for k, err in k_errors.items():
    print(f"k = {k:2d} : RMSE = {err:.4f}")
print(f"Optimal k selected: {best_k}\n")

# Extracted from notebook cell 33
# ==============================================================================
# Strategy 1: Complete-Case Deletion (Listwise Deletion)
# ==============================================================================
df_complete = df.dropna().copy()
X_complete_case = X_encoded.loc[df_complete.index].copy()
y_complete_case = y.loc[df_complete.index].copy()

print("1. Complete-Case Deletion:")
print(
    f"   Retained rows: {len(df_complete)} / {len(df)} ({len(df_complete)/len(df):.1%})"
)
print(f"   Discarded rows: {len(df) - len(df_complete)} (52.7% data loss)\n")


# ==============================================================================
# Strategy 2: Mean / Median Baseline
# ==============================================================================
median_imputer = SimpleImputer(strategy="median")
X_median = pd.DataFrame(
    median_imputer.fit_transform(X_encoded), columns=feature_names
)

print("2. Median Baseline Imputation:")
print(
    f"   Missing values remaining: {X_median.isnull().sum().sum()} (Shape: {X_median.shape})\n"
)


# ==============================================================================
# Strategy 3: KNN Imputation (with tuned k on scaled features)
# ==============================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

knn_imputer = KNNImputer(n_neighbors=best_k)
X_knn_scaled = knn_imputer.fit_transform(X_scaled)

# Inverse transform back to original feature scale
X_knn = pd.DataFrame(
    scaler.inverse_transform(X_knn_scaled), columns=feature_names
)

print(f"3. KNN Imputation (Tuned k = {best_k}):")
print(
    f"   Missing values remaining: {X_knn.isnull().sum().sum()} (Shape: {X_knn.shape})\n"
)


# ==============================================================================
# Strategy 4: MICE (Multivariate Imputation by Chained Equations)
# ==============================================================================
# Scaled data prevents convergence warnings caused by varying feature magnitudes
mice_imputer = IterativeImputer(
    max_iter=25, tol=1e-3, random_state=42, sample_posterior=False
)
X_mice_scaled = mice_imputer.fit_transform(X_scaled)

# Inverse transform back to original feature scale
X_mice = pd.DataFrame(
    scaler.inverse_transform(X_mice_scaled), columns=feature_names
)

print("4. MICE (IterativeImputer with Bayesian Ridge):")
print(f"   Iterations to convergence: {mice_imputer.n_iter_}")
print(
    f"   Missing values remaining: {X_mice.isnull().sum().sum()} (Shape: {X_mice.shape})\n"
)


# ==============================================================================
# Strategy 5: Random Forest Imputation (MissForest)
# ==============================================================================
# Iterative non-parametric tree ensembles capture non-linear relationships & interactions
rf_imputer = IterativeImputer(
    estimator=ExtraTreesRegressor(n_estimators=35, random_state=42, n_jobs=-1),
    max_iter=8,
    random_state=42,
)
X_rf = pd.DataFrame(rf_imputer.fit_transform(X_encoded), columns=feature_names)

print("5. MissForest (Random Forest / ExtraTrees Iterative Imputer):")
print(
    f"   Missing values remaining: {X_rf.isnull().sum().sum()} (Shape: {X_rf.shape})\n"
)

# Extracted from notebook cell 35
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Take the complete cases subset as the ground-truth benchmark (N = 1,420)
df = pd.read_csv(DATA_PATH)
complete_df = df.dropna().copy()

num_cols = [
    "age",
    "tenure_years",
    "credit_score",
    "annual_income",
    "credit_utilization",
    "avg_deposit_balance",
    "monthly_txn_count",
    "products_held",
]

X_comp = complete_df.drop(columns=["customer_id", "churned"]).copy()
X_comp["region"] = X_comp["region"].fillna(X_comp["region"].mode()[0])
X_comp_encoded = pd.get_dummies(
    X_comp, columns=["region", "segment"], drop_first=True
)
feature_names = X_comp_encoded.columns.tolist()

# 2. Artificially mask 15% of observed numeric values at random
rng = np.random.RandomState(42)
mask = rng.rand(*X_comp_encoded[num_cols].shape) < 0.15
masked_encoded = X_comp_encoded.copy()
masked_encoded[num_cols] = masked_encoded[num_cols].mask(mask)

# 3. Apply Imputation Methods
# A. Median
res_median = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(masked_encoded),
    columns=feature_names,
)

# B. KNN (k=15 on scaled data)
scaler = StandardScaler()
scaled_masked = scaler.fit_transform(masked_encoded)
res_knn = pd.DataFrame(
    scaler.inverse_transform(
        KNNImputer(n_neighbors=15).fit_transform(scaled_masked)
    ),
    columns=feature_names,
)

# C. MICE (IterativeImputer with Bayesian Ridge)
res_mice = pd.DataFrame(
    scaler.inverse_transform(
        IterativeImputer(max_iter=25, tol=1e-3, random_state=42).fit_transform(
            scaled_masked
        )
    ),
    columns=feature_names,
)

# D. MissForest (Iterative ExtraTrees)
res_rf = pd.DataFrame(
    IterativeImputer(
        estimator=ExtraTreesRegressor(
            n_estimators=30, random_state=42, n_jobs=-1
        ),
        max_iter=5,
        random_state=42,
    ).fit_transform(masked_encoded),
    columns=feature_names,
)

# 4. Measure Metrics: NRMSE, KS-Distance (Marginal Shape), and Frobenius Norm (Covariance)
true_mat = X_comp_encoded[num_cols].values
true_corr = X_comp_encoded[num_cols].corr().values

eval_records = []
methods = [
    ("Median", res_median),
    ("KNN (k=15)", res_knn),
    ("MICE", res_mice),
    ("MissForest", res_rf),
]

for name, imp_df in methods:
    imp_mat = imp_df[num_cols].values
    actuals, preds = true_mat[mask], imp_mat[mask]

    nrmse = np.sqrt(np.mean((actuals - preds) ** 2)) / np.std(actuals)
    avg_ks = np.mean(
        [
            stats.ks_2samp(X_comp_encoded[c], imp_df[c]).statistic
            for c in num_cols
        ]
    )
    frob_dist = np.linalg.norm(
        true_corr - imp_df[num_cols].corr().values, ord="fro"
    )

    eval_records.append(
        {
            "Method": name,
            "NRMSE": round(nrmse, 4),
            "Avg_KS_Distance": round(avg_ks, 4),
            "Frobenius_Corr_Distance": round(frob_dist, 4),
        }
    )

print(pd.DataFrame(eval_records).to_string(index=False))

# Extracted from notebook cell 38
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score

# 1. Load dataset and split features / target
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["customer_id", "churned"]).copy()
y_all = df["churned"].copy()

# 2. Impute mode for 'region' (MCAR, ~2.7% missing) and one-hot encode
X["region"] = X["region"].fillna(X["region"].mode()[0])
X_all_enc = pd.get_dummies(X, columns=["region", "segment"], drop_first=True)
all_feat_names = X_all_enc.columns.tolist()

# 3. Create all 5 imputed matrices
# Complete-Case
cc_idx = df.dropna().index
X_cc = X_all_enc.loc[cc_idx]
y_cc = y_all.loc[cc_idx]

# Median Baseline
median_imputer = SimpleImputer(strategy="median")
X_median = pd.DataFrame(median_imputer.fit_transform(X_all_enc), columns=all_feat_names)

# Scaler for distance & regression-based models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all_enc)

# KNN (k=15)
knn_imputer = KNNImputer(n_neighbors=15)
X_knn = pd.DataFrame(scaler.inverse_transform(knn_imputer.fit_transform(X_scaled)), columns=all_feat_names)

# MICE (IterativeImputer with Bayesian Ridge)
mice_imputer = IterativeImputer(max_iter=25, tol=1e-3, random_state=42)
X_mice = pd.DataFrame(scaler.inverse_transform(mice_imputer.fit_transform(X_scaled)), columns=all_feat_names)

# MissForest (ExtraTreesRegressor)
rf_imputer = IterativeImputer(
    estimator=ExtraTreesRegressor(n_estimators=30, random_state=42, n_jobs=-1),
    max_iter=5,
    random_state=42
)
X_rf = pd.DataFrame(rf_imputer.fit_transform(X_all_enc), columns=all_feat_names)

# 4. Train churn classifier across each imputed dataset
datasets = {
    "Complete-Case Deletion": (X_cc, y_cc),
    "Median Baseline": (X_median, y_all),
    "KNN (k=15)": (X_knn, y_all),
    "MICE": (X_mice, y_all),
    "MissForest": (X_rf, y_all),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

churn_results = []
for name, (X_data, y_data) in datasets.items():
    proba = cross_val_predict(clf, X_data, y_data, cv=cv, method="predict_proba")[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_data, proba)
    f1 = f1_score(y_data, preds)

    churn_results.append({
        "Imputation Strategy": name,
        "Sample Size": len(X_data),
        "ROC-AUC": round(auc, 4),
        "F1-Score": round(f1, 4)
    })

print(pd.DataFrame(churn_results).to_string(index=False))

# Extracted from notebook cell 40
import pandas as pd
import statsmodels.api as sm

# Test conditional predictability of credit_utilization missingness
y_util_missing = df["credit_utilization"].isnull().astype(int)

# Logistic regression screening against all fully observed covariates
X_covariates = df[["tenure_years", "churned"]].copy()
X_covariates = pd.concat(
    [
        X_covariates,
        pd.get_dummies(df["segment"], drop_first=True, dtype=float),
    ],
    axis=1,
)
X_covariates["age"] = df["age"].fillna(df["age"].median())
X_covariates["credit_score"] = df["credit_score"].fillna(
    df["credit_score"].median()
)
X_covariates = sm.add_constant(X_covariates)

logit_model = sm.Logit(y_util_missing, X_covariates).fit(disp=False)
print(f"Credit Utilization Missingness Pseudo-R2: {logit_model.prsquared:.4f}")
print(
    f"Customer Churn Rate: Observed Util = {df[df['credit_utilization'].notnull()]['churned'].mean():.1%}, Missing Util = {df[df['credit_utilization'].isnull()]['churned'].mean():.1%}"
)

