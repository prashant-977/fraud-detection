import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

# --- Configuration ---
DATA_PATH = "C:/Users/tranq/Downloads/dataset/cp_hist_tran.csv"
MODEL_OUTPUT_PATH = "C:/Users/tranq/Downloads/dataset/lgbm_fraud_model.pkl"

FEATURE_COLS = [
    "tran_amt_16",
    "recency_seconds",
    "frequency_30d",
    "monetary_30d",
    "frequency_7d",
    "monetary_7d",
    "frequency_24h",
    "monetary_24h",
]

def build_rfm_features(csv_path: str) -> pd.DataFrame:
    print(f"Loading switch transactions from {csv_path}...")
    df = pd.read_csv(csv_path, usecols=["tran_id_3", "acct_num_31", "date_create_1", "tran_amt_16"])
    df["switch_time"] = pd.to_datetime(df["date_create_1"], errors="coerce")
    df["tran_amt_16"] = pd.to_numeric(df["tran_amt_16"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["acct_num_31", "switch_time"])

    # Sort chronologically per account for point-in-time rolling windows
    df = df.sort_values(["acct_num_31", "switch_time"]).reset_index(drop=True)

    # Recency (seconds since previous transaction for the same account)
    df["recency_seconds"] = (
        df.groupby("acct_num_31")["switch_time"]
        .diff()
        .dt.total_seconds()
        .fillna(999999.0)
    )

    # Rolling Frequency & Monetary (closed='left' excludes current transaction)
    df_indexed = df.set_index("switch_time")
    for label, window in [("30d", "30D"), ("7d", "7D"), ("24h", "24h")]:
        rolling_grp = (
            df_indexed.groupby("acct_num_31", sort=False)["tran_amt_16"]
            .rolling(window, closed="left")
        )
        df[f"frequency_{label}"] = rolling_grp.count().fillna(0).values
        df[f"monetary_{label}"] = rolling_grp.sum().fillna(0.0).values

    print(f"RFM features built for {len(df):,} transactions.")
    return df

def generate_autoencoder_labels(df: pd.DataFrame) -> pd.DataFrame:
    print("Training unsupervised Autoencoder (20% sample -> 10% train / 10% holdout)...")
    X_all = df[FEATURE_COLS].fillna(0.0)

    # 20% sample, shuffled and split 50/50 into train and holdout
    pdf_sample = X_all.sample(frac=0.2, random_state=42).sample(frac=1, random_state=42)
    split_idx = len(pdf_sample) // 2
    pdf_train, pdf_holdout = pdf_sample.iloc[:split_idx], pdf_sample.iloc[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(pdf_train)

    autoencoder = MLPRegressor(
        hidden_layer_sizes=(4,),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=42,
    )
    autoencoder.fit(X_train, X_train)

    # Compute 99.9th percentile threshold strictly on unseen holdout data
    X_holdout = scaler.transform(pdf_holdout)
    holdout_preds = autoencoder.predict(X_holdout)
    holdout_mse = np.mean(np.power(X_holdout - holdout_preds, 2), axis=1)
    threshold = float(np.percentile(holdout_mse, 99.9))

    # Score full dataset using frozen scaler, model, and holdout threshold
    X_all_scaled = scaler.transform(X_all)
    all_preds = autoencoder.predict(X_all_scaled)
    all_mse = np.mean(np.power(X_all_scaled - all_preds, 2), axis=1)

    df["reconstruction_mse"] = all_mse
    df["is_fraud_pseudo"] = (all_mse > threshold).astype(int)

    fraud_count = int(df["is_fraud_pseudo"].sum())
    print(f"Holdout 99.9th percentile MSE threshold: {threshold:.3e}")
    print(f"Pseudo-labeled fraud cases: {fraud_count:,} / {len(df):,} ({fraud_count/len(df)*100:.3f}%)")
    return df

def train_lightgbm(df: pd.DataFrame):
    X = df[FEATURE_COLS].fillna(0.0)
    y = df["is_fraud_pseudo"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("Fraud_LightGBM_Classifier")

    with mlflow.start_run(run_name="LightGBM_From_Autoencoder_Labels"):
        params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }

        print("Training LightGBM model...")
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )

        val_probs = model.predict_proba(X_val)[:, 1]
        val_pr_auc = average_precision_score(y_val, val_probs)
        val_roc_auc = roc_auc_score(y_val, val_probs)

        mlflow.log_params(params)
        mlflow.log_metric("val_pr_auc", val_pr_auc)
        mlflow.log_metric("val_roc_auc", val_roc_auc)
        mlflow.lightgbm.log_model(model, artifact_path="lightgbm_fraud_model")

        joblib.dump(model, MODEL_OUTPUT_PATH)

        print(f"Validation PR-AUC:  {val_pr_auc:.4f}")
        print(f"Validation ROC-AUC: {val_roc_auc:.4f}")
        print(f"Model saved locally to: {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    df_gold = build_rfm_features(DATA_PATH)
    df_labeled = generate_autoencoder_labels(df_gold)
    train_lightgbm(df_labeled)