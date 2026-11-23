import os
import time
import joblib
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

# --- Configuration ---
DB_CONFIG = {
    "host": "localhost",
    "port": 5439,           # Update if your Postgres container uses 5432
    "dbname": "testdb",   # Update to your database name
    "user": "postgres",     # Update to your Postgres user
    "password": "postgres"
}

DATA_PATH = "C:/Users/tranq/Downloads/dataset/cp_hist_tran.csv"
FEATURE_CACHE_PATH = "C:/Users/tranq/Downloads/dataset/rfm_features_lookup.parquet"
MODEL_PATH = "C:/Users/tranq/Downloads/dataset/lgbm_fraud_model.pkl"

TABLE_NAME = "fraud_dashboard_view"
RISK_THRESHOLD = 0.40
POLL_INTERVAL_SEC = 2
BATCH_SIZE = 200

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


def load_or_build_feature_store() -> pd.DataFrame:
    if os.path.exists(FEATURE_CACHE_PATH):
        print(f"Loading cached RFM feature store from {FEATURE_CACHE_PATH}...")
        lookup = pd.read_parquet(FEATURE_CACHE_PATH)
        lookup.index = lookup.index.astype(str).str.strip()
        return lookup

    print(f"Building RFM feature store from {DATA_PATH} (one-time)...")
    df = pd.read_csv(DATA_PATH, usecols=["tran_id_3", "acct_num_31", "date_create_1", "tran_amt_16"])
    df["switch_time"] = pd.to_datetime(df["date_create_1"], errors="coerce")
    df["tran_amt_16"] = pd.to_numeric(df["tran_amt_16"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["tran_id_3", "acct_num_31", "switch_time"])
    df = df.sort_values(["acct_num_31", "switch_time"]).reset_index(drop=True)

    df["recency_seconds"] = (
        df.groupby("acct_num_31")["switch_time"]
        .diff()
        .dt.total_seconds()
        .fillna(999999.0)
    )

    df_indexed = df.set_index("switch_time")
    for label, window in [("30d", "30D"), ("7d", "7D"), ("24h", "24h")]:
        rolling_grp = (
            df_indexed.groupby("acct_num_31", sort=False)["tran_amt_16"]
            .rolling(window, closed="left")
        )
        df[f"frequency_{label}"] = rolling_grp.count().fillna(0).values
        df[f"monetary_{label}"] = rolling_grp.sum().fillna(0.0).values

    lookup = df.drop_duplicates(subset=["tran_id_3"]).set_index("tran_id_3")[FEATURE_COLS]
    lookup.to_parquet(FEATURE_CACHE_PATH)
    print(f"Cached {len(lookup):,} transaction feature vectors to {FEATURE_CACHE_PATH}.")
    return lookup


def fetch_unscored_rows(conn, limit: int) -> pd.DataFrame:
    query = f"""
        SELECT tran_id_3, tran_amt_16
        FROM {TABLE_NAME}
        WHERE fraud_probability IS NULL
        LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(query, (limit,))
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["tran_id_3", "tran_amt_16"])


def update_scores(conn, updates: list[tuple]):
    query = f"""
        UPDATE {TABLE_NAME}
        SET fraud_probability = %s,
            is_high_risk = %s
        WHERE tran_id_3 = %s;
    """
    with conn.cursor() as cur:
        execute_batch(cur, query, updates)
    conn.commit()


def main():
    feature_store = load_or_build_feature_store()
    print(f"Loading LightGBM model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    conn = psycopg2.connect(**DB_CONFIG)
    print("Connected to PostgreSQL. Real-time scoring worker is running...")

    try:
        while True:
            df_batch = fetch_unscored_rows(conn, BATCH_SIZE)
            if df_batch.empty:
                time.sleep(POLL_INTERVAL_SEC)
                continue

            clean_ids = df_batch["tran_id_3"].astype(str).str.strip()
            X = feature_store.reindex(clean_ids).reset_index(drop=True)

            missing_mask = X["tran_amt_16"].isna()
            missed = int(missing_mask.sum())
            if missed:
                X.loc[missing_mask, "tran_amt_16"] = pd.to_numeric(
                    df_batch.loc[missing_mask, "tran_amt_16"], errors="coerce"
                ).fillna(0.0)
                X.loc[missing_mask, "recency_seconds"] = 999999.0
                X = X.fillna(0.0)

            probs = model.predict_proba(X[FEATURE_COLS])[:, 1]

            updates = [
                (float(prob), bool(prob > RISK_THRESHOLD), str(tid))
                for tid, prob in zip(df_batch["tran_id_3"], probs)
            ]
            update_scores(conn, updates)

            print(
                f"Scored {len(probs)} | Missed lookup: {missed} | "
                f"Prob min={probs.min():.4f}, max={probs.max():.4f} | "
                f"High-risk (> {RISK_THRESHOLD}): {int((probs > RISK_THRESHOLD).sum())}"
            )
    except KeyboardInterrupt:
        print("\nScoring worker stopped.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()