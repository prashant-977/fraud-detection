# Real-Time Banking Fraud Detection & AI Investigation Platform

This platform is an end-to-end Kappa/Lambda hybrid fraud detection system designed to ingest, reconcile, score, and investigate banking transactions in real time. It bridges real-time stream processing (Kafka + PyFlink) with decoupled machine learning inference (Autoencoder + LightGBM), autonomous LLM triage (Gemini API), and live operational BI (Apache Superset).

## 1. Project Objective

Financial institutions face two major challenges in transaction monitoring: reconciling asynchronous multi-source transaction logs (card switch vs. core banking) within tight latency windows, and detecting fraud without pre-existing labeled ground truth while reducing analyst alert fatigue.

This platform addresses both by:

- Reconciling card switch and core banking streams in real time using temporal interval joins.
- Generating fraud labels unsupervised via deep/neural reconstruction error and training a fast gradient-boosted classifier for live scoring.
- Automating Level-1 fraud triage by generating structured action recommendations and natural-language investigation summaries for high-risk alerts.

## 2. End-to-End Architecture & Component Breakdown

```text
[Raw CSVs] ──▶ kafka_producer.py ──▶ [Apache Kafka (Docker)]
                                              │
                                              ▼
                                   flink_fraud_stream.py (PyFlink)
                                   (±5 min Temporal Interval Join)
                                              │
                                              ▼
                              [PostgreSQL: fraud_dashboard_view] ◀──┐
                                    │  ▲                 ▲   │      │
                                    ▼  │                 │   ▼      │
                          scoring_worker.py      fraud_agent_worker.py
                          (LightGBM + RFM)       (Gemini 2.5 Flash)
                                              │
                                              ▼
                                  [Apache Superset (Docker)]
          (Live Counters, Fraud by Channel, Investigation Queue)
```

| Component / File | Environment | Core Functionality |
|---|---|---|
| Docker Services (Kafka, PostgreSQL, Superset) | Docker Compose | Provides the messaging backbone (Kafka), operational serving database (PostgreSQL `fraud_dashboard_view`), and live visualization layer (Apache Superset). |
| `kafka_producer.py` | `spark` Conda env | Loads raw core banking (`hist_tran.csv`) and card switch (`cp_hist_tran.csv`) datasets, merges and sorts them chronologically to prevent out-of-order temporal issues, cleans nulls, and streams events to `raw-switch-events` and `raw-core-transactions` topics. |
| `flink_fraud_stream.py` | `spark` Conda env | Consumes both Kafka topics, normalizes account IDs via regex, parses timestamps, applies watermarks, executes a ±5-minute interval join, and flushes matched pairs instantly into PostgreSQL. |
| `train_fraud_model.py` | `fraud_ml` Conda env | Offline training pipeline. Engineers point-in-time RFM features, generates unsupervised pseudo-labels via an Autoencoder, trains a LightGBM classifier tracked in MLflow, and exports `lgbm_fraud_model.pkl`. |
| `scoring_worker.py` | `fraud_ml` Conda env | Standalone real-time inference worker. Maintains a cached Parquet RFM feature lookup, polls PostgreSQL for unscored rows (`fraud_probability IS NULL`), scores them with LightGBM, and flags high-risk alerts (`is_high_risk = TRUE`). |
| `fraud_agent_worker.py` | `fraud_ml` Conda env | Autonomous AI Investigator. Polls PostgreSQL for flagged high-risk rows (`agent_summary IS NULL`), calls the Gemini API (`gemini-2.5-flash`) with structured JSON output, and persists `agent_recommendation` and `agent_summary`. |

## 3. Core Methodologies & Engineering Decisions

### A. Real-Time Stream Reconciliation (PyFlink)

- **Temporal Interval Join:** Card switch authorizations and core banking ledger postings do not arrive at the exact same millisecond. PyFlink reconciles them in memory using a ±5-minute window (`c.core_time BETWEEN s.switch_time - INTERVAL '5' MINUTE AND s.switch_time + INTERVAL '5' MINUTE`) on regex-cleaned account numbers.
- **Watermark & Flush Reliability:** Setting `parallelism.default = 1` prevents watermark stalls across single-partition Kafka topics, while `sink.buffer-flush.max-rows = 1` guarantees zero-latency visibility of matched transactions in PostgreSQL.

### B. Feature Engineering & Unsupervised-to-Supervised ML

- **Point-in-Time RFM Features:** Computes 8 leakage-free behavioral features per account: transaction amount, `recency_seconds` since the prior transaction, and rolling 24-hour, 7-day, and 30-day transaction frequency and monetary totals.
- **Holdout Autoencoder Pseudo-Labeling:** Because raw banking data lacks labeled fraud ground truth, an `MLPRegressor` Autoencoder is trained on a 10% sample and evaluated on an unseen 10% holdout set to avoid overfitting bias. Setting the anomaly threshold at the 99.9th percentile of holdout Mean Squared Error (MSE) cleanly separates extreme tail outliers from the dense normal traffic cluster.
- **LightGBM Classification:** A LightGBM classifier is trained on the pseudo-labeled dataset (achieving 0.9844 ROC-AUC and 0.6139 PR-AUC on validation) and logged to MLflow for reproducible deployment.

### C. Decoupled Microservice Pattern & Environment Isolation

- **Resolving Dependency Conflicts:** PyFlink requires older NumPy versions that conflict with modern Scikit-Learn/SciPy (`numpy.exceptions`). Rather than forcing ML libraries inside the PyFlink JVM, inference and LLM orchestration are isolated in a dedicated `fraud_ml` Conda environment communicating asynchronously via PostgreSQL.
- **Structured Agentic Triage:** Using Pydantic schemas with Gemini's JSON mode guarantees deterministic outputs (`BLOCK_CARD`, `FREEZE_ACCOUNT`, `STEP_UP_AUTH`, or `MONITOR`) alongside a concise 2-sentence explanation, making the output immediately consumable by downstream BI dashboards without fragile text parsing.

### D. Experiment Tracking & Model Lifecycle Management (MLflow)

- **Reproducible Experiment Logging:** Every offline training run in `train_fraud_model.py` is tracked locally through MLflow under dedicated experiments (such as `Fraud_LightGBM_Classifier` and the earlier `Fraud_Autoencoder_Unsupervised` run).
- **Logged Parameters & Metrics:** MLflow automatically records model hyperparameters (`n_estimators`, `learning_rate`, `num_leaves`, `random_state`), unsupervised anomaly thresholds, and validation performance metrics (ROC-AUC: 0.9844, PR-AUC: 0.6139) so iterations, like tuning the holdout percentile cutoff, can be compared objectively side by side.
- **Artifact Versioning & Serving Readiness:** Trained LightGBM model artifacts are logged alongside their runs (`mlflow.lightgbm.log_model`) and inspected via the local MLflow Tracking UI (<http://127.0.0.1:5000>). This allows `scoring_worker.py` to either load a local `.pkl` export or pull a specific versioned model directly by MLflow Run ID (`runs:/<RUN_ID>/lightgbm_fraud_model`), keeping model governance decoupled from the live streaming pipeline.

## 4. Operational Output (Apache Superset)

The live dashboard connects directly to `fraud_dashboard_view` in PostgreSQL and provides three operational views:

1. **Real-Time Throughput Counters:** Live volume of reconciled transactions flowing through Kafka and PyFlink.
2. **Fraud by Channel / Merchant:** Aggregated breakdown of high-risk alerts across channels (ATM, POS, WEB, MOBILE) and merchants to spot systemic attacks.
3. **The Investigation Queue:** A prioritized table of high-risk transactions ordered by LightGBM `fraud_probability`, enriched with Gemini's `agent_recommendation` and `agent_summary` for rapid analyst sign-off.

This architecture demonstrates a production-style separation of concerns: streaming ingestion and joins stay fast and fault-tolerant in Flink, ML scoring and LLM reasoning scale independently as Python workers, and every stage is observable in real time through PostgreSQL and Superset.
