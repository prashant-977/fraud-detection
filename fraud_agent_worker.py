import os
import time
import psycopg2
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# --- Configuration ---
DB_CONFIG = {
    "host": "localhost",
    "port": 5439,
    "dbname": "testdb",
    "user": "postgres",
    "password": "postgres",
}
TABLE_NAME = "fraud_dashboard_view"
MODEL_NAME = "gemini-2.5-flash"
POLL_INTERVAL_SEC = 5
RATE_LIMIT_SLEEP_SEC = 5  # Keeps requests safely under free-tier RPM limits


class FraudInvestigation(BaseModel):
    recommendation: str = Field(
        description="One of: BLOCK_CARD, FREEZE_ACCOUNT, STEP_UP_AUTH, or MONITOR"
    )
    summary: str = Field(
        description="Concise 2-sentence fraud investigation summary explaining the risk factors."
    )


def fetch_uninvestigated_alerts(conn, limit: int = 10):
    query = f"""
        SELECT tran_id_3, date_create_1, tran_amt_16, tran_curr_17,
               mer_name_27, mcc_23, channel_28, fraud_probability
        FROM {TABLE_NAME}
        WHERE is_high_risk = TRUE
          AND agent_summary IS NULL
        ORDER BY fraud_probability DESC
        LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(query, (limit,))
        return cur.fetchall()


def update_investigation(conn, tran_id: str, recommendation: str, summary: str):
    query = f"""
        UPDATE {TABLE_NAME}
        SET agent_recommendation = %s,
            agent_summary = %s
        WHERE tran_id_3 = %s;
    """
    with conn.cursor() as cur:
        cur.execute(query, (recommendation, summary, tran_id))
    conn.commit()


def investigate_transaction(client: genai.Client, row) -> FraudInvestigation:
    tran_id, ts, amt, curr, merchant, mcc, channel, prob = row
    prompt = f"""
    You are a Senior Banking Fraud Investigator. Analyze this flagged card transaction
    and provide an action recommendation and a brief 2-sentence investigation summary.

    Transaction ID: {tran_id}
    Timestamp: {ts}
    Amount: {amt} {curr}
    Merchant: {merchant} (MCC: {mcc})
    Channel: {channel}
    Model Fraud Probability: {prob:.4f}
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FraudInvestigation,
            temperature=0.2,
        ),
    )
    return FraudInvestigation.model_validate_json(response.text)


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)
    conn = psycopg2.connect(**DB_CONFIG)
    print(f"AI Fraud Investigator running with {MODEL_NAME}. Polling for high-risk alerts...")

    try:
        while True:
            alerts = fetch_uninvestigated_alerts(conn, limit=10)
            if not alerts:
                time.sleep(POLL_INTERVAL_SEC)
                continue

            for row in alerts:
                tran_id = row[0]
                try:
                    result = investigate_transaction(client, row)
                    update_investigation(conn, tran_id, result.recommendation, result.summary)
                    print(f"[{tran_id}] -> {result.recommendation} | {result.summary[:80]}...")
                except Exception as e:
                    print(f"[{tran_id}] Error calling Gemini API: {e}")

                time.sleep(RATE_LIMIT_SLEEP_SEC)
    except KeyboardInterrupt:
        print("\nAI Fraud Investigator stopped.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()