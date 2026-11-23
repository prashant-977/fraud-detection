import time
import json
import pandas as pd
from kafka import KafkaProducer

# Initialize Kafka Producer (Assuming Kafka is running locally on port 9092)
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

path = "C:/Users/tranq/Downloads/dataset/"

print("Loading CSV datasets into memory for streaming...")
pdf_core = pd.read_csv(f"{path}hist_tran.csv")
pdf_switch = pd.read_csv(f"{path}cp_hist_tran.csv")

print("Starting real-time event stream to Kafka...")

# Stream core transactions and switch events concurrently or sequentially
for i in range(len(pdf_core)):
    # Convert row to dictionary
    core_record = pdf_core.iloc[i].to_dict()
    # Replace NaN with None for JSON serialization
    core_record = {k: (None if pd.isna(v) else v) for k, v in core_record.items()}
    
    # Send to Kafka topic
    producer.send('raw-core-transactions', value=core_record)
    
    if i < len(pdf_switch):
        switch_record = pdf_switch.iloc[i].to_dict()
        switch_record = {k: (None if pd.isna(v) else v) for k, v in switch_record.items()}
        producer.send('raw-switch-events', value=switch_record)

    producer.flush()
    
    # Simulate a realistic delay between transactions (e.g., 0.1 seconds = 10 tx/sec)
    time.sleep(0.1)
    
    if (i + 1) % 100 == 0:
        print(f"Streamed {i + 1} transaction pairs to Kafka...")

print("Streaming complete!")