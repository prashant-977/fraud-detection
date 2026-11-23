import os
from pyflink.table import EnvironmentSettings, TableEnvironment

# 1. Initialize Streaming Table Environment
env_settings = EnvironmentSettings.in_streaming_mode()
t_env = TableEnvironment.create(env_settings)

# Add JAR dependencies for Kafka and PostgreSQL JDBC connectors
jars_dir = os.path.abspath("./jars")
t_env.get_config().set(
    "pipeline.jars",
    f"file:///{jars_dir}/flink-sql-connector-kafka-3.1.0-1.18.jar;"
    f"file:///{jars_dir}/flink-connector-jdbc-3.1.2-1.18.jar;"
    f"file:///{jars_dir}/postgresql-42.6.0.jar"
)

# 2. Source Table 1: Card Switch Events (Kafka) with Watermark
t_env.execute_sql("""
    CREATE TABLE kafka_switch_events (
        date_create_1 STRING,
        mti_2 INT,
        tran_id_3 STRING,
        tran_desc_4 STRING,
        card_num_11 STRING,
        tran_amt_16 DOUBLE,
        tran_curr_17 STRING,
        acq_country_cod_22 STRING,
        mcc_23 INT,
        mer_name_27 STRING,
        channel_28 STRING,
        acct_num_31 STRING,
        switch_time AS TO_TIMESTAMP(date_create_1, 'yyyy-MM-dd HH:mm:ss'),
        clean_acct AS REGEXP_REPLACE(acct_num_31, '[^0-9]', ''),
        WATERMARK FOR switch_time AS switch_time - INTERVAL '10' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'raw-switch-events',
        'properties.bootstrap.servers' = 'localhost:9092',
        'properties.group.id' = 'flink-switch-group',
        'scan.startup.mode' = 'earliest-offset',
        'format' = 'json'
    )
""")

# 3. Source Table 2: Core Banking Transactions (Kafka) with Watermark
t_env.execute_sql("""
    CREATE TABLE kafka_core_trans (
        tran_date STRING,
        acid STRING,
        tran_amt DOUBLE,
        tran_particular STRING,
        core_time AS TO_TIMESTAMP(tran_date, 'yyyy-MM-dd HH:mm:ss'),
        clean_acid AS REGEXP_REPLACE(acid, '[^0-9]', ''),
        WATERMARK FOR core_time AS core_time - INTERVAL '10' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'raw-core-transactions',
        'properties.bootstrap.servers' = 'localhost:9092',
        'properties.group.id' = 'flink-core-group',
        'scan.startup.mode' = 'earliest-offset',
        'format' = 'json'
    )
""")

# 4. Sink Table: PostgreSQL Database
t_env.execute_sql("""
    CREATE TABLE postgres_sink (
        tran_id_3 STRING,
        date_create_1 TIMESTAMP(3),
        tran_amt_16 DECIMAL(18, 2),
        tran_curr_17 STRING,
        mer_name_27 STRING,
        mcc_23 INT,
        channel_28 STRING,
        PRIMARY KEY (tran_id_3) NOT ENFORCED
    ) WITH (
        'connector' = 'jdbc',
        'url' = 'jdbc:postgresql://localhost:5432/testdb',
        'table-name' = 'public.fraud_dashboard_view',
        'username' = 'postgres',
        'password' = 'postgres'
    )
""")

# 5. Continuous Interval Join Query (+/- 5 Minutes) and Output to Sink
streaming_join_query = """
    INSERT INTO postgres_sink
    SELECT 
        s.tran_id_3,
        s.switch_time AS date_create_1,
        CAST(s.tran_amt_16 AS DECIMAL(18, 2)),
        s.tran_curr_17,
        s.mer_name_27,
        s.mcc_23,
        s.channel_28
    FROM kafka_switch_events s
    LEFT JOIN kafka_core_trans c
        ON s.clean_acct = c.clean_acid
        AND c.core_time BETWEEN s.switch_time - INTERVAL '5' MINUTE 
                            AND s.switch_time + INTERVAL '5' MINUTE
"""

print("Starting Flink Streaming Job (Interval Join: Switch <-> Core)...")
t_env.execute_sql(streaming_join_query).wait()