import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from google.cloud import aiplatform, storage, bigquery

PROJECT_ID = os.getenv("GCP_PROJECT", "white-library-467506-i4")
BUCKET_NAME = os.getenv("GCP_BUCKET", "loan-vertex-pridection-abhijit")
REGION = os.getenv("GCP_REGION", "us-central1")
experiment_name = "loan-default-prediction"
aiplatform.init(project=PROJECT_ID, location=REGION)

def generate_loan_data(num_records=10000):
    np.random.seed(42)
    data = pd.DataFrame({
        'application_id': [f'APP_{i:06d}' for i in range(num_records)],
        'loan_amount': np.random.lognormal(8, 0.4, num_records).astype(int),
        'interest_rate': np.random.uniform(3, 25, num_records),
        'borrower_income': np.random.lognormal(10, 0.3, num_records).astype(int),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.8, num_records),
        'FICO_score': np.random.randint(300, 850, num_records),
        'employment_length': np.random.choice(['<1y', '1-3y', '3-5y', '5-10y', '10+y'], num_records),
        'loan_purpose': np.random.choice([
            'debt_consolidation', 'home_improvement', 
            'business', 'medical', 'other'
        ], num_records),
    })
    risk_factors = (
        0.5 * (data['debt_to_income_ratio'] > 0.4) +
        0.3 * (data['FICO_score'] < 650) +
        0.2 * (data['employment_length'].isin(['<1y', '1-3y'])) +
        0.1 * (data['interest_rate'] > 15)
    )
    default_proba = 1 / (1 + np.exp(-(risk_factors - 0.5 + np.random.normal(0, 0.2, num_records))))
    data['loan_default'] = np.random.binomial(1, default_proba)
    data['income_to_loan_ratio'] = data['borrower_income'] / data['loan_amount']
    data['installment'] = data['loan_amount'] * (data['interest_rate']/1200) / (1 - (1 + data['interest_rate']/1200)**(-36))
    data['debt_burden'] = data['debt_to_income_ratio'] * data['borrower_income']
    return data

def ingest_and_process_data():
    raw_data = generate_loan_data(5000)
    raw_data.to_parquet("raw_loan_data.parquet")
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"data/raw/{datetime.now().strftime('%Y%m%d')}_loans.parquet")
    blob.upload_from_filename("raw_loan_data.parquet")
    processed_data = raw_data[[
        'application_id', 'loan_amount', 'interest_rate', 
        'borrower_income', 'FICO_score', 'debt_to_income_ratio',
        'income_to_loan_ratio', 'installment', 'debt_burden', 'loan_default'
    ]].copy()
    processed_data.to_parquet("processed_loans.parquet")
    processed_blob = bucket.blob(f"data/processed/{datetime.now().strftime('%Y%m%d')}_features.parquet")
    processed_blob.upload_from_filename("processed_loans.parquet")
    try:
        fs = Featurestore(featurestore_name="lending_fs", project=PROJECT_ID)
    except:
        fs = aiplatform.Featurestore.create(
            featurestore_id="lending_fs",
            online_store_fixed_node_count=1
        )
    try:
        entity_type = fs.get_entity_type(entity_type_id="loan_applications")
    except:
        entity_type = fs.create_entity_type(
            entity_type_id="loan_applications",
            description="Loan application features"
        )
    entity_type.import_feature_values(
        feature_ids=["loan_amount", "interest_rate", "borrower_income", 
                    "FICO_score", "debt_to_income_ratio", "income_to_loan_ratio",
                    "installment", "debt_burden", "loan_default"],
        entity_id_field="application_id",
        feature_time=datetime.now().isoformat(),
        gcs_source_uri=f"gs://{BUCKET_NAME}/data/processed/{datetime.now().strftime('%Y%m%d')}_features.parquet"
    )
    return processed_data

def train_model():
    experiment = aiplatform.Experiment.create(experiment_name)
    experiment_run = experiment.start_run()
    fs = aiplatform.Featurestore(featurestore_name="lending_fs", project=PROJECT_ID)
    features = fs.batch_read(
        entity_type_id="loan_applications",
        feature_ids=["loan_amount", "interest_rate", "borrower_income", 
                    "FICO_score", "debt_to_income_ratio", "income_to_loan_ratio",
                    "installment", "debt_burden", "loan_default"]
    )
    X = features.drop(columns=["loan_default", "application_id"])
    y = features["loan_default"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    params = {
        "n_estimators": 150,
        "max_depth": 10,
        "min_samples_split": 5,
        "class_weight": "balanced"
    }
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    experiment_run.log_params(params)
    experiment_run.log_metrics({"auc": auc})
    model_path = "model.joblib"
    joblib.dump(model, model_path)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    vertex_model = aiplatform.Model.upload(
        display_name=f"loan-default-model-{timestamp}",
        artifact_uri=f"gs://{BUCKET_NAME}/models/{timestamp}/",
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
    )
    vertex_model.versioned_resources.append(experiment_run.resource_name)
    vertex_model.update()
    print(f"Model registered with AUC: {auc:.4f}")
    return vertex_model

def validate_model(vertex_model, threshold=0.80):
    fs = aiplatform.Featurestore(featurestore_name="lending_fs", project=PROJECT_ID)
    val_data = fs.batch_read(
        entity_type_id="loan_applications",
        feature_ids=["loan_amount", "interest_rate", "borrower_income", 
                    "FICO_score", "debt_to_income_ratio", "income_to_loan_ratio",
                    "installment", "debt_burden", "loan_default"]
    ).sample(10000, random_state=42)
    X_val = val_data.drop(columns=["loan_default", "application_id"])
    y_val = val_data["loan_default"]
    eval_job = vertex_model.evaluate(
        X=X_val,
        y=y_val,
        prediction_type="classification",
        target_field_name="loan_default",
        class_labels=[0, 1],
    )
    eval_job.wait()
    metrics = eval_job.metrics
    val_auc = metrics["auRoc"]
    if val_auc < threshold:
        raise ValueError(f"Validation AUC {val_auc:.4f} < threshold {threshold}")
    print(f"Model validation passed with AUC: {val_auc:.4f}")
    return True

def deploy_model(vertex_model):
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="loan-default-endpoint"',
        order_by="create_time desc"
    )
    if endpoints:
        endpoint = endpoints[0]
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name="loan-default-endpoint"
        )
    endpoint.deploy(
        model=vertex_model,
        deployed_model_display_name=vertex_model.display_name,
        traffic_percentage=100,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3
    )
    print(f"Model deployed to endpoint: {endpoint.resource_name}")
    return endpoint

def detect_drift():
    dataset = aiplatform.ModelDeploymentMonitoringDataset.create(
        display_name="loan-monitoring-data",
        gcs_source=[f"gs://{BUCKET_NAME}/data/reference/reference_data.parquet"],
        data_format="parquet"
    )
    monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
        display_name="loan-drift-monitoring",
        endpoint=deploy_model.endpoint,  # From previous deployment
        model_deployment_monitoring_objective_configs=[
            {
                "objective": "feature_drift",
                "feature_drift": {"features": ["FICO_score", "debt_to_income_ratio"]}
            }
        ],
        schedule_config={"monitor_interval": 86400},  # Daily
        analysis_instance_schema_uri="gs://google-cloud-aiplatform/schema/monitoring/instance/1.0.0/monitored_field.json",
        enable_monitoring_pipeline_logs=True,
        dataset=dataset
    )
    drift_status = monitoring_job.state
    if drift_status == aiplatform.JobState.JOB_STATE_FAILED:
        logging.error("Drift monitoring detected issues!")
    return monitoring_job

def run_pipeline():
    logging.info("Starting data ingestion...")
    processed_data = ingest_and_process_data()
    processed_data.sample(10000).to_parquet("reference_data.parquet")
    bucket = storage.Client().bucket(BUCKET_NAME)
    ref_blob = bucket.blob("data/reference/reference_data.parquet")
    ref_blob.upload_from_filename("reference_data.parquet")
    logging.info("Starting model training...")
    vertex_model = train_model()
    logging.info("Validating model...")
    validate_model(vertex_model)
    logging.info("Deploying model...")
    endpoint = deploy_model(vertex_model)
    logging.info("Setting up monitoring...")
    monitoring_job = detect_drift()
    logging.info("Pipeline execution completed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()