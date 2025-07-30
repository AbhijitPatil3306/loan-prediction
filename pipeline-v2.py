import os
from datetime import datetime
from kfp.v2 import dsl
from kfp.v2.dsl import component, Output, Dataset, Input, Model, Artifact
from google.cloud import aiplatform, storage

PROJECT_ID = "bdai-trainings-01"
BUCKET_NAME = "loan-prediction-abhijit"
REGION = "us-central1"
EXPERIMENT_NAME = "loan-default-prediction"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root/"

@component(
    packages_to_install=[
        "pandas", "numpy", "pyarrow", "google-cloud-storage", 
        "google-cloud-aiplatform", "scikit-learn"
    ],
    base_image="python:3.9"
)
def data_ingestion_component(
    project_id: str,
    bucket_name: str,
    region: str,
    processed_data_path: Output[Dataset]
):
    import logging
    import pandas as pd
    import numpy as np
    from google.cloud import aiplatform, storage
    from google.cloud.aiplatform import Featurestore
    logging.basicConfig(level=logging.INFO)
    aiplatform.init(project=project_id, location=region)
    np.random.seed(42)
    num_records = 5000
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
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    raw_path = "raw_loan_data.parquet"
    data.to_parquet(raw_path)
    raw_blob = bucket.blob(f"data/raw/{datetime.now().strftime('%Y%m%d')}_loans.parquet")
    raw_blob.upload_from_filename(raw_path)
    processed_data = data[[
        'application_id', 'loan_amount', 'interest_rate', 
        'borrower_income', 'FICO_score', 'debt_to_income_ratio',
        'income_to_loan_ratio', 'installment', 'debt_burden', 'loan_default'
    ]].copy()
    processed_path = "processed_loans.parquet"
    processed_data.to_parquet(processed_path)
    processed_blob = bucket.blob(f"data/processed/{datetime.now().strftime('%Y%m%d')}_features.parquet")
    processed_blob.upload_from_filename(processed_path)
    processed_data_path.uri = f"gs://{bucket_name}/{processed_blob.name}"
    processed_data.sample(1000).to_parquet("reference_data.parquet")
    ref_blob = bucket.blob("data/reference/reference_data.parquet")
    ref_blob.upload_from_filename("reference_data.parquet")
    try:
        fs = Featurestore(
            featurestore_name="lending_fs", 
            project=project_id, 
            location=region
        )
        logging.info("Feature store already exists")
    except:
        logging.info("Creating new feature store")
        fs = Featurestore.create(
            featurestore_id="lending_fs",
            online_store_fixed_node_count=1,
            project=project_id,
            location=region
        )
    try:
        entity_type = fs.get_entity_type(entity_type_id="loan_applications")
        logging.info("Entity type already exists")
    except:
        logging.info("Creating new entity type")
        entity_type = fs.create_entity_type(
            entity_type_id="loan_applications",
            description="Loan application features"
        )
    feature_types = {
        "loan_amount": "INT64",
        "interest_rate": "DOUBLE",
        "borrower_income": "INT64",
        "FICO_score": "INT64",
        "debt_to_income_ratio": "DOUBLE",
        "income_to_loan_ratio": "DOUBLE",
        "installment": "DOUBLE",
        "debt_burden": "DOUBLE",
        "loan_default": "BOOL"
    }
    for feature_id, value_type in feature_types.items():
        try:
            entity_type.get_feature(feature_id=feature_id)
        except:
            logging.info(f"Creating feature: {feature_id}")
            entity_type.create_feature(
                feature_id=feature_id,
                value_type=value_type
            )
    import_job = entity_type.import_feature_values(
        feature_ids=list(feature_types.keys()),
        entity_id_field="application_id",
        feature_time=datetime.now().isoformat(),
        gcs_source_uri=processed_data_path.uri,
        worker_count=1
    )
    logging.info("Waiting for feature import to complete...")
    import_job.result()
    logging.info("Feature values imported successfully")

@component(
    packages_to_install=[
        "google-cloud-aiplatform", "scikit-learn", "pandas", 
        "joblib", "pyarrow"
    ],
    base_image="python:3.9"
)
def training_component(
    project_id: str,
    region: str,
    bucket_name: str,
    processed_data_path: Input[Dataset],
    model: Output[Model]
):
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from google.cloud import storage, aiplatform
    aiplatform.init(project=project_id, location=region)
    data = pd.read_parquet(processed_data_path.path)
    X = data.drop(columns=["loan_default", "application_id"])
    y = data["loan_default"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    params = {
        "n_estimators": 150,
        "max_depth": 10,
        "min_samples_split": 5,
        "class_weight": "balanced"
    }
    model_clf = RandomForestClassifier(**params, random_state=42)
    model_clf.fit(X_train, y_train)
    y_pred = model_clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print(f"Model trained with AUC: {auc:.4f}")
    model_path = "model.joblib"
    joblib.dump(model_clf, model_path)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_uri = f"gs://{bucket_name}/models/{timestamp}/model.joblib"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    model_blob = bucket.blob(f"models/{timestamp}/model.joblib")
    model_blob.upload_from_filename(model_path)
    model.uri = model_uri
    model.metadata["framework"] = "scikit-learn"
    model.metadata["auc"] = auc

@component(
    packages_to_install=[
        "google-cloud-aiplatform", "scikit-learn", "pandas", 
        "joblib", "pyarrow"
    ],
    base_image="python:3.9"
)
def validation_component(
    model: Input[Model],
    project_id: str,
    region: str,
    threshold: float = 0.75
):
    import joblib
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from google.cloud import storage
    storage_client = storage.Client()
    model_path = model.uri.replace("gs://", "")
    bucket_name = model_path.split("/")[0]
    blob_path = "/".join(model_path.split("/")[1:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    local_path = "local_model.joblib"
    blob.download_to_filename(local_path)
    model_clf = joblib.load(local_path)
    data_path = model.uri.replace("models", "data/processed").rsplit("/", 1)[0] + "_features.parquet"
    data = pd.read_parquet(data_path)
    val_data = data.sample(1000, random_state=42)
    X_val = val_data.drop(columns=["loan_default", "application_id"])
    y_val = val_data["loan_default"]
    y_pred = model_clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_pred)
    if val_auc < threshold:
        raise ValueError(f"Validation failed: AUC {val_auc:.4f} < threshold {threshold}")
    print(f"Validation passed with AUC: {val_auc:.4f}")

@component(
    packages_to_install=["google-cloud-aiplatform"],
    base_image="python:3.9"
)
def deployment_component(
    project_id: str,
    region: str,
    model: Input[Model],
    endpoint: Output[Artifact]
):
    from google.cloud import aiplatform
    aiplatform.init(project=project_id, location=region)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    vertex_model = aiplatform.Model.upload(
        display_name=f"loan-default-model-{timestamp}",
        artifact_uri=model.uri.rsplit("/", 1)[0],
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
    )
    endpoints = aiplatform.Endpoint.list(
        filter='display_name="loan-default-endpoint"',
        order_by="create_time desc",
        project=project_id,
        location=region
    )
    if endpoints:
        endpoint_resource = endpoints[0]
    else:
        endpoint_resource = aiplatform.Endpoint.create(
            display_name="loan-default-endpoint",
            project=project_id,
            location=region
        )
    endpoint_resource.deploy(
        model=vertex_model,
        deployed_model_display_name=f"loan-model-{timestamp}",
        traffic_percentage=100,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3
    )
    endpoint.uri = endpoint_resource.resource_name
    print(f"Model deployed to endpoint: {endpoint.uri}")

@component(
    packages_to_install=["google-cloud-aiplatform"],
    base_image="python:3.9"
)
def monitoring_component(
    project_id: str,
    region: str,
    bucket_name: str,
    endpoint: Input[Artifact]
):
    from google.cloud import aiplatform
    aiplatform.init(project=project_id, location=region)
    monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
        display_name="loan-drift-monitoring",
        endpoint=endpoint.uri,
        logging_sampling_strategy=0.8,
        model_deployment_monitoring_objective_configs=[
            {
                "objective": "feature_drift",
                "feature_drift": {
                    "features": ["FICO_score", "debt_to_income_ratio"]
                }
            }
        ],
        schedule_config={"monitor_interval": 86400},  # Daily
        analysis_instance_schema_uri="gs://google-cloud-aiplatform/schema/monitoring/instance/1.0.0/monitored_field.json",
        enable_monitoring_pipeline_logs=True,
        sample_predict_instance={
            "FICO_score": 720,
            "debt_to_income_ratio": 0.35,
            "interest_rate": 7.5,
            "borrower_income": 85000,
            "loan_amount": 25000,
            "income_to_loan_ratio": 3.4,
            "installment": 750.25,
            "debt_burden": 29750
        }
    )
    print(f"Monitoring job created: {monitoring_job.resource_name}")

@dsl.pipeline(
    name="loan-default-pipeline",
    description="End-to-end loan default prediction pipeline",
    pipeline_root=PIPELINE_ROOT
)
def pipeline(
    project: str = PROJECT_ID,
    region: str = REGION,
    bucket: str = BUCKET_NAME
):
    ingest_task = data_ingestion_component(
        project_id=project,
        bucket_name=bucket,
        region=region
    )

    train_task = training_component(
        project_id=project,
        region=region,
        bucket_name=bucket,
        processed_data_path=ingest_task.outputs["processed_data_path"]
    ).after(ingest_task)
    
    validate_task = validation_component(
        project_id=project,
        region=region,
        model=train_task.outputs["model"]
    ).after(train_task)
    
    deploy_task = deployment_component(
        project_id=project,
        region=region,
        model=train_task.outputs["model"]
    ).after(validate_task)
    
    monitoring_component(
        project_id=project,
        region=region,
        bucket_name=bucket,
        endpoint=deploy_task.outputs["endpoint"]
    ).after(deploy_task)

if __name__ == "__main__":
    from kfp.v2 import compiler
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="loan_default_pipeline.json"
    )
    print("Pipeline compiled successfully")