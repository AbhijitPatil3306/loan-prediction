from google.cloud import aiplatform

PROJECT_ID = "white-library-467506-i4"
REGION = "us-central1"
BUCKET_NAME = "loan-vertex-pridection-abhijit"

aiplatform.init(project=PROJECT_ID, location=REGION)

job = aiplatform.PipelineJob(
    display_name="loan-default-prediction",
    template_path="loan_default_pipeline.json",
    pipeline_root=f"gs://{BUCKET_NAME}/pipeline_root/",
    enable_caching=False
)

job.submit()