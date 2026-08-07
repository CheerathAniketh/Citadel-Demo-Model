"""
deploy_to_sagemaker.py

Uploads the packaged model (artifacts/model.tar.gz from generate_and_train.py)
to S3 and deploys it to a real-time SageMaker endpoint with Data Capture
enabled, so Citadel's monitor_predictions() eventually has real S3 logs to
read instead of mocked data.

Prereqs:
    pip install sagemaker boto3
    AWS CLI configured (aws configure) with a role/user that has SageMaker +
    S3 + IAM pass-role permissions. You already set this up (ap-south-1,
    Aniketh-admin / AdministratorAccess).

You MUST edit the CONFIG section below before running -- at minimum
SAGEMAKER_EXECUTION_ROLE_ARN (see the note above it for how to get one).

Run:
    python3 deploy_to_sagemaker.py
"""

import os
import time

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.model_monitor import DataCaptureConfig

# ─────────────────────────── CONFIG ───────────────────────────
REGION = "ap-south-1"

# SageMaker needs an execution ROLE (not your IAM user) to run the endpoint.
# If you don't have one yet:
#   AWS Console -> IAM -> Roles -> Create role -> Trusted entity: SageMaker
#   -> attach policy: AmazonSageMakerFullAccess -> name it e.g. "citadel-sagemaker-exec-role"
#   -> copy its ARN here.
SAGEMAKER_EXECUTION_ROLE_ARN = "arn:aws:iam::447788060954:role/CitadelSageMakerExecutionRole"

# Any S3 bucket you own in this region. Script will create it if it
# doesn't exist. Bucket names must be globally unique.
S3_BUCKET = "citadel-ai-demo-aniketh-447788060954-v1"

MODEL_TAR_LOCAL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "artifacts", "model.tar.gz"
)

ENDPOINT_NAME = "citadel-biased-hiring-endpoint"
INSTANCE_TYPE = "ml.t2.medium"  # cheapest instance the sklearn container reliably supports
INITIAL_INSTANCE_COUNT = 1

# sklearn version supported by the SageMaker prebuilt SKLearn container.
# Check current supported versions if this fails:
# https://docs.aws.amazon.com/sagemaker/latest/dg/sklearn.html
SKLEARN_FRAMEWORK_VERSION = "1.2-1"
PY_VERSION = "py3"
# ────────────────────────────────────────────────────────────


def ensure_bucket(s3_client, bucket, region):
    existing = [b["Name"] for b in s3_client.list_buckets()["Buckets"]]
    if bucket in existing:
        print(f"Using existing bucket: {bucket}")
        return
    print(f"Creating bucket: {bucket}")
    if region == "us-east-1":
        s3_client.create_bucket(Bucket=bucket)
    else:
        s3_client.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": region},
        )


def main():
    if "<YOUR_ACCOUNT_ID>" in SAGEMAKER_EXECUTION_ROLE_ARN or "<YOUR_UNIQUE_SUFFIX>" in S3_BUCKET:
        raise SystemExit(
            "Edit the CONFIG section at the top of this script first: set "
            "SAGEMAKER_EXECUTION_ROLE_ARN and S3_BUCKET to real values."
        )

    if not os.path.exists(MODEL_TAR_LOCAL_PATH):
        raise SystemExit(
            f"{MODEL_TAR_LOCAL_PATH} not found -- run generate_and_train.py first."
        )

    boto_session = boto3.Session(region_name=REGION)
    sess = sagemaker.Session(boto_session=boto_session)
    s3_client = boto_session.client("s3")

    ensure_bucket(s3_client, S3_BUCKET, REGION)

    model_key = "citadel-demo/model/model.tar.gz"
    print(f"Uploading {MODEL_TAR_LOCAL_PATH} -> s3://{S3_BUCKET}/{model_key}")
    s3_client.upload_file(MODEL_TAR_LOCAL_PATH, S3_BUCKET, model_key)
    model_s3_uri = f"s3://{S3_BUCKET}/{model_key}"

    data_capture_s3_uri = f"s3://{S3_BUCKET}/citadel-demo/data-capture"
    print(f"Data Capture will write to: {data_capture_s3_uri}")

    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,  # capture every request for the demo
        destination_s3_uri=data_capture_s3_uri,
    )

    print("Creating SKLearnModel...")
    sklearn_model = SKLearnModel(
        model_data=model_s3_uri,
        role=SAGEMAKER_EXECUTION_ROLE_ARN,
        entry_point="inference.py",  # already baked into model.tar.gz under code/
        source_dir="artifacts/code",
        framework_version=SKLEARN_FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        sagemaker_session=sess,
    )

    print(
        f"Deploying to endpoint '{ENDPOINT_NAME}' "
        f"({INSTANCE_TYPE} x{INITIAL_INSTANCE_COUNT}) -- this takes 5-10 minutes..."
    )
    start = time.time()
    predictor = sklearn_model.deploy(
        initial_instance_count=INITIAL_INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
        data_capture_config=data_capture_config,
    )
    elapsed = time.time() - start
    print(f"Endpoint deployed in {elapsed:.0f}s: {predictor.endpoint_name}")

    # Quick smoke test
    print("Running a smoke-test invocation...")
    test_payload = {
        "instances": [
            {
                "years_experience": 5.0,
                "education_level": 2,
                "test_score": 75.0,
                "num_previous_companies": 2,
                "age": 30,
                "gender": 1,
            }
        ]
    }
    runtime = boto_session.client("sagemaker-runtime")
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=str(test_payload).replace("'", '"'),
    )
    print("Smoke-test response:", response["Body"].read().decode("utf-8"))

    print("\nDone. Endpoint name for send_traffic.py / Citadel connect mode:")
    print(f"  {ENDPOINT_NAME}")
    print(f"Data Capture logs will land under: {data_capture_s3_uri}")
    print(
        "\nReminder: this endpoint bills per hour while running. "
        f"Delete it when you're done with: aws sagemaker delete-endpoint "
        f"--endpoint-name {ENDPOINT_NAME} --region {REGION}"
    )


if __name__ == "__main__":
    main()