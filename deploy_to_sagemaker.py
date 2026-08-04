"""
Citadel AI — Demo model deployment
Deploys the model trained by generate_and_train.py to a real SageMaker
endpoint, with Data Capture enabled so real inference traffic gets logged
to S3 (which Citadel's monitor_predictions step will eventually read).

Prerequisites:
    - generate_and_train.py already run (model.tar.gz exists in this folder)
    - AWS CLI configured (aws configure) with a user that has
      AdministratorAccess or equivalent SageMaker/S3/IAM permissions
    - pip install boto3 sagemaker

This script:
    1. Creates an S3 bucket for the model artifact + data capture logs
    2. Uploads model.tar.gz to S3
    3. Creates a SageMaker execution role (if one doesn't already exist)
    4. Creates a SageMaker Model, EndpointConfig (with DataCaptureConfig),
       and Endpoint
    5. Waits for the endpoint to be InService

IMPORTANT: this creates a real, billed AWS resource. Run delete_endpoint.py
(or the teardown commands printed at the end) when you're done testing —
don't leave it running idle.
"""

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import time
import json

REGION = "ap-south-1"
INSTANCE_TYPE = "ml.t2.medium"
ENDPOINT_NAME = "citadel-biased-hiring-endpoint"
BUCKET_PREFIX = "citadel-demo"

boto_session = boto3.Session(region_name=REGION)
sm_client = boto_session.client("sagemaker")
iam_client = boto_session.client("iam")
s3_client = boto_session.client("s3")
sts_client = boto_session.client("sts")

account_id = sts_client.get_caller_identity()["Account"]
bucket_name = f"{BUCKET_PREFIX}-{account_id}-{REGION}"

print(f"Account: {account_id}")
print(f"Region: {REGION}")
print(f"Bucket: {bucket_name}")

# ---------------------------------------------------------------------------
# 1. Create S3 bucket (idempotent — skip if it already exists)
# ---------------------------------------------------------------------------
try:
    if REGION == "us-east-1":
        s3_client.create_bucket(Bucket=bucket_name)
    else:
        s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": REGION}
        )
    print(f"Created bucket {bucket_name}")
except s3_client.exceptions.BucketAlreadyOwnedByYou:
    print(f"Bucket {bucket_name} already exists, reusing")

# ---------------------------------------------------------------------------
# 2. Upload model artifact
# ---------------------------------------------------------------------------
model_s3_key = "model/model.tar.gz"
s3_client.upload_file("model.tar.gz", bucket_name, model_s3_key)
model_data_url = f"s3://{bucket_name}/{model_s3_key}"
print(f"Uploaded model to {model_data_url}")

# ---------------------------------------------------------------------------
# 3. Create (or reuse) a SageMaker execution role
# ---------------------------------------------------------------------------
role_name = "CitadelSageMakerExecutionRole"
trust_policy = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "sagemaker.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}

try:
    role_response = iam_client.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="Execution role for Citadel demo SageMaker endpoint"
    )
    role_arn = role_response["Role"]["Arn"]
    iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
    )
    print(f"Created role {role_arn}, waiting for IAM propagation...")
    time.sleep(15)  # IAM roles take a few seconds to propagate
except iam_client.exceptions.EntityAlreadyExistsException:
    role_response = iam_client.get_role(RoleName=role_name)
    role_arn = role_response["Role"]["Arn"]
    print(f"Reusing existing role {role_arn}")

# ---------------------------------------------------------------------------
# 4. Deploy: Model -> EndpointConfig (with Data Capture) -> Endpoint
# ---------------------------------------------------------------------------
sagemaker_session = sagemaker.Session(boto_session=boto_session)

sklearn_model = SKLearnModel(
    model_data=model_data_url,
    role=role_arn,
    entry_point="inference.py",  # see companion file below
    framework_version="1.2-1",
    sagemaker_session=sagemaker_session,
)

data_capture_s3_uri = f"s3://{bucket_name}/data-capture"

print(f"Deploying endpoint '{ENDPOINT_NAME}'... this takes ~5-8 minutes.")

predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type=INSTANCE_TYPE,
    endpoint_name=ENDPOINT_NAME,
    data_capture_config=sagemaker.model_monitor.DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        destination_s3_uri=data_capture_s3_uri,
    ),
)

print("\n" + "=" * 60)
print(f"✅ Endpoint '{ENDPOINT_NAME}' is InService")
print(f"Data Capture writing to: {data_capture_s3_uri}")
print("=" * 60)
print("\nNext: run send_traffic.py to generate real inference traffic.")
print(f"\nWhen done testing, tear down with:")
print(f"  aws sagemaker delete-endpoint --endpoint-name {ENDPOINT_NAME} --region {REGION}")
print(f"  aws sagemaker delete-endpoint-config --endpoint-config-name {ENDPOINT_NAME} --region {REGION}")
print(f"  aws sagemaker delete-model --model-name <check output above for exact name> --region {REGION}")