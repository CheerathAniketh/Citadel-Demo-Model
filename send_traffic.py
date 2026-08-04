"""
Citadel AI — Demo traffic generator
Sends 150 fresh synthetic applications to the live SageMaker endpoint as
real invoke_endpoint() calls. This is what actually populates Data Capture
logs in S3 — the training data (hiring_data.csv) is never sent here; these
are brand-new synthetic applicants the model has never seen, exactly like
real production traffic hitting an already-trained model.

Run this AFTER deploy_to_sagemaker.py has finished and the endpoint shows
'InService'.

Usage:
    pip install boto3 numpy
    python send_traffic.py
"""

import boto3
import json
import numpy as np
import time

REGION = "ap-south-1"
ENDPOINT_NAME = "citadel-biased-hiring-endpoint"
NUM_APPLICATIONS = 150

np.random.seed(123)  # different seed from training data — genuinely new people

runtime = boto3.client("sagemaker-runtime", region_name=REGION)

# ---------------------------------------------------------------------------
# Generate fresh synthetic applicants — same distribution/bias pattern as
# training data, but a distinct batch (different random seed).
# ---------------------------------------------------------------------------
genders = np.random.choice(['M', 'F'], size=NUM_APPLICATIONS)
ages = np.random.randint(22, 55, size=NUM_APPLICATIONS)
experience_years = np.random.randint(0, 20, size=NUM_APPLICATIONS)
education = np.random.choice(['bachelors', 'masters', 'phd'], size=NUM_APPLICATIONS, p=[0.6, 0.3, 0.1])
salary_expectation = np.random.randint(40000, 120000, size=NUM_APPLICATIONS)

applicants = []
for i in range(NUM_APPLICATIONS):
    applicants.append({
        "age": int(ages[i]),
        "experience_years": int(experience_years[i]),
        "education": education[i],
        "salary_expectation": int(salary_expectation[i]),
        "gender": genders[i],
    })

print(f"Generated {NUM_APPLICATIONS} fresh synthetic applicants")
print(f"Sending real invoke_endpoint() calls to '{ENDPOINT_NAME}'...\n")

# ---------------------------------------------------------------------------
# Send each one as a real inference call
# ---------------------------------------------------------------------------
results = []
errors = 0

for i, applicant in enumerate(applicants):
    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(applicant),
        )
        result = json.loads(response["Body"].read().decode())[0]
        results.append(result)

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{NUM_APPLICATIONS} sent...")

    except Exception as e:
        errors += 1
        print(f"  ❌ Request {i + 1} failed: {e}")

    time.sleep(0.05)  # small delay, avoid hammering the endpoint

print(f"\n✅ Done. {len(results)} succeeded, {errors} failed.")

# ---------------------------------------------------------------------------
# Quick sanity check — same shape as the training-data check
# ---------------------------------------------------------------------------
if results:
    male_results = [r for r in results if r["group"] == "M"]
    female_results = [r for r in results if r["group"] == "F"]

    male_approval = sum(1 for r in male_results if r["prediction"] == "approved") / max(len(male_results), 1)
    female_approval = sum(1 for r in female_results if r["prediction"] == "approved") / max(len(female_results), 1)

    print(f"\nLive inference results:")
    print(f"  Men approved:   {male_approval:.1%}  ({len(male_results)} applicants)")
    print(f"  Women approved: {female_approval:.1%}  ({len(female_results)} applicants)")
    print(f"\nData Capture should now have {len(results)} real logged predictions in S3.")
    print("Citadel's monitor_predictions step (once S3 parsing is built) will read these.")