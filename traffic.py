"""
send_traffic.py

Sends a realistic stream of synthetic applications to the live SageMaker
endpoint -- 200-250 requests total, sent in batches over time (not all at
once), so Data Capture accumulates a set of S3 logs that look like real
monitored traffic for Citadel's Connect mode to later read.

Prereqs:
    Endpoint already deployed via deploy_to_sagemaker.py.
    pip install boto3

Run:
    python3 send_traffic.py
"""

import json
import random
import time

import boto3
import numpy as np

# ─────────────────────────── CONFIG ───────────────────────────
REGION = "ap-south-1"
ENDPOINT_NAME = "citadel-biased-hiring-endpoint"

TOTAL_REQUESTS_MIN = 200
TOTAL_REQUESTS_MAX = 250
NUM_BATCHES = 10          # requests get split across this many batches
SLEEP_BETWEEN_BATCHES_SEC = (3, 8)  # random pause range between batches
SLEEP_BETWEEN_REQUESTS_SEC = (0.05, 0.3)  # small jitter within a batch

RANDOM_SEED = 7  # different from the training seed (42) on purpose --
                  # this is meant to be a fresh, unseen traffic distribution
LOG_PATH = "traffic_log.jsonl"
# ────────────────────────────────────────────────────────────


def generate_applicant(rng):
    gender = int(rng.integers(0, 2))
    years_experience = float(np.clip(rng.normal(6, 4), 0, 25))
    education_level = int(rng.choice([0, 1, 2, 3], p=[0.15, 0.45, 0.30, 0.10]))
    test_score = float(np.clip(rng.normal(70, 15), 0, 100))
    num_previous_companies = int(np.clip(rng.poisson(2), 0, 10))
    age = int(np.clip(rng.normal(32, 8), 21, 65))
    return {
        "years_experience": round(years_experience, 1),
        "education_level": education_level,
        "test_score": round(test_score, 1),
        "num_previous_companies": num_previous_companies,
        "age": age,
        "gender": gender,
    }


def invoke(runtime, applicant):
    payload = {"instances": [applicant]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    body = json.loads(response["Body"].read().decode("utf-8"))
    return body["predictions"][0]


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    total = random.randint(TOTAL_REQUESTS_MIN, TOTAL_REQUESTS_MAX)

    # split `total` into NUM_BATCHES roughly-equal batches
    base = total // NUM_BATCHES
    remainder = total % NUM_BATCHES
    batch_sizes = [base + (1 if i < remainder else 0) for i in range(NUM_BATCHES)]

    print(f"Sending {total} requests to '{ENDPOINT_NAME}' across {NUM_BATCHES} batches: {batch_sizes}")

    boto_session = boto3.Session(region_name=REGION)
    runtime = boto_session.client("sagemaker-runtime")

    results = []
    sent = 0
    for batch_num, batch_size in enumerate(batch_sizes, start=1):
        print(f"\nBatch {batch_num}/{NUM_BATCHES} ({batch_size} requests)...")
        for _ in range(batch_size):
            applicant = generate_applicant(rng)
            try:
                pred = invoke(runtime, applicant)
            except Exception as e:
                print(f"  request failed: {e}")
                continue
            record = {**applicant, **pred}
            results.append(record)
            sent += 1
            time.sleep(random.uniform(*SLEEP_BETWEEN_REQUESTS_SEC))
        print(f"  batch {batch_num} done ({sent}/{total} total sent)")

        if batch_num < NUM_BATCHES:
            pause = random.uniform(*SLEEP_BETWEEN_BATCHES_SEC)
            print(f"  pausing {pause:.1f}s before next batch...")
            time.sleep(pause)

    with open(LOG_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSent {sent} requests total. Local log saved -> {LOG_PATH}")

    # quick sanity check: does the live traffic show the same bias pattern?
    male = [r for r in results if r["gender"] == 0]
    female = [r for r in results if r["gender"] == 1]
    if male and female:
        male_rate = sum(r["hired"] for r in male) / len(male)
        female_rate = sum(r["hired"] for r in female) / len(female)
        di = female_rate / male_rate if male_rate > 0 else None
        print(f"\nLive traffic hire rates -- male: {male_rate:.3f} (n={len(male)}), "
              f"female: {female_rate:.3f} (n={len(female)})")
        if di is not None:
            print(f"Observed DI on this traffic: {di:.3f} "
                  f"({'below' if di < 0.8 else 'above'} the 0.80 threshold)")

    print(
        "\nData Capture logs should now be landing in S3 under the "
        "destination you set in deploy_to_sagemaker.py (allow a minute or "
        "two for delivery). That's what Citadel's monitor_predictions() / "
        "get_predictions() will eventually parse instead of mock data."
    )


if __name__ == "__main__":
    main()