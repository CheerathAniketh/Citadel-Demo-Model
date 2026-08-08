# Citadel — SageMaker Deploy Notes

Project: deploying `citadel-biased-hiring-endpoint` (sklearn model, real-time
SageMaker endpoint, Data Capture enabled) from `~/Projects/Model`.

---

## ⏭️ NEXT UP (do this first)

- [ ] **Connect Citadel to the S3 Data Capture logs.** This is the actual
      remaining goal — replace Citadel's mocked `monitor_predictions()` /
      `get_predictions()` data with the real captured logs sitting in S3.
      To do this properly we need to look at:
      1. Wherever `monitor_predictions()` (or its mock) is defined in the
         Citadel codebase — to see what data shape/schema it expects.
      2. Whatever config Citadel uses to point at a data source (env var,
         config file, connection string, etc.) — so we know where to point
         it at the bucket/prefix below.
      3. One real captured `.jsonl` file's actual contents, to confirm it
         matches what Citadel expects to parse (SageMaker Data Capture wraps
         request/response pairs in its own JSON envelope format — may need
         a small parser adapter on the Citadel side).
      - Data is waiting at:
        `s3://citadel-ai-demo-aniketh-447788060954-v1/citadel-demo/data-capture/citadel-biased-hiring-endpoint/AllTraffic/`
      - Sample a file directly from S3 to inspect the format:
        ```bash
        aws s3 cp s3://citadel-ai-demo-aniketh-447788060954-v1/citadel-demo/data-capture/citadel-biased-hiring-endpoint/AllTraffic/2026/08/08/02/35-17-940-757a2824-6167-4cef-919a-5898a4f5122d.jsonl -
        ```
- [ ] If more live traffic / a fresh capture is ever needed: the endpoint is
      currently **shut down** (see below), so redeploy first with
      `python deploy.py`, then rerun `python traffic.py`. Remember to check
      for leftover endpoint configs before redeploying (see "Useful commands").

---

## ✅ DONE (in order — full session so far)

1. **`FileNotFoundError: inference.py`** during `deploy.py`
   - Cause: `SKLearnModel(entry_point="inference.py")` with no `source_dir` looks
     for the file locally, not inside `model.tar.gz`.
   - Fix: added `source_dir="artifacts/code"` to the `SKLearnModel(...)` call in
     `deploy.py` (since `inference.py` actually lives at `artifacts/code/inference.py`).

2. **`ClientError: Could not access model data at s3://...`** (CreateModel)
   - Cause: IAM role `CitadelSageMakerExecutionRole`'s inline policy
     `CitadelS3BucketAccess` was scoped to the *wrong* bucket
     (`citadel-demo-447788060954-ap-south-1` instead of the bucket actually used,
     `citadel-ai-demo-aniketh-447788060954-v1`).
   - Fix: `aws iam put-role-policy` to rewrite `CitadelS3BucketAccess` with the
     correct bucket ARN + `/*`.

3. **`ResourceLimitExceeded: ml.m5.large for endpoint usage is 0 Instances`**
   - Cause: fresh AWS account, quota 0 for `ml.m5.large` endpoints.
   - Fix: checked `list-service-quotas`, found `ml.t2.medium for endpoint usage`
     already had quota `2.0`. Changed `INSTANCE_TYPE = "ml.t2.medium"` in
     `deploy.py` — no quota request needed.

4. **`ValidationException: Cannot create already existing endpoint configuration`**
   - Cause: a previous failed deploy attempt left behind a stale endpoint config
     with the same name.
   - Fix: `aws sagemaker delete-endpoint-config --endpoint-config-name citadel-biased-hiring-endpoint`
     before retrying.

5. **`UnexpectedStatusException: ... did not pass the ping health check`**
   - Root cause (found via CloudWatch logs):
     `ModuleNotFoundError: No module named 'numpy._core.multiarray'` inside
     `inference.py`'s `model_fn()` when unpickling `model.joblib`.
   - Why: model was trained locally with **NumPy 2.5.1**, but the SageMaker
     sklearn container (framework version `1.2-1`, Python 3.9) ships an older
     NumPy that doesn't have the `numpy._core` internal module path
     (renamed from `numpy.core` in NumPy 2.0). Classic train/serve env mismatch.
   - Fix path:
     - Tried downgrading to `numpy==1.24.4` in `venv_train` → failed to build
       (`venv_train` is Python 3.12, and 1.24.4 has no 3.12 wheel, needs
       `distutils` which was removed in 3.12).
     - Installed `numpy==1.26.4` instead (last 1.x line with Python 3.12 wheels).
     - This broke `scipy` (1.18.0, needed numpy>=2.0) and would've broken
       `scikit-learn` (1.9.0) too.
     - Final compatible pin set installed:
       `numpy==1.26.4`, `scipy==1.13.1`, `scikit-learn==1.4.2`, `joblib==1.4.2`
     - `pip check` came back clean.
   - Reran `python train.py` → succeeded, regenerated `model.joblib` and
     `artifacts/model.tar.gz` (now also includes `code/requirements.txt`,
     which presumably pins these versions for the container to install).
   - Ran `pip freeze > requirements1.txt` to lock in the working `venv_train`
     environment so this doesn't silently drift back to NumPy 2.x later.
   - Confirmed no leftover endpoint/config from the failed attempt (both
     `delete-endpoint` / `delete-endpoint-config` returned "not found" — already
     clean).

6. **Second morning attempt still hit `Cannot create already existing endpoint
   configuration`**, even after last night's cleanup.
   - Cause: same collision as issue #4 — an endpoint config with the same name
     had gotten left behind again (deploy attempts partially succeed at
     creating the config before failing later, so this recurs after *any*
     interrupted/failed run, not just the numpy one).
   - Fix: deleted the config again, confirmed `list-endpoints` was empty and
     `delete-endpoint` returned "not found" (clean slate), then retried.

7. **🎉 Deploy succeeded.**
   ```
   Endpoint deployed in 184s: citadel-biased-hiring-endpoint
   Smoke-test response: {"predictions": [{"hired": 0, "probability": 0.103}]}
   ```
   Endpoint went `InService` on `ml.t2.medium`, Data Capture destination
   confirmed active. The numpy/scipy/sklearn version fix from issue #5 held up.

8. **Ran `python traffic.py`** (231 requests, 10 batches, ~1-2 min total).
   - All 231 requests succeeded, logged locally to `traffic_log.jsonl`.
   - **Live bias check on real endpoint traffic:**
     - Male hire rate: 0.369 (n=111)
     - Female hire rate: **0.000** (n=120)
     - Observed DI: **0.000** — far below the 0.80 fairness threshold.
   - This matches/reinforces the training-time bias report (`SPD=0.348`) —
     the deployed model reproduces the same severe bias pattern on fresh,
     unseen traffic (different random seed than training).

9. **Confirmed Data Capture logs landed in S3:**
   ```bash
   aws s3 ls s3://citadel-ai-demo-aniketh-447788060954-v1/citadel-demo/data-capture/ --recursive
   ```
   4 `.jsonl` files present — two small smoke-test captures from the first
   deploy, plus two larger batches (~68KB, ~64KB) from the `traffic.py` run.
   This is real captured data, ready for Citadel to read.

10. **Shut the endpoint down** to stop billing (goal — real S3 Data Capture
    logs for Citadel — was achieved; no need to keep it running):
    ```bash
    aws sagemaker delete-endpoint --endpoint-name citadel-biased-hiring-endpoint --region ap-south-1
    aws sagemaker delete-endpoint-config --endpoint-config-name citadel-biased-hiring-endpoint --region ap-south-1
    ```
    Confirmed via `list-endpoints` that nothing is left running.
    **Important:** the S3 Data Capture logs persist even with the endpoint
    deleted — deleting only stops compute billing, doesn't touch the bucket.

---

## 🗂️ Key facts / config (for future me)

- **Region:** `ap-south-1`
- **Bucket:** `citadel-ai-demo-aniketh-447788060954-v1`
- **Endpoint name:** `citadel-biased-hiring-endpoint`
- **Instance type:** `ml.t2.medium` (quota 2.0; `ml.m5.large` quota is 0 — would
  need a Service Quotas increase request to use it)
- **IAM role:** `arn:aws:iam::447788060954:role/CitadelSageMakerExecutionRole`
  - Inline policy `CitadelS3BucketAccess` → now correctly scoped to the bucket above.
- **Framework:** SKLearn container, `framework_version="1.2-1"`, `py_version="py3"`
  (this container = Python 3.9 internally — NOT the same Python as your local venvs)
- **`entry_point="inference.py"`**, **`source_dir="artifacts/code"`** in `deploy.py`
- **Training venv:** `venv_train` (Python 3.12) — pins now frozen in
  `requirements1.txt` (numpy 1.26.4, scipy 1.13.1, scikit-learn 1.4.2, joblib 1.4.2, ...)
- **Deploy venv:** `venv-dep` — used to run `deploy.py` / AWS CLI
- **Data Capture S3 path:**
  `s3://citadel-ai-demo-aniketh-447788060954-v1/citadel-demo/data-capture/citadel-biased-hiring-endpoint/AllTraffic/`
  (partitioned by date/hour under this prefix, `.jsonl` files)
- **Current endpoint status: SHUT DOWN** (deleted after confirming Data
  Capture worked, to stop billing). Redeploy with `python deploy.py` if
  needed again — model artifact + `requirements.txt` fix are already baked
  into `artifacts/model.tar.gz`, so it should deploy clean.
- **Observed live bias (from `traffic.py` run, 231 requests):** male hire
  rate 0.369 (n=111), female hire rate 0.000 (n=120), DI = 0.000.

---

## 🔧 Useful commands

**Check what's running / billing:**
```bash
aws sagemaker list-endpoints --region ap-south-1 --query 'Endpoints[*].{Name:EndpointName,Status:EndpointStatus}' --output table
```

**Delete endpoint (stop billing) + its config:**
```bash
aws sagemaker delete-endpoint --endpoint-name citadel-biased-hiring-endpoint --region ap-south-1
aws sagemaker delete-endpoint-config --endpoint-config-name citadel-biased-hiring-endpoint --region ap-south-1
```

**Find + read latest CloudWatch logs for the endpoint:**
```bash
aws logs describe-log-streams \
  --log-group-name /aws/sagemaker/Endpoints/citadel-biased-hiring-endpoint \
  --region ap-south-1 --order-by LastEventTime --descending --max-items 5 \
  --query "logStreams[*].{Name:logStreamName,LastEvent:lastEventTimestamp}" --output table

aws logs get-log-events \
  --log-group-name /aws/sagemaker/Endpoints/citadel-biased-hiring-endpoint \
  --log-stream-name "<paste-stream-name>" \
  --region ap-south-1 --limit 100
```

**Check IAM role trust + policies (if permission errors come back):**
```bash
aws iam get-role --role-name CitadelSageMakerExecutionRole --query 'Role.AssumeRolePolicyDocument'
aws iam list-attached-role-policies --role-name CitadelSageMakerExecutionRole
aws iam list-role-policies --role-name CitadelSageMakerExecutionRole
aws iam get-role-policy --role-name CitadelSageMakerExecutionRole --policy-name CitadelS3BucketAccess
```

**Check SageMaker instance quotas:**
```bash
aws service-quotas list-service-quotas \
  --service-code sagemaker --region ap-south-1 \
  --query "Quotas[?contains(QuotaName, 'endpoint usage')].{Name:QuotaName,Code:QuotaCode,Value:Value}" \
  --output table
```

---

## 💡 Lessons learned (avoid repeating these)

- Keep training env NumPy/scipy/sklearn versions **compatible with the
  SageMaker container's Python/NumPy version**, not just "whatever's latest
  locally." The sklearn `1.2-1` container = Python 3.9, pre-NumPy-2.0.
- `requirements1.txt` should be regenerated (`pip freeze`) any time `venv_train`
  deps change, so the training env is reproducible.
- After *any* failed `deploy.py` run, check for and clean up leftover endpoint
  configs before retrying — SageMaker won't silently overwrite same-named ones.
- CloudWatch logs (`/aws/sagemaker/Endpoints/<name>`) are the actual source of
  truth for container-side failures — the local traceback from `deploy.py` often
  just says "health check failed" with no detail.
- Endpoint config name collisions (`Cannot create already existing endpoint
  configuration`) recur after *any* interrupted/failed deploy attempt, not
  just specific failures — get in the habit of running the cleanup check
  before every `deploy.py` run, not only after a known failure.

---

## 🔌 Citadel connection — what's needed

Not done yet. To wire Citadel's `monitor_predictions()` / Connect mode to
read this real S3 data instead of mocks, need to look at:
1. The `monitor_predictions()` / `get_predictions()` code in Citadel — what
   shape/schema does it expect?
2. Citadel's config mechanism for pointing at a data source.
3. An actual sample of the captured `.jsonl` format from S3 (SageMaker Data
   Capture uses its own JSON envelope around request/response pairs — likely
   needs a small parsing adapter).

Bring the relevant Citadel source file(s) next session to figure this out.