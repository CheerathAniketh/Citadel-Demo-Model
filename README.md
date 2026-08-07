# Citadel — SageMaker Deploy Notes

Project: deploying `citadel-biased-hiring-endpoint` (sklearn model, real-time
SageMaker endpoint, Data Capture enabled) from `~/Projects/Model`.

---

## ⏭️ NEXT UP (do this first)

- [ ] Run `python deploy.py` (from `venv-dep`) and see if it deploys cleanly now
      that the model was retrained with NumPy 1.26.4.
- [ ] If it succeeds: note the smoke-test response, then remember the endpoint
      **bills per hour while running** — delete it when done testing:
      ```bash
      aws sagemaker delete-endpoint --endpoint-name citadel-biased-hiring-endpoint --region ap-south-1
      aws sagemaker delete-endpoint-config --endpoint-config-name citadel-biased-hiring-endpoint --region ap-south-1
      ```
- [ ] If it fails: check CloudWatch logs first, before re-running blind —
      ```bash
      aws logs describe-log-streams \
        --log-group-name /aws/sagemaker/Endpoints/citadel-biased-hiring-endpoint \
        --region ap-south-1 --order-by LastEventTime --descending --max-items 5 \
        --query "logStreams[*].{Name:logStreamName,LastEvent:lastEventTimestamp}" --output table
      ```
      then `get-log-events` on the most recent stream (see "Useful commands" below).

---

## ✅ DONE (in order — today's session)

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