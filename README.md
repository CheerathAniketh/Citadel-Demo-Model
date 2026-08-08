# Citadel — SageMaker Deploy Notes

Project: deploying `citadel-biased-hiring-endpoint` (sklearn model, real-time SageMaker endpoint, Data Capture enabled) — this is the demo ground-truth model for the [Citadel-AI](../Citadel-AI) governance platform: an intentionally biased hiring model, used to prove Citadel's bias-detection pipeline works against real live traffic, not just synthetic CSVs.

---

## 👋 New here? Start with this section (Akanksha)

This repo is just the model deployment side — a small, self-contained sklearn model pushed to AWS SageMaker. Your task here is **mechanical, not coding**: clone this repo and deploy the model yourself, so you have real hands-on AWS experience and Citadel has a fresh live endpoint to demo against.

**What to do:**

```bash
python deploy.py
```

That's genuinely most of it. A few things to expect, all already solved once before (see the full debug log below if you want details):

- **You'll probably hit dependency issues** — numpy/scipy/sklearn version mismatches between your local training environment and the SageMaker container. If `deploy.py` fails with something like `ModuleNotFoundError: No module named 'numpy._core.multiarray'`, don't panic — this happened before and is fully documented under "🐛 Issue #5" below. The fix is to match the exact pinned versions in `requirements1.txt` (numpy 1.26.4, scipy 1.13.1, scikit-learn 1.4.2, joblib 1.4.2), not whatever's newest.
- **If you get `ValidationException: Cannot create already existing endpoint configuration`** — that means a leftover config from a previous failed attempt. Delete it first, then retry:
  ```bash
  aws sagemaker delete-endpoint-config --endpoint-config-name citadel-biased-hiring-endpoint --region ap-south-1
  ```
- **If you get a `did not pass the ping health check` error** — check CloudWatch logs (command further down), not just the local traceback, since the real error is usually only visible there.
- **Ping Aniketh if you're stuck for more than 15 minutes** on any one error. Chances are it's already in the debug log below.

**Once deployed successfully**, you'll see something like:
```
Endpoint deployed in 184s: citadel-biased-hiring-endpoint
Smoke-test response: {"predictions": [{"hired": 0, "probability": 0.103}]}
```
That's it — the model's live, Citadel can connect to it, and you've done a real cloud ML deployment.

**Send some traffic (optional but useful)** — this generates real request/response logs in S3 that Citadel reads to compute bias metrics:
```bash
python traffic.py
```

**When you're done testing, shut it down to stop billing:**
```bash
aws sagemaker delete-endpoint --endpoint-name citadel-biased-hiring-endpoint --region ap-south-1
aws sagemaker delete-endpoint-config --endpoint-config-name citadel-biased-hiring-endpoint --region ap-south-1
```
Deleting the endpoint doesn't delete the S3 data it already captured — that's separate and stays intact.

**You don't need to touch any code in this repo** — `deploy.py`, `train.py`, `inference.py`, etc. are all already working and tested. If something genuinely seems broken (not just a known dependency hiccup), message Aniketh before changing anything, since this exact deployment has already been proven to work end-to-end with Citadel.

---

## ✅ Status: Citadel connection — done

The original goal of this repo (get real S3 Data Capture data flowing into Citadel's bias-detection pipeline) is **complete and verified**. Citadel's `get_predictions()` now reads real `.jsonl` capture logs straight from this model's S3 bucket, parses them, and computes real DI/SPD bias metrics — tested end-to-end against 235+ real predictions, with results matching this repo's own `traffic.py` findings (DI=0.000, matching known bias in the model). See the [Citadel-AI README](../Citadel-AI) for the full integration details.

This repo's job now is simple: **be a redeployable source of real live traffic for demos.** Deploy it, generate traffic, let Citadel connect to it, shut it down when done.

---

## ✅ DONE (in order — full session so far)

1. **`FileNotFoundError: inference.py` during `deploy.py`**
   - Cause: `SKLearnModel(entry_point="inference.py")` with no `source_dir` looks for the file locally, not inside `model.tar.gz`.
   - Fix: added `source_dir="artifacts/code"` to the `SKLearnModel(...)` call in `deploy.py` (since `inference.py` actually lives at `artifacts/code/inference.py`).

2. **`ClientError: Could not access model data at s3://...` (CreateModel)**
   - Cause: IAM role `CitadelSageMakerExecutionRole`'s inline policy `CitadelS3BucketAccess` was scoped to the wrong bucket (`citadel-demo-447788060954-ap-south-1` instead of the bucket actually used, `citadel-ai-demo-aniketh-447788060954-v1`).
   - Fix: `aws iam put-role-policy` to rewrite `CitadelS3BucketAccess` with the correct bucket ARN + `/*`.

3. **`ResourceLimitExceeded: ml.m5.large for endpoint usage is 0 Instances`**
   - Cause: fresh AWS account, quota 0 for `ml.m5.large` endpoints.
   - Fix: checked `list-service-quotas`, found `ml.t2.medium` for endpoint usage already had quota 2.0. Changed `INSTANCE_TYPE = "ml.t2.medium"` in `deploy.py` — no quota request needed.

4. **`ValidationException: Cannot create already existing endpoint configuration`**
   - Cause: a previous failed deploy attempt left behind a stale endpoint config with the same name.
   - Fix: `aws sagemaker delete-endpoint-config --endpoint-config-name citadel-biased-hiring-endpoint` before retrying.

5. **`UnexpectedStatusException: ... did not pass the ping health check`**
   - Root cause (found via CloudWatch logs): `ModuleNotFoundError: No module named 'numpy._core.multiarray'` inside `inference.py`'s `model_fn()` when unpickling `model.joblib`.
   - Why: model was trained locally with NumPy 2.5.1, but the SageMaker sklearn container (framework version 1.2-1, Python 3.9) ships an older NumPy that doesn't have the `numpy._core` internal module path (renamed from `numpy.core` in NumPy 2.0). Classic train/serve env mismatch.
   - Fix path:
     - Tried downgrading to `numpy==1.24.4` in `venv_train` → failed to build (`venv_train` is Python 3.12, and 1.24.4 has no 3.12 wheel, needs `distutils` which was removed in 3.12).
     - Installed `numpy==1.26.4` instead (last 1.x line with Python 3.12 wheels).
     - This broke scipy (1.18.0, needed numpy>=2.0) and would've broken scikit-learn (1.9.0) too.
     - Final compatible pin set installed: `numpy==1.26.4`, `scipy==1.13.1`, `scikit-learn==1.4.2`, `joblib==1.4.2`
     - `pip check` came back clean.
     - Reran `python train.py` → succeeded, regenerated `model.joblib` and `artifacts/model.tar.gz` (now also includes `code/requirements.txt`, which pins these versions for the container to install).
     - Ran `pip freeze > requirements1.txt` to lock in the working `venv_train` environment so this doesn't silently drift back to NumPy 2.x later.

6. **Second attempt still hit `Cannot create already existing endpoint configuration`, even after cleanup**
   - Cause: same collision as issue #4 — an endpoint config with the same name had gotten left behind again (deploy attempts partially succeed at creating the config before failing later, so this recurs after any interrupted/failed run, not just the numpy one).
   - Fix: deleted the config again, confirmed `list-endpoints` was empty and `delete-endpoint` returned "not found" (clean slate), then retried.

7. **🎉 Deploy succeeded.**
   - Endpoint deployed in 184s: `citadel-biased-hiring-endpoint`
   - Smoke-test response: `{"predictions": [{"hired": 0, "probability": 0.103}]}`
   - Endpoint went `InService` on `ml.t2.medium`, Data Capture destination confirmed active. The numpy/scipy/sklearn version fix from issue #5 held up.

8. **Ran `python traffic.py`** (231 requests, 10 batches, ~1-2 min total).
   - All 231 requests succeeded, logged locally to `traffic_log.jsonl`.
   - Live bias check on real endpoint traffic:
     - Male hire rate: 0.369 (n=111)
     - Female hire rate: 0.000 (n=120)
     - Observed DI: 0.000 — far below the 0.80 fairness threshold.
   - This matches/reinforces the training-time bias report (SPD=0.348) — the deployed model reproduces the same severe bias pattern on fresh, unseen traffic (different random seed than training).

9. **Confirmed Data Capture logs landed in S3** — `.jsonl` files present, ready for Citadel to read.

10. **Shut the endpoint down to stop billing** once the goal was achieved. S3 Data Capture logs persist even with the endpoint deleted — deleting only stops compute billing, doesn't touch the bucket.

11. **✅ Citadel connection built and verified (Aug 8, in the Citadel-AI repo)** — `get_predictions()` now reads real S3 Data Capture logs from this bucket, parses SageMaker's double-JSON envelope format, and feeds real predictions into Citadel's bias-detection pipeline. Tested end-to-end: 235 real predictions, DI=0.000, matching the numbers above. See the Citadel-AI repo's README for the full integration writeup.

---

## 🗂️ Key facts / config

- **Region:** `ap-south-1`
- **Bucket:** `citadel-ai-demo-aniketh-447788060954-v1`
- **Endpoint name:** `citadel-biased-hiring-endpoint`
- **Instance type:** `ml.t2.medium` (quota 2.0; `ml.m5.large` quota is 0 — would need a Service Quotas increase request to use it)
- **IAM role:** `arn:aws:iam::447788060954:role/CitadelSageMakerExecutionRole`
  - Inline policy `CitadelS3BucketAccess` → correctly scoped to the bucket above.
- **Framework:** SKLearn container, `framework_version="1.2-1"`, `py_version="py3"` (this container = Python 3.9 internally — **not** the same Python as your local venvs)
- `entry_point="inference.py"`, `source_dir="artifacts/code"` in `deploy.py`
- **Training venv:** `venv_train` (Python 3.12) — pins frozen in `requirements1.txt` (numpy 1.26.4, scipy 1.13.1, scikit-learn 1.4.2, joblib 1.4.2, ...)
- **Deploy venv:** `venv-dep` — used to run `deploy.py` / AWS CLI
- **Data Capture S3 path:** `s3://citadel-ai-demo-aniketh-447788060954-v1/citadel-demo/data-capture/citadel-biased-hiring-endpoint/AllTraffic/` (partitioned by date/hour under this prefix, `.jsonl` files)
- **Current endpoint status:** typically shut down between sessions to avoid billing. Redeploy with `python deploy.py` — model artifact + requirements.txt fix are already baked into `artifacts/model.tar.gz`, so it should deploy clean.
- **Observed live bias** (from `traffic.py` run, 231 requests): male hire rate 0.369 (n=111), female hire rate 0.000 (n=120), DI = 0.000.

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

**List captured Data Capture files in S3:**
```bash
aws s3 ls s3://citadel-ai-demo-aniketh-447788060954-v1/citadel-demo/data-capture/citadel-biased-hiring-endpoint/AllTraffic/ --recursive
```

**Pull one real capture file to inspect its format:**
```bash
aws s3 cp s3://citadel-ai-demo-aniketh-447788060954-v1/citadel-demo/data-capture/citadel-biased-hiring-endpoint/AllTraffic/<path>.jsonl ./sample_capture.jsonl
head -n 2 sample_capture.jsonl
```

---

## 💡 Lessons learned (avoid repeating these)

- Keep training env NumPy/scipy/sklearn versions compatible with the SageMaker container's Python/NumPy version, not just "whatever's latest locally." The sklearn 1.2-1 container = Python 3.9, pre-NumPy-2.0.
- `requirements1.txt` should be regenerated (`pip freeze`) any time `venv_train` deps change, so the training env is reproducible.
- After any failed `deploy.py` run, check for and clean up leftover endpoint configs before retrying — SageMaker won't silently overwrite same-named ones.
- CloudWatch logs (`/aws/sagemaker/Endpoints/<name>`) are the actual source of truth for container-side failures — the local traceback from `deploy.py` often just says "health check failed" with no detail.
- Endpoint config name collisions (`Cannot create already existing endpoint configuration`) recur after any interrupted/failed deploy attempt, not just specific failures — get in the habit of running the cleanup check before every `deploy.py` run, not only after a known failure.
- **Region matters everywhere.** This endpoint lives in `ap-south-1`. Any AWS CLI command, and any request to Citadel's Connect-mode API pointing at this endpoint, must use the matching region — this bit us once already (silently returns "0 models found" if the region's wrong, easy to mistake for an auth bug).

---

## 📜 Contribution note

This repo is small and self-contained — deploy/train/traffic scripts, no ongoing feature development expected. If you (Akanksha) hit something genuinely broken (not one of the documented issues above), message Aniketh before changing `deploy.py`, `train.py`, or `inference.py` directly, since this exact setup is already proven working end-to-end with Citadel and unrelated changes risk breaking that.

---

## 📄 About

Demo ground-truth model for [Citadel-AI](../Citadel-AI) — an intentionally biased hiring model deployed to AWS SageMaker with Data Capture enabled, used to prove Citadel's real-time bias-detection pipeline against genuine live cloud traffic.