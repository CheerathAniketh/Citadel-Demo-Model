# Citadel Demo Model

An intentionally biased synthetic hiring classifier (scikit-learn) deployed to an AWS SageMaker real-time endpoint with Data Capture enabled. It is the live-traffic source for [Citadel-AI](https://github.com/CheerathAniketh/Citadel-AI), a bias-monitoring platform, so Citadel can be tested against real endpoint traffic and not only CSVs.

The training data is synthetic and the bias is injected on purpose. This model is not meant for real hiring decisions.

## What's here

| File | Purpose |
|------|---------|
| `generate_data.py` | Generates the synthetic hiring dataset |
| `train.py` | Trains the model and packages `artifacts/model.tar.gz` |
| `artifacts/code/inference.py` | SageMaker inference handler |
| `deploy.py` | Deploys the endpoint (set region, bucket and IAM role in this file) |
| `traffic.py` | Sends test requests and prints a live bias check |

## Run it

```bash
pip install -r requirements1.txt
python train.py
python deploy.py
python traffic.py
```

The endpoint bills while it is up. Delete it when you are done:

```bash
aws sagemaker delete-endpoint --endpoint-name citadel-biased-hiring-endpoint --region ap-south-1
aws sagemaker delete-endpoint-config --endpoint-config-name citadel-biased-hiring-endpoint --region ap-south-1
```

Captured traffic stays in S3 after the endpoint is deleted.

## Results (one run)

- Training-time statistical parity difference: 0.348.
- Live endpoint traffic, 231 requests: male hire rate 0.369 (n=111), female hire rate 0.000 (n=120), disparate impact 0.000 (fairness threshold 0.80).
- Citadel read the captured logs from S3 and reproduced the same metrics on 235 predictions.

## Notes

- The SageMaker sklearn container (framework `1.2-1`) runs Python 3.9 with NumPy 1.x. A model pickled under NumPy 2.x fails to load there with `No module named 'numpy._core.multiarray'`. Training is pinned to numpy 1.26.4, scipy 1.13.1, scikit-learn 1.4.2 and joblib 1.4.2.
- `Cannot create already existing endpoint configuration` means a failed earlier deploy left a config behind. Delete it with the command above and retry.
- `did not pass the ping health check` means the real error is in the endpoint's CloudWatch log group, not in the local traceback.
- The endpoint runs on `ml.t2.medium` because the account's `ml.m5.large` endpoint quota was 0.
