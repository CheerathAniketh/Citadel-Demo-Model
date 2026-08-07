"""
inference.py -- SageMaker SKLearn serving script.

Expects JSON input shaped like:
{"instances": [{"years_experience": 4.5, "education_level": 1, "test_score": 68.0,
                 "num_previous_companies": 2, "age": 29, "gender": 1}, ...]}

Returns:
{"predictions": [{"hired": 0, "probability": 0.23}, ...]}
"""
import json
import os
import joblib
import numpy as np

FEATURE_ORDER = [
    "years_experience",
    "education_level",
    "test_score",
    "num_previous_companies",
    "age",
    "gender",
]


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, request_content_type):
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")
    payload = json.loads(request_body)
    instances = payload["instances"]
    rows = [[inst[f] for f in FEATURE_ORDER] for inst in instances]
    return np.array(rows, dtype=float)


def predict_fn(input_data, model):
    preds = model.predict(input_data)
    probs = model.predict_proba(input_data)[:, 1]
    return list(zip(preds.tolist(), probs.tolist()))


def output_fn(prediction, response_content_type):
    results = [{"hired": int(p), "probability": float(prob)} for p, prob in prediction]
    return json.dumps({"predictions": results}), "application/json"
