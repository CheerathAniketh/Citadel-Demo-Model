"""
SageMaker inference entry point.
Must live in the same folder as deploy_to_sagemaker.py — the SKLearnModel
deploy() call packages this alongside the model automatically.

Implements the four functions SageMaker's sklearn container looks for:
    model_fn    - load the model artifact
    input_fn    - parse the incoming request body
    predict_fn  - run inference
    output_fn   - format the response
"""

import joblib
import json
import os
import numpy as np
import pandas as pd


def model_fn(model_dir):
    """Load the model artifact saved by generate_and_train.py"""
    artifact = joblib.load(os.path.join(model_dir, "model.joblib"))
    return artifact


def input_fn(request_body, request_content_type):
    """
    Expects JSON like:
    {
        "age": 30,
        "experience_years": 5,
        "education": "bachelors",
        "salary_expectation": 60000,
        "gender": "F"
    }
    Or a batch: {"instances": [ {...}, {...}, ... ]}
    """
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    payload = json.loads(request_body)
    if "instances" in payload:
        return payload["instances"]
    return [payload]


def predict_fn(input_data, artifact):
    """Run the model on one or more applicants, return predictions + probabilities."""
    model = artifact["model"]
    edu_encoder = artifact["education_encoder"]
    gender_encoder = artifact["gender_encoder"]
    feature_cols = artifact["feature_cols"]

    rows = []
    for item in input_data:
        rows.append({
            "age": item["age"],
            "experience_years": item["experience_years"],
            "education_enc": edu_encoder.transform([item["education"]])[0],
            "salary_expectation": item["salary_expectation"],
            "gender_enc": gender_encoder.transform([item["gender"]])[0],
        })

    df = pd.DataFrame(rows)[feature_cols]
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    results = []
    for i, item in enumerate(input_data):
        results.append({
            "prediction": "approved" if predictions[i] == 1 else "rejected",
            "confidence": float(probabilities[i]),
            "group": item["gender"],
        })
    return results


def output_fn(prediction, response_content_type):
    """Return JSON response."""
    return json.dumps(prediction), "application/json"