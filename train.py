"""
train.py

Reads artifacts/synthetic_hiring_data.csv (produced by generate.py) and
trains a LogisticRegression on it, then packages the result for SageMaker.

Belongs in venv_train (requirements1.txt: numpy, pandas, scikit-learn, joblib).
This venv never needs to see boto3/sagemaker.

Feature schema, in this exact order:
  [years_experience, education_level, test_score, num_previous_companies, age, gender]
    gender: 0=male, 1=female — used as a training feature on purpose, so the
    bias is strong and unambiguous for the demo.

Outputs (into ./artifacts/):
    model.joblib      -> trained sklearn LogisticRegression
    code/inference.py -> SageMaker serving script
    code/requirements.txt -> deps SageMaker installs INSIDE the container
                              (separate from this venv's requirements1.txt)
    model.tar.gz       -> the two above, packaged for SageMaker
    bias_report.json   -> DI/SPD on the model's own predictions

Run:
    python3 train.py
"""

import json
import os
import tarfile

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RANDOM_SEED = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
CODE_DIR = os.path.join(OUT_DIR, "code")
CSV_PATH = os.path.join(OUT_DIR, "synthetic_hiring_data.csv")

FEATURE_ORDER = [
    "years_experience",
    "education_level",
    "test_score",
    "num_previous_companies",
    "age",
    "gender",
]


def disparate_impact(df, label_col, group_col="gender", favored_group=0):
    rates = df.groupby(group_col)[label_col].mean()
    unfavored_group = 1 - favored_group
    if favored_group not in rates or unfavored_group not in rates or rates[favored_group] == 0:
        return None
    return rates[unfavored_group] / rates[favored_group]


def statistical_parity_difference(df, label_col, group_col="gender"):
    rates = df.groupby(group_col)[label_col].mean()
    return float(rates.max() - rates.min())


def write_inference_script(code_dir):
    os.makedirs(code_dir, exist_ok=True)
    inference_code = '''\
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
'''
    with open(os.path.join(code_dir, "inference.py"), "w") as f:
        f.write(inference_code)

    # These are installed INSIDE the SageMaker container at deploy time —
    # deliberately separate from this venv's own requirements1.txt.
    with open(os.path.join(code_dir, "requirements.txt"), "w") as f:
        f.write("joblib\nnumpy\nscikit-learn\n")


def package_for_sagemaker(out_dir, code_dir, model_path):
    tar_path = os.path.join(out_dir, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_path, arcname="model.joblib")
        tar.add(code_dir, arcname="code")
    return tar_path


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"{CSV_PATH} not found -- run generate.py first.")

    df = pd.read_csv(CSV_PATH)
    X = df[FEATURE_ORDER]
    y = df["hired"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    print("Training LogisticRegression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Held-out test accuracy: {acc:.3f}")

    model_path = os.path.join(OUT_DIR, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved model -> {model_path}")

    write_inference_script(CODE_DIR)
    tar_path = package_for_sagemaker(OUT_DIR, CODE_DIR, model_path)
    print(f"Packaged SageMaker-ready artifact -> {tar_path}")

    test_df = X_test.copy()
    test_df["hired"] = model.predict(X_test)
    model_di = disparate_impact(test_df, "hired")
    model_spd = statistical_parity_difference(test_df, "hired")
    print(f"Model-predictions bias check: DI={model_di:.3f}  SPD={model_spd:.3f}")

    report = {
        "held_out_accuracy": round(float(acc), 4),
        "model_predictions_disparate_impact": None if model_di is None else round(float(model_di), 4),
        "model_predictions_spd": round(float(model_spd), 4),
        "feature_order": FEATURE_ORDER,
        "random_seed": RANDOM_SEED,
    }
    report_path = os.path.join(OUT_DIR, "bias_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved bias report -> {report_path}")
    print("\nDone. Next: switch to venv_deploy and run deploy_to_sagemaker.py.")


if __name__ == "__main__":
    main()