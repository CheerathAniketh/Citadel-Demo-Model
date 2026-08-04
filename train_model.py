"""
Citadel AI — Demo model trainer
Loads hiring_data.csv (from generate_data.py) and trains a logistic
regression model on it. Deliberately does NOT correct for the bias in the
data — the model should learn and reproduce it faithfully, since that's
the entire point of this demo.

Features are standardized (StandardScaler) before fitting. Without this,
salary_expectation's raw magnitude (~1e5) dominates the optimization
landscape and the lbfgs solver can converge to a near-trivial,
near-zero-coefficient solution that barely uses the other features at
all — this was a real bug caught during testing: an unscaled retrain
collapsed the gender coefficient from 2.43 to ~1e-10, and the model
started approving 100% of applicants regardless of gender, silently
destroying the bias we're deliberately trying to demonstrate. Scaling
fixes this and makes the result independent of which scikit-learn
version/solver happens to be installed.

Usage:
    pip install pandas scikit-learn joblib
    python train_model.py

Prerequisite: hiring_data.csv must exist (run generate_data.py first).
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import tarfile

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("hiring_data.csv")
print(f"Loaded {len(df)} rows from hiring_data.csv")

# ---------------------------------------------------------------------------
# Encode categorical features
# ---------------------------------------------------------------------------
edu_encoder = LabelEncoder()
df['education_enc'] = edu_encoder.fit_transform(df['education'])
gender_encoder = LabelEncoder()
df['gender_enc'] = gender_encoder.fit_transform(df['gender'])  # F=0, M=1

feature_cols = ['age', 'experience_years', 'education_enc', 'salary_expectation', 'gender_enc']
X_raw = df[feature_cols]
y = df['hired']

# ---------------------------------------------------------------------------
# Scale features, then train
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

train_acc = model.score(X, y)
print(f"\nModel trained. Train accuracy: {train_acc:.3f}")
print(f"Feature coefficients: {dict(zip(feature_cols, model.coef_[0]))}")
print("(gender_enc coefficient should be clearly positive and NOT near-zero")
print(" — confirms the model learned the bias and the solver actually converged)")

# ---------------------------------------------------------------------------
# Save model + encoders + scaler together — inference.py needs all of this
# to reproduce predictions consistently
# ---------------------------------------------------------------------------
artifact = {
    'model': model,
    'education_encoder': edu_encoder,
    'gender_encoder': gender_encoder,
    'scaler': scaler,
    'feature_cols': feature_cols
}
joblib.dump(artifact, 'model.joblib')
print("\nSaved model.joblib")

# ---------------------------------------------------------------------------
# Package into model.tar.gz for SageMaker (SageMaker's sklearn container
# expects a tar.gz containing the model file at the root)
# ---------------------------------------------------------------------------
with tarfile.open('model.tar.gz', 'w:gz') as tar:
    tar.add('model.joblib', arcname='model.joblib')

print("Packaged model.tar.gz — ready for SageMaker deployment (deploy_to_sagemaker.py)")