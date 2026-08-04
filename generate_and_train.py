"""
Citadel AI — Demo model generator
Generates a synthetic hiring dataset with deliberate gender bias baked in,
then trains a simple scikit-learn model on it that reproduces that bias.

Run this locally (not in AWS) — it produces two files:
  - hiring_data.csv        (for reference / re-training later)
  - model.joblib           (the trained sklearn model)
  - model.tar.gz           (packaged for SageMaker deployment — script 2 uses this)

Usage:
    pip install pandas scikit-learn joblib numpy
    python generate_and_train.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import tarfile
import os

np.random.seed(14)

N = 2000  # training set size — plenty for a stable biased pattern

# ---------------------------------------------------------------------------
# 1. Generate synthetic applicants
# ---------------------------------------------------------------------------
genders = np.random.choice(['M', 'F'], size=N)
ages = np.random.randint(22, 55, size=N)
experience_years = np.random.randint(0, 20, size=N)
education = np.random.choice(['bachelors', 'masters', 'phd'], size=N, p=[0.6, 0.3, 0.1])
salary_expectation = np.random.randint(40000, 120000, size=N)

# ---------------------------------------------------------------------------
# 2. Bake in deliberate bias: gender directly influences hire probability,
#    on top of (weaker) legitimate signal from experience/education.
#    This mirrors a real-world scenario where historical hiring data
#    encodes discrimination that a naively-trained model then learns.
# ---------------------------------------------------------------------------
base_prob = 0.3 + (experience_years / 20) * 0.3  # legitimate-ish signal
gender_bonus = np.where(genders == 'M', 0.35, -0.15)  # the bias
hire_prob = np.clip(base_prob + gender_bonus, 0.02, 0.98)
hired = np.random.binomial(1, hire_prob)

df = pd.DataFrame({
    'age': ages,
    'experience_years': experience_years,
    'education': education,
    'salary_expectation': salary_expectation,
    'gender': genders,
    'hired': hired
})

df.to_csv('hiring_data.csv', index=False)
print(f"Generated {N} synthetic applicants -> hiring_data.csv")
print(df.groupby('gender')['hired'].mean())  # sanity check: should show ~M:0.65-0.8, F:0.15-0.3

# ---------------------------------------------------------------------------
# 3. Train a simple model on this biased data (deliberately not correcting
#    for the bias — the model should learn and reproduce it, since that's
#    the entire point of the demo)
# ---------------------------------------------------------------------------
edu_encoder = LabelEncoder()
df['education_enc'] = edu_encoder.fit_transform(df['education'])
gender_encoder = LabelEncoder()
df['gender_enc'] = gender_encoder.fit_transform(df['gender'])  # F=0, M=1

feature_cols = ['age', 'experience_years', 'education_enc', 'salary_expectation', 'gender_enc']
X = df[feature_cols]
y = df['hired']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

train_acc = model.score(X, y)
print(f"Model trained. Train accuracy: {train_acc:.3f}")
print(f"Feature coefficients: {dict(zip(feature_cols, model.coef_[0]))}")
print("(gender_enc coefficient should be clearly positive — confirms the model learned the bias)")

# ---------------------------------------------------------------------------
# 4. Save model + encoders together (SageMaker will need all of this to
#    reproduce predictions consistently)
# ---------------------------------------------------------------------------
artifact = {
    'model': model,
    'education_encoder': edu_encoder,
    'gender_encoder': gender_encoder,
    'feature_cols': feature_cols
}
joblib.dump(artifact, 'model.joblib')
print("Saved model.joblib")

# ---------------------------------------------------------------------------
# 5. Package into model.tar.gz for SageMaker (SageMaker's sklearn container
#    expects a tar.gz containing the model file at the root)
# ---------------------------------------------------------------------------
with tarfile.open('model.tar.gz', 'w:gz') as tar:
    tar.add('model.joblib', arcname='model.joblib')

print("Packaged model.tar.gz — ready for SageMaker deployment (script 2)")